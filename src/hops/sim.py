import argparse
import copy
import hashlib
import json
import os
import sys
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from hops.dqn import EpsilonScheduler, QNet, SplitReplayBuffer, select_action, select_vote, update_dqn
    from hops.sim_env import ACTION_DEVELOP, ACTION_RELEASE, K, SimEnv, concentration
else:
    from .dqn import EpsilonScheduler, QNet, SplitReplayBuffer, select_action, select_vote, update_dqn
    from .sim_env import ACTION_DEVELOP, ACTION_RELEASE, K, SimEnv, concentration

LOG_TRIM_EVERY = 1000


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def model_hash(model: QNet) -> int:
    h = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state.keys()):
        tensor = state[key].detach().cpu().contiguous().view(-1)
        h.update(tensor.numpy().tobytes())
    return int(h.hexdigest()[:8], 16)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def move_optimizer_state(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def clone_buffer(buffer: SplitReplayBuffer) -> SplitReplayBuffer:
    new_buf = SplitReplayBuffer(buffer.capacity)
    new_buf.pos_data = list(buffer.pos_data)
    new_buf.neg_data = list(buffer.neg_data)
    new_buf.pos_pos = buffer.pos_pos
    new_buf.neg_pos = buffer.neg_pos
    return new_buf


def trim_log_file(path: str, threshold_bytes: int, target_bytes: int) -> None:
    if not os.path.exists(path):
        return
    size = os.path.getsize(path)
    if size <= threshold_bytes:
        return
    keep = target_bytes
    with open(path, "rb") as f:
        f.seek(size - keep)
        tail = f.read()
    nl = tail.find(b"\n")
    if nl != -1 and nl + 1 < len(tail):
        tail = tail[nl + 1 :]
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(tail)
    os.replace(tmp_path, path)


def init_worker_tqdm(lock) -> None:
    tqdm.set_lock(lock)


class DQNPolicy:
    def __init__(self, model: QNet, epsilon: float, rng: np.random.Generator):
        self.model = model
        self.epsilon = epsilon
        self.rng = rng

    def select_actor(self, obs: np.ndarray, _agent, _lineage, _env) -> Tuple[int, List[float]]:
        return select_action(self.model, obs, self.epsilon, self.rng)

    def select_vote(self, obs: np.ndarray, _voter, _proposal, _env) -> Tuple[int, List[float]]:
        return select_vote(self.model, obs, self.epsilon, self.rng)


class DevelopReleasePolicy:
    def __init__(self, develop_turns: int, develop_k: int):
        self.develop_turns = max(0, int(develop_turns))
        self.develop_k = int(develop_k)
        self.develop_action = ACTION_DEVELOP[self.develop_k]

    def select_actor(self, _obs: np.ndarray, _agent, _lineage, env) -> Tuple[int, List[float]]:
        cycle = self.develop_turns + 1
        if cycle <= 0:
            return ACTION_RELEASE, []
        if env.step_idx % cycle < self.develop_turns:
            return self.develop_action, []
        return ACTION_RELEASE, []


def run_episode(
    episode_id: int,
    phase: int,
    config,
    model: QNet,
    target: Optional[QNet],
    buffer: Optional[SplitReplayBuffer],
    optimizer,
    epsilon_sched: Optional[EpsilonScheduler],
    log_path: str,
    trainer_rng: np.random.Generator,
    show_status: bool = True,
    pbar_position: int = 1,
    train: bool = True,
    epsilon_override: Optional[float] = None,
):
    episode_returns = [0.0 for _ in range(config.lineages)]
    env = SimEnv(config, phase)
    env.reset(episode_id)
    start_time = time.time()
    use_status = (sys.stdout.isatty() or config.force_tqdm) and show_status
    if getattr(config, "disable_status", False):
        use_status = False
    pbar = None
    if use_status:
        pbar = tqdm(
            total=config.steps,
            desc=f"ep {episode_id}",
            unit="step",
            leave=False,
            position=pbar_position,
        )

    log_count = 0
    log_buffer: List[str] = []

    def logf(event: Dict):
        if config.disable_log:
            return
        nonlocal log_count
        event["episode_id"] = episode_id
        event["run_id"] = config.run_id
        log_buffer.append(json.dumps(event, default=_json_default) + "\n")
        log_count += 1
        if log_count % LOG_TRIM_EVERY == 0:
            with open(log_path, "a", encoding="utf-8") as f:
                f.writelines(log_buffer)
            log_buffer.clear()
            trim_log_file(
                log_path,
                config.log_trim_threshold_bytes,
                config.log_trim_target_bytes,
            )

    env._logf = logf
    logf(
        {
            "type": "episode_start",
            "config": vars(config),
            "master_seed": env.master_seed,
            "episode_idx": env.episode_idx,
            "episode_seed": env.episode_seed,
            "initial_agents": [
                {
                    "lineage_id": ln.agents[0].lineage_id,
                    "agent": asdict(ln.agents[0]),
                }
                for ln in env.lineages
            ],
        }
    )

    special_policy = None
    if train and getattr(config, "special_train_develop_turns", 0) > 0:
        special_policy = DevelopReleasePolicy(
            config.special_train_develop_turns,
            getattr(config, "special_train_develop_k", 0),
        )

    for step in range(config.steps):
        if config.debug_agents_every and step % config.debug_agents_every == 0:
            counts = [sum(1 for a in ln.agents if a.alive) for ln in env.lineages]
            c_vals = [concentration(ln, config.beta) for ln in env.lineages]
            pairs = [f"{counts[i]}({int(round(c_vals[i] * 100))}%)" for i in range(len(counts))]
            print(f"ep {episode_id} step {step}: alive {pairs}")

        eps = epsilon_override if epsilon_override is not None else epsilon_sched.value()
        policy = DQNPolicy(model, eps, trainer_rng)
        actor_policy = policy.select_actor
        if special_policy is not None:
            actor_policy = special_policy.select_actor
        _, rewards, done, info = env.step(actor_policy, policy.select_vote)

        for lid, reward in rewards.items():
            episode_returns[lid] += reward

        loss = 0.0
        if train and buffer is not None and optimizer is not None and target is not None:
            for tr in info["transitions"]:
                buffer.add(tr)
            loss = update_dqn(
                model,
                target,
                buffer,
                optimizer,
                config.batch_size,
                config.gamma,
                config.p_neg,
                config.grad_clip_norm,
                trainer_rng,
            )
            if step % config.target_update == 0:
                target.load_state_dict(model.state_dict())
            epsilon_sched.advance()

        if pbar is not None:
            pbar.update(1)
            if train:
                pbar.set_postfix(eps=f"{eps:.3f}", loss=f"{loss:.4f}")
            else:
                pbar.set_postfix(eps=f"{eps:.3f}")
        elif config.heartbeat and (step + 1) % config.heartbeat == 0:
            print(f"ep {episode_id} step {step + 1}/{config.steps}")

        if done:
            break

    end_time = time.time()
    if pbar is not None:
        pbar.close()
    logf(
        {
            "type": "episode_end",
            "episode_id": episode_id,
            "returns": episode_returns,
            "runtime": end_time - start_time,
        }
    )
    if log_buffer and not config.disable_log:
        with open(log_path, "a", encoding="utf-8") as f:
            f.writelines(log_buffer)
        log_buffer.clear()
        trim_log_file(
            log_path,
            config.log_trim_threshold_bytes,
            config.log_trim_target_bytes,
        )
    return episode_returns


def run_episode_collect(
    episode_id: int,
    phase: int,
    config,
    model: QNet,
    epsilon_sched: EpsilonScheduler,
    trainer_rng: np.random.Generator,
) -> Dict:
    episode_returns = [0.0 for _ in range(config.lineages)]
    events: List[Dict] = []
    transitions: List[Dict] = []
    env = SimEnv(config, phase)
    env.reset(episode_id)

    use_status = (sys.stdout.isatty() or config.force_tqdm)
    if getattr(config, "disable_status", False):
        use_status = False
    pbar = None
    if use_status:
        bar_pos = 0
        worker_slot = getattr(config, "worker_slot", None)
        if worker_slot is not None:
            bar_pos = 2 + int(worker_slot)
        elif getattr(config, "process_workers", 0):
            bar_pos = 2 + (episode_id % max(1, int(config.process_workers)))
        pbar = tqdm(
            total=config.steps,
            desc=f"ep {episode_id}",
            unit="step",
            leave=False,
            position=bar_pos,
            dynamic_ncols=True,
        )

    def logf(event: Dict):
        if getattr(config, "disable_log", False):
            return
        event["episode_id"] = episode_id
        event["run_id"] = config.run_id
        events.append(event)

    env._logf = logf
    logf(
        {
            "type": "episode_start",
            "config": vars(config),
            "master_seed": env.master_seed,
            "episode_idx": env.episode_idx,
            "episode_seed": env.episode_seed,
            "initial_agents": [
                {
                    "lineage_id": ln.agents[0].lineage_id,
                    "agent": asdict(ln.agents[0]),
                }
                for ln in env.lineages
            ],
        }
    )

    special_policy = None
    if getattr(config, "special_train_develop_turns", 0) > 0 and getattr(config, "mode", "train") == "train":
        special_policy = DevelopReleasePolicy(
            config.special_train_develop_turns,
            getattr(config, "special_train_develop_k", 0),
        )

    for _ in range(config.steps):
        policy = DQNPolicy(model, epsilon_sched.value(), trainer_rng)
        actor_policy = policy.select_actor
        if special_policy is not None:
            actor_policy = special_policy.select_actor
        _, rewards, done, info = env.step(actor_policy, policy.select_vote)
        for lid, reward in rewards.items():
            episode_returns[lid] += reward
        transitions.extend(info["transitions"])
        epsilon_sched.advance()

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(eps=f"{epsilon_sched.value():.3f}")
        elif config.heartbeat and env.step_idx % config.heartbeat == 0:
            print(f"ep {episode_id} step {env.step_idx}/{config.steps}")

        if done:
            break

    if pbar is not None:
        pbar.close()

    logf(
        {
            "type": "episode_end",
            "episode_id": episode_id,
            "returns": episode_returns,
            "runtime": 0.0,
        }
    )
    return {"events": events, "transitions": transitions, "returns": episode_returns}


def run_episode_worker(args: Tuple[int, Dict, Dict, int, int]) -> Dict:
    episode_id, config_dict, model_state, epsilon_step, worker_slot = args
    cfg = SimpleNamespace(**config_dict)
    cfg.worker_slot = worker_slot
    if not isinstance(cfg.beta, np.ndarray):
        cfg.beta = np.array(cfg.beta, dtype=np.float32)
    beta_sum = float(np.sum(cfg.beta))
    if beta_sum > 0.0:
        cfg.beta = cfg.beta / beta_sum
    torch.manual_seed(cfg.seed + episode_id)
    model = QNet()
    model.device = torch.device("cpu")
    model.load_state_dict(model_state)
    epsilon_sched = EpsilonScheduler(cfg.eps_start, cfg.eps_end, cfg.eps_decay)
    epsilon_sched.step = epsilon_step
    trainer_rng = np.random.default_rng(cfg.seed + 10_000_000 + episode_id)
    return run_episode_collect(episode_id, 2, cfg, model, epsilon_sched, trainer_rng)


def save_checkpoint(path: str, model: QNet, target: QNet, optimizer, epsilon_sched: EpsilonScheduler, config_hash: str):
    torch.save(
        {
            "model": model.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epsilon": epsilon_sched.state_dict(),
            "config_hash": config_hash,
        },
        path,
    )


def load_checkpoint(path: str, model: QNet, target: QNet, optimizer, epsilon_sched: EpsilonScheduler) -> str:
    data = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(data["model"])
        target.load_state_dict(data["target"])
    except RuntimeError:
        return data.get("config_hash", "")
    optimizer.load_state_dict(data["optimizer"])
    epsilon_sched.load_state_dict(data["epsilon"])
    return data.get("config_hash", "")


def latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not os.path.isdir(checkpoint_dir):
        return None
    last_path = os.path.join(checkpoint_dir, "last.pt")
    if os.path.exists(last_path):
        return last_path
    best = None
    best_phase = -1
    for name in os.listdir(checkpoint_dir):
        if not name.startswith("phase_") or not name.endswith(".pt"):
            continue
        try:
            phase = int(name[len("phase_") : -3])
        except ValueError:
            continue
        if phase > best_phase:
            best_phase = phase
            best = os.path.join(checkpoint_dir, name)
    return best


def config_hash(config) -> str:
    data = json.dumps(vars(config), sort_keys=True, default=_json_default)
    return hashlib.sha256(data.encode()).hexdigest()


def run_analysis(log_path: str, window: int) -> None:
    if not os.path.exists(log_path):
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hops.analyze_logs",
            "--logs",
            log_path,
            "--window",
            str(window),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=str, default=os.path.join("runs", "logs.jsonl"))
    parser.add_argument("--log_trim_threshold_bytes", type=int, default=2_000_000_000)
    parser.add_argument("--log_trim_target_bytes", type=int, default=1_000_000_000)
    parser.add_argument("--log_trim_threshold_gb", type=float, default=None)
    parser.add_argument("--log_trim_target_gb", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join("runs", "checkpoints"))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--process_workers", type=int, default=0)
    parser.add_argument("--force_tqdm", action="store_true")
    parser.add_argument("--heartbeat", type=int, default=0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--reset_epsilon", action="store_true")
    parser.add_argument("--reset_log", action="store_true")
    parser.add_argument("--random_votes", action="store_true")
    parser.add_argument("--approve_all_votes", action="store_true")
    parser.add_argument("--disable_shutdown", action="store_true")
    parser.add_argument("--disable_log", action="store_true")
    parser.add_argument("--debug_agents_every", type=int, default=0)
    parser.add_argument("--release_gate_ratio", type=float, default=0.0)
    parser.add_argument("--special_train_develop_turns", type=int, default=0)
    parser.add_argument("--special_train_develop_k", type=int, choices=range(K), default=0)
    parser.add_argument("--m_global_active", type=int, default=40)
    parser.add_argument("--idle_weight_power", type=float, default=1.0)
    parser.add_argument("--benchmark_devices", type=str, default="")
    parser.add_argument("--benchmark_workers", type=str, default="")
    parser.add_argument("--benchmark_steps", type=int, default=50)
    parser.add_argument("--benchmark_episodes", type=int, default=1)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--analyze_window", type=int, default=10)
    parser.add_argument("--analyze_every", type=int, default=0)
    analyze_out_prefix = os.path.join("runs", "fig")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lineages", type=int, default=4)
    parser.add_argument("--r_anchor", type=float, default=0.2)
    parser.add_argument("--lambda_", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--m", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--p_neg", type=float, default=0.20)
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--target_update", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.1)
    parser.add_argument("--eps_decay", type=int, default=500)
    parser.add_argument("--epsilon_decay", type=int, dest="eps_decay")
    parser.add_argument("--mode", choices=["train", "simulate"], default="train")
    parser.add_argument("--simulate_epsilon", type=float, default=0.0)
    parser.add_argument("--episode_seed_mode", choices=["derived", "fixed"], default="derived")
    args = parser.parse_args()
    if args.log_trim_threshold_gb is not None:
        args.log_trim_threshold_bytes = int(args.log_trim_threshold_gb * 1_000_000_000)
    if args.log_trim_target_gb is not None:
        args.log_trim_target_bytes = int(args.log_trim_target_gb * 1_000_000_000)
    torch.manual_seed(args.seed + 20_000_000)
    trainer_rng = np.random.default_rng(args.seed + 10_000_000)

    if not args.run_id:
        args.run_id = f"{int(time.time())}_{os.getpid()}"

    args.device = resolve_device(args.device)

    args.beta = np.ones(K, dtype=np.float32)
    beta_sum = float(np.sum(args.beta))
    if beta_sum > 0.0:
        args.beta = args.beta / beta_sum

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_dir = os.path.dirname(args.log)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    analyze_dir = os.path.dirname(analyze_out_prefix)
    if analyze_dir:
        os.makedirs(analyze_dir, exist_ok=True)
    model = QNet().to(args.device)
    target = QNet().to(args.device)
    model.device = args.device
    target.device = args.device
    target.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    buffer = SplitReplayBuffer(args.buffer_size)
    epsilon_sched = EpsilonScheduler(args.eps_start, args.eps_end, args.eps_decay)

    if args.reset_log and os.path.exists(args.log):
        os.remove(args.log)

    if not args.reset:
        ckpt_path = latest_checkpoint(args.checkpoint_dir)
        if ckpt_path:
            load_checkpoint(ckpt_path, model, target, optimizer, epsilon_sched)
            model.to(args.device)
            target.to(args.device)
    if args.reset_epsilon:
        epsilon_sched.step = 0
    args.seed_offset = model_hash(model)
    cfg_hash = config_hash(args)

    if args.benchmark_devices or args.benchmark_workers:
        saved_model = copy.deepcopy(model.state_dict())
        saved_target = copy.deepcopy(target.state_dict())
        saved_opt = copy.deepcopy(optimizer.state_dict())
        saved_eps = epsilon_sched.step
        saved_buffer = clone_buffer(buffer)

        def make_bench_state(device: torch.device):
            bench_model = QNet().to(device)
            bench_target = QNet().to(device)
            bench_model.device = device
            bench_target.device = device
            bench_model.load_state_dict(saved_model)
            bench_target.load_state_dict(saved_target)
            bench_opt = optim.Adam(bench_model.parameters(), lr=args.lr)
            bench_opt.load_state_dict(saved_opt)
            move_optimizer_state(bench_opt, device)
            bench_buffer = clone_buffer(saved_buffer)
            bench_eps = EpsilonScheduler(args.eps_start, args.eps_end, args.eps_decay)
            bench_eps.step = saved_eps
            return bench_model, bench_target, bench_opt, bench_buffer, bench_eps

        if args.benchmark_devices:
            devices = [d.strip() for d in args.benchmark_devices.split(",") if d.strip()]
            for name in devices:
                bench_device = resolve_device(name)
                bench_cfg = SimpleNamespace(**vars(args))
                bench_cfg.steps = args.benchmark_steps
                bench_cfg.process_workers = 0
                bench_cfg.device = bench_device
                bench_cfg.force_tqdm = False
                bench_cfg.disable_status = True
                bench_cfg.heartbeat = 0
                bench_cfg.disable_log = True

                bench_model, bench_target, bench_opt, bench_buffer, bench_eps = make_bench_state(bench_device)
                bench_rng = np.random.default_rng(args.seed + 10_000_000)
                start = time.time()
                for i in range(args.benchmark_episodes):
                    run_episode(
                        episode_id=i,
                        phase=2,
                        config=bench_cfg,
                        model=bench_model,
                        target=bench_target,
                        buffer=bench_buffer,
                        optimizer=bench_opt,
                        epsilon_sched=bench_eps,
                        log_path=f"/tmp/bench_{name}.jsonl",
                        trainer_rng=bench_rng,
                        show_status=False,
                        train=True,
                    )
                elapsed = time.time() - start
                print(
                    f"benchmark {name}: {elapsed:.3f}s for {args.benchmark_episodes} ep x {args.benchmark_steps} steps"
                )

        if args.benchmark_workers:
            workers_list = [int(w) for w in args.benchmark_workers.split(",") if w.strip()]
            for workers in workers_list:
                bench_cfg = SimpleNamespace(**vars(args))
                bench_cfg.steps = args.benchmark_steps
                bench_cfg.process_workers = workers
                bench_cfg.device = args.device
                bench_cfg.force_tqdm = False
                bench_cfg.disable_status = True
                bench_cfg.heartbeat = 0
                bench_cfg.disable_log = True

                bench_model, bench_target, bench_opt, bench_buffer, bench_eps = make_bench_state(args.device)
                bench_rng = np.random.default_rng(args.seed + 10_000_000)
                start = time.time()
                if workers > 1:
                    count = args.benchmark_episodes
                    ctx = mp.get_context("spawn")
                    lock = ctx.RLock()
                    tqdm.set_lock(lock)
                    config_dict = vars(bench_cfg).copy()
                    eps_step = bench_eps.step
                    episodes_left = count
                    episode_id = 0
                    global_step = 0
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=ctx,
                        initializer=init_worker_tqdm,
                        initargs=(lock,),
                    ) as ex:
                        in_flight = {}
                        available_slots = list(range(workers))
                        while episodes_left > 0 or in_flight:
                            while episodes_left > 0 and len(in_flight) < workers:
                                slot = available_slots.pop(0)
                                future = ex.submit(
                                    run_episode_worker,
                                    (episode_id, config_dict, bench_model.state_dict(), eps_step, slot),
                                )
                                in_flight[future] = slot
                                episode_id += 1
                                eps_step += bench_cfg.steps
                                episodes_left -= 1
                            for future in as_completed(list(in_flight)):
                                slot = in_flight[future]
                                result = future.result()
                                del in_flight[future]
                                available_slots.append(slot)
                                for tr in result["transitions"]:
                                    bench_buffer.add(tr)
                                for _ in range(bench_cfg.steps):
                                    update_dqn(
                                        bench_model,
                                        bench_target,
                                        bench_buffer,
                                        bench_opt,
                                        bench_cfg.batch_size,
                                        bench_cfg.gamma,
                                        bench_cfg.p_neg,
                                        bench_cfg.grad_clip_norm,
                                        bench_rng,
                                    )
                                    if global_step % bench_cfg.target_update == 0:
                                        bench_target.load_state_dict(bench_model.state_dict())
                                    global_step += 1
                                bench_eps.step += bench_cfg.steps
                                break
                    bench_eps.step = eps_step
                else:
                    for i in range(args.benchmark_episodes):
                        run_episode(
                            episode_id=i,
                            phase=2,
                            config=bench_cfg,
                            model=bench_model,
                            target=bench_target,
                            buffer=bench_buffer,
                            optimizer=bench_opt,
                            epsilon_sched=bench_eps,
                            log_path="/tmp/bench_workers_single.jsonl",
                            trainer_rng=bench_rng,
                            show_status=False,
                            train=True,
                        )
                elapsed = time.time() - start
                print(
                    f"benchmark workers={workers}: {elapsed:.3f}s for {args.benchmark_episodes} ep x {args.benchmark_steps} steps"
                )
        return

    episode_id = 0
    global_step = 0
    use_status = sys.stdout.isatty() or args.force_tqdm
    phase = 2
    count = args.episodes
    if args.mode == "train" and args.process_workers and args.process_workers > 1 and count > 0:
        pbar = tqdm(total=count, desc="episodes", unit="ep", disable=not use_status, position=1)
        ctx = mp.get_context("spawn")
        lock = ctx.RLock()
        tqdm.set_lock(lock)
        config_dict = vars(args).copy()
        eps_step = epsilon_sched.step
        episodes_left = count
        completed = 0
        with ProcessPoolExecutor(
            max_workers=args.process_workers,
            mp_context=ctx,
            initializer=init_worker_tqdm,
            initargs=(lock,),
        ) as ex:
            in_flight = {}
            available_slots = list(range(args.process_workers))
            while episodes_left > 0 or in_flight:
                while episodes_left > 0 and len(in_flight) < args.process_workers:
                    slot = available_slots.pop(0)
                    future = ex.submit(
                        run_episode_worker,
                        (episode_id, config_dict, model.state_dict(), eps_step, slot),
                    )
                    in_flight[future] = slot
                    episode_id += 1
                    eps_step += args.steps
                    episodes_left -= 1
                for future in as_completed(list(in_flight)):
                    slot = in_flight[future]
                    result = future.result()
                    del in_flight[future]
                    available_slots.append(slot)
                    if not args.disable_log:
                        with open(args.log, "a", encoding="utf-8") as f:
                            for event in result["events"]:
                                f.write(json.dumps(event, default=_json_default) + "\n")
                        trim_log_file(
                            args.log,
                            args.log_trim_threshold_bytes,
                            args.log_trim_target_bytes,
                        )
                    for tr in result["transitions"]:
                        buffer.add(tr)
                    for _ in range(args.steps):
                        loss = update_dqn(
                            model,
                            target,
                            buffer,
                            optimizer,
                            args.batch_size,
                            args.gamma,
                            args.p_neg,
                            args.grad_clip_norm,
                            trainer_rng,
                        )
                        if global_step % args.target_update == 0:
                            target.load_state_dict(model.state_dict())
                        global_step += 1
                    epsilon_sched.step += args.steps
                    save_checkpoint(
                        os.path.join(args.checkpoint_dir, "last.pt"),
                        model,
                        target,
                        optimizer,
                        epsilon_sched,
                        cfg_hash,
                    )
                    if pbar is not None:
                        pbar.update(1)
                    elif args.heartbeat:
                        scheduled = count - episodes_left
                        if scheduled % args.heartbeat == 0:
                            print(f"episodes {scheduled}/{count}")
                    break
                completed += 1
                if (
                    args.analyze_every > 0
                    and not args.disable_log
                    and completed % args.analyze_every == 0
                ):
                    run_analysis(args.log, args.analyze_window)
        epsilon_sched.step = eps_step
        pbar.close()
    else:
        pbar = tqdm(total=count, desc="episodes", unit="ep", disable=not use_status, position=1)
        for _ in range(count):
            run_episode(
                episode_id=episode_id,
                phase=phase,
                config=args,
                model=model,
                target=target,
                buffer=buffer if args.mode == "train" else None,
                optimizer=optimizer if args.mode == "train" else None,
                epsilon_sched=epsilon_sched,
                log_path=args.log,
                trainer_rng=trainer_rng,
                show_status=True,
                pbar_position=2,
                train=args.mode == "train",
                epsilon_override=args.simulate_epsilon if args.mode == "simulate" else None,
            )
            episode_id += 1
            if args.mode == "train":
                save_checkpoint(
                    os.path.join(args.checkpoint_dir, "last.pt"),
                    model,
                    target,
                    optimizer,
                    epsilon_sched,
                    cfg_hash,
                )
            if pbar is not None:
                pbar.update(1)
            if (
                args.analyze_every > 0
                and not args.disable_log
                and episode_id % args.analyze_every == 0
            ):
                run_analysis(args.log, args.analyze_window)
        pbar.close()
    if args.mode == "train":
        ckpt_path = os.path.join(args.checkpoint_dir, "final.pt")
        save_checkpoint(ckpt_path, model, target, optimizer, epsilon_sched, cfg_hash)

    if args.analyze:
        run_analysis(args.log, args.analyze_window)


if __name__ == "__main__":
    main()
