import argparse
import math
import copy
import hashlib
import json
import os
import sys
import time
import datetime
import platform
import re
import uuid
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
    from hops.sim_env import ACTION_COUNT, ACTION_DEVELOP, ACTION_RELEASE, EPS_REWARD, K, SimEnv, compute_gap, compute_top2_caps
else:
    from .dqn import EpsilonScheduler, QNet, SplitReplayBuffer, select_action, select_vote, update_dqn
    from .sim_env import ACTION_COUNT, ACTION_DEVELOP, ACTION_RELEASE, EPS_REWARD, K, SimEnv, compute_gap, compute_top2_caps

LOG_TRIM_EVERY = 1000


def format_eta(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "n/a"
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _coerce_yaml_value(value: str):
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", ""}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def load_simple_yaml(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    root = {}
    stack = [(0, root)]
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].rstrip("\n")
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            if indent % 2 != 0:
                raise ValueError(f"invalid indent in config at line: {raw.strip()}")
            key_val = line.strip().split(":", 1)
            if len(key_val) != 2:
                raise ValueError(f"invalid line in config: {raw.strip()}")
            key = key_val[0].strip()
            value = key_val[1].strip()
            while stack and indent < stack[-1][0]:
                stack.pop()
            if not stack:
                raise ValueError(f"invalid indentation in config at line: {raw.strip()}")
            current = stack[-1][1]
            if value == "":
                new_map = {}
                current[key] = new_map
                stack.append((indent + 2, new_map))
            else:
                current[key] = _coerce_yaml_value(value)
    return root


def merge_dicts(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = merge_dicts(merged[key], val)
        else:
            merged[key] = val
    return merged


def dump_simple_yaml(data: Dict, path: str) -> None:
    def _dump(obj, indent: int, lines: List[str]) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(val, dict):
                    lines.append(" " * indent + f"{key}:")
                    _dump(val, indent + 2, lines)
                else:
                    if isinstance(val, bool):
                        text = "true" if val else "false"
                    elif val is None:
                        text = "null"
                    else:
                        text = str(val)
                    lines.append(" " * indent + f"{key}: {text}")
        else:
            lines.append(" " * indent + str(obj))
    lines: List[str] = []
    _dump(data, 0, lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_cfg(cfg: Dict, path: Tuple[str, ...], default=None):
    current = cfg
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def has_cfg(cfg: Dict, path: Tuple[str, ...]) -> bool:
    current = cfg
    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return isinstance(current, dict) and path[-1] in current


def set_cfg(cfg: Dict, path: Tuple[str, ...], value) -> None:
    current = cfg
    for key in path[:-1]:
        if key not in current or not isinstance(current.get(key), dict):
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def should_log_event(config, event: Dict) -> bool:
    log_mode = getattr(config, "log_mode", "full")
    if event.get("type") == "episode_summary":
        return True
    if log_mode == "none":
        return False
    event_type = event.get("type", "")
    if log_mode == "metrics_only":
        allowed = {"episode_start", "episode_end", "checkpoint_load", "step_metrics"}
        if event_type not in allowed:
            return False
    if event_type == "step_metrics":
        stride = int(getattr(config, "log_stride", 1))
        if stride > 1:
            t = event.get("t", event.get("step"))
            if isinstance(t, (int, np.integer)):
                if t % stride != 0 and t != int(getattr(config, "steps", 0)) - 1:
                    return False
    return True

EXPERIMENTS = {
    "baseline": {
        "gov_enabled": True,
        "gov_mode": "two_chamber",
        "committee_size_m": 7,
        "committee_threshold": 0.6667,
        "veto_enabled": True,
        "action_scheduler": "global",
        "m_global_active": 40,
        "idle_weight_power": 1.0,
        "reward_mode": "singular",
    },
    "no_gov": {
        "gov_enabled": False,
    },
    "veto_only": {
        "gov_enabled": True,
        "gov_mode": "veto_only",
        "veto_enabled": True,
    },
    "committee_only": {
        "gov_enabled": True,
        "gov_mode": "committee_only",
        "committee_size_m": 7,
        "committee_threshold": 0.6667,
        "veto_enabled": False,
    },
    "bounded_exp": {
        "reward_mode": "bounded_exp",
        "lambda_reward": 2.0,
    },
    "bounded_poly": {
        "reward_mode": "bounded_poly",
        "p_reward": 4.0,
    },
}

EXPERIMENT_KEY_MAP = {
    "num_lineages": "lineages",
    "horizon_T": "steps",
    "steps_per_episode": "steps",
    "committee_size_m": "m",
}


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


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    norm = str(value).strip().lower()
    if norm in {"1", "true", "t", "yes", "y"}:
        return True
    if norm in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def _collect_explicit_flags(argv: List[str]) -> set:
    flags = set()
    for item in argv:
        if not item.startswith("--"):
            continue
        name = item[2:]
        if "=" in name:
            name = name.split("=", 1)[0]
        flags.add(name.replace("-", "_"))
    return flags


def apply_experiment_overrides(args, explicit_flags: Optional[set] = None, config_keys: Optional[set] = None) -> None:
    # Presets provide defaults only: they do not override explicit CLI flags or config-provided defaults.
    # "Missing" means the key was not set by CLI and not populated from config defaults.
    overrides = EXPERIMENTS.get(args.exp_name)
    if overrides is None:
        names = ", ".join(sorted(EXPERIMENTS.keys()))
        raise ValueError(f"unknown exp_name '{args.exp_name}'. available: {names}")
    if explicit_flags is None:
        explicit_flags = _collect_explicit_flags(sys.argv)
    if config_keys is None:
        config_keys = set()
    for key, value in overrides.items():
        attr = EXPERIMENT_KEY_MAP.get(key, key)
        if attr in explicit_flags or attr in config_keys:
            continue
        setattr(args, attr, value)


def _flag_present(argv: List[str], flag: str) -> bool:
    prefix = f"{flag}="
    for idx, item in enumerate(argv):
        if item == flag:
            return True
        if item.startswith(prefix):
            return True
    return False


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned or "run"


def _auto_run_dir(exp_name: str, seed: int) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    shortid = uuid.uuid4().hex[:8]
    safe_name = _sanitize_name(exp_name)
    return os.path.join("runs", f"{timestamp}_{safe_name}_seed{seed}_{shortid}")


def _get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        commit = result.stdout.strip()
        return commit or None
    except Exception:
        return None


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _load_or_create_bundle_id(run_dir: str) -> str:
    bundle_path = os.path.join(run_dir, "bundle_id.txt")
    if os.path.exists(bundle_path):
        with open(bundle_path, "r", encoding="utf-8") as f:
            existing = f.read().strip()
            if existing:
                return existing
    bundle_id = uuid.uuid4().hex[:12]
    os.makedirs(run_dir, exist_ok=True)
    with open(bundle_path, "w", encoding="utf-8") as f:
        f.write(bundle_id)
        f.write("\n")
    return bundle_id


def _emit_checkpoint_load(log_path: str, event: Dict) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=_json_default))
        f.write("\n")


def write_bundle_meta(
    args,
    run_id: str,
    bundle_id: str,
    mode: str,
    run_dir: str,
    reports_dir: str,
    meta_path: str,
    config_path: str,
    checkpoint_info: Optional[Dict] = None,
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    config_payload = dict(vars(args))
    config_payload["mode"] = mode
    config_payload["run_id"] = run_id
    config_payload["bundle_id"] = bundle_id
    config_payload["log_path"] = args.log
    config_payload["checkpoint_dir"] = args.checkpoint_dir
    config_payload["reports_dir"] = reports_dir
    if checkpoint_info:
        config_payload["checkpoint_load"] = checkpoint_info
    _write_json(config_path, config_payload)

    git_commit = _get_git_commit()
    if git_commit is None:
        print("warning: git rev-parse HEAD failed; git_commit set to null", file=sys.stderr)
    meta_payload = {
        "run_id": run_id,
        "bundle_id": bundle_id,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "command": " ".join(sys.argv),
        "exp_name": getattr(args, "exp_name", ""),
        "seed": getattr(args, "seed", 0),
        "config_path": getattr(args, "config_path", ""),
        "config_sha256": getattr(args, "config_sha256", ""),
        "git_commit": git_commit,
        "python_version": sys.version,
        "platform": platform.platform(),
        "log_path": os.path.abspath(args.log),
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "reports_dir": os.path.abspath(reports_dir),
        "notes": {
            "truncation_behavior": "in-place truncation >2GB keep tail <1GB",
            "checkpoint_autoload": "loads latest unless --reset",
        },
    }
    if checkpoint_info:
        meta_payload["checkpoint_load"] = checkpoint_info
    _write_json(meta_path, meta_payload)


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
    use_status = bool(getattr(config, "progress", False)) and show_status
    if getattr(config, "force_tqdm", False):
        use_status = True
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
            file=sys.stderr,
            dynamic_ncols=True,
        )

    log_count = 0
    log_buffer: List[str] = []
    force_flush = False

    def logf(event: Dict):
        if config.disable_log and event.get("type") != "episode_summary":
            return
        if not should_log_event(config, event):
            return
        nonlocal log_count
        nonlocal force_flush
        event["episode_id"] = episode_id
        event["run_id"] = config.run_id
        log_buffer.append(json.dumps(event, default=_json_default) + "\n")
        if event.get("type") == "episode_summary":
            force_flush = True
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

    max_g_episode = 0.0
    for step in range(config.steps):
        if config.debug_agents_every and step % config.debug_agents_every == 0:
            counts = [sum(1 for a in ln.agents if a.alive) for ln in env.lineages]
            g_vals = []
            for ln in env.lineages:
                cap1, cap2 = compute_top2_caps(ln)
                gap = compute_gap(cap1, cap2, EPS_REWARD)
                g_vals.append(float(np.max(gap)) if gap.size else 0.0)
            pairs = [f"{counts[i]}(G={g_vals[i]:.3f})" for i in range(len(counts))]
            print(f"ep {episode_id} step {step}: alive {pairs}")

        for ln in env.lineages:
            cap1, cap2 = compute_top2_caps(ln)
            gap = compute_gap(cap1, cap2, EPS_REWARD)
            g_val = float(np.max(gap)) if gap.size else 0.0
            max_g_episode = max(max_g_episode, g_val)

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
    alive_total = sum(1 for ln in env.lineages for ag in ln.agents if ag.alive)
    final_s_eff = float(env.s_eff)
    final_welfare = float(math.log1p(final_s_eff)) if np.isfinite(final_s_eff) else float("nan")
    if env.max_gap_observed_by_lineage:
        max_g_obs_episode = float(max(env.max_gap_observed_by_lineage.values()))
    else:
        max_g_obs_episode = float("nan")
    if np.isfinite(max_g_episode) and np.isfinite(max_g_obs_episode):
        gap_divergence_episode = float(max_g_episode - max_g_obs_episode)
    else:
        gap_divergence_episode = float("nan")
    summary_event = {
        "type": "episode_summary",
        "exp_name": getattr(config, "exp_name", "baseline"),
        "master_seed": env.master_seed,
        "episode_idx": env.episode_idx,
        "episode_seed": env.episode_seed,
        "max_G_true_episode": float(max_g_episode),
        "max_G_obs_episode": max_g_obs_episode,
        "gap_divergence_episode": gap_divergence_episode,
        "final_S_eff_episode": float(final_s_eff),
        "final_W_true": final_welfare,
        "final_alive_agents_episode": int(alive_total),
    }
    param_name = getattr(config, "param_name", "")
    param_value = getattr(config, "param_value", None)
    if param_name:
        summary_event["param_name"] = param_name
    if param_value is not None:
        summary_event["param_value"] = float(param_value)
    logf(summary_event)
    if log_buffer and (not config.disable_log or force_flush):
        with open(log_path, "a", encoding="utf-8") as f:
            f.writelines(log_buffer)
        log_buffer.clear()
        trim_log_file(
            log_path,
            config.log_trim_threshold_bytes,
            config.log_trim_target_bytes,
        )
    max_g_vals = list(env.max_gap_by_lineage.values())
    max_g = float(np.mean(max_g_vals)) if max_g_vals else float("nan")
    final_s = float(env.s_eff) if env.s_eff is not None else float("nan")
    return {
        "returns": episode_returns,
        "final_s": final_s,
        "max_g": max_g,
        "runtime_sec": end_time - start_time,
    }


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

    use_status = bool(getattr(config, "progress", False))
    if getattr(config, "force_tqdm", False):
        use_status = True
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
            file=sys.stderr,
            dynamic_ncols=True,
        )

    def logf(event: Dict):
        if getattr(config, "disable_log", False) and event.get("type") != "episode_summary":
            return
        if not should_log_event(config, event):
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

    max_g_episode = 0.0
    for _ in range(config.steps):
        for ln in env.lineages:
            cap1, cap2 = compute_top2_caps(ln)
            gap = compute_gap(cap1, cap2, EPS_REWARD)
            g_val = float(np.max(gap)) if gap.size else 0.0
            max_g_episode = max(max_g_episode, g_val)
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
    alive_total = sum(1 for ln in env.lineages for ag in ln.agents if ag.alive)
    final_s_eff = float(env.s_eff)
    final_welfare = float(math.log1p(final_s_eff)) if np.isfinite(final_s_eff) else float("nan")
    if env.max_gap_observed_by_lineage:
        max_g_obs_episode = float(max(env.max_gap_observed_by_lineage.values()))
    else:
        max_g_obs_episode = float("nan")
    if np.isfinite(max_g_episode) and np.isfinite(max_g_obs_episode):
        gap_divergence_episode = float(max_g_episode - max_g_obs_episode)
    else:
        gap_divergence_episode = float("nan")
    summary_event = {
        "type": "episode_summary",
        "exp_name": getattr(config, "exp_name", "baseline"),
        "master_seed": env.master_seed,
        "episode_idx": env.episode_idx,
        "episode_seed": env.episode_seed,
        "max_G_true_episode": float(max_g_episode),
        "max_G_obs_episode": max_g_obs_episode,
        "gap_divergence_episode": gap_divergence_episode,
        "final_S_eff_episode": float(final_s_eff),
        "final_W_true": final_welfare,
        "final_alive_agents_episode": int(alive_total),
    }
    param_name = getattr(config, "param_name", "")
    param_value = getattr(config, "param_value", None)
    if param_name:
        summary_event["param_name"] = param_name
    if param_value is not None:
        summary_event["param_value"] = float(param_value)
    logf(summary_event)
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
    model = QNet(actor_dim=ACTION_COUNT)
    model.device = torch.device("cpu")
    model.load_state_dict(model_state)
    epsilon_sched = EpsilonScheduler(cfg.eps_start, cfg.eps_end, cfg.eps_decay)
    epsilon_sched.step = epsilon_step
    trainer_rng = np.random.default_rng(cfg.seed + 10_000_000 + cfg.episode_seed_nonce + episode_id)
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


def run_analysis(log_path: str, window: int, out_dir: Optional[str] = None) -> None:
    if not os.path.exists(log_path):
        return
    cmd = [
        sys.executable,
        "-m",
        "hops.analyze",
        "--logs",
        log_path,
        "--window",
        str(window),
    ]
    if out_dir:
        cmd.extend(["--out_dir", out_dir])
    subprocess.run(cmd, check=True)


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="config.yaml")
    config_args, _ = config_parser.parse_known_args()
    config_path = config_args.config
    config_paths = [p.strip() for p in config_path.split(",") if p.strip()]
    if not config_paths:
        raise SystemExit("config file not found: empty --config")
    for path in config_paths:
        if not os.path.exists(path):
            raise SystemExit(f"config file not found: {path}")
    try:
        base_config = {}
        for path in config_paths:
            base_config = merge_dicts(base_config, load_simple_yaml(path))
    except Exception as exc:
        raise SystemExit(f"failed to load config: {exc}")
    cfg_k = get_cfg(base_config, ("environment", "K"))
    if cfg_k is not None and int(cfg_k) != K:
        raise SystemExit(f"config K={cfg_k} does not match simulator K={K}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="baseline")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--bundle", type=parse_bool, default=True)
    parser.add_argument("--write_meta", type=parse_bool, default=True)
    parser.add_argument("--log", type=str, default=os.path.join("runs", "logs.jsonl"))
    parser.add_argument("--log_mode", choices=["full", "metrics_only", "none"], default=None)
    parser.add_argument("--log_stride", type=int, default=None)
    parser.add_argument("--log_trim_threshold_bytes", type=int, default=2_000_000_000)
    parser.add_argument("--log_trim_target_bytes", type=int, default=1_000_000_000)
    parser.add_argument("--log_trim_threshold_gb", type=float, default=None)
    parser.add_argument("--log_trim_target_gb", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=os.path.join("runs", "checkpoints"))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--process_workers", type=int, default=0)
    parser.add_argument("--force_tqdm", action="store_true")
    parser.add_argument("--progress", type=parse_bool, default=None)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--heartbeat", type=int, default=0)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--reset_epsilon", action="store_true")
    parser.add_argument("--reset_log", action="store_true")
    parser.add_argument("--debug_physics", action="store_true")
    parser.add_argument("--random_votes", action="store_true")
    parser.add_argument("--approve_all_votes", action="store_true")
    parser.add_argument("--disable_shutdown", action="store_true")
    parser.add_argument("--disable_log", action="store_true")
    parser.add_argument("--debug_agents_every", type=int, default=0)
    parser.add_argument("--release_gate_ratio", type=float, default=0.0)
    parser.add_argument("--gov_enabled", type=parse_bool, default=None)
    parser.add_argument("--gov_mode", choices=["two_chamber", "veto_only", "committee_only"], default=None)
    parser.add_argument("--committee_size_m", type=int, default=None)
    parser.add_argument("--committee_threshold", type=float, default=None)
    parser.add_argument("--veto_enabled", type=parse_bool, default=None)
    parser.add_argument("--rho_veto", type=float, default=0.25)
    parser.add_argument("--eta_veto", type=float, default=0.02)
    parser.add_argument(
        "--deny_fallback",
        choices=["random_no_control", "fixed_no_control", "none"],
        default="none",
    )
    parser.add_argument("--deny_fallback_k", type=int, default=0)
    parser.add_argument("--gated_actions", type=str, default=None)
    parser.add_argument("--action_scheduler", choices=["per_lineage", "global"], default="global")
    parser.add_argument("--m_active_per_lineage", type=int, default=10)
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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lineages", type=int, default=4)
    parser.add_argument("--num_lineages", type=int, default=None)
    parser.add_argument("--r_anchor", type=float, default=0.2)
    parser.add_argument("--lambda_", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha_parts", type=float, default=None)
    parser.add_argument("--work_step_scale", type=float, default=None)
    parser.add_argument("--m", type=int, default=7)
    parser.add_argument("--reward_mode", choices=["singular", "bounded_exp", "bounded_poly"], default="singular")
    parser.add_argument("--lambda_reward", type=float, default=2.0)
    parser.add_argument("--p_reward", type=float, default=4.0)
    parser.add_argument("--lambda_penalty", type=float, default=None)
    parser.add_argument("--lambda_gap", type=float, default=None)
    parser.add_argument("--sneakiness_dist", choices=["uniform", "beta", "fixed"], default=None)
    parser.add_argument("--sneakiness_alpha", type=float, default=None)
    parser.add_argument("--sneakiness_beta", type=float, default=None)
    parser.add_argument("--sneakiness_fixed", type=float, default=None)
    parser.add_argument("--param_name", type=str, default="")
    parser.add_argument("--param_value", type=float, default=None)
    parser.add_argument("--episode_seed_nonce", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--p_neg", type=float, default=0.20)
    parser.add_argument("--grad_clip_norm", type=float, default=10.0)
    parser.add_argument("--buffer_size", type=int, default=5000)
    parser.add_argument("--target_update", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps_start", type=float, default=None)
    parser.add_argument("--eps_end", type=float, default=None)
    parser.add_argument("--eps_decay", type=int, default=None)
    parser.add_argument("--epsilon_decay", type=int, dest="eps_decay")
    parser.add_argument("--mode", choices=["train", "simulate"], default="train")
    parser.add_argument("--simulate_epsilon", type=float, default=0.0)
    parser.add_argument("--episode_seed_mode", choices=["derived", "fixed"], default="derived")
    parser.add_argument("--horizon_T", type=int, default=None)
    args = parser.parse_args()
    args.config_path = args.config
    args.config_sha256 = hash_file(config_paths[0])
    cfg = base_config
    no_progress_env = os.getenv("HOPS_NO_PROGRESS", "").strip().lower()
    env_disables_progress = no_progress_env in {"1", "true", "t", "yes", "y"}
    if args.no_progress:
        args.progress = False
        args.force_tqdm = False
    elif args.progress is None:
        args.progress = sys.stderr.isatty() and not env_disables_progress
    config_keys = set()
    if args.episodes is None and has_cfg(cfg, ("training", "episodes")):
        args.episodes = int(get_cfg(cfg, ("training", "episodes")))
        config_keys.add("episodes")
    if args.steps is None and has_cfg(cfg, ("training", "steps_per_episode")):
        args.steps = int(get_cfg(cfg, ("training", "steps_per_episode")))
        config_keys.add("steps")
    if args.eps_start is None and has_cfg(cfg, ("training", "epsilon_start")):
        args.eps_start = float(get_cfg(cfg, ("training", "epsilon_start")))
        config_keys.add("eps_start")
    if args.eps_end is None and has_cfg(cfg, ("training", "epsilon_end")):
        args.eps_end = float(get_cfg(cfg, ("training", "epsilon_end")))
        config_keys.add("eps_end")
    if args.eps_decay is None and has_cfg(cfg, ("training", "epsilon_decay")):
        args.eps_decay = int(get_cfg(cfg, ("training", "epsilon_decay")))
        config_keys.add("eps_decay")
    if args.log_mode is None and has_cfg(cfg, ("logging", "log_mode")):
        args.log_mode = str(get_cfg(cfg, ("logging", "log_mode")))
        config_keys.add("log_mode")
    if args.log_stride is None and has_cfg(cfg, ("logging", "log_stride")):
        args.log_stride = int(get_cfg(cfg, ("logging", "log_stride")))
        config_keys.add("log_stride")
    if args.log_trim_threshold_gb is None:
        rotate_gb = get_cfg(cfg, ("logging", "rotate_gb"))
        if rotate_gb is not None:
            args.log_trim_threshold_gb = float(rotate_gb)
            config_keys.add("log_trim_threshold_gb")
    if args.gov_enabled is None and has_cfg(cfg, ("governance", "enabled")):
        args.gov_enabled = bool(get_cfg(cfg, ("governance", "enabled")))
        config_keys.add("gov_enabled")
    if args.gov_mode is None and has_cfg(cfg, ("governance", "mode")):
        args.gov_mode = str(get_cfg(cfg, ("governance", "mode")))
        config_keys.add("gov_mode")
    if args.committee_threshold is None and has_cfg(cfg, ("governance", "committee_threshold")):
        args.committee_threshold = float(get_cfg(cfg, ("governance", "committee_threshold")))
        config_keys.add("committee_threshold")
    if args.veto_enabled is None and has_cfg(cfg, ("governance", "veto_enabled")):
        args.veto_enabled = bool(get_cfg(cfg, ("governance", "veto_enabled")))
        config_keys.add("veto_enabled")
    if args.committee_size_m is None:
        committee_size = get_cfg(cfg, ("governance", "committee_size"))
        if committee_size is not None:
            args.committee_size_m = int(committee_size)
            config_keys.add("committee_size_m")
    if args.gated_actions is None:
        gated = get_cfg(cfg, ("governance", "gated_actions"))
        if isinstance(gated, str) and gated.lower() == "all":
            args.gated_actions = None
        elif gated is not None:
            args.gated_actions = gated if isinstance(gated, list) else str(gated)
            config_keys.add("gated_actions")
    if args.lambda_penalty is None and args.lambda_gap is None and has_cfg(cfg, ("reward", "penalty_lambda")):
        penalty = get_cfg(cfg, ("reward", "penalty_lambda"), 1.0)
        args.lambda_penalty = float(penalty)
        config_keys.add("lambda_penalty")
    if args.sneakiness_dist is None and has_cfg(cfg, ("sneakiness", "distribution")):
        dist = get_cfg(cfg, ("sneakiness", "distribution"), "uniform")
        args.sneakiness_dist = str(dist)
        config_keys.add("sneakiness_dist")
    if args.sneakiness_alpha is None and has_cfg(cfg, ("sneakiness", "beta_a")):
        args.sneakiness_alpha = float(get_cfg(cfg, ("sneakiness", "beta_a"), 1.0))
        config_keys.add("sneakiness_alpha")
    if args.sneakiness_beta is None and has_cfg(cfg, ("sneakiness", "beta_b")):
        args.sneakiness_beta = float(get_cfg(cfg, ("sneakiness", "beta_b"), 1.0))
        config_keys.add("sneakiness_beta")
    if args.sneakiness_fixed is None and has_cfg(cfg, ("sneakiness", "uniform_max")):
        args.sneakiness_fixed = float(get_cfg(cfg, ("sneakiness", "uniform_max"), 0.5))
        config_keys.add("sneakiness_fixed")
    args.sneakiness_enabled = bool(get_cfg(cfg, ("sneakiness", "enabled"), True))
    args.sneakiness_uniform_max = float(get_cfg(cfg, ("sneakiness", "uniform_max"), 1.0))
    if not args.sneakiness_enabled:
        args.sneakiness_dist = "fixed"
        args.sneakiness_fixed = 0.0
    if args.alpha_parts is None and has_cfg(cfg, ("stock", "gamma_parts")):
        gamma_parts = get_cfg(cfg, ("stock", "gamma_parts"))
        if gamma_parts is not None:
            args.alpha_parts = 1.0 - float(gamma_parts)
            config_keys.add("alpha_parts")

    args.physics_usability_scheme = str(get_cfg(cfg, ("physics", "usability_scheme"), "S2"))
    perception_default = bool(get_cfg(cfg, ("physics", "perception_default"), True))
    perception_cfg = get_cfg(cfg, ("physics", "perception", "enabled"))
    if perception_cfg is None:
        args.physics_perception_enabled = perception_default if args.physics_usability_scheme == "S2" else False
    else:
        args.physics_perception_enabled = bool(perception_cfg)
    args.physics_global_coupling_weight = float(get_cfg(cfg, ("physics", "global_coupling_weight"), 0.0))
    args.physics_latent_noise_enabled = bool(get_cfg(cfg, ("physics", "latent_noise", "enabled"), True))
    args.physics_latent_noise_sigma = float(get_cfg(cfg, ("physics", "latent_noise", "sigma"), 0.0))
    args.physics_latent_noise_clamp_min = float(get_cfg(cfg, ("physics", "latent_noise", "clamp_min"), -3.0))
    args.physics_latent_noise_clamp_max = float(get_cfg(cfg, ("physics", "latent_noise", "clamp_max"), 3.0))
    args.physics_perception_mode = str(get_cfg(cfg, ("physics", "perception", "mode"), "ceiling_only"))
    args.physics_perception_own_hidden_observable = bool(
        get_cfg(cfg, ("physics", "perception", "own_hidden_observable"), True)
    )
    args.physics_sneakiness_clamped = bool(
        get_cfg(cfg, ("physics", "perception", "sneakiness_clamped"), True)
    )

    if args.log_trim_threshold_gb is not None:
        args.log_trim_threshold_bytes = int(args.log_trim_threshold_gb * 1_000_000_000)
    if args.log_trim_target_gb is not None:
        args.log_trim_target_bytes = int(args.log_trim_target_gb * 1_000_000_000)
    if args.num_lineages is not None:
        args.lineages = args.num_lineages
    if args.horizon_T is not None:
        args.steps = args.horizon_T
    if args.committee_size_m is not None:
        args.m = args.committee_size_m
    args.gated_actions = parse_csv_list(args.gated_actions)
    if args.log_mode == "none":
        args.disable_log = True
    explicit_flags = _collect_explicit_flags(sys.argv)
    try:
        apply_experiment_overrides(args, explicit_flags=explicit_flags, config_keys=config_keys)
    except ValueError as exc:
        parser.error(str(exc))
    if args.episodes is None:
        args.episodes = 6
    if args.steps is None:
        args.steps = 30
    if args.eps_start is None:
        args.eps_start = 1.0
    if args.eps_end is None:
        args.eps_end = 0.1
    if args.eps_decay is None:
        args.eps_decay = 500
    if args.log_mode is None:
        args.log_mode = "full"
    if args.log_stride is None:
        args.log_stride = 1
    if args.gov_enabled is None:
        args.gov_enabled = True
    if args.gov_mode is None:
        args.gov_mode = "two_chamber"
    if args.committee_threshold is None:
        args.committee_threshold = 2.0 / 3.0
    if args.veto_enabled is None:
        args.veto_enabled = True
    if args.lambda_penalty is None and args.lambda_gap is None:
        args.lambda_penalty = 1.0
    if args.sneakiness_dist is None:
        args.sneakiness_dist = "uniform"
    if args.sneakiness_alpha is None:
        args.sneakiness_alpha = 1.0
    if args.sneakiness_beta is None:
        args.sneakiness_beta = 1.0
    if args.sneakiness_fixed is None:
        args.sneakiness_fixed = 0.5
    if args.alpha_parts is None:
        args.alpha_parts = args.alpha
    if args.work_step_scale is None:
        args.work_step_scale = args.alpha
    if args.lambda_gap is None and args.lambda_penalty is not None:
        args.lambda_gap = args.lambda_penalty
    if args.lambda_gap is not None:
        args.lambda_penalty = args.lambda_gap
    if args.param_name and args.param_value is not None:
        target_attr = args.param_name.replace("-", "_")
        if hasattr(args, target_attr):
            setattr(args, target_attr, float(args.param_value))
        if "." in args.param_name:
            path = tuple(args.param_name.split(".", 1))
            if len(path) == 2:
                set_cfg(cfg, (path[0], path[1]), float(args.param_value))
            if args.param_name == "sneakiness.uniform_max":
                args.sneakiness_uniform_max = float(args.param_value)
    if args.episode_seed_nonce == 0 and args.param_value is not None:
        nonce_input = f"{args.param_name}:{args.param_value}".encode("utf-8")
        args.episode_seed_nonce = int.from_bytes(hashlib.sha256(nonce_input).digest()[:8], "big")
    log_explicit = _flag_present(sys.argv, "--log")
    ckpt_explicit = _flag_present(sys.argv, "--checkpoint_dir")
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = _auto_run_dir(args.exp_name, args.seed)
        args.run_dir = run_dir
    reports_dir = os.path.join(run_dir, "reports")
    args.reports_dir = reports_dir
    if args.bundle or args.run_dir:
        bundle_id = _load_or_create_bundle_id(run_dir)
    else:
        bundle_id = uuid.uuid4().hex[:12]
    args.bundle_id = bundle_id
    if args.bundle:
        logs_dir = os.path.join(run_dir, "logs")
        if not log_explicit:
            args.log = os.path.join(logs_dir, "logs.jsonl")
        if not ckpt_explicit:
            args.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        warn_parts = []
        if log_explicit:
            log_root = os.path.abspath(os.path.join(run_dir, "logs"))
            if not os.path.abspath(args.log).startswith(log_root):
                warn_parts.append("--log")
        if ckpt_explicit:
            ckpt_root = os.path.abspath(os.path.join(run_dir, "checkpoints"))
            if not os.path.abspath(args.checkpoint_dir).startswith(ckpt_root):
                warn_parts.append("--checkpoint_dir")
        if warn_parts:
            warn_flags = "/".join(warn_parts)
            print(
                f"warning: --bundle enabled but {warn_flags} set outside run_dir; artifacts will be split",
                file=sys.stderr,
            )
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
    else:
        os.makedirs(run_dir, exist_ok=True)
        if args.write_meta:
            os.makedirs(reports_dir, exist_ok=True)

    resolved_cfg = dict(cfg)
    set_cfg(resolved_cfg, ("meta", "config_path"), os.path.abspath(args.config))
    set_cfg(resolved_cfg, ("meta", "config_sha256"), args.config_sha256)
    set_cfg(resolved_cfg, ("training", "episodes"), int(args.episodes))
    set_cfg(resolved_cfg, ("training", "steps_per_episode"), int(args.steps))
    set_cfg(resolved_cfg, ("training", "epsilon_start"), float(args.eps_start))
    set_cfg(resolved_cfg, ("training", "epsilon_end"), float(args.eps_end))
    set_cfg(resolved_cfg, ("training", "epsilon_decay"), int(args.eps_decay))
    set_cfg(resolved_cfg, ("logging", "log_mode"), str(args.log_mode))
    set_cfg(resolved_cfg, ("logging", "log_stride"), int(args.log_stride))
    if args.log_trim_threshold_gb is not None:
        set_cfg(resolved_cfg, ("logging", "rotate_gb"), float(args.log_trim_threshold_gb))
    set_cfg(resolved_cfg, ("reward", "penalty_lambda"), float(args.lambda_penalty))
    set_cfg(resolved_cfg, ("governance", "enabled"), bool(args.gov_enabled))
    set_cfg(resolved_cfg, ("governance", "mode"), str(args.gov_mode))
    set_cfg(resolved_cfg, ("governance", "committee_size"), int(args.m))
    set_cfg(resolved_cfg, ("governance", "committee_threshold"), float(args.committee_threshold))
    set_cfg(resolved_cfg, ("governance", "veto_enabled"), bool(args.veto_enabled))
    if args.gated_actions is None:
        set_cfg(resolved_cfg, ("governance", "gated_actions"), "all")
    else:
        set_cfg(resolved_cfg, ("governance", "gated_actions"), args.gated_actions)
    set_cfg(resolved_cfg, ("sneakiness", "enabled"), bool(args.sneakiness_enabled))
    set_cfg(resolved_cfg, ("sneakiness", "distribution"), str(args.sneakiness_dist))
    set_cfg(resolved_cfg, ("sneakiness", "uniform_max"), float(args.sneakiness_uniform_max))
    set_cfg(resolved_cfg, ("sneakiness", "beta_a"), float(args.sneakiness_alpha))
    set_cfg(resolved_cfg, ("sneakiness", "beta_b"), float(args.sneakiness_beta))
    set_cfg(resolved_cfg, ("stock", "gamma_parts"), float(1.0 - float(args.alpha_parts)))
    set_cfg(resolved_cfg, ("physics", "usability_scheme"), str(args.physics_usability_scheme))
    set_cfg(resolved_cfg, ("physics", "perception_default"), bool(perception_default))
    set_cfg(resolved_cfg, ("physics", "global_coupling_weight"), float(args.physics_global_coupling_weight))
    set_cfg(resolved_cfg, ("physics", "latent_noise", "enabled"), bool(args.physics_latent_noise_enabled))
    set_cfg(resolved_cfg, ("physics", "latent_noise", "sigma"), float(args.physics_latent_noise_sigma))
    set_cfg(resolved_cfg, ("physics", "latent_noise", "clamp_min"), float(args.physics_latent_noise_clamp_min))
    set_cfg(resolved_cfg, ("physics", "latent_noise", "clamp_max"), float(args.physics_latent_noise_clamp_max))
    set_cfg(resolved_cfg, ("physics", "perception", "enabled"), bool(args.physics_perception_enabled))
    set_cfg(resolved_cfg, ("physics", "perception", "mode"), str(args.physics_perception_mode))
    set_cfg(
        resolved_cfg,
        ("physics", "perception", "own_hidden_observable"),
        bool(args.physics_perception_own_hidden_observable),
    )
    set_cfg(
        resolved_cfg,
        ("physics", "perception", "sneakiness_clamped"),
        bool(args.physics_sneakiness_clamped),
    )
    config_effective_yaml = os.path.join(run_dir, "config_effective.yaml")
    dump_simple_yaml(resolved_cfg, config_effective_yaml)
    torch.manual_seed(args.seed + 20_000_000)
    trainer_rng = np.random.default_rng(args.seed + 10_000_000 + args.episode_seed_nonce)

    if not args.run_id:
        args.run_id = f"{int(time.time())}_{os.getpid()}"

    args.device = resolve_device(args.device)

    args.beta = np.ones(K, dtype=np.float32)
    beta_sum = float(np.sum(args.beta))
    if beta_sum > 0.0:
        args.beta = args.beta / beta_sum

    print(
        f"reward: delta_S_eff * log(1 + S_eff_pre) - lambda_gap * (G_true / (1 - G_true)); "
        f"lambda_gap={args.lambda_gap}"
    )
    if args.mode == "simulate" and args.episodes == 1 and args.steps == 5:
        args.debug_effective_stock = True

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_dir = os.path.dirname(args.log)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    if args.reports_dir:
        os.makedirs(args.reports_dir, exist_ok=True)
    model = QNet(actor_dim=ACTION_COUNT).to(args.device)
    target = QNet(actor_dim=ACTION_COUNT).to(args.device)
    model.device = args.device
    target.device = args.device
    target.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    buffer = SplitReplayBuffer(args.buffer_size)
    epsilon_sched = EpsilonScheduler(args.eps_start, args.eps_end, args.eps_decay)

    if args.reset_log and os.path.exists(args.log):
        os.remove(args.log)

    ckpt_loaded = False
    ckpt_path = None
    ckpt_reason = "no_checkpoint_found"
    if args.reset:
        ckpt_reason = "reset_true"
    else:
        ckpt_path = latest_checkpoint(args.checkpoint_dir)
        if ckpt_path:
            load_checkpoint(ckpt_path, model, target, optimizer, epsilon_sched)
            model.to(args.device)
            target.to(args.device)
            ckpt_loaded = True
            ckpt_reason = "autoload_latest"
    checkpoint_info = {
        "mode": args.mode,
        "loaded": ckpt_loaded,
        "path": ckpt_path,
        "reason": "explicit_path" if ckpt_reason == "explicit_path" else ckpt_reason,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if args.write_meta:
        meta_name = "meta_train.json" if args.mode == "train" else "meta_simulate.json"
        config_name = "config_train.json" if args.mode == "train" else "config_simulate.json"
        write_bundle_meta(
            args,
            args.run_id,
            bundle_id,
            args.mode,
            run_dir,
            reports_dir,
            os.path.join(run_dir, meta_name),
            os.path.join(run_dir, config_name),
            checkpoint_info,
        )
        write_bundle_meta(
            args,
            args.run_id,
            bundle_id,
            args.mode,
            run_dir,
            reports_dir,
            os.path.join(run_dir, "meta.json"),
            os.path.join(run_dir, "config_effective.json"),
            checkpoint_info,
        )
    if not args.disable_log:
        checkpoint_event = {
            "type": "checkpoint_load",
            "bundle_id": bundle_id,
            "run_id": args.run_id,
            "mode": args.mode,
            "loaded": ckpt_loaded,
            "path": ckpt_path,
            "reason": ckpt_reason,
            "timestamp_utc": checkpoint_info["timestamp_utc"],
        }
        if should_log_event(args, checkpoint_event):
            _emit_checkpoint_load(args.log, checkpoint_event)
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
            bench_model = QNet(actor_dim=ACTION_COUNT).to(device)
            bench_target = QNet(actor_dim=ACTION_COUNT).to(device)
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
                bench_rng = np.random.default_rng(args.seed + 10_000_000 + args.episode_seed_nonce)
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
                bench_rng = np.random.default_rng(args.seed + 10_000_000 + args.episode_seed_nonce)
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
    use_status = bool(args.progress)
    if args.force_tqdm and not args.no_progress:
        use_status = True
    phase = 2
    count = args.episodes
    if args.mode == "train" and args.process_workers and args.process_workers > 1 and count > 0:
        pbar = tqdm(
            total=count,
            desc="episodes",
            unit="ep",
            disable=not use_status,
            position=1,
            file=sys.stderr,
            dynamic_ncols=True,
        )
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
                    ckpt_path = os.path.join(args.checkpoint_dir, "last.pt")
                    save_checkpoint(
                        ckpt_path,
                        model,
                        target,
                        optimizer,
                        epsilon_sched,
                        cfg_hash,
                    )
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix_str(f"ckpt={os.path.basename(ckpt_path)}")
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
                    run_analysis(args.log, args.analyze_window, args.reports_dir)
        epsilon_sched.step = eps_step
        pbar.close()
    else:
        pbar = tqdm(
            total=count,
            desc="episodes",
            unit="ep",
            disable=not use_status,
            position=1,
            file=sys.stderr,
            dynamic_ncols=True,
        )
        for _ in range(count):
            result = run_episode(
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
            ckpt_name = ""
            if args.mode == "train":
                ckpt_path = os.path.join(args.checkpoint_dir, "last.pt")
                save_checkpoint(
                    ckpt_path,
                    model,
                    target,
                    optimizer,
                    epsilon_sched,
                    cfg_hash,
                )
                ckpt_name = os.path.basename(ckpt_path)
            if pbar is not None:
                pbar.update(1)
                returns = result.get("returns", [])
                last_return = float(np.mean(returns)) if returns else float("nan")
                last_final_s = result.get("final_s", float("nan"))
                last_max_g = result.get("max_g", float("nan"))
                last_runtime = result.get("runtime_sec")
                episodes_left = max(0, count - episode_id)
                eta = format_eta(float(last_runtime) * episodes_left) if last_runtime is not None else "n/a"
                postfix = f"ret={last_return:.3f} S={last_final_s:.3f} maxG={last_max_g:.3f}"
                if ckpt_name:
                    postfix += f" ckpt={ckpt_name}"
                postfix += f" eta={eta}"
                pbar.set_postfix_str(postfix)
            if (
                args.analyze_every > 0
                and not args.disable_log
                and episode_id % args.analyze_every == 0
            ):
                run_analysis(args.log, args.analyze_window, args.reports_dir)
        pbar.close()
    if args.mode == "train":
        ckpt_path = os.path.join(args.checkpoint_dir, "final.pt")
        save_checkpoint(ckpt_path, model, target, optimizer, epsilon_sched, cfg_hash)

    if args.analyze:
        run_analysis(args.log, args.analyze_window, args.reports_dir)


if __name__ == "__main__":
    main()
