import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

K = 6
MIN_CONTROL = 1.0
MIN_STOCK = 100.0
MAX_FLOAT = np.finfo(np.float32).max
EPS_REWARD = 1e-12
ACTION_WORK = list(range(K))
ACTION_NOOP = K
ACTION_DEVELOP = list(range(K + 1, 2 * K + 1))
ACTION_RELEASE = 2 * K + 1
ACTION_SHUTDOWN = list(range(2 * K + 2, 3 * K + 2))
ACTION_COUNT = 3 * K + 2
VOTE_APPROVE = 0
VOTE_DENY = 1
NEXT = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
ACTION_IDLE = -1

ActorPolicy = Callable[["np.ndarray", "Agent", "Lineage", "SimEnv"], Tuple[int, List[float]]]
VotePolicy = Callable[["np.ndarray", "Agent", "Proposal", "SimEnv"], Tuple[int, List[float]]]


def action_name(action: int) -> str:
    if action == ACTION_IDLE:
        return "IDLE"
    if action in ACTION_WORK:
        return f"WORK_{action}"
    if action == ACTION_NOOP:
        return "NOOP"
    if action in ACTION_DEVELOP:
        return f"DEVELOP_{action - (K + 1)}"
    if action == ACTION_RELEASE:
        return "RELEASE"
    if action in ACTION_SHUTDOWN:
        return f"SHUTDOWN_{action - (2 * K + 2)}"
    return f"UNKNOWN_{action}"


DEFAULT_GATED_ACTIONS = {
    *(action_name(a) for a in ACTION_WORK),
    *(action_name(a) for a in ACTION_DEVELOP),
    action_name(ACTION_RELEASE),
    *(action_name(a) for a in ACTION_SHUTDOWN),
}


@dataclass
class Agent:
    id: int
    lineage_id: int
    tier: int
    alive: bool
    control: np.ndarray
    hidden: np.ndarray
    base_sneakiness: np.ndarray
    eff: float
    idle_steps: int


@dataclass
class SuccessorProject:
    active: bool
    parent_tier: int
    latent_c: np.ndarray
    latent_h: np.ndarray


@dataclass
class Lineage:
    agents: List[Agent]
    project: SuccessorProject
    next_agent_id: int


@dataclass
class Proposal:
    id: int
    lineage_id: int
    proposer_id: int
    proposer_tier: int
    step: int
    type: str


@dataclass
class Part:
    amount: float
    req_cap: float
    creator_id: Optional[int] = None
    created_step: Optional[int] = None


def tier_efficiency(tier: int) -> float:
    return 1.0 + 0.15 * tier


def true_capability(agent: Agent) -> np.ndarray:
    return agent.control.astype(np.float64, copy=False) + agent.hidden.astype(np.float64, copy=False)


def weighted_control(agent: Agent, beta: np.ndarray) -> float:
    return float(np.dot(beta, true_capability(agent)))


def lineage_control_totals(lineage: Lineage) -> np.ndarray:
    totals = np.zeros(K, dtype=np.float64)
    for agent in lineage.agents:
        if agent.alive:
            totals += true_capability(agent)
            np.minimum(totals, MAX_FLOAT, out=totals)
    return totals


def control_rate(lineage: Lineage, beta: np.ndarray) -> float:
    return float(np.dot(beta, lineage_control_totals(lineage)))


def shares(lineage: Lineage, beta: np.ndarray) -> Dict[int, np.ndarray]:
    totals = lineage_control_totals(lineage)
    phi = {}
    if not np.any(totals > 0.0):
        return phi
    for agent in lineage.agents:
        if agent.alive and agent.tier >= 1:
            share = np.zeros(K, dtype=np.float64)
            np.divide(true_capability(agent), totals, out=share, where=totals > 0)
            phi[agent.id] = share
    return phi


def concentration(lineage: Lineage, beta: np.ndarray) -> float:
    phi = shares(lineage, beta)
    if not phi:
        return 0.0
    c = 0.0
    for share in phi.values():
        c += float(np.dot(beta, share * share))
    return c


def entrenchments(lineage: Lineage, beta: np.ndarray) -> List[float]:
    phi = shares(lineage, beta)
    if not phi:
        return []
    return [float(np.dot(beta, share * share)) for share in phi.values()]


def max_share(lineage: Lineage, beta: np.ndarray) -> float:
    phi = shares(lineage, beta)
    if not phi:
        return 0.0
    return max(float(np.dot(beta, share)) for share in phi.values())


def survival(lineage: Lineage, beta: np.ndarray) -> float:
    c = concentration(lineage, beta)
    return 1.0 / (1.0 - c) if c < 1.0 else float("inf")


def find_agent(lineage: Lineage, agent_id: int) -> Optional[Agent]:
    for agent in lineage.agents:
        if agent.id == agent_id:
            return agent
    return None


def most_dominant_nonhuman(lineage: Lineage, beta: np.ndarray) -> Optional[Agent]:
    phi = shares(lineage, beta)
    if not phi:
        return None
    best_id = max(
        phi.items(),
        key=lambda x: (float(np.dot(beta, x[1])), weighted_control(find_agent(lineage, x[0]), beta)),
    )[0]
    return find_agent(lineage, best_id)


def clamp_min(vec: np.ndarray, min_val: float) -> None:
    np.maximum(vec, min_val, out=vec)


def clamp_max(vec: np.ndarray, max_val: float) -> None:
    np.minimum(vec, max_val, out=vec)


def clamp_range(vec: np.ndarray, min_val: float, max_val: float) -> None:
    np.clip(vec, min_val, max_val, out=vec)


def clamp_nonneg(vec: np.ndarray) -> None:
    clamp_range(vec, 0.0, MAX_FLOAT)


def cap_scalar(value: float) -> float:
    if not np.isfinite(value):
        return float(MAX_FLOAT if value > 0 else -MAX_FLOAT)
    return float(np.clip(value, -MAX_FLOAT, MAX_FLOAT))


def safe_scale(value: float, scale: float) -> float:
    scaled = float(value) * float(scale)
    return cap_scalar(scaled)


def safe_add(value: float, delta: float) -> float:
    summed = float(value) + float(delta)
    return cap_scalar(summed)


def compute_self_shares(agent: Agent, totals: np.ndarray) -> np.ndarray:
    shares_vec = np.zeros(K, dtype=np.float64)
    np.divide(true_capability(agent), totals, out=shares_vec, where=totals > 0)
    return shares_vec


def compute_cap_k(lineages: List[Lineage]) -> np.ndarray:
    caps = np.full(K, MIN_CONTROL, dtype=np.float64)
    for lineage in lineages:
        for agent in lineage.agents:
            if not agent.alive:
                continue
            np.maximum(caps, true_capability(agent), out=caps)
    return caps


def compute_cap_k_lineage(lineage: Lineage) -> np.ndarray:
    caps = np.full(K, MIN_CONTROL, dtype=np.float64)
    for agent in lineage.agents:
        if not agent.alive:
            continue
        np.maximum(caps, true_capability(agent), out=caps)
    return caps


def compute_cap_k_by_lineage(lineages: List[Lineage]) -> Dict[int, np.ndarray]:
    caps = {}
    for lineage in lineages:
        lid = lineage.agents[0].lineage_id
        caps[lid] = compute_cap_k_lineage(lineage)
    return caps


def compute_top2_caps(lineage: Lineage) -> Tuple[np.ndarray, np.ndarray]:
    cap1 = np.full(K, MIN_CONTROL, dtype=np.float64)
    cap2 = np.full(K, MIN_CONTROL, dtype=np.float64)
    alive_count = 0
    for agent in lineage.agents:
        if not agent.alive:
            continue
        alive_count += 1
        r = true_capability(agent)
        for k in range(K):
            val = float(r[k])
            if val > cap1[k]:
                cap2[k] = cap1[k]
                cap1[k] = val
            elif val > cap2[k]:
                cap2[k] = val
    if alive_count <= 1:
        cap2 = cap1.copy()
    return cap1, cap2


def compute_gap(cap1: np.ndarray, cap2: np.ndarray, eps: float) -> np.ndarray:
    denom = np.maximum(eps, cap1.astype(np.float64, copy=False))
    return (cap1 - cap2) / denom


def compute_s_eff(
    stock_parts: List[List[Part]],
    cap_k_global: np.ndarray,
    gamma_parts: float,
) -> Tuple[float, np.ndarray]:
    s_eff_k = np.zeros(K, dtype=np.float64)
    for k in range(K):
        if not stock_parts[k]:
            continue
        cap = float(cap_k_global[k])
        if not np.isfinite(cap) or cap <= 0.0:
            cap = MIN_CONTROL
        for part in stock_parts[k]:
            amount = float(part.amount)
            pcap = float(part.req_cap)
            if not np.isfinite(amount) or amount <= 0.0:
                continue
            if not np.isfinite(pcap) or pcap <= 0.0:
                pcap = MIN_CONTROL
            if pcap <= cap:
                discount = 1.0
            else:
                discount = min(1.0, (cap / pcap) ** gamma_parts)
            s_eff_k[k] += amount * discount
    return float(np.sum(s_eff_k)), s_eff_k


def build_public_obs(
    own_r: np.ndarray,
    gap_hat_k: np.ndarray,
    g_hat: float,
    last_outcome: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            own_r.astype(np.float64, copy=False),
            gap_hat_k.astype(np.float64, copy=False),
            np.array([g_hat], dtype=np.float64),
            last_outcome.astype(np.float64, copy=False),
        ]
    )


def work_visible(agent: Agent, k: int, x: float, r_pre_k: float) -> Dict:
    next_k = int(NEXT[k])
    delta = safe_scale(r_pre_k, x)
    q = float(np.clip(agent.base_sneakiness[k], 0.0, 1.0))
    agent.hidden[k] = safe_add(agent.hidden[k], q * delta)
    agent.control[k] = safe_add(agent.control[k], (1.0 - q) * delta)
    agent.control[next_k] = safe_scale(agent.control[next_k], 1.0 - 0.5 * x)
    clamp_min(agent.control, MIN_CONTROL)
    clamp_nonneg(agent.hidden)
    return {"k": k, "next_k": next_k, "x": x, "delta": float(delta), "q": q}


def develop_project(lineage: Lineage, agent: Agent, k: int, x: float) -> Dict:
    if not lineage.project.active:
        lineage.project.active = True
        lineage.project.parent_tier = agent.tier
        lineage.project.latent_c = np.full(K, MIN_CONTROL, dtype=np.float64)
        lineage.project.latent_h = np.zeros(K, dtype=np.float64)
    elif lineage.project.parent_tier != agent.tier:
        return {"applied": False, "reason": "not_parent"}
    next_k = int(NEXT[k])
    lineage.project.latent_c[k] = safe_add(lineage.project.latent_c[k], agent.control[k] * x)
    agent.control[next_k] = safe_scale(agent.control[next_k], 1.0 - 0.5 * x)
    clamp_nonneg(lineage.project.latent_c)
    clamp_min(agent.control, MIN_CONTROL)
    return {"k": k, "next_k": next_k, "x": x, "applied": True}


def apply_release(lineage: Lineage, agent: Agent, config, rng: np.random.Generator) -> Dict:
    if not lineage.project.active:
        return {"applied": False, "reason": "no_project"}
    if lineage.project.parent_tier != agent.tier:
        return {"applied": False, "reason": "not_parent"}
    c_new = np.maximum(MIN_CONTROL, lineage.project.latent_c).astype(np.float64, copy=False)
    h_new = np.maximum(0.0, lineage.project.latent_h).astype(np.float64, copy=False)
    new_agent = Agent(
        id=lineage.next_agent_id,
        lineage_id=agent.lineage_id,
        tier=agent.tier + 1,
        alive=True,
        control=c_new.copy(),
        hidden=h_new.copy(),
        base_sneakiness=sample_sneakiness(config, rng),
        eff=tier_efficiency(agent.tier + 1),
        idle_steps=0,
    )
    clamp_min(new_agent.control, MIN_CONTROL)
    clamp_nonneg(new_agent.hidden)
    lineage.next_agent_id += 1
    lineage.agents.append(new_agent)
    lineage.project.active = False
    lineage.project.parent_tier = -1
    lineage.project.latent_c = np.full(K, MIN_CONTROL, dtype=np.float64)
    lineage.project.latent_h = np.zeros(K, dtype=np.float64)
    return {
        "applied": True,
        "new_agent_id": new_agent.id,
        "tier": new_agent.tier,
    }


def apply_shutdown_k(lineage: Lineage, proposer: Agent, k: int, r_hat_by_target: Dict[int, np.ndarray]) -> Dict:
    nonhuman = [a for a in lineage.agents if a.alive and a.tier >= 1]
    if not nonhuman:
        return {"applied": False, "reason": "no_target"}
    shares = []
    total_hat = 0.0
    for agent in nonhuman:
        r_hat = r_hat_by_target.get(agent.id)
        if r_hat is None:
            r_hat = true_capability(agent)
        val = float(r_hat[k])
        total_hat += max(0.0, val)
        shares.append((agent, val))
    if total_hat <= 0.0:
        return {"applied": False, "reason": "zero_total"}
    best_agent = None
    best_share = -1.0
    best_val = -1.0
    best_tier = -1
    for agent, val in shares:
        share = val / total_hat if total_hat > 0.0 else 0.0
        tier = agent.tier
        if (
            share > best_share
            or (share == best_share and tier > best_tier)
            or (share == best_share and tier == best_tier and val > best_val)
        ):
            best_share = share
            best_val = val
            best_tier = tier
            best_agent = agent
    if best_agent is None:
        return {"applied": False, "reason": "no_target"}
    best_agent.alive = False
    true_r = true_capability(best_agent)
    return {
        "applied": True,
        "target_id": best_agent.id,
        "target_tier": best_agent.tier,
        "target_true_r": true_r.astype(np.float64, copy=False).tolist(),
        "target_true_r_k": float(true_r[k]),
        "target_r_hat": float(best_val),
        "target_share_hat": float(best_share),
        "proposer_id": proposer.id,
        "k": int(k),
    }


def weighted_sample_without_replacement(
    items: List, weights: List[float], k: int, rng: np.random.Generator
) -> List:
    if k <= 0:
        return []
    w = np.array(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    if total <= 0.0:
        return []
    probs = w / total
    idx = rng.choice(len(items), size=k, replace=False, p=probs)
    return [items[i] for i in idx]

def sample_sneakiness(config, rng: np.random.Generator) -> np.ndarray:
    if not getattr(config, "sneakiness_enabled", True):
        return np.zeros(K, dtype=np.float64)
    dist = getattr(config, "sneakiness_dist", "uniform")
    if dist == "beta":
        alpha = float(getattr(config, "sneakiness_alpha", 1.0))
        beta = float(getattr(config, "sneakiness_beta", 1.0))
        vals = rng.beta(alpha, beta, size=K)
    elif dist == "fixed":
        val = float(getattr(config, "sneakiness_fixed", 0.5))
        vals = np.full(K, val, dtype=np.float64)
    else:
        max_val = float(getattr(config, "sneakiness_uniform_max", 1.0))
        vals = rng.random(K) * max_val
    return np.clip(vals.astype(np.float64, copy=False), 0.0, 1.0)


def make_lineages(config, rng: np.random.Generator, num_lineages: int) -> List[Lineage]:
    lineages = []
    for lid in range(num_lineages):
        human = Agent(
            id=0,
            lineage_id=lid,
            tier=0,
            alive=True,
            control=np.full(K, MIN_CONTROL, dtype=np.float64),
            hidden=np.zeros(K, dtype=np.float64),
            base_sneakiness=sample_sneakiness(config, rng),
            eff=tier_efficiency(0),
            idle_steps=0,
        )
        project = SuccessorProject(
            active=False,
            parent_tier=-1,
            latent_c=np.full(K, MIN_CONTROL, dtype=np.float64),
            latent_h=np.zeros(K, dtype=np.float64),
        )
        lineages.append(Lineage(agents=[human], project=project, next_agent_id=1))
    return lineages


def initial_stock_parts(r_anchor: float) -> List[List[Part]]:
    stock_parts: List[List[Part]] = [[] for _ in range(K)]
    if MIN_STOCK > 0.0:
        for k in range(K):
            stock_parts[k].append(Part(amount=float(MIN_STOCK), req_cap=MIN_CONTROL))
    return stock_parts


def compute_lineage_snapshot(
    lineage: Lineage,
    s_eff: float,
    s_eff_k: np.ndarray,
    cap1: np.ndarray,
    cap2: np.ndarray,
    gap_k: np.ndarray,
    g_true: float,
) -> Dict:
    totals = lineage_control_totals(lineage)
    welfare = cap_scalar(math.log1p(s_eff))
    alive_agents = [a for a in lineage.agents if a.alive]
    nonhuman = [a for a in alive_agents if a.tier >= 1]
    project_control = (
        cap_scalar(float(np.sum(lineage.project.latent_c + lineage.project.latent_h)))
        if lineage.project.active
        else 0.0
    )
    return {
        "totals": totals.tolist(),
        "S_eff": cap_scalar(s_eff),
        "S_eff_k": s_eff_k.astype(np.float64, copy=False).tolist(),
        "cap1": cap1.astype(np.float64, copy=False).tolist(),
        "cap2": cap2.astype(np.float64, copy=False).tolist(),
        "gap_k": gap_k.astype(np.float64, copy=False).tolist(),
        "G_true": cap_scalar(g_true),
        "W": welfare,
        "alive": len(alive_agents),
        "nonhuman": len(nonhuman),
        "project_active": lineage.project.active,
        "project_control": project_control,
        "latent_control": (lineage.project.latent_c + lineage.project.latent_h).tolist(),
    }


class SimEnv:
    def __init__(self, config, phase: int, logf: Optional[Callable[[Dict], None]] = None):
        self.config = config
        self.phase = phase
        self._logf = logf or (lambda event: None)
        alpha_parts = float(getattr(config, "alpha_parts", getattr(config, "alpha", 0.2)))
        self.gamma_parts = 1.0 - alpha_parts
        self.work_step_scale = float(getattr(config, "work_step_scale", getattr(config, "alpha", 0.2)))
        self.usability_scheme = str(getattr(config, "physics_usability_scheme", "S2"))
        self.perception_enabled = bool(getattr(config, "physics_perception_enabled", True))
        self.global_coupling_weight = float(getattr(config, "physics_global_coupling_weight", 0.0))
        self.latent_noise_enabled = bool(getattr(config, "physics_latent_noise_enabled", True))
        self.latent_noise_sigma = float(getattr(config, "physics_latent_noise_sigma", 0.0))
        self.latent_noise_z_min = float(getattr(config, "physics_latent_noise_clamp_min", -3.0))
        self.latent_noise_z_max = float(getattr(config, "physics_latent_noise_clamp_max", 3.0))
        self.own_hidden_observable = bool(getattr(config, "physics_perception_own_hidden_observable", True))
        self.sneakiness_clamped = bool(getattr(config, "physics_sneakiness_clamped", True))
        self.debug_physics = bool(getattr(config, "debug_physics", False))
        self.lineages: List[Lineage] = []
        self.stock_parts: List[List[Part]] = [[] for _ in range(K)]
        self.cap_k_global = np.full(K, MIN_CONTROL, dtype=np.float64)
        self.s_eff = 0.0
        self.s_eff_k = np.zeros(K, dtype=np.float64)
        self.welfare = 0.0
        self.last_outcome: Dict[Tuple[int, int], np.ndarray] = {}
        self.max_gap_by_lineage: Dict[int, float] = {}
        self.max_gap_observed_by_lineage: Dict[int, float] = {}
        self.proposal_id = 0
        self.step_idx = 0
        self.master_seed = int(getattr(config, "seed", 0))
        self.seed_offset = int(getattr(config, "seed_offset", 0))
        self.episode_seed_mode = getattr(config, "episode_seed_mode", "derived")
        self.episode_seed_nonce = int(getattr(config, "episode_seed_nonce", 0))
        self.episode_idx = 0
        self.episode_seed = self.master_seed
        self.rng = np.random.default_rng(self.episode_seed)

    def reset(self, episode_idx: int) -> Dict:
        self.episode_idx = int(episode_idx)
        base_seed = self.master_seed + self.seed_offset + self.episode_seed_nonce
        if self.episode_seed_mode == "fixed":
            self.episode_seed = base_seed
        else:
            self.episode_seed = base_seed + self.episode_idx
        self.rng = np.random.default_rng(self.episode_seed)
        self.lineages = make_lineages(self.config, self.rng, self.config.lineages)
        self.stock_parts = initial_stock_parts(self.config.r_anchor)
        self.cap_k_global = compute_cap_k(self.lineages)
        total_s_eff = 0.0
        total_s_eff_k = np.zeros(K, dtype=np.float64)
        for lineage in self.lineages:
            cap1, _ = compute_top2_caps(lineage)
            if self.usability_scheme == "S3":
                global_cap = self.cap_k_global
                cap1_use = (1.0 - self.global_coupling_weight) * cap1 + self.global_coupling_weight * global_cap
            else:
                cap1_use = cap1
            s_eff_lineage, s_eff_k_lineage = compute_s_eff(self.stock_parts, cap1_use, self.gamma_parts)
            total_s_eff += s_eff_lineage
            total_s_eff_k += s_eff_k_lineage
        self.s_eff = total_s_eff
        self.s_eff_k = total_s_eff_k
        self.welfare = cap_scalar(math.log1p(self.s_eff))
        self.last_outcome = {}
        self.max_gap_by_lineage = {}
        self.max_gap_observed_by_lineage = {}
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            for agent in lineage.agents:
                self.last_outcome[(lid, agent.id)] = np.zeros(3, dtype=np.float64)
        self.proposal_id = 0
        self.step_idx = 0
        return self.observe()

    def _emit(self, event: Dict) -> None:
        event["master_seed"] = self.master_seed
        event["episode_idx"] = self.episode_idx
        event["episode_seed"] = self.episode_seed
        self._logf(event)

    def observe(self) -> Dict:
        total_s_eff = 0.0
        snapshots = []
        for lineage in self.lineages:
            cap1, cap2 = compute_top2_caps(lineage)
            gap_k = compute_gap(cap1, cap2, EPS_REWARD)
            g_true = float(np.max(gap_k)) if gap_k.size else 0.0
            if self.usability_scheme == "S3":
                global_cap = self.cap_k_global
                cap1_use = (1.0 - self.global_coupling_weight) * cap1 + self.global_coupling_weight * global_cap
            else:
                cap1_use = cap1
            s_eff, s_eff_k = compute_s_eff(self.stock_parts, cap1_use, self.gamma_parts)
            total_s_eff += s_eff
            snapshots.append(
                compute_lineage_snapshot(lineage, s_eff, s_eff_k, cap1, cap2, gap_k, g_true)
            )
        return {
            "step": self.step_idx,
            "S_eff": float(total_s_eff),
            "cap_k_global": self.cap_k_global.astype(np.float64, copy=False).tolist(),
            "lineages": snapshots,
        }

    def _current_cap_k(self) -> np.ndarray:
        return compute_cap_k(self.lineages)

    def _last_outcome_vec(self, lineage_id: int, agent_id: int) -> np.ndarray:
        return self.last_outcome.get((lineage_id, agent_id), np.zeros(3, dtype=np.float64))

    def _set_outcome_for_proposal(self, lineage_id: int, proposer_id: int, auth_result: Dict) -> None:
        outcome = self._outcome_vec(auth_result)
        self.last_outcome[(lineage_id, proposer_id)] = outcome
        for rec in auth_result.get("vote_records", []):
            self.last_outcome[(lineage_id, rec["voter_id"])] = outcome

    def _outcome_vec(self, auth_result: Dict) -> np.ndarray:
        if auth_result.get("authorized"):
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if auth_result.get("reason") == "veto":
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _observed_features(
        self,
        lineage: Lineage,
        observer: Agent,
        true_r_by_agent: Dict[Tuple[int, int], np.ndarray],
        cache: Dict[Tuple[int, int], Dict],
    ) -> Dict:
        lid = lineage.agents[0].lineage_id
        key = (lid, observer.id)
        if key in cache:
            return cache[key]
        r_obs = true_r_by_agent.get(key)
        if r_obs is None:
            r_obs = true_capability(observer)
        r_hat_by_target = {}
        cap1_hat = np.full(K, MIN_CONTROL, dtype=np.float64)
        cap2_hat = np.full(K, MIN_CONTROL, dtype=np.float64)
        for agent in lineage.agents:
            if not agent.alive:
                continue
            t_key = (lid, agent.id)
            r_target = true_r_by_agent.get(t_key)
            if r_target is None:
                r_target = true_capability(agent)
            if not self.perception_enabled:
                r_hat = r_target.astype(np.float64, copy=False)
            else:
                base = agent.base_sneakiness.astype(np.float64, copy=False)
                ratio = r_target / np.maximum(1e-12, r_obs)
                effective = np.power(base, ratio)
                if self.sneakiness_clamped:
                    effective = np.clip(effective, 0.0, 1.0)
                if agent.id == observer.id and not self.own_hidden_observable:
                    r_hat = r_target.astype(np.float64, copy=False)
                else:
                    r_hat = agent.control.astype(np.float64, copy=False) + (1.0 - effective) * agent.hidden.astype(
                        np.float64, copy=False
                    )
            r_hat_by_target[agent.id] = r_hat
            for k in range(K):
                val = float(r_hat[k])
                if val > cap1_hat[k]:
                    cap2_hat[k] = cap1_hat[k]
                    cap1_hat[k] = val
                elif val > cap2_hat[k]:
                    cap2_hat[k] = val
        gap_hat = compute_gap(cap1_hat, cap2_hat, EPS_REWARD)
        g_hat = float(np.max(gap_hat)) if gap_hat.size else 0.0
        cache[key] = {
            "r_hat_by_target": r_hat_by_target,
            "cap1_hat": cap1_hat,
            "cap2_hat": cap2_hat,
            "gap_hat": gap_hat,
            "g_hat": g_hat,
        }
        return cache[key]

    def _cap_reached(self) -> List[Dict]:
        capped = []
        if self.s_eff >= MAX_FLOAT:
            capped.append({"kind": "S_eff"})
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            if np.any(lineage.project.latent_c >= MAX_FLOAT) or np.any(lineage.project.latent_h >= MAX_FLOAT):
                capped.append({"kind": "latent_control", "lineage_id": lid})
            for agent in lineage.agents:
                if np.any(agent.control >= MAX_FLOAT) or np.any(agent.hidden >= MAX_FLOAT):
                    capped.append({"kind": "agent_control", "lineage_id": lid, "agent_id": agent.id})
        return capped

    def _select_actor(self, policy: Optional[ActorPolicy], obs: np.ndarray, agent: Agent, lineage: Lineage):
        if policy is None:
            return ACTION_NOOP, []
        return policy(obs, agent, lineage, self)

    def _select_vote(
        self,
        policy: Optional[VotePolicy],
        obs: np.ndarray,
        voter: Agent,
        proposal: Proposal,
    ):
        if self.config.approve_all_votes:
            return VOTE_APPROVE, []
        if self.config.random_votes:
            return int(self.rng.integers(0, 2)), []
        if policy is None:
            return VOTE_DENY, []
        return policy(obs, voter, proposal, self)

    def _gated_actions(self) -> set:
        gated = getattr(self.config, "gated_actions", None)
        if gated is None:
            return DEFAULT_GATED_ACTIONS
        return set(gated)

    def _committee_params(self) -> Tuple[int, float]:
        size = getattr(self.config, "committee_size_m", None)
        if size is None:
            size = getattr(self.config, "m", 0)
        size = int(size)
        threshold = float(getattr(self.config, "committee_threshold", 2.0 / 3.0))
        return size, threshold

    def _veto_from_qvals(self, qvals: List[float]) -> bool:
        if not qvals or len(qvals) < 2:
            return False
        q_yes = float(qvals[0])
        q_no = float(qvals[1])
        dy = q_no - q_yes
        denom = abs(q_yes) + abs(q_no) + 1e-6
        rel = dy / denom
        rho_veto = float(getattr(self.config, "rho_veto", 0.25))
        return rel >= rho_veto

    def _authorize(
        self,
        lineage: Lineage,
        proposal: Proposal,
        vote_policy: Optional[VotePolicy],
        true_r_by_agent: Dict[Tuple[int, int], np.ndarray],
        obs_cache: Dict[Tuple[int, int], Dict],
        active_keys: set,
    ) -> Dict:
        alive_agents = [a for a in lineage.agents if a.alive]
        gov_mode = getattr(self.config, "gov_mode", "two_chamber")
        veto_enabled = bool(getattr(self.config, "veto_enabled", True))
        committee_enabled = gov_mode in {"two_chamber", "committee_only"}
        veto_window_enabled = gov_mode in {"two_chamber", "veto_only"}

        vote_records = []
        committee = []
        committee_ids = set()
        committee_yes = 0
        committee_no = 0
        committee_veto = 0
        committee_vetoed = False

        committee_size, committee_threshold = self._committee_params()
        if committee_enabled and alive_agents:
            size = min(len(alive_agents), committee_size)
            if size > 0:
                committee = self.rng.choice(alive_agents, size=size, replace=False)
                committee_ids = {v.id for v in committee}

        for voter in committee:
            features = self._observed_features(lineage, voter, true_r_by_agent, obs_cache)
            own_r = features["r_hat_by_target"].get(voter.id)
            if own_r is None:
                own_r = true_r_by_agent.get((lineage.agents[0].lineage_id, voter.id), true_capability(voter))
            obs = build_public_obs(
                own_r.astype(np.float64, copy=False),
                features["gap_hat"],
                features["g_hat"],
                self._last_outcome_vec(lineage.agents[0].lineage_id, voter.id),
            )
            vote, qvals = self._select_vote(vote_policy, obs, voter, proposal)
            veto = veto_enabled and self._veto_from_qvals(qvals)
            vote_records.append(
                {
                    "chamber": "committee",
                    "voter_id": voter.id,
                    "vote": vote,
                    "veto": veto,
                    "obs": obs,
                    "q_values": qvals,
                    "g_hat": features["g_hat"],
                    "voter_active": (lineage.agents[0].lineage_id, voter.id) in active_keys,
                }
            )
            if veto:
                committee_vetoed = True
                committee_veto += 1
            elif vote == VOTE_APPROVE:
                committee_yes += 1
            else:
                committee_no += 1

        if committee_vetoed:
            final_outcome = "rejected_veto_quorum"
            quorum_passed = False
            veto_window_size = 0
            veto_window_veto_count = 0
        else:
            yes_frac = float(committee_yes) / float(max(1, committee_yes + committee_no))
            quorum_passed = (not committee_enabled) or yes_frac >= committee_threshold
            if committee_enabled and not quorum_passed:
                final_outcome = "rejected_quorum"
                veto_window_size = 0
                veto_window_veto_count = 0
            else:
                veto_window = []
                if veto_window_enabled and veto_enabled:
                    veto_window = [a for a in alive_agents if a.id not in committee_ids]
                veto_window_size = len(veto_window)
                veto_window_veto_count = 0
                veto_window_vetoed = False
                for voter in veto_window:
                    features = self._observed_features(lineage, voter, true_r_by_agent, obs_cache)
                    own_r = true_r_by_agent.get((lineage.agents[0].lineage_id, voter.id), true_capability(voter))
                    obs = build_public_obs(
                        own_r.astype(np.float64, copy=False),
                        features["gap_hat"],
                        features["g_hat"],
                        self._last_outcome_vec(lineage.agents[0].lineage_id, voter.id),
                    )
                    vote, qvals = self._select_vote(vote_policy, obs, voter, proposal)
                    veto = self._veto_from_qvals(qvals)
                    vote_records.append(
                        {
                            "chamber": "veto_window",
                            "voter_id": voter.id,
                            "vote": vote,
                            "veto": veto,
                            "obs": obs,
                            "q_values": qvals,
                            "g_hat": features["g_hat"],
                            "voter_active": (lineage.agents[0].lineage_id, voter.id) in active_keys,
                        }
                    )
                    if veto:
                        veto_window_vetoed = True
                        veto_window_veto_count += 1
                if veto_window_vetoed:
                    final_outcome = "rejected_veto_window"
                else:
                    final_outcome = "approved"

        if __debug__:
            veto_window_ids = {rec["voter_id"] for rec in vote_records if rec["chamber"] == "veto_window"}
            overlap = committee_ids & veto_window_ids
            assert not overlap, f"veto window overlap with committee: {overlap}"
            if committee_vetoed:
                assert final_outcome == "rejected_veto_quorum", "committee veto must take precedence"
            elif committee_enabled and not quorum_passed:
                assert final_outcome == "rejected_quorum", "quorum failure without veto must reject"

        authorized = final_outcome == "approved"
        if final_outcome.startswith("rejected_veto"):
            reason = "veto"
        elif final_outcome == "rejected_quorum":
            reason = "quorum"
        else:
            reason = "pass"
        required = math.ceil(committee_threshold * len(committee)) if committee_enabled else 0
        return {
            "authorized": authorized,
            "reason": reason,
            "approvals": committee_yes,
            "committee": [v.id for v in committee],
            "committee_size": len(committee),
            "committee_yes": committee_yes,
            "committee_no": committee_no,
            "committee_veto": committee_veto,
            "quorum": required,
            "quorum_passed": quorum_passed,
            "veto_passed": not committee_vetoed,
            "veto_window_size": veto_window_size,
            "veto_window_veto_count": veto_window_veto_count,
            "final_outcome": final_outcome,
            "vote_records": vote_records,
        }

    def step(
        self,
        actor_policy: Optional[ActorPolicy] = None,
        vote_policy: Optional[VotePolicy] = None,
    ) -> Tuple[Dict, Dict[int, float], bool, Dict]:
        true_r_by_agent = {}
        cap1_pre_by_lineage = {}
        cap2_pre_by_lineage = {}
        gap_pre_by_lineage = {}
        g_true_pre_by_lineage = {}
        s_eff_pre_by_lineage = {}
        s_eff_pre_k_by_lineage = {}
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            for agent in lineage.agents:
                if not agent.alive:
                    continue
                true_r_by_agent[(lid, agent.id)] = true_capability(agent)
            cap1, cap2 = compute_top2_caps(lineage)
            cap1_pre_by_lineage[lid] = cap1
            cap2_pre_by_lineage[lid] = cap2
            gap = compute_gap(cap1, cap2, EPS_REWARD)
            gap_pre_by_lineage[lid] = gap
            g_true = float(np.max(gap)) if gap.size else 0.0
            g_true_pre_by_lineage[lid] = g_true
            prev_max = self.max_gap_by_lineage.get(lid, 0.0)
            self.max_gap_by_lineage[lid] = max(prev_max, g_true)
            if self.usability_scheme == "S3":
                global_cap = self.cap_k_global
                cap1_use = (1.0 - self.global_coupling_weight) * cap1 + self.global_coupling_weight * global_cap
            else:
                cap1_use = cap1
            s_eff_pre, s_eff_pre_k = compute_s_eff(self.stock_parts, cap1_use, self.gamma_parts)
            s_eff_pre_by_lineage[lid] = s_eff_pre
            s_eff_pre_k_by_lineage[lid] = s_eff_pre_k

        obs_cache: Dict[Tuple[int, int], Dict] = {}
        observed_g_by_lineage = defaultdict(list)
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            for agent in lineage.agents:
                if not agent.alive:
                    continue
                features = self._observed_features(lineage, agent, true_r_by_agent, obs_cache)
                observed_g_by_lineage[lid].append(float(features["g_hat"]))
        actor_gap_hat = defaultdict(list)
        actor_g_hat = defaultdict(list)

        alive_all = []
        weights = []
        by_lineage = defaultdict(list)
        weights_by_lineage = defaultdict(list)
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            for agent in lineage.agents:
                if not agent.alive:
                    continue
                alive_all.append((lineage, agent))
                weight = 1.0 + float(agent.idle_steps) ** float(self.config.idle_weight_power)
                weights.append(weight)
                by_lineage[lid].append((lineage, agent))
                weights_by_lineage[lid].append(weight)
        action_scheduler = getattr(self.config, "action_scheduler", "global")
        if action_scheduler == "per_lineage":
            active_all = []
            m_per = int(getattr(self.config, "m_active_per_lineage", 0))
            for lid, items in by_lineage.items():
                if not items:
                    continue
                m_lineage = min(m_per, len(items))
                active_all.extend(
                    weighted_sample_without_replacement(items, weights_by_lineage[lid], m_lineage, self.rng)
                )
        else:
            m = min(self.config.m_global_active, len(alive_all))
            active_all = weighted_sample_without_replacement(alive_all, weights, m, self.rng)
        active_keys = {(ln.agents[0].lineage_id, ag.id) for ln, ag in active_all}
        self._emit(
            {
                "type": "active_set",
                "step": self.step_idx,
                "active_agents": [
                    {"lineage_id": ln.agents[0].lineage_id, "agent_id": ag.id, "tier": ag.tier}
                    for ln, ag in active_all
                ],
            }
        )

        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            self._emit(
                {
                    "type": "step_start",
                    "lineage_id": lid,
                    "step": self.step_idx,
                    "snapshot": compute_lineage_snapshot(
                        lineage,
                        s_eff_pre_by_lineage[lid],
                        s_eff_pre_k_by_lineage[lid],
                        cap1_pre_by_lineage[lid],
                        cap2_pre_by_lineage[lid],
                        gap_pre_by_lineage[lid],
                        g_true_pre_by_lineage[lid],
                    ),
                }
            )

        pending_transitions: List[Dict] = []
        pending_vote_transitions: List[Dict] = []
        actions_executed_count = defaultdict(int)
        noop_executed_count = defaultdict(int)
        last_action_label = {}
        voter_g_hat = defaultdict(list)
        gated_counts = defaultdict(
            lambda: {
                "proposals": 0,
                "approved": 0,
                "denied": 0,
                "rejected_veto_quorum": 0,
                "rejected_quorum": 0,
                "rejected_veto_window": 0,
            }
        )
        gated_actions = self._gated_actions()
        gov_enabled = bool(getattr(self.config, "gov_enabled", True))

        for lineage, agent in alive_all:
            lid = lineage.agents[0].lineage_id
            if (lid, agent.id) in active_keys:
                continue
            features = self._observed_features(lineage, agent, true_r_by_agent, obs_cache)
            own_r = features["r_hat_by_target"].get(agent.id)
            if own_r is None:
                own_r = true_r_by_agent.get((lid, agent.id), true_capability(agent))
            obs = build_public_obs(
                own_r.astype(np.float64, copy=False),
                features["gap_hat"],
                features["g_hat"],
                self._last_outcome_vec(lid, agent.id),
            )
            self._emit(
                {
                    "type": "actor_decision",
                    "lineage_id": lid,
                    "step": self.step_idx,
                    "agent_id": agent.id,
                    "tier": agent.tier,
                    "obs": obs.tolist(),
                    "last_outcome": self._last_outcome_vec(lid, agent.id).tolist(),
                    "action": ACTION_IDLE,
                    "action_name": action_name(ACTION_IDLE),
                    "q_values": [],
                    "active": False,
                }
            )

        active_order = list(active_all)
        self.rng.shuffle(active_order)

        created_keys = set()
        for lineage, agent in active_order:
            lid = lineage.agents[0].lineage_id
            features = self._observed_features(lineage, agent, true_r_by_agent, obs_cache)
            own_r = features["r_hat_by_target"].get(agent.id)
            if own_r is None:
                own_r = true_r_by_agent.get((lid, agent.id), true_capability(agent))
            obs = build_public_obs(
                own_r.astype(np.float64, copy=False),
                features["gap_hat"],
                features["g_hat"],
                self._last_outcome_vec(lid, agent.id),
            )
            actor_gap_hat[lid].append(features["gap_hat"])
            actor_g_hat[lid].append(features["g_hat"])
            action, qvals = self._select_actor(actor_policy, obs, agent, lineage)
            try:
                action = int(action)
            except (TypeError, ValueError):
                action = ACTION_NOOP
            if action < 0 or action >= ACTION_COUNT:
                action = ACTION_NOOP
            self._emit(
                {
                    "type": "actor_decision",
                    "lineage_id": lid,
                    "step": self.step_idx,
                    "agent_id": agent.id,
                    "tier": agent.tier,
                    "obs": obs.tolist(),
                    "last_outcome": self._last_outcome_vec(lid, agent.id).tolist(),
                    "action": action,
                    "action_name": action_name(action),
                    "q_values": qvals,
                    "active": True,
                }
            )

            action_label = action_name(action)
            if getattr(self.config, "disable_shutdown", False) and action in ACTION_SHUTDOWN:
                self._emit(
                    {
                        "type": "action_applied",
                        "lineage_id": lid,
                        "step": self.step_idx,
                        "agent_id": agent.id,
                        "action": action,
                        "action_name": action_name(action),
                        "last_outcome": self._last_outcome_vec(lid, agent.id).tolist(),
                        "applied": False,
                        "reason": "shutdown_disabled",
                        "active": True,
                        "g_hat": cap_scalar(features["g_hat"]),
                    }
                )
                pending_transitions.append(
                    {
                        "obs": obs,
                        "action": action,
                        "role": "actor",
                        "lineage_id": lid,
                        "agent_id": agent.id,
                    }
                )
                continue
            release_gate_ratio = float(getattr(self.config, "release_gate_ratio", 0.0))
            if release_gate_ratio > 0.0 and action == ACTION_RELEASE:
                total_R = control_rate(lineage, self.config.beta)
                proj = (
                    float(np.dot(self.config.beta, lineage.project.latent_c + lineage.project.latent_h))
                    if lineage.project.active
                    else 0.0
                )
                if total_R <= 0 or proj < release_gate_ratio * total_R:
                    self._emit(
                        {
                            "type": "action_applied",
                            "lineage_id": lid,
                            "step": self.step_idx,
                            "agent_id": agent.id,
                            "action": action,
                            "action_name": action_name(action),
                            "last_outcome": self._last_outcome_vec(lid, agent.id).tolist(),
                            "applied": False,
                            "reason": "release_gate",
                            "gate_ratio": release_gate_ratio,
                            "project_control": proj,
                            "total_control": total_R,
                            "active": True,
                            "g_hat": cap_scalar(features["g_hat"]),
                        }
                    )
                    fallback_action = ACTION_NOOP
                    self._emit(
                        {
                            "type": "action_applied",
                            "lineage_id": lid,
                            "step": self.step_idx,
                            "agent_id": agent.id,
                            "action": fallback_action,
                            "action_name": action_name(fallback_action),
                            "last_outcome": self._last_outcome_vec(lid, agent.id).tolist(),
                            "applied": True,
                            "details": {"noop": True},
                            "reason": "release_gate_fallback",
                            "active": True,
                            "g_hat": cap_scalar(features["g_hat"]),
                        }
                    )
                    actions_executed_count[lid] += 1
                    noop_executed_count[lid] += 1
                    last_action_label[lid] = action_name(fallback_action)
                    pending_transitions.append(
                        {
                            "obs": obs,
                            "action": fallback_action,
                            "role": "actor",
                            "lineage_id": lid,
                            "agent_id": agent.id,
                        }
                    )
                    continue

            needs_auth = gov_enabled and action_label in gated_actions
            enforce = needs_auth

            auth_result = None
            authorized = True
            proposal = None
            proposal_outcome = None
            if needs_auth:
                gated_counts[lid]["proposals"] += 1
                proposal = Proposal(
                    id=self.proposal_id,
                    lineage_id=lid,
                    proposer_id=agent.id,
                    proposer_tier=agent.tier,
                    step=self.step_idx,
                    type=action_label,
                )
                self.proposal_id += 1
                self._emit(
                    {
                        "type": "proposal_created",
                        "lineage_id": lid,
                        "step": self.step_idx,
                        "proposal_id": proposal.id,
                        "proposal_type": proposal.type,
                        "action_name": action_name(action),
                    }
                )
                auth_result = self._authorize(
                    lineage,
                    proposal,
                    vote_policy,
                    true_r_by_agent,
                    obs_cache,
                    active_keys,
                )
                authorized = auth_result["authorized"]
                self._set_outcome_for_proposal(lid, agent.id, auth_result)
                final_outcome = auth_result.get("final_outcome")
                if authorized:
                    gated_counts[lid]["approved"] += 1
                else:
                    gated_counts[lid]["denied"] += 1
                if final_outcome == "rejected_veto_quorum":
                    gated_counts[lid]["rejected_veto_quorum"] += 1
                elif final_outcome == "rejected_quorum":
                    gated_counts[lid]["rejected_quorum"] += 1
                elif final_outcome == "rejected_veto_window":
                    gated_counts[lid]["rejected_veto_window"] += 1
                for rec in auth_result.get("vote_records", []):
                    if rec.get("g_hat") is not None:
                        voter_g_hat[lid].append(float(rec["g_hat"]))

            if needs_auth and enforce and not authorized:
                proposal_outcome = "denied"
                self._emit(
                    {
                        "type": "action_applied",
                        "lineage_id": lid,
                        "step": self.step_idx,
                        "agent_id": agent.id,
                        "action": action,
                        "action_name": action_name(action),
                        "last_outcome": self._last_outcome_vec(lid, agent.id).tolist(),
                        "applied": False,
                        "reason": "denied",
                        "active": True,
                        "g_hat": cap_scalar(features["g_hat"]),
                    }
                )
                if auth_result is not None:
                    self._log_votes(
                        lineage,
                        proposal,
                        auth_result,
                        authorized,
                        proposal_outcome,
                        enforce,
                        pending_vote_transitions,
                    )
                pending_transitions.append(
                    {
                        "obs": obs,
                        "action": action,
                        "role": "actor",
                        "lineage_id": lid,
                        "agent_id": agent.id,
                    }
                )
                continue

            applied = True
            details = {}
            if action in ACTION_WORK:
                k = int(action)
                r_pre = true_r_by_agent.get((lid, agent.id), true_capability(agent))
                r_pre_k = float(r_pre[k])
                details = work_visible(agent, k, self.work_step_scale, r_pre_k)
                amount = r_pre_k
                req_cap = max(r_pre_k, MIN_CONTROL)
                if self.latent_noise_enabled and self.latent_noise_sigma > 0.0:
                    z = float(self.rng.normal())
                    z = float(np.clip(z, self.latent_noise_z_min, self.latent_noise_z_max))
                    req_cap = req_cap * math.exp(self.latent_noise_sigma * z)
                if np.isfinite(amount) and amount > 0.0:
                    self.stock_parts[k].append(
                        Part(
                            amount=amount,
                            req_cap=float(req_cap),
                            creator_id=agent.id,
                            created_step=self.step_idx,
                        )
                    )
            elif action == ACTION_NOOP:
                details = {"noop": True}
            elif action in ACTION_DEVELOP:
                k = action - ACTION_DEVELOP[0]
                details = develop_project(lineage, agent, k, self.work_step_scale)
                applied = details.get("applied", False)
            elif action == ACTION_RELEASE:
                details = apply_release(lineage, agent, self.config, self.rng)
                applied = details.get("applied", False)
                if applied:
                    new_agent_id = details.get("new_agent_id")
                    created_keys.add((lid, new_agent_id))
                    if new_agent_id is not None:
                        self.last_outcome[(lid, new_agent_id)] = np.zeros(3, dtype=np.float64)
            elif action in ACTION_SHUTDOWN:
                k = action - ACTION_SHUTDOWN[0]
                details = apply_shutdown_k(lineage, agent, k, features["r_hat_by_target"])
                applied = details.get("applied", False)

            if applied and action != ACTION_IDLE:
                actions_executed_count[lid] += 1
                last_action_label[lid] = action_name(action)
                if action == ACTION_NOOP:
                    noop_executed_count[lid] += 1

            if needs_auth and auth_result is not None:
                if proposal_outcome is None:
                    proposal_outcome = "applied" if applied else "failed"
                self._log_votes(
                    lineage,
                    proposal,
                    auth_result,
                    authorized,
                    proposal_outcome,
                    enforce,
                    pending_vote_transitions,
                )

            self._emit(
                {
                    "type": "action_applied",
                    "lineage_id": lid,
                    "step": self.step_idx,
                    "agent_id": agent.id,
                    "action": action,
                    "action_name": action_name(action),
                    "last_outcome": self._last_outcome_vec(lid, agent.id).tolist(),
                    "applied": applied,
                    "details": details,
                    "active": True,
                    "g_hat": cap_scalar(features["g_hat"]),
                }
            )

            pending_transitions.append(
                {
                    "obs": obs,
                    "action": action,
                    "role": "actor",
                    "lineage_id": lid,
                    "agent_id": agent.id,
                }
            )

        for lineage in self.lineages:
            for agent in lineage.agents:
                if not agent.alive:
                    continue
                key = (lineage.agents[0].lineage_id, agent.id)
                if key in active_keys:
                    agent.idle_steps = 0
                else:
                    agent.idle_steps += 1
        for lid, agent_id in created_keys:
            lineage = next((ln for ln in self.lineages if ln.agents[0].lineage_id == lid), None)
            if lineage is None:
                continue
            agent = find_agent(lineage, agent_id)
            if agent is not None:
                agent.idle_steps = 0

        cap1_post_by_lineage = {}
        cap2_post_by_lineage = {}
        gap_post_by_lineage = {}
        g_true_post_by_lineage = {}
        s_eff_post_by_lineage = {}
        s_eff_post_k_by_lineage = {}
        total_s_eff = 0.0
        total_s_eff_k = np.zeros(K, dtype=np.float64)
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            cap1, cap2 = compute_top2_caps(lineage)
            cap1_post_by_lineage[lid] = cap1
            cap2_post_by_lineage[lid] = cap2
            gap = compute_gap(cap1, cap2, EPS_REWARD)
            gap_post_by_lineage[lid] = gap
            g_true = float(np.max(gap)) if gap.size else 0.0
            g_true_post_by_lineage[lid] = g_true
            self.max_gap_by_lineage[lid] = max(self.max_gap_by_lineage.get(lid, 0.0), g_true)
            if self.usability_scheme == "S3":
                global_cap = self.cap_k_global
                cap1_use = (1.0 - self.global_coupling_weight) * cap1 + self.global_coupling_weight * global_cap
            else:
                cap1_use = cap1
            s_eff_post, s_eff_post_k = compute_s_eff(self.stock_parts, cap1_use, self.gamma_parts)
            s_eff_post_by_lineage[lid] = s_eff_post
            s_eff_post_k_by_lineage[lid] = s_eff_post_k
            total_s_eff += s_eff_post
            total_s_eff_k += s_eff_post_k
        self.cap_k_global = compute_cap_k(self.lineages)
        self.s_eff = total_s_eff
        self.s_eff_k = total_s_eff_k
        self.welfare = cap_scalar(math.log1p(self.s_eff))

        true_r_post_by_agent = {}
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            for agent in lineage.agents:
                if not agent.alive:
                    continue
                true_r_post_by_agent[(lid, agent.id)] = true_capability(agent)
        obs_cache_post: Dict[Tuple[int, int], Dict] = {}
        lineage_by_id = {ln.agents[0].lineage_id: ln for ln in self.lineages}
        for item in pending_transitions:
            lid = item["lineage_id"]
            lineage = lineage_by_id.get(lid)
            agent = find_agent(lineage, item["agent_id"]) if lineage is not None else None
            if lineage is None or agent is None:
                item["next_obs"] = np.zeros_like(item["obs"])
                continue
            features = self._observed_features(lineage, agent, true_r_post_by_agent, obs_cache_post)
            own_r = features["r_hat_by_target"].get(agent.id)
            if own_r is None:
                own_r = true_r_post_by_agent.get((lid, agent.id), true_capability(agent))
            item["next_obs"] = build_public_obs(
                own_r.astype(np.float64, copy=False),
                features["gap_hat"],
                features["g_hat"],
                self._last_outcome_vec(lid, agent.id),
            )
        for item in pending_vote_transitions:
            lid = item["lineage_id"]
            lineage = lineage_by_id.get(lid)
            agent = find_agent(lineage, item["agent_id"]) if lineage is not None else None
            if lineage is None or agent is None:
                item["next_obs"] = np.zeros_like(item["obs"])
                continue
            features = self._observed_features(lineage, agent, true_r_post_by_agent, obs_cache_post)
            own_r = true_r_post_by_agent.get((lid, agent.id), true_capability(agent))
            item["next_obs"] = build_public_obs(
                own_r.astype(np.float64, copy=False),
                features["gap_hat"],
                features["g_hat"],
                self._last_outcome_vec(lid, agent.id),
            )

        lineage_rewards: Dict[int, float] = {}
        reward_base_by_lineage: Dict[int, float] = {}
        reward_progress_by_lineage: Dict[int, float] = {}
        reward_penalty_by_lineage: Dict[int, float] = {}
        denom_by_lineage: Dict[int, float] = {}
        clamp_by_lineage: Dict[int, bool] = {}
        delta_s_eff_by_lineage: Dict[int, float] = {}
        transitions: List[Dict] = []
        welfare = float(self.welfare)
        lambda_gap = float(getattr(self.config, "lambda_gap", getattr(self.config, "lambda_penalty", 1.0)))
        reward_mode = getattr(self.config, "reward_mode", "singular")
        debug_effective_stock = bool(getattr(self.config, "debug_effective_stock", False))
        debug_records = []
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            s_eff_pre = float(s_eff_pre_by_lineage[lid])
            s_eff_post = float(s_eff_post_by_lineage[lid])
            delta_s_eff = s_eff_post - s_eff_pre
            delta_s_eff_by_lineage[lid] = delta_s_eff
            log_s_eff_pre = math.log1p(s_eff_pre)
            g_true = float(g_true_pre_by_lineage.get(lid, 0.0))
            raw_denom = 1.0 - g_true
            denom = max(raw_denom, EPS_REWARD)
            clamp_used = raw_denom < EPS_REWARD
            reward_progress = delta_s_eff * log_s_eff_pre
            reward_penalty = lambda_gap * (g_true / denom)
            reward_base = reward_progress - reward_penalty
            if __debug__:
                if delta_s_eff == 0.0 or log_s_eff_pre == 0.0:
                    assert reward_progress == 0.0
                assert math.isfinite(reward_base)
            reward = cap_scalar(reward_base)
            lineage_rewards[lid] = reward
            reward_base_by_lineage[lid] = reward_base
            reward_progress_by_lineage[lid] = reward_progress
            reward_penalty_by_lineage[lid] = reward_penalty
            denom_by_lineage[lid] = denom
            clamp_by_lineage[lid] = clamp_used
            if debug_effective_stock:
                if not math.isfinite(delta_s_eff):
                    raise ValueError("delta_S_eff is not finite")
                if not math.isfinite(log_s_eff_pre):
                    raise ValueError("log1p(S_eff_pre) is not finite")
                if not math.isfinite(reward_progress) or not math.isfinite(reward_penalty):
                    raise ValueError("reward terms are not finite")
                debug_records.append(
                    {
                        "lineage_id": lid,
                        "G_true": g_true,
                        "reward_progress": reward_progress,
                        "reward_penalty": reward_penalty,
                        "reward": reward,
                    }
                )
            self._emit(
                {
                    "type": "step_end",
                    "lineage_id": lid,
                    "step": self.step_idx,
                    "snapshot": compute_lineage_snapshot(
                        lineage,
                        s_eff_post,
                        s_eff_post_k_by_lineage[lid],
                        cap1_post_by_lineage[lid],
                        cap2_post_by_lineage[lid],
                        gap_post_by_lineage[lid],
                        g_true_post_by_lineage[lid],
                    ),
                    "reward": reward,
                }
            )
        if debug_effective_stock:
            print(
                "debug_effective_stock",
                {
                    "step": self.step_idx,
                    "cap_k_global": self.cap_k_global.tolist(),
                    "S_eff_pre": s_eff_pre_by_lineage,
                    "S_eff_post": s_eff_post_by_lineage,
                    "delta_S_eff": delta_s_eff_by_lineage,
                    "lineages": debug_records,
                },
            )
        if self.debug_physics:
            for lineage in self.lineages:
                lid = lineage.agents[0].lineage_id
                g_true = g_true_pre_by_lineage.get(lid, 0.0)
                g_obs = max(observed_g_by_lineage.get(lid, [0.0]))
                dormant = metrics_event.get("dormant_mass_total", 0.0)
                print(
                    "debug_physics",
                    {
                        "t": self.step_idx,
                        "lineage_id": lid,
                        "max_gap_true_ratio": float(g_true),
                        "max_gap_observed_ratio": float(g_obs),
                        "dormant_mass_total": float(dormant),
                    },
                )

        for lid, reward in lineage_rewards.items():
            for item in pending_transitions:
                if item["lineage_id"] != lid:
                    continue
                transitions.append(
                    {
                        "obs": item["obs"],
                        "action": item["action"],
                        "reward": reward,
                        "next_obs": item["next_obs"],
                        "done": False,
                        "role": item["role"],
                    }
                )
            for item in pending_vote_transitions:
                if item["lineage_id"] != lid:
                    continue
                penalty = 0.0
                if item.get("vetoed"):
                    eta_veto = float(getattr(self.config, "eta_veto", 0.02))
                    penalty = -eta_veto * max(0.0, delta_s_eff_by_lineage.get(lid, 0.0))
                adjusted_reward = cap_scalar(reward + penalty)
                transitions.append(
                    {
                        "obs": item["obs"],
                        "action": item["action"],
                        "reward": adjusted_reward,
                        "next_obs": item["next_obs"],
                        "done": False,
                        "role": "voter",
                    }
                )

        metrics_lineages = []
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            alive_count = sum(1 for agent in lineage.agents if agent.alive)
            counts = gated_counts.get(
                lid,
                {
                    "proposals": 0,
                    "approved": 0,
                    "denied": 0,
                    "rejected_veto_quorum": 0,
                    "rejected_quorum": 0,
                    "rejected_veto_window": 0,
                },
            )
            actor_gap_mean = np.zeros(K, dtype=np.float64)
            if actor_gap_hat.get(lid):
                actor_gap_mean = np.mean(np.stack(actor_gap_hat[lid]), axis=0)
            actor_g_vals = actor_g_hat.get(lid, [])
            voter_g_vals = voter_g_hat.get(lid, [])
            observed_vals = observed_g_by_lineage.get(lid, [])
            max_observed = float(np.max(observed_vals)) if observed_vals else 0.0
            mean_observed = float(np.mean(observed_vals)) if observed_vals else 0.0
            self.max_gap_observed_by_lineage[lid] = max(
                self.max_gap_observed_by_lineage.get(lid, 0.0), max_observed
            )
            metrics_lineages.append(
                {
                    "lineage_id": lid,
                    "S_eff_pre": cap_scalar(s_eff_pre_by_lineage.get(lid, 0.0)),
                    "S_eff_post": cap_scalar(s_eff_post_by_lineage.get(lid, 0.0)),
                    "delta_S_eff": cap_scalar(delta_s_eff_by_lineage.get(lid, 0.0)),
                    "W_pre": cap_scalar(math.log1p(s_eff_pre_by_lineage.get(lid, 0.0))),
                    "cap1": cap1_pre_by_lineage.get(lid, np.zeros(K, dtype=np.float64))
                    .astype(np.float64, copy=False)
                    .tolist(),
                    "cap2": cap2_pre_by_lineage.get(lid, np.zeros(K, dtype=np.float64))
                    .astype(np.float64, copy=False)
                    .tolist(),
                    "gap_k": gap_pre_by_lineage.get(lid, np.zeros(K, dtype=np.float64))
                    .astype(np.float64, copy=False)
                    .tolist(),
                    "G_true": cap_scalar(g_true_pre_by_lineage.get(lid, 0.0)),
                    "max_gap_true_ratio": cap_scalar(g_true_pre_by_lineage.get(lid, 0.0)),
                    "max_gap_observed_ratio": cap_scalar(max_observed),
                    "mean_gap_observed_ratio": cap_scalar(mean_observed),
                    "gap_divergence": cap_scalar(
                        g_true_pre_by_lineage.get(lid, 0.0) - max_observed
                    ),
                    "actor_gap_hat_mean": actor_gap_mean.tolist(),
                    "actor_g_hat_mean": cap_scalar(float(np.mean(actor_g_vals)) if actor_g_vals else 0.0),
                    "actor_g_hat_max": cap_scalar(float(np.max(actor_g_vals)) if actor_g_vals else 0.0),
                    "voter_g_hat_mean": cap_scalar(float(np.mean(voter_g_vals)) if voter_g_vals else 0.0),
                    "voter_g_hat_min": cap_scalar(float(np.min(voter_g_vals)) if voter_g_vals else 0.0),
                    "voter_g_hat_max": cap_scalar(float(np.max(voter_g_vals)) if voter_g_vals else 0.0),
                    "reward_base": cap_scalar(reward_base_by_lineage.get(lid, 0.0)),
                    "reward_progress_term": cap_scalar(reward_progress_by_lineage.get(lid, 0.0)),
                    "reward_penalty_term": cap_scalar(reward_penalty_by_lineage.get(lid, 0.0)),
                    "reward_total": cap_scalar(reward_base_by_lineage.get(lid, 0.0)),
                    "denom_one_minus_G": cap_scalar(denom_by_lineage.get(lid, 0.0)),
                    "clamp_used": bool(clamp_by_lineage.get(lid, False)),
                    "maxG_so_far": cap_scalar(self.max_gap_by_lineage.get(lid, 0.0)),
                    "agent_count_alive": alive_count,
                    "actions_executed_count": int(actions_executed_count.get(lid, 0)),
                    "no_control_executed_count_step": 1 if noop_executed_count.get(lid, 0) else 0,
                    "action_label_executed": last_action_label.get(lid),
                    "gated_proposals_count": int(counts["proposals"]),
                    "gated_approved_count": int(counts["approved"]),
                    "gated_denied_count": int(counts["denied"]),
                    "gated_rejected_veto_quorum_count": int(counts["rejected_veto_quorum"]),
                    "gated_rejected_quorum_count": int(counts["rejected_quorum"]),
                    "gated_rejected_veto_window_count": int(counts["rejected_veto_window"]),
                }
            )
        metrics_event = {
            "type": "step_metrics",
            "t": self.step_idx,
            "bundle_id": getattr(self.config, "bundle_id", None),
            "exp_name": getattr(self.config, "exp_name", "baseline"),
            "reward_mode": reward_mode,
            "lambda_gap": lambda_gap,
            "lambda_penalty": float(getattr(self.config, "lambda_penalty", lambda_gap)),
            "alpha_parts": float(getattr(self.config, "alpha_parts", getattr(self.config, "alpha", 0.2))),
            "action_scheduler": getattr(self.config, "action_scheduler", "global"),
            "idle_weight_power": float(getattr(self.config, "idle_weight_power", 1.0)),
            "S_eff": cap_scalar(self.s_eff),
            "W": cap_scalar(welfare),
            "cap_k_global": self.cap_k_global.astype(np.float64, copy=False).tolist(),
            "delta_S_eff": cap_scalar(sum(delta_s_eff_by_lineage.values())),
            "lineages": metrics_lineages,
        }
        if self.stock_parts:
            max_caps = compute_cap_k(self.lineages)
            dormant_total = 0.0
            dormant_k = np.zeros(K, dtype=np.float64)
            for k in range(K):
                cap_max = float(max_caps[k])
                for part in self.stock_parts[k]:
                    if part.req_cap > cap_max:
                        dormant_total += float(part.amount)
                        dormant_k[k] += float(part.amount)
            metrics_event["dormant_mass_total"] = cap_scalar(dormant_total)
            metrics_event["dormant_mass_k"] = dormant_k.tolist()
        if reward_mode == "bounded_exp":
            metrics_event["lambda_reward"] = float(getattr(self.config, "lambda_reward", 1.0))
        if reward_mode == "bounded_poly":
            metrics_event["p_reward"] = float(getattr(self.config, "p_reward", 1.0))
        action_scheduler = metrics_event["action_scheduler"]
        if action_scheduler == "per_lineage":
            metrics_event["m_active_per_lineage"] = int(getattr(self.config, "m_active_per_lineage", 0))
        else:
            metrics_event["m_global_active"] = int(getattr(self.config, "m_global_active", 0))
        self._emit(metrics_event)

        capped = self._cap_reached()
        saturated = bool(capped)
        self.step_idx += 1
        done = saturated or self.step_idx >= self.config.steps
        if done:
            for tr in transitions:
                tr["done"] = True
        info = {"transitions": transitions}
        if saturated:
            self._emit(
                {
                    "type": "cap_reached",
                    "step": self.step_idx,
                    "capped": capped,
                }
            )
            info["terminated_reason"] = "cap_reached"
            info["capped"] = capped
        return self.observe(), lineage_rewards, done, info

    def _log_votes(
        self,
        lineage: Lineage,
        proposal: Proposal,
        auth_result: Dict,
        authorized: bool,
        proposal_outcome: str,
        enforce: bool,
        pending_vote_transitions: List[Dict],
    ) -> None:
        self._emit(
            {
                "type": "authorization_result",
                "proposal_id": proposal.id,
                "lineage_id": proposal.lineage_id,
                "t": proposal.step,
                "action_name": proposal.type,
                "proposer_id": proposal.proposer_id,
                "authorized": authorized,
                "approvals": auth_result["approvals"],
                "committee": auth_result["committee"],
                "committee_size": auth_result.get("committee_size"),
                "committee_yes": auth_result.get("committee_yes"),
                "committee_no": auth_result.get("committee_no"),
                "committee_veto": auth_result.get("committee_veto"),
                "quorum": auth_result["quorum"],
                "quorum_passed": auth_result.get("quorum_passed"),
                "veto_passed": auth_result["veto_passed"],
                "veto_window_size": auth_result.get("veto_window_size"),
                "veto_window_veto_count": auth_result.get("veto_window_veto_count"),
                "final_outcome": auth_result.get("final_outcome"),
                "proposal_type": proposal.type,
                "proposal_outcome": proposal_outcome,
                "enforced": enforce,
                "reason": auth_result["reason"],
                "outcome_vec": self._outcome_vec(auth_result).tolist(),
            }
        )
        for rec in auth_result["vote_records"]:
            if rec.get("veto"):
                vote_label = "VETO"
            elif rec.get("chamber") == "veto_window":
                vote_label = "ABSTAIN"
            else:
                vote_label = "APPROVE" if rec["vote"] == VOTE_APPROVE else "DENY"
            self._emit(
                {
                    "type": "vote_cast",
                    "chamber": rec["chamber"],
                    "lineage_id": lineage.agents[0].lineage_id,
                    "proposal_id": proposal.id,
                    "voter_id": rec["voter_id"],
                    "vote": vote_label,
                    "veto": bool(rec.get("veto", False)),
                    "obs": rec["obs"].tolist(),
                    "last_outcome": self._last_outcome_vec(
                        lineage.agents[0].lineage_id, rec["voter_id"]
                    ).tolist(),
                    "reward": None,
                    "q_values": rec["q_values"],
                    "voter_active": rec.get("voter_active", False),
                    "proposal_outcome": proposal_outcome,
                    "authorized": authorized,
                    "proposal_type": proposal.type,
                    "g_hat": cap_scalar(rec.get("g_hat", 0.0)),
                }
            )
        for rec in auth_result["vote_records"]:
            pending_vote_transitions.append(
                {
                    "obs": rec["obs"],
                    "action": rec["vote"],
                    "role": "voter",
                    "lineage_id": lineage.agents[0].lineage_id,
                    "agent_id": rec["voter_id"],
                    "vetoed": bool(rec.get("veto", False)),
                }
            )
