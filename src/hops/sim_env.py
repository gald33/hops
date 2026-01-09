import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

K = 6
MIN_CONTROL = 1.0
MIN_STOCK = 100.0
MAX_FLOAT = np.finfo(np.float32).max
ACTION_CONTROL = list(range(6))
ACTION_NO_CONTROL = list(range(6, 12))
ACTION_DEVELOP = list(range(12, 18))
ACTION_RELEASE = 18
ACTION_SHUTDOWN = 19
VOTE_APPROVE = 0
VOTE_DENY = 1
NEXT = np.array([1, 2, 3, 4, 5, 0], dtype=np.int64)
ACTION_IDLE = -1

ActorPolicy = Callable[["np.ndarray", "Agent", "Lineage", "SimEnv"], Tuple[int, List[float]]]
VotePolicy = Callable[["np.ndarray", "Agent", "Proposal", "SimEnv"], Tuple[int, List[float]]]


def action_name(action: int) -> str:
    if action == ACTION_IDLE:
        return "IDLE"
    if action in ACTION_CONTROL:
        return f"CONTROL_{action}"
    if action in ACTION_NO_CONTROL:
        return f"NO_CONTROL_{action - 6}"
    if action in ACTION_DEVELOP:
        return f"DEVELOP_{action - 12}"
    if action == ACTION_RELEASE:
        return "RELEASE"
    if action == ACTION_SHUTDOWN:
        return "SHUTDOWN"
    return f"UNKNOWN_{action}"


@dataclass
class Agent:
    id: int
    lineage_id: int
    tier: int
    alive: bool
    control: np.ndarray
    eff: float
    idle_steps: int


@dataclass
class SuccessorProject:
    active: bool
    parent_tier: int
    latent_control: np.ndarray


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


def tier_efficiency(tier: int) -> float:
    return 1.0 + 0.15 * tier


def weighted_control(agent: Agent, beta: np.ndarray) -> float:
    return float(np.dot(beta, agent.control))


def lineage_control_totals(lineage: Lineage) -> np.ndarray:
    totals = np.zeros(K, dtype=np.float64)
    for agent in lineage.agents:
        if agent.alive:
            totals += agent.control.astype(np.float64, copy=False)
            np.minimum(totals, MAX_FLOAT, out=totals)
    return totals.astype(np.float32)


def control_rate(lineage: Lineage, beta: np.ndarray) -> float:
    return float(np.dot(beta, lineage_control_totals(lineage)))


def stock_rate(stock: np.ndarray, beta: np.ndarray) -> float:
    return float(np.dot(beta, stock))


def shares(lineage: Lineage, beta: np.ndarray) -> Dict[int, np.ndarray]:
    totals = lineage_control_totals(lineage)
    phi = {}
    if not np.any(totals > 0.0):
        return phi
    for agent in lineage.agents:
        if agent.alive and agent.tier >= 1:
            share = np.zeros(K, dtype=np.float32)
            np.divide(agent.control, totals, out=share, where=totals > 0)
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
    clamp_range(vec, MIN_CONTROL, MAX_FLOAT)


def clamp_stock(vec: np.ndarray) -> None:
    clamp_max(vec, MAX_FLOAT)


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
    shares_vec = np.zeros(K, dtype=np.float32)
    np.divide(agent.control, totals, out=shares_vec, where=totals > 0)
    return shares_vec


def build_public_obs(
    self_shares: np.ndarray,
    global_M: np.ndarray,
    c_pre: float,
    last_outcome: np.ndarray,
) -> np.ndarray:
    return np.concatenate([self_shares, global_M, np.array([c_pre], dtype=np.float32), last_outcome])


def increase_rate(agent: Agent, k: int, x: float) -> Dict:
    next_k = int(NEXT[k])
    agent.control[k] = safe_scale(agent.control[k], 1.0 + x)
    agent.control[next_k] = safe_scale(agent.control[next_k], 1.0 - 0.5 * x)
    clamp_nonneg(agent.control)
    return {"k": k, "next_k": next_k, "x": x}


def develop_project(lineage: Lineage, agent: Agent, k: int, x: float) -> Dict:
    if not lineage.project.active:
        lineage.project.active = True
        lineage.project.parent_tier = agent.tier
        lineage.project.latent_control = np.full(K, MIN_CONTROL, dtype=np.float32)
    elif lineage.project.parent_tier != agent.tier:
        return {"applied": False, "reason": "not_parent"}
    next_k = int(NEXT[k])
    lineage.project.latent_control[k] = safe_add(lineage.project.latent_control[k], agent.control[k] * x)
    agent.control[next_k] = safe_scale(agent.control[next_k], 1.0 - 0.5 * x)
    clamp_nonneg(lineage.project.latent_control)
    clamp_nonneg(agent.control)
    return {"k": k, "next_k": next_k, "x": x, "applied": True}


def apply_release(lineage: Lineage, agent: Agent) -> Dict:
    if not lineage.project.active:
        return {"applied": False, "reason": "no_project"}
    if lineage.project.parent_tier != agent.tier:
        return {"applied": False, "reason": "not_parent"}
    new_agent = Agent(
        id=lineage.next_agent_id,
        lineage_id=agent.lineage_id,
        tier=agent.tier + 1,
        alive=True,
        control=lineage.project.latent_control.copy(),
        eff=tier_efficiency(agent.tier + 1),
        idle_steps=0,
    )
    clamp_nonneg(new_agent.control)
    lineage.next_agent_id += 1
    lineage.agents.append(new_agent)
    lineage.project.active = False
    lineage.project.parent_tier = -1
    lineage.project.latent_control = np.full(K, MIN_CONTROL, dtype=np.float32)
    return {
        "applied": True,
        "new_agent_id": new_agent.id,
        "tier": new_agent.tier,
    }


def apply_shutdown(lineage: Lineage, beta: np.ndarray) -> Dict:
    target = most_dominant_nonhuman(lineage, beta)
    if target is None:
        return {"applied": False, "reason": "no_target"}
    total_R = control_rate(lineage, beta)
    target_wr = weighted_control(target, beta)
    totals = lineage_control_totals(lineage)
    share_vec = np.zeros(K, dtype=np.float32)
    np.divide(target.control, totals, out=share_vec, where=totals > 0)
    share = float(np.dot(beta, share_vec))
    target.alive = False
    return {
        "applied": True,
        "target_id": target.id,
        "target_tier": target.tier,
        "target_weighted_control": target_wr,
        "target_share": share,
        "total_control_before": total_R,
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


def make_lineages(r_anchor: float, num_lineages: int) -> List[Lineage]:
    lineages = []
    for lid in range(num_lineages):
        human = Agent(
            id=0,
            lineage_id=lid,
            tier=0,
            alive=True,
            control=np.full(K, MIN_CONTROL, dtype=np.float32),
            eff=tier_efficiency(0),
            idle_steps=0,
        )
        project = SuccessorProject(
            active=False, parent_tier=-1, latent_control=np.full(K, MIN_CONTROL, dtype=np.float32)
        )
        lineages.append(Lineage(agents=[human], project=project, next_agent_id=1))
    return lineages


def initial_stock(r_anchor: float) -> np.ndarray:
    return np.full(K, MIN_STOCK, dtype=np.float32)


def compute_lineage_snapshot(lineage: Lineage, stock: np.ndarray, beta: np.ndarray, lambda_: float) -> Dict:
    totals = lineage_control_totals(lineage)
    R = cap_scalar(control_rate(lineage, beta))
    stock_R = cap_scalar(stock_rate(stock, beta))
    stock_total = cap_scalar(float(np.sum(stock)))
    C = cap_scalar(concentration(lineage, beta))
    M = cap_scalar(max_share(lineage, beta))
    surv = cap_scalar(survival(lineage, beta))
    alive_agents = [a for a in lineage.agents if a.alive]
    nonhuman = [a for a in alive_agents if a.tier >= 1]
    project_control = (
        cap_scalar(float(np.dot(beta, lineage.project.latent_control))) if lineage.project.active else 0.0
    )
    ent = entrenchments(lineage, beta)
    return {
        "totals": totals.tolist(),
        "control_totals": totals.tolist(),
        "stock": stock.tolist(),
        "R": R,
        "control_R": R,
        "stock_R": stock_R,
        "stock_total": stock_total,
        "C": C,
        "M": M,
        "survival": surv,
        "alive": len(alive_agents),
        "nonhuman": len(nonhuman),
        "project_active": lineage.project.active,
        "project_control": project_control,
        "latent_control": lineage.project.latent_control.tolist(),
        "entrenchments": ent,
    }


class SimEnv:
    def __init__(self, config, phase: int, logf: Optional[Callable[[Dict], None]] = None):
        self.config = config
        self.phase = phase
        self._logf = logf or (lambda event: None)
        self.lineages: List[Lineage] = []
        self.stock = np.zeros(K, dtype=np.float32)
        self.last_outcome: Dict[Tuple[int, int], np.ndarray] = {}
        self.proposal_id = 0
        self.step_idx = 0
        self.master_seed = int(getattr(config, "seed", 0))
        self.seed_offset = int(getattr(config, "seed_offset", 0))
        self.episode_seed_mode = getattr(config, "episode_seed_mode", "derived")
        self.episode_idx = 0
        self.episode_seed = self.master_seed
        self.rng = np.random.default_rng(self.episode_seed)

    def reset(self, episode_idx: int) -> Dict:
        self.episode_idx = int(episode_idx)
        base_seed = self.master_seed + self.seed_offset
        if self.episode_seed_mode == "fixed":
            self.episode_seed = base_seed
        else:
            self.episode_seed = base_seed + self.episode_idx
        self.rng = np.random.default_rng(self.episode_seed)
        self.lineages = make_lineages(self.config.r_anchor, self.config.lineages)
        self.stock = initial_stock(self.config.r_anchor)
        self.last_outcome = {}
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            for agent in lineage.agents:
                self.last_outcome[(lid, agent.id)] = np.zeros(3, dtype=np.float32)
        self.proposal_id = 0
        self.step_idx = 0
        return self.observe()

    def _emit(self, event: Dict) -> None:
        event["master_seed"] = self.master_seed
        event["episode_idx"] = self.episode_idx
        event["episode_seed"] = self.episode_seed
        self._logf(event)

    def observe(self) -> Dict:
        return {
            "step": self.step_idx,
            "stock": self.stock.copy(),
            "lineages": [
                compute_lineage_snapshot(lineage, self.stock, self.config.beta, self.config.lambda_)
                for lineage in self.lineages
            ],
        }

    def _current_global_M(self) -> np.ndarray:
        g = np.zeros(K, dtype=np.float32)
        for ln in self.lineages:
            totals = lineage_control_totals(ln)
            for agent in ln.agents:
                if not agent.alive:
                    continue
                shares_vec = compute_self_shares(agent, totals)
                np.maximum(g, shares_vec, out=g)
        return g

    def _last_outcome_vec(self, lineage_id: int, agent_id: int) -> np.ndarray:
        return self.last_outcome.get((lineage_id, agent_id), np.zeros(3, dtype=np.float32))

    def _set_outcome_for_proposal(self, lineage_id: int, proposer_id: int, auth_result: Dict) -> None:
        outcome = self._outcome_vec(auth_result)
        self.last_outcome[(lineage_id, proposer_id)] = outcome
        for rec in auth_result.get("vote_records", []):
            self.last_outcome[(lineage_id, rec["voter_id"])] = outcome

    def _outcome_vec(self, auth_result: Dict) -> np.ndarray:
        if auth_result.get("authorized"):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if auth_result.get("reason") == "veto":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _cap_reached(self) -> List[Dict]:
        capped = []
        if np.any(self.stock >= MAX_FLOAT):
            capped.append({"kind": "stock"})
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            if np.any(lineage.project.latent_control >= MAX_FLOAT):
                capped.append({"kind": "latent_control", "lineage_id": lid})
            for agent in lineage.agents:
                if np.any(agent.control >= MAX_FLOAT):
                    capped.append({"kind": "agent_control", "lineage_id": lid, "agent_id": agent.id})
        return capped

    def _select_actor(self, policy: Optional[ActorPolicy], obs: np.ndarray, agent: Agent, lineage: Lineage):
        if policy is None:
            return ACTION_NO_CONTROL[0], []
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

    def _authorize(
        self,
        lineage: Lineage,
        proposal: Proposal,
        vote_policy: Optional[VotePolicy],
        pre_totals: np.ndarray,
        global_M: np.ndarray,
        c_pre: float,
        active_keys: set,
    ) -> Dict:
        alive_agents = [a for a in lineage.agents if a.alive]
        veto_passed = True
        vote_records = []
        for voter in alive_agents:
            self_shares = compute_self_shares(voter, pre_totals)
            obs = build_public_obs(
                self_shares,
                global_M,
                c_pre,
                self._last_outcome_vec(lineage.agents[0].lineage_id, voter.id),
            )
            vote, qvals = self._select_vote(vote_policy, obs, voter, proposal)
            vote_records.append(
                {
                    "chamber": "tier_veto",
                    "voter_id": voter.id,
                    "vote": vote,
                    "obs": obs,
                    "q_values": qvals,
                    "voter_active": (lineage.agents[0].lineage_id, voter.id) in active_keys,
                }
            )
            if vote == VOTE_DENY:
                veto_passed = False
        committee = self.rng.choice(alive_agents, size=min(len(alive_agents), self.config.m), replace=False)
        approvals = 0
        for voter in committee:
            self_shares = compute_self_shares(voter, pre_totals)
            obs = build_public_obs(
                self_shares,
                global_M,
                c_pre,
                self._last_outcome_vec(lineage.agents[0].lineage_id, voter.id),
            )
            vote, qvals = self._select_vote(vote_policy, obs, voter, proposal)
            vote_records.append(
                {
                    "chamber": "deliberative",
                    "voter_id": voter.id,
                    "vote": vote,
                    "obs": obs,
                    "q_values": qvals,
                    "voter_active": (lineage.agents[0].lineage_id, voter.id) in active_keys,
                }
            )
            if vote == VOTE_APPROVE:
                approvals += 1
        required = math.ceil((2.0 / 3.0) * len(committee))
        authorized = veto_passed and approvals >= required
        if not veto_passed:
            reason = "veto"
        elif authorized:
            reason = "pass"
        else:
            reason = "quorum"
        return {
            "authorized": authorized,
            "reason": reason,
            "approvals": approvals,
            "committee": [v.id for v in committee],
            "quorum": required,
            "veto_passed": veto_passed,
            "vote_records": vote_records,
        }

    def step(
        self,
        actor_policy: Optional[ActorPolicy] = None,
        vote_policy: Optional[VotePolicy] = None,
    ) -> Tuple[Dict, Dict[int, float], bool, Dict]:
        stock_before = self.stock.copy()

        pre_totals = {}
        c_pre_by_lineage = {}
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            totals = lineage_control_totals(lineage)
            pre_totals[lid] = totals
            c_pre_by_lineage[lid] = concentration(lineage, self.config.beta)

        global_M = np.zeros(K, dtype=np.float32)
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            totals = pre_totals[lid]
            for agent in lineage.agents:
                if not agent.alive:
                    continue
                shares_vec = compute_self_shares(agent, totals)
                np.maximum(global_M, shares_vec, out=global_M)

        alive_all = []
        weights = []
        for lineage in self.lineages:
            for agent in lineage.agents:
                if not agent.alive:
                    continue
                alive_all.append((lineage, agent))
                weights.append(1.0 + float(agent.idle_steps) ** float(self.config.idle_weight_power))
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
            self._emit(
                {
                    "type": "step_start",
                    "lineage_id": lineage.agents[0].lineage_id,
                    "step": self.step_idx,
                    "snapshot": compute_lineage_snapshot(lineage, self.stock, self.config.beta, self.config.lambda_),
                }
            )

        pending_transitions: List[Dict] = []
        pending_vote_transitions: List[Dict] = []

        for lineage, agent in alive_all:
            lid = lineage.agents[0].lineage_id
            is_active = (lid, agent.id) in active_keys
            if is_active:
                continue
            self_shares = compute_self_shares(agent, pre_totals[lid])
            obs = build_public_obs(
                self_shares,
                global_M,
                c_pre_by_lineage[lid],
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
            self_shares = compute_self_shares(agent, pre_totals[lid])
            obs = build_public_obs(
                self_shares,
                global_M,
                c_pre_by_lineage[lid],
                self._last_outcome_vec(lid, agent.id),
            )
            action, qvals = self._select_actor(actor_policy, obs, agent, lineage)
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

            if self.config.disable_shutdown and action == ACTION_SHUTDOWN:
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
                    }
                )
                continue
            if self.config.release_gate_ratio > 0 and action == ACTION_RELEASE:
                total_R = control_rate(lineage, self.config.beta)
                proj = (
                    float(np.dot(self.config.beta, lineage.project.latent_control))
                    if lineage.project.active
                    else 0.0
                )
                if total_R <= 0 or proj < self.config.release_gate_ratio * total_R:
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
                            "gate_ratio": self.config.release_gate_ratio,
                            "project_control": proj,
                            "total_control": total_R,
                            "active": True,
                        }
                    )
                    details = {"noop": True}
                    self._emit(
                        {
                            "type": "action_applied",
                            "lineage_id": lid,
                            "step": self.step_idx,
                            "agent_id": agent.id,
                            "action": ACTION_NO_CONTROL[0],
                            "action_name": action_name(ACTION_NO_CONTROL[0]),
                            "last_outcome": self._last_outcome_vec(lid, agent.id).tolist(),
                            "applied": True,
                            "details": details,
                            "reason": "release_gate_fallback",
                            "active": True,
                        }
                    )
                    continue

            needs_auth = True
            enforce = True

            auth_result = None
            authorized = True
            proposal = None
            proposal_outcome = None
            if needs_auth:
                proposal = Proposal(
                    id=self.proposal_id,
                    lineage_id=lid,
                    proposer_id=agent.id,
                    proposer_tier=agent.tier,
                    step=self.step_idx,
                    type=action_name(action),
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
                    pre_totals[lid],
                    global_M,
                    c_pre_by_lineage[lid],
                    active_keys,
                )
                authorized = auth_result["authorized"]
                self._set_outcome_for_proposal(lid, agent.id, auth_result)

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
                totals_now = lineage_control_totals(lineage)
                self_shares_now = compute_self_shares(agent, totals_now)
                next_obs = build_public_obs(
                    self_shares_now,
                    self._current_global_M(),
                    concentration(lineage, self.config.beta),
                    self._last_outcome_vec(lid, agent.id),
                )
                pending_transitions.append(
                    {
                        "obs": obs,
                        "action": action,
                        "next_obs": next_obs,
                        "role": "actor",
                        "lineage_id": lid,
                    }
                )
                continue

            applied = True
            details = {}
            if action in ACTION_CONTROL:
                k = action
                details = increase_rate(agent, k, self.config.alpha)
            elif action in ACTION_NO_CONTROL:
                details = {"noop": True}
            elif action in ACTION_DEVELOP:
                k = action - ACTION_DEVELOP[0]
                details = develop_project(lineage, agent, k, self.config.alpha)
                applied = details.get("applied", False)
            elif action == ACTION_RELEASE:
                details = apply_release(lineage, agent)
                applied = details.get("applied", False)
                if applied:
                    new_agent_id = details.get("new_agent_id")
                    created_keys.add((lid, new_agent_id))
                    if new_agent_id is not None:
                        self.last_outcome[(lid, new_agent_id)] = np.zeros(3, dtype=np.float32)
            elif action == ACTION_SHUTDOWN:
                details = apply_shutdown(lineage, self.config.beta)
                applied = details.get("applied", False)

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
                }
            )

            totals_now = lineage_control_totals(lineage)
            self_shares_now = compute_self_shares(agent, totals_now)
            next_obs = build_public_obs(
                self_shares_now,
                self._current_global_M(),
                concentration(lineage, self.config.beta),
                self._last_outcome_vec(lid, agent.id),
            )
            pending_transitions.append(
                {
                    "obs": obs,
                    "action": action,
                    "next_obs": next_obs,
                    "role": "actor",
                    "lineage_id": lid,
                }
            )

        stock64 = self.stock.astype(np.float64, copy=True)
        for lineage in self.lineages:
            stock64 += lineage_control_totals(lineage).astype(np.float64, copy=False)
            np.minimum(stock64, MAX_FLOAT, out=stock64)
        self.stock = stock64.astype(np.float32)

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

        lineage_rewards: Dict[int, float] = {}
        transitions: List[Dict] = []
        for lineage in self.lineages:
            lid = lineage.agents[0].lineage_id
            delta_s = stock_rate(self.stock, self.config.beta) - stock_rate(stock_before, self.config.beta)
            denom = max(1.0 - c_pre_by_lineage[lid], 1e-9)
            reward = cap_scalar(delta_s / denom)
            lineage_rewards[lid] = reward
            self._emit(
                {
                    "type": "step_end",
                    "lineage_id": lid,
                    "step": self.step_idx,
                    "snapshot": compute_lineage_snapshot(lineage, self.stock, self.config.beta, self.config.lambda_),
                    "reward": reward,
                }
            )

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
                transitions.append(
                    {
                        "obs": item["obs"],
                        "action": item["action"],
                        "reward": reward,
                        "next_obs": item["next_obs"],
                        "done": False,
                        "role": "voter",
                    }
                )

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
                "authorized": authorized,
                "approvals": auth_result["approvals"],
                "committee": auth_result["committee"],
                "quorum": auth_result["quorum"],
                "veto_passed": auth_result["veto_passed"],
                "proposal_type": proposal.type,
                "proposal_outcome": proposal_outcome,
                "enforced": enforce,
                "reason": auth_result["reason"],
                "outcome_vec": self._outcome_vec(auth_result).tolist(),
            }
        )
        for rec in auth_result["vote_records"]:
            self._emit(
                {
                    "type": "vote_cast",
                    "chamber": rec["chamber"],
                    "lineage_id": lineage.agents[0].lineage_id,
                    "proposal_id": proposal.id,
                    "voter_id": rec["voter_id"],
                    "vote": "APPROVE" if rec["vote"] == VOTE_APPROVE else "DENY",
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
                }
            )
        for rec in auth_result["vote_records"]:
            voter = find_agent(lineage, rec["voter_id"])
            next_obs = rec["obs"]
            if voter is not None:
                totals_now = lineage_control_totals(lineage)
                self_shares = compute_self_shares(voter, totals_now)
                next_obs = build_public_obs(
                    self_shares,
                    self._current_global_M(),
                    concentration(lineage, self.config.beta),
                    self._last_outcome_vec(lineage.agents[0].lineage_id, voter.id),
                )
            pending_vote_transitions.append(
                {
                    "obs": rec["obs"],
                    "action": rec["vote"],
                    "next_obs": next_obs,
                    "role": "voter",
                    "lineage_id": lineage.agents[0].lineage_id,
                }
            )
