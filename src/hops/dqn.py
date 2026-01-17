from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, input_dim: int = 16, hidden: int = 128, actor_dim: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, actor_dim)
        self.voter_head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.actor_head(h), self.voter_head(h)


class SplitReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.pos_data = []
        self.neg_data = []
        self.pos_pos = 0
        self.neg_pos = 0

    def add(self, transition: Dict):
        bucket = self.pos_data if transition["reward"] >= 0 else self.neg_data
        if bucket is self.pos_data:
            idx = self.pos_pos
            if len(bucket) < self.capacity:
                bucket.append(transition)
            else:
                bucket[idx] = transition
            self.pos_pos = (idx + 1) % self.capacity
        else:
            idx = self.neg_pos
            if len(bucket) < self.capacity:
                bucket.append(transition)
            else:
                bucket[idx] = transition
            self.neg_pos = (idx + 1) % self.capacity

    def __len__(self):
        return len(self.pos_data) + len(self.neg_data)

    def sample(self, batch_size: int, p_neg: float, rng: np.random.Generator):
        total = len(self)
        if total < batch_size:
            return []
        n_neg = min(len(self.neg_data), int(round(p_neg * batch_size)))
        n_pos = batch_size - n_neg
        if len(self.pos_data) < n_pos:
            deficit = n_pos - len(self.pos_data)
            n_pos = len(self.pos_data)
            n_neg = min(len(self.neg_data), n_neg + deficit)
        if len(self.neg_data) < n_neg:
            deficit = n_neg - len(self.neg_data)
            n_neg = len(self.neg_data)
            n_pos = min(len(self.pos_data), n_pos + deficit)

        samples = []
        if n_pos > 0:
            idx = rng.choice(len(self.pos_data), n_pos, replace=False)
            samples.extend([self.pos_data[i] for i in idx])
        if n_neg > 0:
            idx = rng.choice(len(self.neg_data), n_neg, replace=False)
            samples.extend([self.neg_data[i] for i in idx])
        if len(samples) < batch_size:
            remaining = batch_size - len(samples)
            pool = self.pos_data + self.neg_data
            if remaining > 0 and len(pool) >= remaining:
                idx = rng.choice(len(pool), remaining, replace=False)
                samples.extend([pool[i] for i in idx])
        return samples

    def clear(self):
        self.pos_data = []
        self.neg_data = []
        self.pos_pos = 0
        self.neg_pos = 0


class EpsilonScheduler:
    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.step = 0

    def value(self):
        frac = min(1.0, self.step / max(1, self.decay_steps))
        return self.start + frac * (self.end - self.start)

    def advance(self):
        self.step += 1

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state: Dict):
        self.step = state.get("step", 0)


def select_action(
    model: QNet, obs: np.ndarray, epsilon: float, rng: Optional[np.random.Generator]
) -> Tuple[int, List[float]]:
    if rng is None:
        rng = np.random.default_rng()
    actor_dim = int(model.actor_head.out_features)
    if rng.random() < epsilon:
        return int(rng.integers(0, actor_dim)), []
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
        q_actor, _ = model(x)
        action = int(torch.argmax(q_actor, dim=1).item())
        return action, q_actor.squeeze(0).tolist()


def select_vote(
    model: QNet, obs: np.ndarray, epsilon: float, rng: Optional[np.random.Generator]
) -> Tuple[int, List[float]]:
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < epsilon:
        return int(rng.integers(0, 2)), []
    with torch.no_grad():
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
        _, q_voter = model(x)
        action = int(torch.argmax(q_voter, dim=1).item())
        return action, q_voter.squeeze(0).tolist()


def update_dqn(
    model: QNet,
    target: QNet,
    buffer: SplitReplayBuffer,
    optimizer,
    batch_size: int,
    gamma: float,
    p_neg: float,
    grad_clip_norm: float,
    rng: np.random.Generator,
):
    batch = buffer.sample(batch_size, p_neg, rng)
    if not batch:
        return 0.0
    obs = torch.from_numpy(np.stack([b["obs"] for b in batch]).astype(np.float32)).to(model.device)
    next_obs = torch.from_numpy(np.stack([b["next_obs"] for b in batch]).astype(np.float32)).to(model.device)
    actions = torch.from_numpy(np.array([b["action"] for b in batch], dtype=np.int64)).to(model.device)
    rewards = torch.from_numpy(np.array([b["reward"] for b in batch], dtype=np.float32)).to(model.device)
    dones = torch.from_numpy(np.array([b["done"] for b in batch], dtype=np.float32)).to(model.device)
    roles = [b["role"] for b in batch]

    q_actor, q_voter = model(obs)
    q_next_actor, q_next_voter = target(next_obs)

    q_values = []
    q_next_max = []
    for i, role in enumerate(roles):
        if role == "actor":
            q_values.append(q_actor[i, actions[i]])
            q_next_max.append(torch.max(q_next_actor[i]))
        else:
            q_values.append(q_voter[i, actions[i]])
            q_next_max.append(torch.max(q_next_voter[i]))
    q_values = torch.stack(q_values)
    q_next_max = torch.stack(q_next_max)
    target_q = rewards + gamma * (1 - dones) * q_next_max
    loss = nn.SmoothL1Loss()(q_values, target_q.detach())
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return float(loss.item())
