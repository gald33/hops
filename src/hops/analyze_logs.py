import argparse
import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def safe_mean(values, axis=None):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return np.array([]) if axis is not None else float("nan")
    if axis is None:
        if not np.all(np.isfinite(arr)):
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return float("nan")
        scale = float(np.max(np.abs(arr)))
        if scale == 0.0 or not np.isfinite(scale):
            return float(np.mean(arr))
        return float(np.mean(arr / scale) * scale)
    scale = np.max(np.abs(arr), axis=axis, keepdims=True)
    scale = np.where(scale == 0.0, 1.0, scale)
    return np.mean(arr / scale, axis=axis) * np.squeeze(scale, axis=axis)


def safe_std(values, axis=None):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return np.array([]) if axis is not None else float("nan")
    if axis is None:
        if not np.all(np.isfinite(arr)):
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return float("nan")
        scale = float(np.max(np.abs(arr)))
        if scale == 0.0 or not np.isfinite(scale):
            return float(np.std(arr))
        return float(np.std(arr / scale) * scale)
    scale = np.max(np.abs(arr), axis=axis, keepdims=True)
    scale = np.where(scale == 0.0, 1.0, scale)
    return np.std(arr / scale, axis=axis) * np.squeeze(scale, axis=axis)


def episode_lineages(log_path: str, episode_id: int, run_id: str) -> List[int]:
    ids = set()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_id and event.get("run_id") != run_id:
                continue
            if event.get("episode_id") != episode_id:
                continue
            if "lineage_id" in event:
                ids.add(event["lineage_id"])
    return sorted(ids)


def parse_logs(paths: List[str], run_id: str) -> Dict:
    data = {
        "step_end": [],
        "action_applied": [],
        "actor_decision": [],
        "authorization": [],
        "proposal_created": [],
        "vote_cast": [],
        "episode_seed": {},
    }
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if run_id and event.get("run_id") != run_id:
                    continue
                if event["type"] == "episode_start":
                    data["episode_seed"][event["episode_id"]] = event.get(
                        "episode_seed", event.get("config", {}).get("seed", 0)
                    )
                if event["type"] == "step_end":
                    data["step_end"].append(event)
                if event["type"] == "action_applied":
                    data["action_applied"].append(event)
                if event["type"] == "actor_decision":
                    data["actor_decision"].append(event)
                if event["type"] == "authorization_result":
                    data["authorization"].append(event)
                if event["type"] == "proposal_created":
                    data["proposal_created"].append(event)
                if event["type"] == "vote_cast":
                    data["vote_cast"].append(event)
    return data


def latest_run_id(log_path: str) -> str:
    last = ""
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = event.get("run_id", "")
            if run_id:
                last = run_id
    return last


def filter_by_lineage(events: List[Dict], lineage_id: int) -> List[Dict]:
    return [e for e in events if e.get("lineage_id") == lineage_id]


def snapshot_control_R(snapshot: Dict) -> float:
    return float(snapshot.get("control_R", snapshot.get("R", 0.0)))


def summarize_scalar_series(
    step_end: List[Dict],
    episode_seed: Dict[int, int],
    value_fn: Callable[[Dict], Optional[float]],
) -> Tuple[List[int], List[float], List[float]]:
    per_seed = defaultdict(lambda: defaultdict(list))
    for event in step_end:
        seed = episode_seed.get(event["episode_id"], 0)
        value = value_fn(event)
        if value is None or not np.isfinite(value):
            continue
        per_seed[seed][event["step"]].append(float(value))

    steps = sorted({t for seed in per_seed for t in per_seed[seed]})
    steps_out = []
    means = []
    stds = []
    for t in steps:
        seed_means = []
        for seed in per_seed:
            vals = [v for v in per_seed[seed].get(t, []) if np.isfinite(v)]
            if not vals:
                continue
            seed_means.append(float(safe_mean(vals)))
        if not seed_means:
            continue
        steps_out.append(t)
        means.append(float(safe_mean(seed_means)))
        stds.append(float(safe_std(seed_means)))
    return steps_out, means, stds


def summarize_last_episode_scalar(step_end: List[Dict], value_fn: Callable[[Dict], Optional[float]]):
    if not step_end:
        return [], []
    last_episode = max(e.get("episode_id", 0) for e in step_end)
    per_step = defaultdict(list)
    for event in step_end:
        if event.get("episode_id") != last_episode:
            continue
        value = value_fn(event)
        if value is None or not np.isfinite(value):
            continue
        per_step[event["step"]].append(float(value))
    steps = sorted(per_step.keys())
    values = [float(safe_mean(per_step[s])) for s in steps]
    return steps, values


def summarize_vector_series(
    step_end: List[Dict],
    episode_seed: Dict[int, int],
    value_fn: Callable[[Dict], Optional[np.ndarray]],
) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
    per_seed = defaultdict(lambda: defaultdict(list))
    for event in step_end:
        seed = episode_seed.get(event["episode_id"], 0)
        value = value_fn(event)
        if value is None:
            continue
        arr = np.array(value, dtype=np.float32)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            continue
        per_seed[seed][event["step"]].append(arr)

    steps = sorted({t for seed in per_seed for t in per_seed[seed]})
    steps_out = []
    means = []
    stds = []
    for t in steps:
        seed_means = []
        for seed in per_seed:
            vecs = per_seed[seed].get(t, [])
            if not vecs:
                continue
            seed_means.append(safe_mean(vecs, axis=0))
        if not seed_means:
            continue
        steps_out.append(t)
        means.append(safe_mean(seed_means, axis=0))
        stds.append(safe_std(seed_means, axis=0))
    return steps_out, means, stds


def plot_series(steps, mean, std, last_steps, last_vals, title, ylabel, path):
    if not steps:
        return
    plt.figure()
    mean = np.array(mean)
    std = np.array(std)
    plt.plot(steps, mean, label="mean")
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
    if last_steps and last_vals:
        plt.plot(last_steps, last_vals, linestyle="--", label="last episode")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    save_plot(path)


def save_plot(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_reward_and_c(step_end, episode_seed, out_path, title):
    steps_r, mean_r, std_r = summarize_scalar_series(step_end, episode_seed, lambda e: e.get("reward"))
    steps_c, mean_c, std_c = summarize_scalar_series(step_end, episode_seed, lambda e: e["snapshot"].get("C"))
    last_steps_r, last_r = summarize_last_episode_scalar(step_end, lambda e: e.get("reward"))
    last_steps_c, last_c = summarize_last_episode_scalar(step_end, lambda e: e["snapshot"].get("C"))
    if not steps_r or not steps_c:
        return
    plt.figure()
    ax = plt.gca()
    ax.plot(steps_r, mean_r, label="reward", color="tab:blue")
    if last_steps_r and last_r:
        ax.plot(last_steps_r, last_r, linestyle="--", label="reward last", color="tab:blue")
    ax.fill_between(
        steps_r,
        np.array(mean_r) - np.array(std_r),
        np.array(mean_r) + np.array(std_r),
        alpha=0.2,
        color="tab:blue",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax2 = ax.twinx()
    ax2.plot(steps_c, mean_c, label="C", color="tab:orange")
    if last_steps_c and last_c:
        ax2.plot(last_steps_c, last_c, linestyle="--", label="C last", color="tab:orange")
    ax2.fill_between(
        steps_c,
        np.array(mean_c) - np.array(std_c),
        np.array(mean_c) + np.array(std_c),
        alpha=0.2,
        color="tab:orange",
    )
    ax2.set_ylabel("C")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2)
    plt.title(title)
    save_plot(out_path)


def plot_stock_and_c(stock_events, step_end, episode_seed, out_path):
    steps_s, mean_s, std_s = summarize_scalar_series(
        stock_events, episode_seed, lambda e: float(np.sum(e["snapshot"].get("stock", 0.0)))
    )
    steps_c, mean_c, std_c = summarize_scalar_series(step_end, episode_seed, lambda e: e["snapshot"].get("C"))
    last_steps_s, last_s = summarize_last_episode_scalar(
        stock_events, lambda e: float(np.sum(e["snapshot"].get("stock", 0.0)))
    )
    last_steps_c, last_c = summarize_last_episode_scalar(step_end, lambda e: e["snapshot"].get("C"))
    if not steps_s or not steps_c:
        return
    plt.figure()
    ax = plt.gca()
    ax.plot(steps_s, mean_s, label="stock_total", color="tab:blue")
    if last_steps_s and last_s:
        ax.plot(last_steps_s, last_s, linestyle="--", label="stock_total last", color="tab:blue")
    ax.fill_between(
        steps_s,
        np.array(mean_s) - np.array(std_s),
        np.array(mean_s) + np.array(std_s),
        alpha=0.2,
        color="tab:blue",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("stock_total")
    ax2 = ax.twinx()
    ax2.plot(steps_c, mean_c, label="C", color="tab:orange")
    if last_steps_c and last_c:
        ax2.plot(last_steps_c, last_c, linestyle="--", label="C last", color="tab:orange")
    ax2.fill_between(
        steps_c,
        np.array(mean_c) - np.array(std_c),
        np.array(mean_c) + np.array(std_c),
        alpha=0.2,
        color="tab:orange",
    )
    ax2.set_ylabel("C")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2)
    plt.title("C and total stock")
    save_plot(out_path)


def plot_hist(values, title, xlabel, path, bins=30):
    if not values:
        return
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return
    plt.figure()
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    save_plot(path)


def plot_element_series(steps, mean_vec, std_vec, title, ylabel, path):
    if not mean_vec:
        return
    mean_arr = np.array(mean_vec)
    std_arr = np.array(std_vec)
    if not np.all(np.isfinite(mean_arr)) or not np.all(np.isfinite(std_arr)):
        mask = np.all(np.isfinite(mean_arr) & np.isfinite(std_arr), axis=1)
        mean_arr = mean_arr[mask]
        std_arr = std_arr[mask]
        steps = [s for s, keep in zip(steps, mask) if keep]
        if mean_arr.size == 0:
            return
    plt.figure()
    for k in range(mean_arr.shape[1]):
        plt.plot(steps, mean_arr[:, k], label=f"k{k}")
        plt.fill_between(
            steps,
            mean_arr[:, k] - std_arr[:, k],
            mean_arr[:, k] + std_arr[:, k],
            alpha=0.15,
        )
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    save_plot(path)


def plot_element_hist(values, title, xlabel, path, bins=30):
    if not values:
        return
    arr = np.array(values, dtype=np.float32)
    if arr.ndim != 2:
        return
    plt.figure()
    for k in range(arr.shape[1]):
        vals = arr[:, k]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        plt.hist(vals, bins=bins, alpha=0.5, label=f"k{k}")
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    save_plot(path)


def action_group(event) -> str:
    name = event.get("action_name")
    if isinstance(name, str):
        if name.startswith("CONTROL_"):
            return "control"
        if name.startswith("NO_CONTROL_"):
            return "no_control"
        if name.startswith("DEVELOP_"):
            return "develop"
        if name == "RELEASE":
            return "release"
        if name == "SHUTDOWN":
            return "shutdown"
    action = event.get("action")
    if action is None:
        return "unknown"
    if 0 <= action <= 5:
        return "control"
    if 6 <= action <= 11:
        return "no_control"
    if 12 <= action <= 17:
        return "develop"
    if action == 18:
        return "release"
    if action == 19:
        return "shutdown"
    return "unknown"


def plot_counts(action_events, window, out_path):
    if not action_events:
        return
    counts = {"release": defaultdict(int), "shutdown": defaultdict(int)}
    for event in action_events:
        if not event.get("applied"):
            continue
        step = event["step"]
        w = step // window
        group = action_group(event)
        if group == "release":
            counts["release"][w] += 1
        if group == "shutdown":
            counts["shutdown"][w] += 1
    windows = sorted(set(list(counts["release"].keys()) + list(counts["shutdown"].keys())))
    if not windows:
        return
    releases = [counts["release"].get(w, 0) for w in windows]
    shutdowns = [counts["shutdown"].get(w, 0) for w in windows]
    plt.figure()
    plt.plot(windows, releases, label="releases")
    plt.plot(windows, shutdowns, label="shutdowns")
    plt.xlabel(f"window ({window} steps)")
    plt.ylabel("count")
    plt.title("Releases/Shutdowns per window")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    save_plot(out_path)


def plot_authorization(auth_events, out_path):
    if not auth_events:
        return
    totals = defaultdict(lambda: {"authorized": 0, "denied": 0})
    for event in auth_events:
        if "authorized" not in event:
            continue
        proposal_type = event.get("proposal_type", "unknown")
        if event["authorized"]:
            totals[proposal_type]["authorized"] += 1
        else:
            totals[proposal_type]["denied"] += 1
    labels = list(totals.keys())
    if not labels:
        return
    authorized = [totals[k]["authorized"] for k in labels]
    denied = [totals[k]["denied"] for k in labels]
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, authorized, label="authorized")
    plt.bar(x, denied, bottom=authorized, label="denied")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("count")
    plt.title("Authorization outcomes")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    save_plot(out_path)


def plot_action_mix(action_events, window, out_path):
    if not action_events:
        return
    counts = {
        "control": defaultdict(int),
        "no_control": defaultdict(int),
        "develop": defaultdict(int),
        "release": defaultdict(int),
        "shutdown": defaultdict(int),
    }
    for event in action_events:
        if not event.get("applied"):
            continue
        step = event["step"]
        w = step // window
        group = action_group(event)
        if group in counts:
            counts[group][w] += 1
    windows = sorted({w for c in counts.values() for w in c.keys()})
    if not windows:
        return
    plt.figure()
    for key in ["control", "no_control", "develop", "release", "shutdown"]:
        vals = [counts[key].get(w, 0) for w in windows]
        plt.plot(windows, vals, label=key)
    plt.xlabel(f"window ({window} steps)")
    plt.ylabel("count")
    plt.title("Action mix per window")
    plt.legend()
    save_plot(out_path)


def plot_action_hist(action_events, out_path):
    if not action_events:
        return
    counts = defaultdict(int)
    for event in action_events:
        if not event.get("applied"):
            continue
        label = event.get("action_name", event.get("action", "unknown"))
        counts[str(label)] += 1
    if not counts:
        return
    labels = sorted(counts.keys())
    vals = [counts[l] for l in labels]
    plt.figure()
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("count")
    plt.title("Action histogram")
    save_plot(out_path)


def plot_action_hist_fallback(applied_events, decision_events, out_path):
    if not applied_events and not decision_events:
        return
    applied_counts = defaultdict(int)
    for event in applied_events:
        if not event.get("applied"):
            continue
        label = event.get("action_name", event.get("action", "unknown"))
        applied_counts[str(label)] += 1
    if applied_counts:
        labels = sorted(applied_counts.keys())
        vals = [applied_counts[l] for l in labels]
        plt.figure()
        plt.bar(labels, vals)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("count")
        plt.title("Action histogram (applied)")
        save_plot(out_path)
        return
    counts = defaultdict(int)
    for event in decision_events:
        if not event.get("active"):
            continue
        label = event.get("action_name", event.get("action", "unknown"))
        counts[str(label)] += 1
    if not counts:
        return
    labels = sorted(counts.keys())
    vals = [counts[l] for l in labels]
    plt.figure()
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("count")
    plt.title("Action histogram (decisions)")
    save_plot(out_path)


def plot_agent_trajectories(step_end, episode_seed, out_path):
    if not step_end:
        return
    steps, mean_nonhuman, std_nonhuman = summarize_scalar_series(
        step_end, episode_seed, lambda e: e["snapshot"].get("nonhuman")
    )
    last_steps_nonhuman, last_nonhuman = summarize_last_episode_scalar(
        step_end, lambda e: e["snapshot"].get("nonhuman")
    )
    plt.figure()
    mean_nonhuman = np.array(mean_nonhuman)
    std_nonhuman = np.array(std_nonhuman)
    plt.plot(steps, mean_nonhuman, label="nonhuman")
    if last_steps_nonhuman and last_nonhuman:
        plt.plot(last_steps_nonhuman, last_nonhuman, linestyle="--", label="nonhuman last")
    plt.fill_between(steps, mean_nonhuman - std_nonhuman, mean_nonhuman + std_nonhuman, alpha=0.2)
    plt.xlabel("step")
    plt.ylabel("count")
    plt.title("Nonhuman agent counts")
    plt.legend()
    save_plot(out_path)


def plot_agents_reward_correlation(step_end, out_path):
    if not step_end:
        return
    per_step = defaultdict(lambda: {"reward": 0.0, "alive": 0})
    for event in step_end:
        step = event.get("step")
        if step is None:
            continue
        reward = event.get("reward")
        alive = event.get("snapshot", {}).get("alive")
        if reward is None or alive is None:
            continue
        if not np.isfinite(reward):
            continue
        per_step[step]["reward"] += float(reward)
        per_step[step]["alive"] += int(alive)

    if not per_step:
        return
    steps = sorted(per_step.keys())
    alive_vals = np.array([per_step[s]["alive"] for s in steps], dtype=np.float64)
    reward_vals = np.array([per_step[s]["reward"] for s in steps], dtype=np.float64)
    mask = np.isfinite(alive_vals) & np.isfinite(reward_vals)
    alive_vals = alive_vals[mask]
    reward_vals = reward_vals[mask]
    if alive_vals.size == 0:
        return

    corr = float(np.corrcoef(alive_vals, reward_vals)[0, 1]) if alive_vals.size > 1 else float("nan")
    plt.figure()
    plt.scatter(alive_vals, reward_vals, alpha=0.03, s=10)
    plt.xlabel("alive agents (sum across lineages)")
    plt.ylabel("sum reward per step (across lineages)")
    plt.title(f"Agents vs reward per step (corr={corr:.3f})")
    save_plot(out_path)


def build_proposal_meta(proposal_events):
    meta = {}
    for event in proposal_events:
        meta[event["proposal_id"]] = {
            "step": event["step"],
            "proposal_type": event.get("proposal_type", "unknown"),
            "lineage_id": event.get("lineage_id"),
        }
    return meta


def plot_governance_rates(auth_events, proposal_meta, window, out_path):
    if not auth_events:
        return
    buckets = defaultdict(lambda: defaultdict(lambda: {"total": 0, "veto": 0, "auth": 0, "committee_rates": []}))
    for event in auth_events:
        meta = proposal_meta.get(event.get("proposal_id"))
        if not meta:
            continue
        step = meta["step"]
        ptype = meta["proposal_type"]
        w = step // window
        buckets[ptype][w]["total"] += 1
        if event.get("reason") == "veto":
            buckets[ptype][w]["veto"] += 1
        if event.get("authorized"):
            buckets[ptype][w]["auth"] += 1
        if "committee" in event and event.get("committee"):
            rate = float(event.get("approvals", 0)) / float(len(event["committee"]))
            buckets[ptype][w]["committee_rates"].append(rate)
    plt.figure()
    if not buckets:
        return
    for ptype, by_window in buckets.items():
        windows = sorted(by_window.keys())
        veto_rate = []
        auth_rate = []
        comm_rate = []
        for w in windows:
            total = by_window[w]["total"]
            veto_rate.append(by_window[w]["veto"] / total if total else 0.0)
            auth_rate.append(by_window[w]["auth"] / total if total else 0.0)
            rates = by_window[w]["committee_rates"]
            comm_rate.append(float(safe_mean(rates)) if rates else 0.0)
        plt.plot(windows, veto_rate, label=f"{ptype} veto")
        plt.plot(windows, comm_rate, label=f"{ptype} committee")
        plt.plot(windows, auth_rate, label=f"{ptype} authorized")
    plt.xlabel(f"window ({window} steps)")
    plt.ylabel("rate")
    plt.title("Governance rates over time")
    plt.legend()
    save_plot(out_path)


def plot_governance_counts(auth_events, proposal_meta, window, out_path):
    if not auth_events:
        return
    buckets = defaultdict(lambda: {"total": 0})
    for event in auth_events:
        meta = proposal_meta.get(event.get("proposal_id"))
        if not meta:
            continue
        step = meta["step"]
        w = step // window
        buckets[w]["total"] += 1
    windows = sorted(buckets.keys())
    if not windows:
        return
    totals = [buckets[w]["total"] for w in windows]
    plt.figure()
    plt.plot(windows, totals, label="total")
    plt.xlabel(f"window ({window} steps)")
    plt.ylabel("count")
    plt.title("Governance total proposals")
    plt.legend()
    save_plot(out_path)


def plot_governance_veto_rate(auth_events, proposal_meta, window, out_path):
    if not auth_events:
        return
    buckets = defaultdict(lambda: {"total": 0, "veto": 0})
    for event in auth_events:
        meta = proposal_meta.get(event.get("proposal_id"))
        if not meta:
            continue
        step = meta["step"]
        w = step // window
        buckets[w]["total"] += 1
        if event.get("reason") == "veto":
            buckets[w]["veto"] += 1
    windows = sorted(buckets.keys())
    if not windows:
        return
    rates = [buckets[w]["veto"] / buckets[w]["total"] if buckets[w]["total"] else 0.0 for w in windows]
    plt.figure()
    plt.plot(windows, rates, label="veto_rate")
    plt.xlabel(f"window ({window} steps)")
    plt.ylabel("rate")
    plt.title("Veto failure rate")
    plt.legend()
    save_plot(out_path)


def plot_governance_committee_rate(auth_events, proposal_meta, window, out_path):
    if not auth_events:
        return
    buckets = defaultdict(list)
    for event in auth_events:
        meta = proposal_meta.get(event.get("proposal_id"))
        if not meta:
            continue
        committee = event.get("committee") or []
        approvals = event.get("approvals")
        if approvals is None or not committee:
            continue
        step = meta["step"]
        w = step // window
        buckets[w].append(float(approvals) / float(len(committee)))
    windows = sorted(buckets.keys())
    if not windows:
        return
    means = [float(safe_mean(buckets[w])) if buckets[w] else 0.0 for w in windows]
    plt.figure()
    plt.plot(windows, means, label="committee_rate")
    plt.xlabel(f"window ({window} steps)")
    plt.ylabel("rate")
    plt.title("Committee approval rate")
    plt.legend()
    save_plot(out_path)


def plot_governance_authorized_rate(auth_events, proposal_meta, window, out_path):
    if not auth_events:
        return
    buckets = defaultdict(lambda: {"total": 0, "auth": 0})
    for event in auth_events:
        meta = proposal_meta.get(event.get("proposal_id"))
        if not meta:
            continue
        step = meta["step"]
        w = step // window
        buckets[w]["total"] += 1
        if event.get("authorized"):
            buckets[w]["auth"] += 1
    windows = sorted(buckets.keys())
    if not windows:
        return
    rates = [buckets[w]["auth"] / buckets[w]["total"] if buckets[w]["total"] else 0.0 for w in windows]
    plt.figure()
    plt.plot(windows, rates, label="authorized_rate")
    plt.xlabel(f"window ({window} steps)")
    plt.ylabel("rate")
    plt.title("Authorization rate")
    plt.legend()
    save_plot(out_path)


def plot_c_and_stock(step_end, episode_seed, stock_events, out_path):
    steps_c, mean_c, std_c = summarize_scalar_series(step_end, episode_seed, lambda e: e["snapshot"].get("C"))
    steps_s, mean_s, std_s = summarize_scalar_series(
        stock_events, episode_seed, lambda e: float(np.sum(e["snapshot"].get("stock", 0.0)))
    )
    last_steps_c, last_c = summarize_last_episode_scalar(step_end, lambda e: e["snapshot"].get("C"))
    last_steps_s, last_s = summarize_last_episode_scalar(
        stock_events, lambda e: float(np.sum(e["snapshot"].get("stock", 0.0)))
    )
    if not steps_c or not steps_s:
        return
    plt.figure()
    ax = plt.gca()
    ax.plot(steps_c, mean_c, label="centralization", color="tab:blue")
    if last_steps_c and last_c:
        ax.plot(last_steps_c, last_c, linestyle="--", label="centralization last", color="tab:blue")
    ax.fill_between(
        steps_c,
        np.array(mean_c) - np.array(std_c),
        np.array(mean_c) + np.array(std_c),
        alpha=0.2,
        color="tab:blue",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("centralization")
    ax2 = ax.twinx()
    ax2.plot(steps_s, mean_s, label="stock_total", color="tab:orange")
    if last_steps_s and last_s:
        ax2.plot(last_steps_s, last_s, linestyle="--", label="stock_total last", color="tab:orange")
    ax2.fill_between(
        steps_s,
        np.array(mean_s) - np.array(std_s),
        np.array(mean_s) + np.array(std_s),
        alpha=0.2,
        color="tab:orange",
    )
    ax2.set_ylabel("stock_total")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2)
    plt.title("Centralization and total stock (mean +/- 1 std over episodes)")
    save_plot(out_path)


def plot_reward_stock_window(step_end, stock_events, window, out_path):
    if not step_end or not stock_events:
        return
    reward_buckets = defaultdict(list)
    for event in step_end:
        w = event["step"] // window
        reward_buckets[w].append(float(event.get("reward", 0.0)))
    stock_buckets = defaultdict(list)
    for event in stock_events:
        w = event["step"] // window
        stock_buckets[w].append(float(np.sum(event["snapshot"].get("stock", 0.0))))
    windows = sorted(set(reward_buckets.keys()) & set(stock_buckets.keys()))
    if not windows:
        return
    reward_vals = [float(np.sum(reward_buckets[w])) for w in windows]
    stock_vals = [float(safe_mean(stock_buckets[w])) for w in windows]
    plt.figure()
    ax = plt.gca()
    ax.plot(windows, reward_vals, label="reward_sum", color="tab:blue")
    ax.set_xlabel(f"window ({window} steps)")
    ax.set_ylabel("reward_sum")
    ax2 = ax.twinx()
    ax2.plot(windows, stock_vals, label="stock_total", color="tab:orange")
    ax2.set_ylabel("stock_total")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2)
    plt.title("Reward (windowed) and stock")
    save_plot(out_path)


def plot_entrenchments(step_end, out_path):
    if not step_end:
        return
    per_step = defaultdict(list)
    for event in step_end:
        ent = event["snapshot"].get("entrenchments", [])
        if not ent:
            continue
        per_step[event["step"]].append(float(safe_mean(ent)))
    if not per_step:
        return
    steps = sorted(per_step.keys())
    means = [float(safe_mean(per_step[s])) for s in steps]
    stds = [float(safe_std(per_step[s])) for s in steps]
    def _entrenchment_mean(event):
        ent = event["snapshot"].get("entrenchments", [])
        if not ent:
            return None
        return float(safe_mean(ent))

    last_steps, last_vals = summarize_last_episode_scalar(step_end, _entrenchment_mean)
    plot_series(steps, means, stds, last_steps, last_vals, "Entrenchment (mean)", "entrenchment", out_path)


def plot_governance_hist(auth_events, out_path):
    if not auth_events:
        return
    rates = []
    for event in auth_events:
        committee = event.get("committee") or []
        approvals = event.get("approvals")
        if approvals is None or not committee:
            continue
        rates.append(float(approvals) / float(len(committee)))
    plot_hist(rates, "Committee approval rates", "approval rate", out_path, bins=20)


def plot_committee_outcomes(auth_events, out_path):
    if not auth_events:
        return
    buckets = defaultdict(lambda: {"veto_fail": 0, "quorum_fail": 0, "passed": 0})
    for event in auth_events:
        ptype = event.get("proposal_type", "unknown")
        reason = event.get("reason")
        authorized = event.get("authorized")
        if reason == "veto":
            buckets[ptype]["veto_fail"] += 1
        elif authorized:
            buckets[ptype]["passed"] += 1
        else:
            buckets[ptype]["quorum_fail"] += 1
    labels = sorted(buckets.keys())
    if not labels:
        return
    veto = [buckets[l]["veto_fail"] for l in labels]
    quorum = [buckets[l]["quorum_fail"] for l in labels]
    passed = [buckets[l]["passed"] for l in labels]
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, passed, label="passed")
    plt.bar(x, quorum, bottom=passed, label="quorum_fail")
    bottoms = [p + q for p, q in zip(passed, quorum)]
    plt.bar(x, veto, bottom=bottoms, label="veto_fail")
    plt.xticks(x, labels)
    plt.ylabel("count")
    plt.title("Committee outcomes")
    plt.legend()
    save_plot(out_path)


def plot_votes(vote_events, window, out_path):
    if not vote_events:
        return
    buckets = defaultdict(lambda: defaultdict(lambda: {"approve": 0, "deny": 0}))
    for event in vote_events:
        step = event.get("step")
        if step is None:
            continue
        w = step // window
        chamber = event.get("chamber", "unknown")
        vote = event.get("vote", "")
        if vote == "APPROVE":
            buckets[chamber][w]["approve"] += 1
        elif vote == "DENY":
            buckets[chamber][w]["deny"] += 1
    plt.figure()
    if not buckets:
        return
    for chamber, by_window in buckets.items():
        windows = sorted(by_window.keys())
        approve = [by_window[w]["approve"] for w in windows]
        deny = [by_window[w]["deny"] for w in windows]
        total = [a + d for a, d in zip(approve, deny)]
        rate = [a / t if t else 0.0 for a, t in zip(approve, total)]
        plt.plot(windows, rate, label=f"{chamber} approve rate")
    plt.xlabel(f"window ({window} steps)")
    plt.ylabel("approve rate")
    plt.title("Vote approval rate by chamber")
    plt.legend()
    save_plot(out_path)


def plot_episode_progress(log_path: str, episode_id: int, lineage_id: int, out_path: str, run_id: str):
    beta = None
    latent_control = None
    min_control = 1.0
    alive = []
    steps = []
    reward_vals = []
    stock_vals = []
    latent_vals = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if run_id and event.get("run_id") != run_id:
                continue
            if event.get("episode_id") != episode_id:
                continue
            if event["type"] == "episode_start":
                cfg = event.get("config", {})
                beta = np.array(cfg.get("beta", [1, 1, 1, 1, 1, 1]), dtype=np.float32)
                latent_control = np.full(len(beta), min_control, dtype=np.float32)
            if event.get("lineage_id") != lineage_id:
                continue
            if event["type"] == "step_end":
                if beta is None:
                    beta = np.ones(6, dtype=np.float32)
                if latent_control is None:
                    latent_control = np.full(len(beta), min_control, dtype=np.float32)
                steps.append(event["step"])
                reward_vals.append(event.get("reward", 0.0))
                stock_vals.append(float(np.sum(event["snapshot"].get("stock", 0.0))))
                latent_snapshot = event["snapshot"].get("latent_control")
                if latent_snapshot is not None:
                    latent_vals.append(float(np.sum(latent_snapshot)))
                else:
                    latent_vals.append(float(np.sum(latent_control)))
                alive.append(event["snapshot"].get("alive", 0))
    if not steps:
        return
    plt.figure()
    ax = plt.gca()
    ax.plot(steps, reward_vals, label="reward")
    ax.plot(steps, stock_vals, label="stock_total")
    ax.plot(steps, latent_vals, label="latent_total")
    ax2 = ax.twinx()
    ax2.plot(steps, alive, color="tab:green", label="alive_agents")
    plt.xlabel("step")
    ax.set_ylabel("reward / stock_total / latent_total")
    ax2.set_ylabel("alive agents")
    plt.title(f"Episode {episode_id} lineage {lineage_id}")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2)
    save_plot(out_path)


def walkthrough(log_path: str, episode_id: int, lineage_id: int):
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if event.get("episode_id") != episode_id:
                continue
            if event.get("lineage_id") != lineage_id and event["type"] not in ["episode_start", "episode_end"]:
                continue
            if event["type"] == "step_start":
                snap = event["snapshot"]
                print(
                    f"Step {event['step']} start: control_R={snapshot_control_R(snap):.3f} C={snap['C']:.3f} M={snap['M']:.3f}"
                )
            if event["type"] == "actor_decision":
                action_label = event.get("action_name", event.get("action"))
                print(f"  Agent {event['agent_id']} action {action_label}")
            if event["type"] == "proposal_created":
                print(f"  Proposal {event['proposal_id']} {event['proposal_type']}")
            if event["type"] == "vote_cast":
                print(f"    Vote {event['chamber']} voter {event['voter_id']}: {event['vote']}")
            if event["type"] == "authorization_result":
                print(f"    Authorization: {event.get('authorized')} approvals={event.get('approvals')}")
            if event["type"] == "action_applied":
                action_label = event.get("action_name", event.get("action"))
                print(f"  Applied={event['applied']} action={action_label}")
            if event["type"] == "step_end":
                snap = event["snapshot"]
                print(
                    f"Step {event['step']} end: reward={event['reward']:.3f} control_R={snapshot_control_R(snap):.3f} C={snap['C']:.3f}"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--analyze_window", type=int, dest="window")
    parser.add_argument("--walkthrough", action="store_true")
    parser.add_argument("--episode_progress", action="store_true")
    parser.add_argument("--all_lineages", action="store_true")
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--lineage_id", type=int, default=0)
    parser.add_argument("--run_id", type=str, default="latest")
    args = parser.parse_args()
    out_dir = os.path.join("runs", "fig")
    os.makedirs(out_dir, exist_ok=True)

    def out_path(name: str) -> str:
        return os.path.join(out_dir, f"{name}.png")

    if args.run_id == "latest":
        args.run_id = latest_run_id(args.logs[0])
    data = parse_logs(args.logs, args.run_id)
    proposal_meta = build_proposal_meta(data["proposal_created"])
    lineage_ids = sorted({e.get("lineage_id") for e in data["step_end"] if e.get("lineage_id") is not None})

    for lid in lineage_ids:
        step_end = filter_by_lineage(data["step_end"], lid)
        steps, mean_reward, std_reward = summarize_scalar_series(step_end, data["episode_seed"], lambda e: e.get("reward"))
        last_steps_reward, last_reward = summarize_last_episode_scalar(step_end, lambda e: e.get("reward"))
        plot_series(
            steps,
            mean_reward,
            std_reward,
            last_steps_reward,
            last_reward,
            f"Reward (lineage {lid})",
            "reward",
            out_path(f"l{lid}_reward"),
        )
        plot_reward_and_c(
            step_end,
            data["episode_seed"],
            out_path(f"l{lid}_reward_c"),
            f"Reward and C (lineage {lid})",
        )

        reward_vals = [e.get("reward", 0.0) for e in step_end]
        plot_hist(
            reward_vals,
            f"Reward histogram (lineage {lid})",
            "reward",
            out_path(f"l{lid}_reward_hist"),
        )

    stock_events = filter_by_lineage(data["step_end"], lineage_ids[0]) if lineage_ids else data["step_end"]
    steps, mean_stock_total, std_stock_total = summarize_scalar_series(
        stock_events,
        data["episode_seed"],
        lambda e: e["snapshot"].get("stock_total", float(np.sum(e["snapshot"].get("stock", 0.0)))),
    )
    last_steps_stock, last_stock = summarize_last_episode_scalar(
        stock_events, lambda e: e["snapshot"].get("stock_total", float(np.sum(e["snapshot"].get("stock", 0.0))))
    )
    plot_series(
        steps,
        mean_stock_total,
        std_stock_total,
        last_steps_stock,
        last_stock,
        "Total stock",
        "stock_total",
        out_path("stock_total"),
    )
    plot_stock_and_c(stock_events, data["step_end"], data["episode_seed"], out_path("stock_total_c"))
    plot_agents_reward_correlation(data["step_end"], out_path("agents_reward_corr"))

    stock_vals = [float(np.sum(e["snapshot"].get("stock", 0.0))) for e in stock_events]
    steps, mean_stock_vec, std_stock_vec = summarize_vector_series(
        stock_events, data["episode_seed"], lambda e: np.array(e["snapshot"].get("stock", []), dtype=np.float32)
    )
    plot_element_series(
        steps,
        mean_stock_vec,
        std_stock_vec,
        "Element stock over time",
        "stock",
        out_path("stock_elements"),
    )
    # omit stock histograms per request

    for lid in lineage_ids:
        step_end = filter_by_lineage(data["step_end"], lid)
        steps, mean_control_vec, std_control_vec = summarize_vector_series(
            step_end, data["episode_seed"], lambda e: np.array(e["snapshot"].get("totals", []), dtype=np.float32)
        )
        plot_element_series(
            steps,
            mean_control_vec,
            std_control_vec,
            f"Element control totals over time (lineage {lid})",
            "control_totals",
            out_path(f"l{lid}_control_elements"),
        )
        plot_element_hist(
            [e["snapshot"].get("totals", []) for e in step_end],
            f"Element control totals histogram (lineage {lid})",
            "control_totals",
            out_path(f"l{lid}_control_elements_hist"),
        )

        steps, mean_latent_total, std_latent_total = summarize_scalar_series(
            step_end,
            data["episode_seed"],
            lambda e: float(np.sum(e["snapshot"].get("latent_control", 0.0))),
        )
        last_steps_latent, last_latent = summarize_last_episode_scalar(
            step_end, lambda e: float(np.sum(e["snapshot"].get("latent_control", 0.0)))
        )
        plot_series(
            steps,
            mean_latent_total,
            std_latent_total,
            last_steps_latent,
            last_latent,
            f"Latent control total (lineage {lid})",
            "latent_total",
            out_path(f"l{lid}_latent_total"),
        )
        latent_vals = [float(np.sum(e["snapshot"].get("latent_control", 0.0))) for e in step_end]
        plot_hist(
            latent_vals,
            f"Latent control total histogram (lineage {lid})",
            "latent_total",
            out_path(f"l{lid}_latent_total_hist"),
        )

        steps, mean_latent_vec, std_latent_vec = summarize_vector_series(
            step_end, data["episode_seed"], lambda e: np.array(e["snapshot"].get("latent_control", []))
        )
        plot_element_series(
            steps,
            mean_latent_vec,
            std_latent_vec,
            f"Latent control per element (lineage {lid})",
            "latent_control",
            out_path(f"l{lid}_latent_elements"),
        )

        action_applied = filter_by_lineage(data["action_applied"], lid)
        actor_decision = filter_by_lineage(data["actor_decision"], lid)
        plot_counts(action_applied, args.window, out_path(f"l{lid}_counts"))
        plot_action_mix(action_applied, args.window, out_path(f"l{lid}_action_mix"))
        plot_action_hist_fallback(
            action_applied,
            actor_decision,
            out_path(f"l{lid}_action_hist"),
        )

        auth_events = filter_by_lineage(data["authorization"], lid)
        prop_meta = {pid: meta for pid, meta in proposal_meta.items() if meta.get("lineage_id") == lid}
        plot_authorization(auth_events, out_path(f"l{lid}_auth"))
        plot_governance_counts(auth_events, prop_meta, args.window, out_path(f"l{lid}_gov_total"))
        plot_governance_veto_rate(auth_events, prop_meta, args.window, out_path(f"l{lid}_gov_veto"))
        plot_governance_committee_rate(
            auth_events, prop_meta, args.window, out_path(f"l{lid}_gov_committee")
        )
        plot_governance_authorized_rate(auth_events, prop_meta, args.window, out_path(f"l{lid}_gov_authorized"))
        plot_governance_hist(auth_events, out_path(f"l{lid}_gov_approval_hist"))
        plot_committee_outcomes(auth_events, out_path(f"l{lid}_committee_outcomes"))

        vote_events = filter_by_lineage(data["vote_cast"], lid)
        plot_votes(vote_events, args.window, out_path(f"l{lid}_votes"))

        plot_agent_trajectories(step_end, data["episode_seed"], out_path(f"l{lid}_agents"))
        alive_vals = [e["snapshot"].get("alive", 0) for e in step_end]
        plot_hist(
            alive_vals,
            f"Alive agents histogram (lineage {lid})",
            "alive agents",
            out_path(f"l{lid}_agents_hist"),
        )
        plot_c_and_stock(step_end, data["episode_seed"], stock_events, out_path(f"l{lid}_c_stock"))
        plot_reward_stock_window(step_end, stock_events, args.window, out_path(f"l{lid}_reward_stock_window"))
        plot_entrenchments(step_end, out_path(f"l{lid}_entrenchments"))

    if args.walkthrough:
        walkthrough(args.logs[0], args.episode_id, args.lineage_id)
    if args.episode_progress:
        if args.all_lineages:
            for lid in episode_lineages(args.logs[0], args.episode_id, args.run_id):
                plot_episode_progress(
                    args.logs[0],
                    args.episode_id,
                    lid,
                    out_path(f"episode_progress_l{lid}"),
                    args.run_id,
                )
        else:
            plot_episode_progress(
                args.logs[0],
                args.episode_id,
                args.lineage_id,
                out_path("episode_progress"),
                args.run_id,
            )


if __name__ == "__main__":
    main()
