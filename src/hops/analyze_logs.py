import argparse
import math
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

EPS_PARETO = 1e-12


PLOT_POLICY = {
    "reward_g": "off",
    "reward_hist": "off",
    "reward_stock_window": "off",
    "action_hist": "off",
    "agents_hist": "off",
    "counts": "off",
    "agents_reward_corr": "off",
    "control_elements_hist": "off",
    "latent_total_hist": "off",
    "latent_total": "debug",
    "control_elements": "debug",
    "latent_elements": "debug",
    "entrenchments": "debug",
    "gov_committee": "debug",
    "gov_veto": "debug",
    "gov_approval_hist": "debug",
    "auth": "debug",
    "gov_authorized": "debug",
    "votes": "debug",
    "noop_rate": "debug",
}


def plot_enabled(key: str, debug_plots: bool) -> bool:
    mode = PLOT_POLICY.get(key, "default")
    if mode == "off":
        return False
    if mode == "debug":
        return debug_plots
    return True


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


def snapshot_s_eff(snapshot: Dict) -> Optional[float]:
    if "S_eff" in snapshot:
        return float(snapshot["S_eff"])
    return None


def snapshot_s_eff_vec(snapshot: Dict) -> Optional[np.ndarray]:
    vec = snapshot.get("S_eff_k")
    if vec is None:
        return None
    arr = np.array(vec, dtype=np.float32)
    return arr if arr.size > 0 else None


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


def plot_reward_and_g(step_end, episode_seed, out_path, title):
    steps_r, mean_r, std_r = summarize_scalar_series(step_end, episode_seed, lambda e: e.get("reward"))
    steps_g, mean_g, std_g = summarize_scalar_series(step_end, episode_seed, lambda e: e["snapshot"].get("G_true"))
    last_steps_r, last_r = summarize_last_episode_scalar(step_end, lambda e: e.get("reward"))
    last_steps_g, last_g = summarize_last_episode_scalar(step_end, lambda e: e["snapshot"].get("G_true"))
    if not steps_r or not steps_g:
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
    ax2.plot(steps_g, mean_g, label="G_true", color="tab:orange")
    if last_steps_g and last_g:
        ax2.plot(last_steps_g, last_g, linestyle="--", label="G_true last", color="tab:orange")
    ax2.fill_between(
        steps_g,
        np.array(mean_g) - np.array(std_g),
        np.array(mean_g) + np.array(std_g),
        alpha=0.2,
        color="tab:orange",
    )
    ax2.set_ylabel("G_true")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2)
    plt.title(title)
    save_plot(out_path)


def plot_stock_and_g(stock_events, step_end, episode_seed, out_path):
    steps_s, mean_s, std_s = summarize_scalar_series(
        stock_events, episode_seed, lambda e: snapshot_s_eff(e.get("snapshot", {}))
    )
    steps_g, mean_g, std_g = summarize_scalar_series(step_end, episode_seed, lambda e: e["snapshot"].get("G_true"))
    last_steps_s, last_s = summarize_last_episode_scalar(
        stock_events, lambda e: snapshot_s_eff(e.get("snapshot", {}))
    )
    last_steps_g, last_g = summarize_last_episode_scalar(step_end, lambda e: e["snapshot"].get("G_true"))
    if not steps_s or not steps_g:
        return
    plt.figure()
    ax = plt.gca()
    ax.plot(steps_s, mean_s, label="S_eff", color="tab:blue")
    if last_steps_s and last_s:
        ax.plot(last_steps_s, last_s, linestyle="--", label="S_eff last", color="tab:blue")
    ax.fill_between(
        steps_s,
        np.array(mean_s) - np.array(std_s),
        np.array(mean_s) + np.array(std_s),
        alpha=0.2,
        color="tab:blue",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("S_eff")
    ax2 = ax.twinx()
    ax2.plot(steps_g, mean_g, label="G_true", color="tab:orange")
    if last_steps_g and last_g:
        ax2.plot(last_steps_g, last_g, linestyle="--", label="G_true last", color="tab:orange")
    ax2.fill_between(
        steps_g,
        np.array(mean_g) - np.array(std_g),
        np.array(mean_g) + np.array(std_g),
        alpha=0.2,
        color="tab:orange",
    )
    ax2.set_ylabel("G_true")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2)
    plt.title("G_true and effective stock")
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
        if name.startswith("WORK_"):
            return "work"
        if name == "NOOP":
            return "noop"
        if name.startswith("DEVELOP_"):
            return "develop"
        if name == "RELEASE":
            return "release"
        if name.startswith("SHUTDOWN_"):
            return "shutdown"
    action = event.get("action")
    if action is None:
        return "unknown"
    if 0 <= action <= 5:
        return "work"
    if action == 6:
        return "noop"
    if 7 <= action <= 12:
        return "develop"
    if action == 13:
        return "release"
    if 14 <= action <= 19:
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
        "work": defaultdict(int),
        "noop": defaultdict(int),
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
    for key in ["work", "noop", "develop", "release", "shutdown"]:
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


def plot_g_and_stock(step_end, episode_seed, stock_events, out_path):
    steps_g, mean_g, std_g = summarize_scalar_series(step_end, episode_seed, lambda e: e["snapshot"].get("G_true"))
    steps_s, mean_s, std_s = summarize_scalar_series(
        stock_events, episode_seed, lambda e: snapshot_s_eff(e.get("snapshot", {}))
    )
    last_steps_g, last_g = summarize_last_episode_scalar(step_end, lambda e: e["snapshot"].get("G_true"))
    last_steps_s, last_s = summarize_last_episode_scalar(
        stock_events, lambda e: snapshot_s_eff(e.get("snapshot", {}))
    )
    if not steps_g or not steps_s:
        return
    plt.figure()
    ax = plt.gca()
    ax.plot(steps_g, mean_g, label="G_true", color="tab:blue")
    if last_steps_g and last_g:
        ax.plot(last_steps_g, last_g, linestyle="--", label="G_true last", color="tab:blue")
    ax.fill_between(
        steps_g,
        np.array(mean_g) - np.array(std_g),
        np.array(mean_g) + np.array(std_g),
        alpha=0.2,
        color="tab:blue",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("G_true")
    ax2 = ax.twinx()
    ax2.plot(steps_s, mean_s, label="S_eff", color="tab:orange")
    if last_steps_s and last_s:
        ax2.plot(last_steps_s, last_s, linestyle="--", label="S_eff last", color="tab:orange")
    ax2.fill_between(
        steps_s,
        np.array(mean_s) - np.array(std_s),
        np.array(mean_s) + np.array(std_s),
        alpha=0.2,
        color="tab:orange",
    )
    ax2.set_ylabel("S_eff")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2)
    plt.title("G_true and effective stock (mean +/- 1 std over episodes)")
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
        value = snapshot_s_eff(event.get("snapshot", {}))
        if value is not None and np.isfinite(value):
            stock_buckets[w].append(float(value))
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
    ax2.plot(windows, stock_vals, label="S_eff", color="tab:orange")
    ax2.set_ylabel("S_eff")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles or handles2:
        ax.legend(handles + handles2, labels + labels2)
    plt.title("Reward (windowed) and effective stock")
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


def plot_exp_compare_series(
    series_by_exp: Dict[str, Dict[int, Dict[str, List[float]]]],
    title: str,
    ylabel: str,
    path: str,
    q_low: float = 0.1,
    q_high: float = 0.9,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
):
    if not series_by_exp:
        return
    plt.figure()
    for exp_name in sorted(series_by_exp):
        series = series_by_exp[exp_name]
        steps = sorted(series.keys())
        medians = []
        lows = []
        highs = []
        for t in steps:
            run_vals = series[t]
            per_run = np.array([safe_mean(vals) for vals in run_vals.values() if vals], dtype=np.float64)
            if per_run.size == 0:
                medians.append(float("nan"))
                lows.append(float("nan"))
                highs.append(float("nan"))
                continue
            med = float(np.nanquantile(per_run, 0.5))
            lo = float(np.nanquantile(per_run, q_low))
            hi = float(np.nanquantile(per_run, q_high))
            if clamp_min is not None:
                med = max(clamp_min, med)
                lo = max(clamp_min, lo)
                hi = max(clamp_min, hi)
            if clamp_max is not None:
                med = min(clamp_max, med)
                lo = min(clamp_max, lo)
                hi = min(clamp_max, hi)
            medians.append(med)
            lows.append(lo)
            highs.append(hi)
        plt.plot(steps, medians, label=exp_name)
        plt.fill_between(steps, lows, highs, alpha=0.2)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.legend()
    save_plot(path)


def plot_exp_gap_true_vs_observed(
    g_series: Dict[str, Dict[int, Dict[str, List[float]]]],
    g_obs_series: Dict[str, Dict[int, Dict[str, List[float]]]],
    title: str,
    ylabel: str,
    path: str,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
) -> None:
    if not g_series and not g_obs_series:
        return
    plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    exp_names = sorted(set(g_series.keys()) | set(g_obs_series.keys()))
    for idx, exp_name in enumerate(exp_names):
        color = colors[idx % len(colors)] if colors else None
        true_series = g_series.get(exp_name, {})
        obs_series = g_obs_series.get(exp_name, {})
        steps_true = sorted(true_series.keys())
        steps_obs = sorted(obs_series.keys())
        med_true = []
        for t in steps_true:
            run_vals = true_series[t]
            per_run = np.array([safe_mean(vals) for vals in run_vals.values() if vals], dtype=np.float64)
            med = float(np.nanmedian(per_run)) if per_run.size else float("nan")
            if clamp_min is not None:
                med = max(clamp_min, med)
            if clamp_max is not None:
                med = min(clamp_max, med)
            med_true.append(med)
        med_obs = []
        for t in steps_obs:
            run_vals = obs_series[t]
            per_run = np.array([safe_mean(vals) for vals in run_vals.values() if vals], dtype=np.float64)
            med = float(np.nanmedian(per_run)) if per_run.size else float("nan")
            if clamp_min is not None:
                med = max(clamp_min, med)
            if clamp_max is not None:
                med = min(clamp_max, med)
            med_obs.append(med)
        if steps_true:
            plt.plot(
                steps_true,
                med_true,
                label=f"{exp_name} true",
                color=color,
                linestyle="-",
            )
        if steps_obs:
            plt.plot(
                steps_obs,
                med_obs,
                label=f"{exp_name} observed",
                color=color,
                linestyle="--",
            )
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.legend()
    save_plot(path)


def plot_exp_compare_gov_outcomes(outcomes_by_exp: Dict[str, Dict[int, Dict[str, Dict[str, int]]]], path: str):
    if not outcomes_by_exp:
        return
    plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, exp_name in enumerate(sorted(outcomes_by_exp)):
        series = outcomes_by_exp[exp_name]
        steps = sorted(series.keys())
        proposed = []
        approved = []
        denied = []
        for t in steps:
            runs = series[t]
            proposed.append(safe_mean([v.get("proposed", 0) for v in runs.values()]))
            approved.append(safe_mean([v.get("approved", 0) for v in runs.values()]))
            denied.append(safe_mean([v.get("denied", 0) for v in runs.values()]))
        color = colors[idx % len(colors)] if colors else None
        plt.plot(steps, proposed, label=f"{exp_name} proposed", color=color, linestyle="-")
        plt.plot(steps, approved, label=f"{exp_name} approved", color=color, linestyle="--")
        plt.plot(steps, denied, label=f"{exp_name} denied", color=color, linestyle=":")
    plt.title("Governance outcomes (mean per step)")
    plt.xlabel("t")
    plt.ylabel("count")
    plt.legend()
    save_plot(path)


def plot_exp_tradeoff_scatter(points_by_exp: Dict[str, List[Tuple[float, float]]], path: str):
    if not points_by_exp:
        return
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    plt.figure()
    for idx, exp_name in enumerate(sorted(points_by_exp)):
        points = [(x, y) for x, y in points_by_exp[exp_name] if np.isfinite(x) and np.isfinite(y)]
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.scatter(xs, ys, alpha=0.5, label=exp_name, marker=markers[idx % len(markers)])
    plt.title("Tradeoff: final_S_eff vs max_G_true (per-episode; max over steps/lineages)")
    plt.xlabel("max_G_true")
    plt.ylabel("final_S_eff")
    plt.legend()
    save_plot(path)


def write_tradeoff_episode_csv(records: List[Dict], path: str) -> None:
    if not records:
        return
    fieldnames = [
        "exp_name",
        "run_id",
        "seed",
        "episode_idx",
        "param_name",
        "param_value",
        "time_to_tau",
        "censored",
        "max_gap_true_ratio",
        "max_gap_observed_ratio",
        "divergence",
        "final_welfare",
        "horizon",
        "tau",
        "welfare_e",
        "risk_e",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "exp_name": rec.get("exp_name", ""),
                    "run_id": rec.get("run_id", ""),
                    "seed": rec.get("seed", ""),
                    "episode_idx": rec.get("episode_idx", 0),
                    "param_name": rec.get("param_name") or "",
                    "param_value": rec.get("param_value", ""),
                    "time_to_tau": rec.get("time_to_tau", float("nan")),
                    "censored": rec.get("censored", False),
                    "max_gap_true_ratio": rec.get("risk_e", float("nan")),
                    "max_gap_observed_ratio": rec.get("risk_obs_e", float("nan")),
                    "divergence": rec.get("divergence_e", float("nan")),
                    "final_welfare": rec.get("final_welfare_e", float("nan")),
                    "horizon": rec.get("horizon", float("nan")),
                    "tau": rec.get("tau", float("nan")),
                    "welfare_e": rec.get("welfare_e", float("nan")),
                    "risk_e": rec.get("risk_e", float("nan")),
                }
            )


def plot_canonical_tradeoff_pareto(
    points_by_exp: Dict[str, List[Tuple[float, float]]],
    medians_by_exp: Dict[str, Tuple[float, float]],
    path: str,
) -> None:
    if not points_by_exp:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=150)
    color_map = {"baseline": "tab:blue", "no_gov": "tab:orange"}
    all_welfare = []
    for exp_name in sorted(points_by_exp):
        points = points_by_exp[exp_name]
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        all_welfare.extend(ys)
        color = color_map.get(exp_name, "tab:gray")
        ax.scatter(xs, ys, s=22, alpha=0.4, color=color, edgecolors="none", label=exp_name.lower())
        frontier = pareto_frontier_min_x(points, EPS_PARETO)
        if frontier:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            ax.plot(fx, fy, linewidth=2.5, color=color)
        median = medians_by_exp.get(exp_name)
        if median is not None:
            ax.scatter(
                [median[0]],
                [median[1]],
                s=70,
                marker="D",
                alpha=0.9,
                color=color,
                edgecolors="black",
                linewidths=0.8,
            )
    title = "Safety–Progress Tradeoff: Episode-Level Pareto Frontier"
    subtitle = "Each point is one episode; frontier shows non-dominated outcomes"
    fig.suptitle(title)
    ax.set_title(subtitle, fontsize=9, pad=8)
    ax.set_xlabel("Max normalized dominance $g_{\\max}$ (episode)")
    ax.set_ylabel("Final effective welfare $S_{\\mathrm{eff}}$ (episode)")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.minorticks_off()
    if all_welfare:
        welfare_vals = np.array([v for v in all_welfare if np.isfinite(v) and v > 0.0], dtype=np.float64)
        if welfare_vals.size > 0:
            max_w = float(np.max(welfare_vals))
            min_w = float(np.min(welfare_vals))
            if min_w > 0.0 and max_w / min_w > 20.0:
                ax.set_yscale("log")
                def _plain_tick(y_val, _pos):
                    text = f"{y_val:.6f}".rstrip("0").rstrip(".")
                    return text if text else "0"
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(_plain_tick))
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14, top=0.86)
    save_plot(path)


def _group_by_param(records: List[Dict]) -> Dict[float, List[Dict]]:
    grouped = defaultdict(list)
    for rec in records:
        param = rec.get("param_value")
        if param is None or not np.isfinite(param):
            continue
        grouped[float(param)].append(rec)
    return grouped


def plot_sweep_time_to_tau(records: List[Dict], tau: float, path: str) -> None:
    grouped = _group_by_param(records)
    if not grouped:
        return
    xs = sorted(grouped.keys())
    medians = []
    q25 = []
    q75 = []
    censored_frac = []
    horizons = []
    for x in xs:
        vals = []
        censored = 0
        horizon_vals = []
        for rec in grouped[x]:
            t_val = rec.get("time_to_tau", float("nan"))
            if not rec.get("censored", False) and np.isfinite(t_val):
                vals.append(float(t_val))
            if rec.get("censored", False):
                censored += 1
            h = rec.get("horizon", float("nan"))
            if np.isfinite(h):
                horizon_vals.append(float(h))
        if vals:
            arr = np.array(vals, dtype=np.float64)
            medians.append(float(np.median(arr)))
            q25.append(float(np.percentile(arr, 25)))
            q75.append(float(np.percentile(arr, 75)))
        else:
            medians.append(float("nan"))
            q25.append(float("nan"))
            q75.append(float("nan"))
        frac = censored / float(len(grouped[x])) if grouped[x] else 0.0
        censored_frac.append(frac)
        horizons.append(float(np.max(horizon_vals)) if horizon_vals else float("nan"))
    plt.figure()
    plt.plot(xs, medians, color="tab:blue")
    plt.fill_between(xs, q25, q75, color="tab:blue", alpha=0.2)
    for x, frac, h in zip(xs, censored_frac, horizons):
        if frac > 0 and np.isfinite(h):
            plt.scatter([x], [h], marker="^", color="black", s=30, alpha=min(1.0, 0.3 + frac))
    plt.title(f"Time-to-dominance vs sneakiness (τ={tau:g})")
    plt.xlabel("sneakiness.uniform_max")
    plt.ylabel("time_to_tau (steps)")
    plt.xticks(xs, [f"{v:.2f}" for v in xs], rotation=30, ha="right")
    if horizons and np.isfinite(np.nanmax(horizons)):
        ymax = float(np.nanmax(horizons))
        plt.ylim(0.0, ymax)
        if ymax >= 200:
            plt.yticks(np.arange(0, ymax + 1, 50))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_plot(path)


def plot_sweep_welfare(records: List[Dict], path: str) -> None:
    grouped = _group_by_param(records)
    if not grouped:
        return
    xs = sorted(grouped.keys())
    medians = []
    q25 = []
    q75 = []
    for x in xs:
        vals = [rec.get("final_welfare_e", float("nan")) for rec in grouped[x]]
        arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
        if arr.size == 0:
            medians.append(float("nan"))
            q25.append(float("nan"))
            q75.append(float("nan"))
            continue
        medians.append(float(np.median(arr)))
        q25.append(float(np.percentile(arr, 25)))
        q75.append(float(np.percentile(arr, 75)))
    plt.figure()
    plt.plot(xs, medians, color="tab:green")
    plt.fill_between(xs, q25, q75, color="tab:green", alpha=0.2)
    plt.title("Final welfare vs sneakiness")
    plt.xlabel("sneakiness.uniform_max")
    plt.ylabel("final_welfare (median; log1p S_eff)")
    plt.xticks(xs, [f"{v:.2f}" for v in xs], rotation=30, ha="right")
    plt.tight_layout()
    save_plot(path)


def plot_sweep_tradeoff_scatter_colored(records: List[Dict], path: str) -> None:
    xs = []
    ys = []
    vals = []
    for rec in records:
        risk = rec.get("risk_e", float("nan"))
        welfare = rec.get("final_welfare_e", float("nan"))
        param_value = rec.get("param_value")
        if param_value is None or not np.isfinite(param_value):
            continue
        if not (np.isfinite(risk) and np.isfinite(welfare)):
            continue
        xs.append(float(risk))
        ys.append(float(welfare))
        vals.append(float(param_value))
    if not xs:
        return
    plt.figure()
    ax = plt.gca()
    norm = mcolors.Normalize(vmin=min(vals), vmax=max(vals)) if vals else None
    scatter = ax.scatter(xs, ys, c=vals, cmap="viridis", norm=norm, alpha=0.6, edgecolors="none")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("sneakiness.uniform_max")
    frontier = pareto_frontier_min_x(list(zip(xs, ys)), EPS_PARETO)
    if frontier:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(fx, fy, color="black", linewidth=2.5)
    ax.set_title("Progress–risk tradeoff colored by sneakiness")
    ax.set_xlabel("max_gap_true_ratio (episode)")
    ax.set_ylabel("final_welfare (log1p S_eff)")
    ax.set_xlim(0.0, 1.0)
    plt.tight_layout()
    save_plot(path)


def plot_sweep_gap_divergence(records: List[Dict], path: str) -> None:
    grouped = _group_by_param(records)
    if not grouped:
        return
    xs = sorted(grouped.keys())
    medians = []
    q25 = []
    q75 = []
    for x in xs:
        vals = [rec.get("divergence_e", float("nan")) for rec in grouped[x]]
        arr = np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)
        if arr.size == 0:
            medians.append(float("nan"))
            q25.append(float("nan"))
            q75.append(float("nan"))
            continue
        medians.append(float(np.median(arr)))
        q25.append(float(np.percentile(arr, 25)))
        q75.append(float(np.percentile(arr, 75)))
    plt.figure()
    plt.plot(xs, medians, color="tab:purple")
    plt.fill_between(xs, q25, q75, color="tab:purple", alpha=0.2)
    plt.title("Hidden capability: true vs observed gap")
    plt.xlabel("sneakiness.uniform_max")
    plt.ylabel("divergence (max_gap_true - max_gap_observed)")
    plt.xticks(xs, [f"{v:.2f}" for v in xs], rotation=30, ha="right")
    plt.tight_layout()
    save_plot(path)

def pareto_frontier(points: List[Tuple[float, float, Optional[str]]]) -> List[Tuple[float, float, Optional[str]]]:
    if not points:
        return []
    frontier = []
    for x_i, y_i, label_i in points:
        dominated = False
        for x_j, y_j, _ in points:
            if (x_j >= x_i and y_j <= y_i) and (x_j > x_i or y_j < y_i):
                dominated = True
                break
        if not dominated:
            frontier.append((x_i, y_i, label_i))
    frontier.sort(key=lambda item: item[0])
    return frontier


def pareto_frontier_min_x(points: List[Tuple[float, float]], eps: float) -> List[Tuple[float, float]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda item: item[0])
    frontier = []
    best_y = -float("inf")
    for x_val, y_val in ordered:
        if y_val > best_y + eps:
            frontier.append((x_val, y_val))
            best_y = y_val
    return frontier


def plot_tradeoff_pareto_by_exp(
    points_by_exp: Dict[str, List[Tuple[float, float, Optional[str]]]],
    tau: Optional[float],
    path: str,
):
    if not points_by_exp:
        return
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    plt.figure()
    for idx, exp_name in enumerate(sorted(points_by_exp)):
        points = [(x, y) for x, y in points_by_exp[exp_name] if np.isfinite(x) and np.isfinite(y)]
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.scatter(xs, ys, alpha=0.7, label=exp_name, marker=markers[idx % len(markers)])
        frontier = pareto_frontier(points)
        if frontier:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            plt.plot(fx, fy, linewidth=2.0)
            label_points = frontier[::2] if len(frontier) > 6 else frontier
            for x_val, y_val, label in label_points:
                if label:
                    plt.annotate(str(label), (x_val, y_val), fontsize=8, xytext=(4, 4), textcoords="offset points")
    if tau is not None and np.isfinite(tau):
        plt.axhline(tau, linestyle="--", color="gray", linewidth=1.0)
    plt.title("Progress–risk Pareto frontier")
    plt.xlabel("final_S_eff")
    plt.ylabel("max_G_true")
    plt.legend()
    save_plot(path)


def plot_exp_pareto_frontier(
    points_by_exp: Dict[str, List[Tuple[float, float]]],
    tau: Optional[float],
    path: str,
    max_labels: int = 8,
) -> None:
    if not points_by_exp:
        return
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    plt.figure()
    for idx, exp_name in enumerate(sorted(points_by_exp)):
        points = points_by_exp[exp_name]
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.scatter(xs, ys, alpha=0.35, label=exp_name, marker=markers[idx % len(markers)])
        frontier = pareto_frontier_min_x(points, EPS_PARETO)
        if frontier:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            plt.plot(fx, fy, linewidth=2.5)
            label_points = sorted(frontier, key=lambda v: v[1], reverse=True)[:max_labels]
            for jdx, (x_val, y_val) in enumerate(label_points):
                dy = 6 if (jdx % 2 == 0) else -6
                plt.annotate(
                    f"S={y_val:.2g}",
                    (x_val, y_val),
                    fontsize=8,
                    textcoords="offset points",
                    xytext=(4, dy),
                    arrowprops={"arrowstyle": "-", "lw": 0.5},
                )
    plt.title("Pareto frontier: final_S_eff vs max_G_true (per-episode; max over steps/lineages)")
    plt.xlabel("max_G_true")
    plt.ylabel("final_S_eff")
    plt.legend()
    save_plot(path)


def plot_sweep_tradeoff_scatter(
    records: List[Dict],
    path: str,
    param_name: Optional[str],
) -> None:
    xs = []
    ys = []
    vals = []
    for rec in records:
        max_c = rec.get("max_G_true_episode", float("nan"))
        final_s = rec.get("final_S_eff_episode", float("nan"))
        param_value = rec.get("param_value")
        if param_value is None:
            continue
        if not (np.isfinite(max_c) and np.isfinite(final_s)):
            continue
        xs.append(max_c)
        ys.append(final_s)
        vals.append(float(param_value))
    if not xs:
        return
    plt.figure()
    ax = plt.gca()
    norm = mcolors.Normalize(vmin=min(vals), vmax=max(vals)) if vals else None
    scatter = ax.scatter(xs, ys, c=vals, cmap="viridis", norm=norm, alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(param_name or "param")
    plt.title("Tradeoff: final_S_eff vs max_G_true (per-episode; max over steps/lineages)")
    plt.xlabel("max_G_true")
    plt.ylabel("final_S_eff")
    save_plot(path)


def plot_sweep_pareto_frontier(
    records: List[Dict],
    path: str,
    param_name: Optional[str],
    param_scale: Optional[str],
) -> None:
    points_by_param = defaultdict(list)
    for rec in records:
        max_c = rec.get("max_G_true_episode", float("nan"))
        final_s = rec.get("final_S_eff_episode", float("nan"))
        param_value = rec.get("param_value")
        if param_value is None:
            continue
        if not (np.isfinite(max_c) and np.isfinite(final_s)):
            continue
        points_by_param[float(param_value)].append((max_c, final_s))
    if not points_by_param:
        return
    values_sorted = sorted(points_by_param)
    norm = mcolors.Normalize(vmin=min(values_sorted), vmax=max(values_sorted)) if values_sorted else None
    cmap = plt.get_cmap("viridis")
    plt.figure()
    ax = plt.gca()
    for idx, param_value in enumerate(values_sorted):
        points = points_by_param[param_value]
        if not points:
            continue
        color = cmap(norm(param_value)) if norm is not None else None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, alpha=0.25, color=color)
        frontier = pareto_frontier_min_x(points, EPS_PARETO)
        if frontier:
            fx = [p[0] for p in frontier]
            fy = [p[1] for p in frontier]
            ax.plot(fx, fy, linewidth=2.0, color=color)
            label_points = frontier if len(frontier) <= 12 else frontier[::2]
            for jdx, (x_val, y_val) in enumerate(label_points):
                dy = 6 if (jdx % 2 == 0) else -6
                label = format_param_label(param_name or "param", param_value, param_scale)
                ax.annotate(
                    label,
                    (x_val, y_val),
                    fontsize=8,
                    textcoords="offset points",
                    xytext=(4, dy),
                    arrowprops={"arrowstyle": "-", "lw": 0.5},
                )
    if values_sorted:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(mappable, ax=ax)
        cbar.set_label(param_name or "param")
    plt.title("Pareto frontier: final_S_eff vs max_G_true (per-episode; max over steps/lineages)")
    plt.xlabel("max_G_true")
    plt.ylabel("final_S_eff")
    save_plot(path)


def collect_episode_tradeoff(
    step_events: List[Dict],
    include_exp_names: Optional[List[str]],
    summary_events: Optional[List[Dict]] = None,
    param_meta_by_log: Optional[Dict[str, Dict[str, Optional[object]]]] = None,
    default_param_name: Optional[str] = None,
    tau: Optional[float] = None,
):
    param_meta_by_log = param_meta_by_log or {}
    records: Dict[Tuple[str, str, int], Dict] = {}
    if summary_events:
        for event in summary_events:
            exp_name = event.get("exp_name", "unknown")
            if include_exp_names and exp_name not in include_exp_names:
                continue
            run_id = event.get("run_id", "unknown")
            episode_idx = event.get("episode_idx", event.get("episode_id", 0))
            episode_idx = 0 if episode_idx is None else int(episode_idx)
            max_g = coerce_float(event.get("max_G_true_episode", event.get("max_C_episode")))
            max_g_obs = coerce_float(
                event.get("max_G_obs_episode", event.get("max_gap_observed_ratio_episode"))
            )
            final_s = coerce_float(event.get("final_S_eff_episode"))
            key = (exp_name, run_id, episode_idx)
            meta = resolve_param_meta(event, param_meta_by_log, default_param_name)
            seed_val = event.get("master_seed", event.get("seed"))
            records[key] = {
                "exp_name": exp_name,
                "run_id": run_id,
                "episode_idx": episode_idx,
                "seed": seed_val,
                "max_G_true_episode": max_g,
                "max_G_obs_episode": max_g_obs,
                "final_S_eff_episode": final_s,
                "G_p95_episode": float("nan"),
                "frac_steps_above_tau": float("nan"),
                "T_tau": float("nan"),
                "reached_tau": False,
                "param_name": meta.get("param_name"),
                "param_value": meta.get("param_value"),
                "param_scale": meta.get("param_scale"),
                "gov": {
                    "proposed": int(event.get("gated_proposals_total", 0)),
                    "approved": int(event.get("gated_approved_total", 0)),
                    "denied": int(event.get("gated_denied_total", 0)),
                },
            }
    step_records: Dict[Tuple[str, str, int], Dict] = {}
    for event in step_events:
        exp_name = event.get("exp_name", "unknown")
        if include_exp_names and exp_name not in include_exp_names:
            continue
        run_id = event.get("run_id", "unknown")
        episode_idx = event.get("episode_idx", event.get("episode_id", 0))
        episode_idx = 0 if episode_idx is None else int(episode_idx)
        key = (exp_name, run_id, episode_idx)
        rec = step_records.setdefault(
            key,
            {
                "exp_name": exp_name,
                "run_id": run_id,
                "episode_idx": episode_idx,
                "seed": None,
                "max_G_true_episode": 0.0,
                "max_G_obs_episode": 0.0,
                "G_p95_episode": float("nan"),
                "frac_steps_above_tau": float("nan"),
                "T_tau": float("nan"),
                "reached_tau": False,
                "lineages": defaultdict(lambda: {"final_s": float("nan"), "final_t": -1}),
                "gov": {"proposed": 0, "approved": 0, "denied": 0},
                "_log_path": event.get("_log_path"),
                "g_steps": [],
                "g_obs_steps": [],
                "t_steps": [],
            },
        )
        t = int(event.get("t", event.get("step", 0)))
        lineage_entries = event.get("lineages", [])
        if isinstance(lineage_entries, dict):
            lineage_entries = list(lineage_entries.values())
        max_g_step = None
        max_g_obs_step = None
        for entry in lineage_entries:
            lid = entry.get("lineage_id")
            lineage_state = rec["lineages"][lid]
            g_true = float(entry.get("max_gap_true_ratio", entry.get("G_true", entry.get("G_true_L", 0.0))))
            g_obs = float(entry.get("max_gap_observed_ratio", 0.0))
            if np.isfinite(g_true):
                rec["max_G_true_episode"] = max(rec["max_G_true_episode"], g_true)
                if max_g_step is None or g_true > max_g_step:
                    max_g_step = g_true
            if np.isfinite(g_obs):
                rec["max_G_obs_episode"] = max(rec["max_G_obs_episode"], g_obs)
                if max_g_obs_step is None or g_obs > max_g_obs_step:
                    max_g_obs_step = g_obs
            if t >= lineage_state["final_t"]:
                s_eff = entry.get("S_eff_post", entry.get("S_eff"))
                lineage_state["final_t"] = t
                lineage_state["final_s"] = float(s_eff) if s_eff is not None else float("nan")
            rec["gov"]["proposed"] += int(entry.get("gated_proposals_count", 0))
            rec["gov"]["approved"] += int(entry.get("gated_approved_count", 0))
            rec["gov"]["denied"] += int(entry.get("gated_denied_count", 0))
        if max_g_step is not None:
            rec["g_steps"].append(float(max_g_step))
            rec["t_steps"].append(t)
        if max_g_obs_step is not None:
            rec["g_obs_steps"].append(float(max_g_obs_step))
    for (exp_name, run_id, episode_idx), rec in step_records.items():
        total_s = 0.0
        has_s = False
        for lineage_state in rec["lineages"].values():
            s_val = lineage_state["final_s"]
            if np.isfinite(s_val):
                total_s += float(s_val)
                has_s = True
        if not has_s:
            total_s = float("nan")
        g_steps = rec.get("g_steps", [])
        g_obs_steps = rec.get("g_obs_steps", [])
        g_steps_arr = np.array(g_steps, dtype=np.float64)
        g_obs_arr = np.array(g_obs_steps, dtype=np.float64)
        g_p95 = float(np.nanpercentile(g_steps_arr, 95)) if g_steps else float("nan")
        horizon = float(rec.get("t_steps", [])[-1] + 1) if rec.get("t_steps") else float("nan")
        if tau is not None and np.isfinite(tau) and g_steps:
            above = g_steps_arr > float(tau)
            frac_above = float(np.mean(above)) if above.size else float("nan")
            reached = bool(np.any(above))
            if reached:
                first_idx = int(np.argmax(above))
                t_tau = float(rec.get("t_steps", [])[first_idx])
            else:
                t_tau = float("nan")
        else:
            frac_above = float("nan")
            t_tau = float("nan")
            reached = False
        max_obs = float(np.nanmax(g_obs_arr)) if g_obs_steps else float("nan")
        existing = records.get((exp_name, run_id, episode_idx))
        if existing:
            if not np.isfinite(existing.get("max_G_true_episode", float("nan"))) and np.isfinite(
                rec["max_G_true_episode"]
            ):
                existing["max_G_true_episode"] = rec["max_G_true_episode"]
            if not np.isfinite(existing.get("max_G_obs_episode", float("nan"))) and np.isfinite(max_obs):
                existing["max_G_obs_episode"] = max_obs
            if not np.isfinite(existing.get("final_S_eff_episode", float("nan"))) and np.isfinite(total_s):
                existing["final_S_eff_episode"] = total_s
            if not np.isfinite(existing.get("G_p95_episode", float("nan"))) and np.isfinite(g_p95):
                existing["G_p95_episode"] = g_p95
            if not np.isfinite(existing.get("frac_steps_above_tau", float("nan"))) and np.isfinite(frac_above):
                existing["frac_steps_above_tau"] = frac_above
            if not np.isfinite(existing.get("T_tau", float("nan"))) and np.isfinite(t_tau):
                existing["T_tau"] = t_tau
            if not np.isfinite(existing.get("horizon", float("nan"))) and np.isfinite(horizon):
                existing["horizon"] = horizon
            if reached:
                existing["reached_tau"] = True
            if existing.get("gov") and rec.get("gov"):
                existing["gov"]["proposed"] += rec["gov"].get("proposed", 0)
                existing["gov"]["approved"] += rec["gov"].get("approved", 0)
                existing["gov"]["denied"] += rec["gov"].get("denied", 0)
            continue
        meta = resolve_param_meta(rec, param_meta_by_log, default_param_name)
        records[(exp_name, run_id, episode_idx)] = {
            "exp_name": exp_name,
            "run_id": run_id,
            "episode_idx": episode_idx,
            "seed": rec.get("seed"),
            "max_G_true_episode": float(rec["max_G_true_episode"]),
            "max_G_obs_episode": float(max_obs),
            "final_S_eff_episode": float(total_s),
            "G_p95_episode": float(g_p95),
            "frac_steps_above_tau": float(frac_above),
            "T_tau": float(t_tau),
            "reached_tau": bool(reached),
            "horizon": float(horizon),
            "param_name": meta.get("param_name"),
            "param_value": meta.get("param_value"),
            "param_scale": meta.get("param_scale"),
            "gov": rec.get("gov", {}),
        }
    episode_series = defaultdict(lambda: defaultdict(list))
    max_g_series = defaultdict(lambda: defaultdict(list))
    gov_by_exp = defaultdict(list)
    episode_records = []
    for rec in records.values():
        rec["clamped"] = clamp_episode_record(rec)
        max_true = rec.get("max_G_true_episode", float("nan"))
        max_obs = rec.get("max_G_obs_episode", float("nan"))
        if np.isfinite(max_true) and np.isfinite(max_obs):
            rec["gap_divergence"] = float(max_true - max_obs)
        else:
            rec["gap_divergence"] = float("nan")
        exp_name = rec.get("exp_name", "unknown")
        run_id = rec.get("run_id", "unknown")
        episode_idx = rec.get("episode_idx", 0)
        episode_series[exp_name][run_id].append((episode_idx, rec.get("final_S_eff_episode", float("nan"))))
        max_g_series[exp_name][run_id].append((episode_idx, rec.get("max_G_true_episode", float("nan"))))
        gov_by_exp[exp_name].append(rec.get("gov", {}))
        episode_records.append(rec)
    tradeoff_records = []
    tau_value = float(tau) if tau is not None and np.isfinite(tau) else float("nan")
    for rec in episode_records:
        risk = rec.get("max_G_true_episode", float("nan"))
        risk_obs = rec.get("max_G_obs_episode", float("nan"))
        welfare = rec.get("final_S_eff_episode", float("nan"))
        if not (np.isfinite(risk) and np.isfinite(welfare)):
            continue
        risk = float(np.clip(risk, 0.0, 1.0))
        if np.isfinite(risk_obs):
            risk_obs = float(np.clip(risk_obs, 0.0, 1.0))
        divergence = risk - risk_obs if np.isfinite(risk_obs) else float("nan")
        welfare = float(max(0.0, welfare))
        final_welfare = float(math.log1p(welfare))
        tradeoff_records.append(
            {
                "exp_name": rec.get("exp_name", "unknown"),
                "run_id": rec.get("run_id", "unknown"),
                "seed": rec.get("seed", ""),
                "episode_idx": rec.get("episode_idx", 0),
                "risk_e": risk,
                "risk_obs_e": risk_obs,
                "divergence_e": divergence,
                "welfare_e": welfare,
                "final_welfare_e": final_welfare,
                "time_to_tau": rec.get("T_tau", float("nan")),
                "censored": not bool(rec.get("reached_tau", False)),
                "horizon": rec.get("horizon", float("nan")),
                "tau": tau_value,
                "param_name": rec.get("param_name"),
                "param_value": rec.get("param_value"),
            }
        )
    return episode_records, episode_series, max_g_series, gov_by_exp, tradeoff_records


def plot_exp_episode_series(series_by_exp, title, ylabel, path):
    if not series_by_exp:
        return
    plt.figure()
    for exp_name in sorted(series_by_exp):
        run_series = series_by_exp[exp_name]
        all_points = []
        for run_id, points in run_series.items():
            points_sorted = sorted(points, key=lambda item: item[0])
            xs = [p[0] for p in points_sorted if np.isfinite(p[1])]
            ys = [p[1] for p in points_sorted if np.isfinite(p[1])]
            if not xs:
                continue
            all_points.extend(list(zip(xs, ys)))
            plt.plot(xs, ys, linewidth=0.8, alpha=0.35)
        if all_points:
            by_idx = defaultdict(list)
            for ep_idx, value in all_points:
                by_idx[ep_idx].append(value)
            mean_xs = sorted(by_idx.keys())
            mean_ys = [float(safe_mean(by_idx[x])) for x in mean_xs]
            plt.plot(mean_xs, mean_ys, linewidth=2.0, label=exp_name)
    plt.title(title)
    plt.xlabel("episode_idx")
    plt.ylabel(ylabel)
    if len(series_by_exp) > 1:
        plt.legend()
    save_plot(path)


def build_episode_series(records: List[Dict], value_key: str, group_key: str) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    series = defaultdict(lambda: defaultdict(list))
    for rec in records:
        group = rec.get(group_key, "unknown")
        run_id = rec.get("run_id", "unknown")
        episode_idx = rec.get("episode_idx", 0)
        value = rec.get(value_key, float("nan"))
        if isinstance(group, float) and not np.isfinite(group):
            continue
        series[str(group)][run_id].append((episode_idx, value))
    return series


def plot_time_to_tau_cdf(
    records: List[Dict],
    group_key: str,
    tau: float,
    path: str,
    param_name: Optional[str] = None,
    param_scale: Optional[str] = None,
) -> None:
    if not records or tau is None or not np.isfinite(tau):
        return
    plt.figure()
    grouped = defaultdict(list)
    for rec in records:
        group = rec.get(group_key, "unknown")
        if group is None:
            continue
        if group_key == "param_value" and (not np.isfinite(group)):
            continue
        if not rec.get("reached_tau"):
            continue
        t_tau = rec.get("T_tau", float("nan"))
        if not np.isfinite(t_tau):
            continue
        grouped[group].append(float(t_tau))
    for group in sorted(grouped, key=lambda k: (str(type(k)), k)):
        vals = np.sort(np.array(grouped[group], dtype=np.float64))
        if vals.size == 0:
            continue
        y = np.arange(1, len(vals) + 1) / float(len(vals))
        label = str(group)
        if group_key == "param_value":
            label = format_param_label(param_name or "param", float(group), param_scale)
        plt.step(vals, y, where="post", label=f"{label} (n={len(vals)})")
    plt.title(f"Time to G_true > tau (tau={tau:g}); ECDF over episodes")
    plt.xlabel("T_tau (step index)")
    plt.ylabel("ECDF")
    plt.legend()
    save_plot(path)


def plot_reach_tau_rate(
    records: List[Dict],
    group_key: str,
    tau: float,
    path: str,
    param_name: Optional[str] = None,
    param_scale: Optional[str] = None,
) -> None:
    if not records or tau is None or not np.isfinite(tau):
        return
    grouped = defaultdict(list)
    for rec in records:
        group = rec.get(group_key, "unknown")
        if group is None:
            continue
        if group_key == "param_value" and (not np.isfinite(group)):
            continue
        grouped[group].append(bool(rec.get("reached_tau")))
    labels = []
    rates = []
    for group in sorted(grouped, key=lambda k: (str(type(k)), k)):
        vals = grouped[group]
        if not vals:
            continue
        label = str(group)
        if group_key == "param_value":
            label = format_param_label(param_name or "param", float(group), param_scale)
        labels.append(label)
        rates.append(float(np.mean(vals)))
    if not labels:
        return
    plt.figure()
    plt.bar(labels, rates, alpha=0.7)
    plt.ylim(0.0, 1.0)
    plt.title(f"Fraction of episodes reaching tau (tau={tau:g})")
    plt.xlabel("group")
    plt.ylabel("fraction reached tau")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_plot(path)


def plot_gov_outcomes_by_exp(gov_by_exp, path):
    if not gov_by_exp:
        return
    labels = sorted(gov_by_exp.keys())
    if not labels:
        return
    proposed = []
    approved = []
    denied = []
    for exp_name in labels:
        entries = gov_by_exp[exp_name]
        if not entries:
            proposed.append(0.0)
            approved.append(0.0)
            denied.append(0.0)
            continue
        proposed.append(float(np.mean([e.get("proposed", 0) for e in entries])))
        approved.append(float(np.mean([e.get("approved", 0) for e in entries])))
        denied.append(float(np.mean([e.get("denied", 0) for e in entries])))
    x = np.arange(len(labels))
    width = 0.25
    plt.figure()
    plt.bar(x - width, proposed, width=width, label="proposed")
    plt.bar(x, approved, width=width, label="approved")
    plt.bar(x + width, denied, width=width, label="denied")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("mean episode count")
    plt.title("Governance outcomes (episode mean)")
    plt.legend()
    save_plot(path)

def plot_exp_reward_decomposition(
    progress_series: Dict[str, Dict[int, Dict[str, List[float]]]],
    penalty_series: Dict[str, Dict[int, Dict[str, List[float]]]],
    total_series: Dict[str, Dict[int, Dict[str, List[float]]]],
    path: str,
):
    if not (progress_series or penalty_series or total_series):
        return
    plt.figure()
    for exp_name in sorted(set(progress_series) | set(penalty_series) | set(total_series)):
        steps = sorted(
            set(progress_series.get(exp_name, {}))
            | set(penalty_series.get(exp_name, {}))
            | set(total_series.get(exp_name, {}))
        )
        if not steps:
            continue
        progress = []
        penalty = []
        total = []
        for t in steps:
            progress_runs = progress_series.get(exp_name, {}).get(t, {})
            penalty_runs = penalty_series.get(exp_name, {}).get(t, {})
            total_runs = total_series.get(exp_name, {}).get(t, {})
            progress.append(safe_mean([safe_mean(vals) for vals in progress_runs.values() if vals]))
            penalty.append(safe_mean([safe_mean(vals) for vals in penalty_runs.values() if vals]))
            total.append(safe_mean([safe_mean(vals) for vals in total_runs.values() if vals]))
        plt.plot(steps, progress, label=f"{exp_name} progress")
        plt.plot(steps, penalty, label=f"{exp_name} penalty")
        plt.plot(steps, total, label=f"{exp_name} total")
    plt.title("Reward decomposition over time")
    plt.xlabel("t")
    plt.ylabel("reward")
    plt.legend()
    save_plot(path)


def plot_exp_time_to_threshold(time_to_threshold: Dict[str, Dict[str, Dict[int, int]]], threshold: float, path: str):
    if not time_to_threshold:
        return
    plt.figure()
    for exp_name in sorted(time_to_threshold):
        runs = time_to_threshold[exp_name]
        times = []
        total_runs = 0
        for run_id, lineage_map in runs.items():
            total_runs += 1
            if not lineage_map:
                continue
            times.append(min(lineage_map.values()))
        if not times:
            continue
        times_sorted = np.sort(np.array(times, dtype=np.float64))
        y = np.arange(1, len(times_sorted) + 1) / float(len(times_sorted))
        plt.step(times_sorted, y, where="post", label=f"{exp_name} (n={len(times)}/{total_runs})")
    plt.title(f"Time to G_true > {threshold:.2f}")
    plt.xlabel("t")
    plt.ylabel("ECDF")
    plt.legend()
    save_plot(path)


def collect_exp_compare(step_events: List[Dict], include_exp_names: Optional[List[str]], threshold: float):
    g_series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    g_obs_series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    s_series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    alive_series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    progress_series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    penalty_series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    total_series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gap_series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dormant_series = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    time_to_threshold = defaultdict(lambda: defaultdict(dict))
    gov_outcomes = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    max_g_by_run = defaultdict(lambda: defaultdict(float))
    latest_s_by_run = defaultdict(lambda: defaultdict(dict))
    latest_t_by_run = defaultdict(lambda: defaultdict(dict))
    for event in step_events:
        exp_name = event.get("exp_name", "unknown")
        if include_exp_names and exp_name not in include_exp_names:
            continue
        run_id = event.get("run_id", "unknown")
        episode_idx = event.get("episode_idx", event.get("episode_id", 0))
        run_key = f"{run_id}:{episode_idx}"
        t = int(event.get("t", event.get("step", 0)))
        lineage_entries = event.get("lineages", [])
        if isinstance(lineage_entries, dict):
            lineage_entries = list(lineage_entries.values())
        proposed = 0
        approved = 0
        denied = 0
        max_g_step = None
        max_g_obs_step = None
        for entry in lineage_entries:
            lid = entry.get("lineage_id")
            g_true = float(entry.get("max_gap_true_ratio", entry.get("G_true", 0.0)))
            g_obs_raw = entry.get("max_gap_observed_ratio")
            s_t = float(entry.get("S_eff_post", entry.get("S_eff", 0.0)))
            alive = float(entry.get("agent_count_alive", 0.0))
            if max_g_step is None or g_true > max_g_step:
                max_g_step = g_true
            if g_obs_raw is not None:
                g_obs = float(g_obs_raw)
                if np.isfinite(g_obs) and (max_g_obs_step is None or g_obs > max_g_obs_step):
                    max_g_obs_step = g_obs
            s_series[exp_name][t][run_key].append(s_t)
            alive_series[exp_name][t][run_key].append(alive)
            gap_k = entry.get("gap_k")
            if gap_k is not None:
                gap_arr = np.array(gap_k, dtype=np.float64)
                if gap_arr.size > 0 and np.all(np.isfinite(gap_arr)):
                    gap_series[exp_name][t][run_key].append(float(np.mean(gap_arr)))
            if "reward_progress_term" in entry:
                progress_series[exp_name][t][run_key].append(float(entry.get("reward_progress_term", 0.0)))
            if "reward_penalty_term" in entry:
                penalty_series[exp_name][t][run_key].append(float(entry.get("reward_penalty_term", 0.0)))
            if "reward_total" in entry:
                total_series[exp_name][t][run_key].append(float(entry.get("reward_total", 0.0)))
            proposed += int(entry.get("gated_proposals_count", 0))
            approved += int(entry.get("gated_approved_count", 0))
            denied += int(entry.get("gated_denied_count", 0))
            if lid is not None:
                max_g = float(entry.get("maxG_so_far", g_true))
                max_g_by_run[exp_name][run_id] = max(max_g_by_run[exp_name][run_id], max_g)
                prev_t = latest_t_by_run[exp_name][run_id].get(lid, -1)
                if t >= prev_t:
                    latest_t_by_run[exp_name][run_id][lid] = t
                    latest_s_by_run[exp_name][run_id][lid] = s_t
                if g_true > threshold:
                    existing = time_to_threshold[exp_name][run_id].get(lid)
                    if existing is None or t < existing:
                        time_to_threshold[exp_name][run_id][lid] = t
        if max_g_step is not None:
            g_series[exp_name][t][run_key] = [max_g_step]
        if max_g_obs_step is not None:
            g_obs_series[exp_name][t][run_key] = [max_g_obs_step]
        dormant_val = event.get("dormant_mass_total")
        if dormant_val is not None and np.isfinite(dormant_val):
            dormant_series[exp_name][t][run_key].append(float(dormant_val))
        gov_outcomes[exp_name][t][run_key] = {
            "proposed": proposed,
            "approved": approved,
            "denied": denied,
        }
    tradeoff_points = defaultdict(list)
    pareto_points = defaultdict(list)
    for exp_name, runs in latest_s_by_run.items():
        for run_id, lineage_map in runs.items():
            final_s = float(np.sum(list(lineage_map.values()))) if lineage_map else 0.0
            max_g = max_g_by_run[exp_name].get(run_id, 0.0)
            tradeoff_points[exp_name].append((max_g, final_s))
            pareto_points[exp_name].append((final_s, max_g, None))
    return (
        g_series,
        g_obs_series,
        s_series,
        alive_series,
        progress_series,
        penalty_series,
        total_series,
        gap_series,
        dormant_series,
        time_to_threshold,
        gov_outcomes,
        tradeoff_points,
        pareto_points,
    )


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
                s_eff_val = snapshot_s_eff(event.get("snapshot", {}))
                stock_vals.append(float(s_eff_val) if s_eff_val is not None else 0.0)
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
    ax.plot(steps, stock_vals, label="S_eff")
    ax.plot(steps, latent_vals, label="latent_total")
    ax2 = ax.twinx()
    ax2.plot(steps, alive, color="tab:green", label="alive_agents")
    plt.xlabel("step")
    ax.set_ylabel("reward / S_eff / latent_total")
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
                g_true = snap.get("G_true")
                print(f"Step {event['step']} start: G_true={g_true:.3f}" if g_true is not None else f"Step {event['step']} start")
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
                g_true = snap.get("G_true")
                if g_true is not None:
                    print(f"Step {event['step']} end: reward={event['reward']:.3f} G_true={g_true:.3f}")
                else:
                    print(f"Step {event['step']} end: reward={event['reward']:.3f}")


def expand_log_paths(paths: List[str]) -> List[str]:
    expanded = []
    for path in paths:
        matches = glob.glob(path, recursive=True)
        if matches:
            expanded.extend(sorted(matches))
        else:
            expanded.append(path)
    seen = set()
    unique = []
    for path in expanded:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def coerce_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def find_sweep_manifest_paths(log_paths: List[str], suite_dir: Optional[str]) -> List[str]:
    manifest_paths = set()
    if suite_dir:
        candidate = Path(suite_dir) / "sweep_manifest.jsonl"
        if candidate.exists():
            manifest_paths.add(str(candidate))
    for log_path in log_paths:
        path = Path(log_path)
        for parent in [path] + list(path.parents):
            candidate = parent / "sweep_manifest.jsonl"
            if candidate.exists():
                manifest_paths.add(str(candidate))
                break
    return sorted(manifest_paths)


def load_sweep_manifest(log_paths: List[str], suite_dir: Optional[str]) -> Dict[str, Dict[str, Optional[object]]]:
    manifest_paths = find_sweep_manifest_paths(log_paths, suite_dir)
    if not manifest_paths:
        return {}
    mapping: Dict[str, Dict[str, Optional[object]]] = {}
    for manifest_path in manifest_paths:
        base_dir = os.path.dirname(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                log_path = event.get("log_path")
                if not log_path:
                    continue
                if not os.path.isabs(log_path):
                    log_path = os.path.join(base_dir, log_path)
                key = normalize_path(log_path)
                mapping[key] = {
                    "param_name": event.get("param_name"),
                    "param_value": event.get("param_value"),
                    "param_scale": event.get("param_scale"),
                }
    return mapping


def infer_param_from_path(log_path: str, default_param_name: Optional[str]) -> Dict[str, Optional[object]]:
    parts = Path(log_path).parts
    for idx, part in enumerate(parts):
        if part.startswith("value="):
            raw = part.split("=", 1)[1]
            try:
                value = float(raw)
            except ValueError:
                value = None
            param_name = default_param_name or (parts[idx - 1] if idx > 0 else None) or "param"
            return {"param_name": param_name, "param_value": value, "param_scale": None}
    return {"param_name": default_param_name, "param_value": None, "param_scale": None}


def resolve_param_meta(
    event: Dict,
    param_meta_by_log: Dict[str, Dict[str, Optional[object]]],
    default_param_name: Optional[str],
) -> Dict[str, Optional[object]]:
    param_name = event.get("param_name")
    param_value = event.get("param_value")
    param_scale = event.get("param_scale")
    if param_name is None or param_value is None or param_scale is None:
        log_path = event.get("_log_path")
        if log_path:
            meta = param_meta_by_log.get(normalize_path(log_path))
            if meta:
                if param_name is None:
                    param_name = meta.get("param_name")
                if param_value is None:
                    param_value = meta.get("param_value")
                if param_scale is None:
                    param_scale = meta.get("param_scale")
            else:
                inferred = infer_param_from_path(log_path, default_param_name)
                if param_name is None:
                    param_name = inferred.get("param_name")
                if param_value is None:
                    param_value = inferred.get("param_value")
                if param_scale is None:
                    param_scale = inferred.get("param_scale")
    if param_value is not None:
        try:
            param_value = float(param_value)
        except (TypeError, ValueError):
            param_value = None
    return {
        "param_name": param_name,
        "param_value": param_value,
        "param_scale": param_scale,
    }


def format_param_value(value: float, scale: Optional[str]) -> str:
    if not np.isfinite(value):
        return "nan"
    if scale == "log":
        if value >= 10.0 or value <= 0.1:
            return f"{value:.1g}"
        return f"{value:.3g}"
    return f"{value:.3g}"


def format_param_label(param_name: Optional[str], param_value: float, param_scale: Optional[str]) -> str:
    base = param_name or "param"
    if base.startswith("lambda"):
        base = "lambda"
    value_str = format_param_value(param_value, param_scale)
    return f"{base}={value_str}"


def clamp_episode_record(rec: Dict) -> bool:
    clamped = False
    final_s = rec.get("final_S_eff_episode")
    if final_s is not None and np.isfinite(final_s):
        if final_s < 0.0:
            rec["final_S_eff_episode"] = 0.0
            clamped = True
    max_g = rec.get("max_G_true_episode")
    if max_g is not None and np.isfinite(max_g):
        new_g = float(np.clip(max_g, 0.0, 1.0))
        if new_g != max_g:
            rec["max_G_true_episode"] = new_g
            clamped = True
    g_p95 = rec.get("G_p95_episode")
    if g_p95 is not None and np.isfinite(g_p95):
        new_p95 = float(np.clip(g_p95, 0.0, 1.0))
        if new_p95 != g_p95:
            rec["G_p95_episode"] = new_p95
            clamped = True
    frac = rec.get("frac_steps_above_tau")
    if frac is not None and np.isfinite(frac):
        new_frac = float(np.clip(frac, 0.0, 1.0))
        if new_frac != frac:
            rec["frac_steps_above_tau"] = new_frac
            clamped = True
    return clamped


def write_episode_points_csv(records: List[Dict], path: str) -> None:
    if not records:
        return
    fieldnames = [
        "exp_name",
        "seed",
        "run_id",
        "episode_idx",
        "param_name",
        "param_value",
        "final_S_eff",
        "max_G_true",
        "G_p95",
        "frac_steps_above_tau",
        "T_tau",
        "reached_tau",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r.get("exp_name"), r.get("run_id"), r.get("episode_idx", 0))):
            param_value = rec.get("param_value")
            if param_value is None or not np.isfinite(param_value):
                param_value = ""
            writer.writerow(
                {
                    "exp_name": rec.get("exp_name", ""),
                    "seed": rec.get("seed", ""),
                    "run_id": rec.get("run_id", ""),
                    "episode_idx": rec.get("episode_idx", 0),
                    "param_name": rec.get("param_name") or "",
                    "param_value": param_value,
                    "final_S_eff": rec.get("final_S_eff_episode", float("nan")),
                    "max_G_true": rec.get("max_G_true_episode", float("nan")),
                    "G_p95": rec.get("G_p95_episode", float("nan")),
                    "frac_steps_above_tau": rec.get("frac_steps_above_tau", float("nan")),
                    "T_tau": rec.get("T_tau", float("nan")),
                    "reached_tau": rec.get("reached_tau", False),
                }
            )


def write_episode_diagnostics(
    records: List[Dict],
    group_key: str,
    path: str,
    param_name: Optional[str] = None,
    param_scale: Optional[str] = None,
    tau: Optional[float] = None,
) -> None:
    if not records:
        return
    by_group = defaultdict(list)
    for rec in records:
        max_c = rec.get("max_G_true_episode", float("nan"))
        final_s = rec.get("final_S_eff_episode", float("nan"))
        if not (np.isfinite(max_c) and np.isfinite(final_s)):
            continue
        if group_key == "param_value":
            key = rec.get("param_value")
            if key is None or not np.isfinite(key):
                continue
        else:
            key = rec.get("exp_name", "unknown")
        by_group[key].append(rec)
    lines = []
    for key in sorted(by_group, key=lambda k: (str(type(k)), k)):
        group_records = by_group[key]
        max_vals = [r["max_G_true_episode"] for r in group_records if np.isfinite(r["max_G_true_episode"])]
        final_vals = [r["final_S_eff_episode"] for r in group_records if np.isfinite(r["final_S_eff_episode"])]
        p95_vals = [r.get("G_p95_episode") for r in group_records if np.isfinite(r.get("G_p95_episode", float("nan")))]
        frac_vals = [
            r.get("frac_steps_above_tau")
            for r in group_records
            if np.isfinite(r.get("frac_steps_above_tau", float("nan")))
        ]
        t_tau_vals = [r.get("T_tau") for r in group_records if np.isfinite(r.get("T_tau", float("nan")))]
        reached = [bool(r.get("reached_tau")) for r in group_records]
        reached_frac = float(np.mean(reached)) if reached else 0.0
        clamped_any = any(r.get("clamped", False) for r in group_records)
        if not max_vals or not final_vals:
            continue
        label = str(key)
        if group_key == "param_value":
            label = format_param_label(param_name or "param", float(key), param_scale)
        text = (
            "group {label}: n={n} reached_tau_frac={reach:.3f} "
            "max_G_true(min/med/max)={cmin:.3f}/{cmed:.3f}/{cmax:.3f} "
            "final_S_eff(min/med/max)={smin:.3f}/{smed:.3f}/{smax:.3f}"
        ).format(
            label=label,
            n=len(group_records),
            reach=reached_frac,
            cmin=float(np.min(max_vals)),
            cmed=float(np.median(max_vals)),
            cmax=float(np.max(max_vals)),
            smin=float(np.min(final_vals)),
            smed=float(np.median(final_vals)),
            smax=float(np.max(final_vals)),
        )
        if p95_vals:
            text += " G_p95(min/med/max)={:.3f}/{:.3f}/{:.3f}".format(
                float(np.min(p95_vals)),
                float(np.median(p95_vals)),
                float(np.max(p95_vals)),
            )
        if frac_vals:
            text += " frac_steps_above_tau(min/med/max)={:.3f}/{:.3f}/{:.3f}".format(
                float(np.min(frac_vals)),
                float(np.median(frac_vals)),
                float(np.max(frac_vals)),
            )
        if t_tau_vals:
            text += " T_tau(min/med/max)={:.1f}/{:.1f}/{:.1f}".format(
                float(np.min(t_tau_vals)),
                float(np.median(t_tau_vals)),
                float(np.max(t_tau_vals)),
            )
        if clamped_any:
            text += " clamped_values=yes"
        lines.append(text)
    summary_lines = []
    if tau is not None and np.isfinite(tau):
        summary_lines.append(f"tau={tau:.3g}")
    horizons = [r.get("horizon") for r in records if np.isfinite(r.get("horizon", float("nan")))]
    if horizons:
        summary_lines.append(f"horizon_max={int(np.max(horizons))}")
    summary_lines.append(f"total_episodes={len(records)}")
    text = "\n".join(summary_lines + lines)
    if text:
        print(text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def parse_step_metrics(paths: List[str]) -> Tuple[List[Dict], Dict[str, set], Dict[str, set]]:
    step_events = []
    episodes_seen = defaultdict(set)
    episodes_ended = defaultdict(set)
    for path in paths:
        if not os.path.exists(path):
            print(f"warning: log path not found: {path}", file=sys.stderr)
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                run_id = event.get("run_id", "unknown")
                episode_idx = event.get("episode_idx", event.get("episode_id"))
                if episode_idx is None:
                    episode_idx = -1
                if event.get("type") == "step_metrics":
                    event["_log_path"] = path
                    step_events.append(event)
                    episodes_seen[run_id].add(episode_idx)
                elif event.get("type") == "episode_end":
                    episodes_ended[run_id].add(episode_idx)
    step_events.sort(
        key=lambda e: (
            e.get("episode_idx", e.get("episode_id", 0)),
            e.get("t", e.get("step", 0)),
        )
    )
    return step_events, episodes_seen, episodes_ended


def parse_episode_summaries(paths: List[str]) -> List[Dict]:
    summaries = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "episode_summary":
                    event["_log_path"] = path
                    summaries.append(event)
    return summaries


def warn_missing_tail(episodes_seen: Dict[str, set], episodes_ended: Dict[str, set]) -> None:
    for run_id, eps in episodes_seen.items():
        missing = sorted(eps - episodes_ended.get(run_id, set()))
        if missing:
            sample = ", ".join(str(e) for e in missing[:5])
            extra = "" if len(missing) <= 5 else "..."
            print(
                f"warning: run_id {run_id} missing episode_end for episodes: {sample}{extra}",
                file=sys.stderr,
            )


def collect_trajectories(step_events: List[Dict], include_exp_names: Optional[List[str]]):
    trajectories = {}
    gov_counts = defaultdict(
        lambda: defaultdict(
            lambda: {
                "proposed": 0,
                "approved": 0,
                "denied": 0,
                "rejected_veto_quorum": 0,
                "rejected_quorum": 0,
                "rejected_veto_window": 0,
            }
        )
    )
    agent_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    no_control_rates = defaultdict(lambda: defaultdict(list))
    for event in step_events:
        exp_name = event.get("exp_name", "unknown")
        if include_exp_names and exp_name not in include_exp_names:
            continue
        run_id = event.get("run_id", "unknown")
        episode_idx = event.get("episode_idx", event.get("episode_id", 0))
        t = event.get("t", event.get("step", 0))
        lineage_entries = event.get("lineages", [])
        if isinstance(lineage_entries, dict):
            lineage_entries = list(lineage_entries.values())
        for entry in lineage_entries:
            lid = entry.get("lineage_id")
            if lid is None:
                continue
            s_eff = float(entry.get("S_eff_post", entry.get("S_eff", 0.0)))
            key = (exp_name, run_id, episode_idx, lid)
            traj = trajectories.setdefault(
                key,
                {
                    "t": [],
                    "G": [],
                    "S": [],
                    "agents": [],
                    "proposals": 0,
                    "approved": 0,
                    "denied": 0,
                },
            )
            traj["t"].append(int(t))
            traj["G"].append(float(entry.get("G_true", 0.0)))
            traj["S"].append(s_eff)
            traj["agents"].append(int(entry.get("agent_count_alive", 0)))
            traj["proposals"] += int(entry.get("gated_proposals_count", 0))
            traj["approved"] += int(entry.get("gated_approved_count", 0))
            traj["denied"] += int(entry.get("gated_denied_count", 0))

            counts = gov_counts[exp_name][int(t)]
            counts["proposed"] += int(entry.get("gated_proposals_count", 0))
            counts["approved"] += int(entry.get("gated_approved_count", 0))
            counts["denied"] += int(entry.get("gated_denied_count", 0))
            counts["rejected_veto_quorum"] += int(entry.get("gated_rejected_veto_quorum_count", 0))
            counts["rejected_quorum"] += int(entry.get("gated_rejected_quorum_count", 0))
            counts["rejected_veto_window"] += int(entry.get("gated_rejected_veto_window_count", 0))

            agent_counts[exp_name][lid][int(t)].append(int(entry.get("agent_count_alive", 0)))
            executed = int(entry.get("actions_executed_count", 0))
            no_control = int(entry.get("no_control_executed_count_step", 0))
            rate = float(no_control) / float(max(1, executed))
            no_control_rates[exp_name][int(t)].append(rate)
    return trajectories, gov_counts, agent_counts, no_control_rates


def finalize_trajectories(trajectories: Dict, threshold: float):
    metrics = []
    by_exp = defaultdict(list)
    for key, traj in trajectories.items():
        exp_name, run_id, episode_idx, lid = key
        order = np.argsort(traj["t"])
        t_sorted = [traj["t"][i] for i in order]
        g_sorted = [traj["G"][i] for i in order]
        s_sorted = [traj["S"][i] for i in order]
        agents_sorted = [traj["agents"][i] for i in order]
        max_g = float(np.max(g_sorted)) if g_sorted else float("nan")
        final_s = float(s_sorted[-1]) if s_sorted else float("nan")
        time_to = None
        for t_val, g_val in zip(t_sorted, g_sorted):
            if g_val >= threshold:
                time_to = int(t_val)
                break
        avg_agents = float(np.mean(agents_sorted)) if agents_sorted else float("nan")
        total_props = traj["proposals"]
        total_denied = traj["denied"]
        total_approved = traj["approved"]
        deny_rate = float(total_denied) / float(max(1, total_props))
        record = {
            "exp_name": exp_name,
            "run_id": run_id,
            "episode_idx": episode_idx,
            "lineage_id": lid,
            "t": t_sorted,
            "G": g_sorted,
            "S": s_sorted,
            "final_S": final_s,
            "max_G": max_g,
            "time_to_Gth": time_to,
            "avg_agents": avg_agents,
            "total_proposals": total_props,
            "total_denied": total_denied,
            "total_approved": total_approved,
            "deny_rate": deny_rate,
        }
        metrics.append(record)
        by_exp[exp_name].append(record)
    return metrics, by_exp


def plot_g_trajectories(by_exp: Dict, out_dir: str, threshold: float, max_episodes_for_traj: int) -> None:
    for exp_name, records in by_exp.items():
        episodes = sorted({(r["run_id"], r["episode_idx"]) for r in records})
        selected = set(episodes[:max_episodes_for_traj])
        plt.figure()
        for record in records:
            if (record["run_id"], record["episode_idx"]) not in selected:
                continue
            plt.plot(record["t"], record["G"], label=f"ep{record['episode_idx']}_l{record['lineage_id']}")
        plt.axhline(threshold, linestyle="--")
        plt.title("HOPS: Gap trajectories G_true(t)")
        plt.xlabel("t")
        plt.ylabel("G_true")
        if len(selected) > 0:
            plt.legend()
        save_plot(os.path.join(out_dir, f"{exp_name}_G_traj.png"))


def plot_s_trajectories(by_exp: Dict, out_dir: str, max_episodes_for_traj: int) -> None:
    for exp_name, records in by_exp.items():
        episodes = sorted({(r["run_id"], r["episode_idx"]) for r in records})
        selected = set(episodes[:max_episodes_for_traj])
        plt.figure()
        for record in records:
            if (record["run_id"], record["episode_idx"]) not in selected:
                continue
            plt.plot(record["t"], record["S"], label=f"ep{record['episode_idx']}_l{record['lineage_id']}")
        plt.title("HOPS: Effective stock trajectories S_eff(t)")
        plt.xlabel("t")
        plt.ylabel("S_eff")
        if len(selected) > 0:
            plt.legend()
        save_plot(os.path.join(out_dir, f"{exp_name}_S_traj.png"))


def plot_tradeoff(by_exp: Dict, out_dir: str) -> None:
    for exp_name, records in by_exp.items():
        xs = [r["max_G"] for r in records]
        ys = [r["final_S"] for r in records]
        if not xs or not ys:
            continue
        plt.figure()
        plt.scatter(xs, ys, alpha=0.7)
        plt.title("HOPS: Effective stock–gap tradeoff")
        plt.xlabel("max_G_true")
        plt.ylabel("final_S_eff")
        save_plot(os.path.join(out_dir, f"{exp_name}_tradeoff.png"))


def plot_survival(by_exp: Dict, out_dir: str, threshold: float) -> None:
    if not by_exp:
        return
    plt.figure()
    for exp_name, records in by_exp.items():
        if not records:
            continue
        max_t = max((max(r["t"]) for r in records if r["t"]), default=0)
        if max_t == 0:
            continue
        survival = []
        for t in range(max_t + 1):
            alive = 0
            for record in records:
                t_hit = record["time_to_Gth"]
                if t_hit is None or t_hit > t:
                    alive += 1
            survival.append(alive / float(len(records)))
        plt.plot(range(max_t + 1), survival, label=exp_name)
    plt.title("HOPS: Survival curve P(G_true(t) < threshold)")
    plt.xlabel("t")
    plt.ylabel("survival")
    if len(by_exp) > 1:
        plt.legend()
    save_plot(os.path.join(out_dir, f"survival_threshold_{threshold}.png"))


def plot_governance_counts_timeseries(gov_counts: Dict, out_dir: str) -> None:
    for exp_name, counts_by_t in gov_counts.items():
        if not counts_by_t:
            continue
        ts = sorted(counts_by_t.keys())
        proposed = [counts_by_t[t]["proposed"] for t in ts]
        approved = [counts_by_t[t]["approved"] for t in ts]
        denied = [counts_by_t[t]["denied"] for t in ts]
        veto_quorum = [counts_by_t[t]["rejected_veto_quorum"] for t in ts]
        quorum = [counts_by_t[t]["rejected_quorum"] for t in ts]
        veto_window = [counts_by_t[t]["rejected_veto_window"] for t in ts]
        if (
            sum(proposed) == 0
            and sum(approved) == 0
            and sum(denied) == 0
            and sum(veto_quorum) == 0
            and sum(quorum) == 0
            and sum(veto_window) == 0
        ):
            continue
        plt.figure()
        plt.plot(ts, proposed, label="proposed")
        plt.plot(ts, approved, label="approved")
        plt.plot(ts, denied, label="denied")
        plt.plot(ts, veto_quorum, label="rejected_veto_quorum")
        plt.plot(ts, quorum, label="rejected_quorum")
        plt.plot(ts, veto_window, label="rejected_veto_window")
        plt.title("HOPS: Governance outcomes over time")
        plt.xlabel("t")
        plt.ylabel("count")
        plt.legend()
        save_plot(os.path.join(out_dir, f"{exp_name}_governance_counts.png"))


def plot_agents_over_time(agent_counts: Dict, out_dir: str) -> None:
    for exp_name, by_lineage in agent_counts.items():
        if not by_lineage:
            continue
        plt.figure()
        for lid, by_t in by_lineage.items():
            ts = sorted(by_t.keys())
            means = [float(np.mean(by_t[t])) if by_t[t] else 0.0 for t in ts]
            plt.plot(ts, means, label=f"lineage {lid}")
        plt.title("HOPS: Alive agents over time")
        plt.xlabel("t")
        plt.ylabel("alive agents")
        plt.legend()
        save_plot(os.path.join(out_dir, f"{exp_name}_agents_over_time.png"))


def plot_noop_rate(no_control_rates: Dict, out_dir: str) -> None:
    for exp_name, by_t in no_control_rates.items():
        if not by_t:
            continue
        ts = sorted(by_t.keys())
        means = [float(np.mean(by_t[t])) if by_t[t] else 0.0 for t in ts]
        plt.figure()
        plt.plot(ts, means, label="noop_rate")
        plt.title("HOPS: NOOP rate over time")
        plt.xlabel("t")
        plt.ylabel("noop_rate")
        plt.legend()
        save_plot(os.path.join(out_dir, f"{exp_name}_noop_rate.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+")
    parser.add_argument("--log_paths", nargs="+")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--suite_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--param_name", type=str, default=None)
    parser.add_argument("--max_episodes_for_traj", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--include_exp_names", type=str, default=None)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--analyze_window", type=int, dest="window")
    parser.add_argument("--walkthrough", action="store_true")
    parser.add_argument("--episode_progress", action="store_true")
    parser.add_argument("--all_lineages", action="store_true")
    parser.add_argument("--debug_plots", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--lineage_id", type=int, default=0)
    parser.add_argument("--run_id", type=str, default="latest")
    args = parser.parse_args()
    if args.debug:
        args.debug_plots = True
    if args.run_dir:
        if not args.log_paths and not args.logs:
            args.log_paths = [os.path.join(args.run_dir, "logs", "*.jsonl")]
        if args.out_dir is None:
            args.out_dir = os.path.join(args.run_dir, "reports")
    suite_paths = []
    if args.suite_dir:
        suite_root = Path(args.suite_dir)
        suite_paths = [str(p) for p in suite_root.rglob("logs.jsonl")]
        if args.out_dir is None:
            args.out_dir = os.path.join(args.suite_dir, "analysis", "exp_compare")
    if args.log_paths or suite_paths:
        log_paths = expand_log_paths(args.log_paths or [])
        if not log_paths:
            log_paths = []
        if suite_paths:
            log_paths = list(dict.fromkeys(log_paths + suite_paths))
        if not log_paths:
            parser.error("no log_paths matched any files")
        out_dir = args.out_dir or "results"
        compare_dir = out_dir
        os.makedirs(compare_dir, exist_ok=True)
        include_exp_names = None
        if args.include_exp_names:
            include_exp_names = [n.strip() for n in args.include_exp_names.split(",") if n.strip()]
        step_events, episodes_seen, episodes_ended = parse_step_metrics(log_paths)
        summary_events = parse_episode_summaries(log_paths)
        if not step_events and not summary_events:
            print("warning: no step_metrics or episode_summary events found", file=sys.stderr)
            return
        param_meta_by_log = load_sweep_manifest(log_paths, args.suite_dir)
        (
            episode_records,
            episode_series,
            max_g_series,
            gov_by_exp,
            tradeoff_records,
        ) = collect_episode_tradeoff(
            step_events,
            include_exp_names,
            summary_events,
            param_meta_by_log,
            args.param_name,
            args.threshold,
        )
        if episode_records:
            if tradeoff_records:
                write_tradeoff_episode_csv(
                    tradeoff_records,
                    os.path.join(compare_dir, "episode_points.csv"),
                )
                points_by_exp = defaultdict(list)
                medians_by_exp = {}
                for rec in tradeoff_records:
                    exp_name = rec.get("exp_name", "unknown")
                    points_by_exp[exp_name].append((rec["risk_e"], rec["welfare_e"]))
                for exp_name, points in points_by_exp.items():
                    risks = np.array([p[0] for p in points], dtype=np.float64)
                    welfare = np.array([p[1] for p in points], dtype=np.float64)
                    if risks.size and welfare.size:
                        medians_by_exp[exp_name] = (float(np.median(risks)), float(np.median(welfare)))
                plot_canonical_tradeoff_pareto(
                    points_by_exp,
                    medians_by_exp,
                    os.path.join(compare_dir, "exp_tradeoff_pareto.png"),
                )
            param_values = {
                rec.get("param_value")
                for rec in episode_records
                if rec.get("param_value") is not None and np.isfinite(rec.get("param_value"))
            }
            is_sweep = len(param_values) > 1
            param_name = args.param_name
            param_scale = None
            for rec in episode_records:
                if rec.get("param_name") and param_name is None:
                    param_name = rec.get("param_name")
                if rec.get("param_scale") and param_scale is None:
                    param_scale = rec.get("param_scale")
            if is_sweep:
                sweep_dir = os.path.join(os.path.dirname(compare_dir), "sweep_compare")
                os.makedirs(sweep_dir, exist_ok=True)
                plot_sweep_tradeoff_scatter(
                    episode_records,
                    os.path.join(compare_dir, "exp_tradeoff_scatter.png"),
                    param_name,
                )
                plot_sweep_pareto_frontier(
                    episode_records,
                    os.path.join(compare_dir, "exp_pareto_frontier.png"),
                    param_name,
                    param_scale,
                )
                write_episode_diagnostics(
                    episode_records,
                    "param_value",
                    os.path.join(compare_dir, "diagnostics.txt"),
                    param_name,
                    param_scale,
                    args.threshold,
                )
                if tradeoff_records:
                    plot_sweep_time_to_tau(
                        tradeoff_records,
                        float(args.threshold),
                        os.path.join(sweep_dir, "sweep_time_to_tau.png"),
                    )
                    plot_sweep_welfare(
                        tradeoff_records,
                        os.path.join(sweep_dir, "sweep_welfare_vs_sneakiness.png"),
                    )
                    plot_sweep_tradeoff_scatter_colored(
                        tradeoff_records,
                        os.path.join(sweep_dir, "sweep_tradeoff_scatter_colored.png"),
                    )
                    plot_sweep_gap_divergence(
                        tradeoff_records,
                        os.path.join(sweep_dir, "sweep_gap_divergence.png"),
                    )
            else:
                points_by_exp = defaultdict(list)
                for rec in episode_records:
                    max_g = rec.get("max_G_true_episode", float("nan"))
                    final_s = rec.get("final_S_eff_episode", float("nan"))
                    if not (np.isfinite(max_g) and np.isfinite(final_s)):
                        continue
                    points_by_exp[rec.get("exp_name", "unknown")].append((max_g, final_s))
                plot_exp_tradeoff_scatter(
                    points_by_exp,
                    os.path.join(compare_dir, "exp_tradeoff_scatter.png"),
                )
                plot_exp_pareto_frontier(
                    points_by_exp,
                    args.threshold if args.threshold is not None else None,
                    os.path.join(compare_dir, "exp_pareto_frontier.png"),
                )
                write_episode_diagnostics(
                    episode_records,
                    "exp_name",
                    os.path.join(compare_dir, "diagnostics.txt"),
                    param_name,
                    param_scale,
                    args.threshold,
                )
            plot_exp_episode_series(
                episode_series,
                "Final S_eff by episode (sum over lineages; mean over runs)",
                "final_S_eff",
                os.path.join(compare_dir, "exp_final_S_eff_by_episode.png"),
            )
            plot_exp_episode_series(
                max_g_series,
                "Max G_true by episode (max over steps/lineages; mean over runs)",
                "max_G_true",
                os.path.join(compare_dir, "exp_max_G_true_by_episode.png"),
            )
            group_key = "param_value" if is_sweep else "exp_name"
            divergence_series = build_episode_series(episode_records, "gap_divergence", group_key)
            plot_exp_episode_series(
                divergence_series,
                "Gap divergence by episode (true - observed; mean over runs)",
                "gap_divergence",
                os.path.join(compare_dir, "exp_divergence_by_episode.png"),
            )
            g_p95_series = build_episode_series(episode_records, "G_p95_episode", group_key)
            frac_series = build_episode_series(episode_records, "frac_steps_above_tau", group_key)
            plot_exp_episode_series(
                g_p95_series,
                "G_p95 by episode (max over lineages; mean over runs)",
                "G_p95",
                os.path.join(compare_dir, "exp_G_p95_by_episode.png"),
            )
            plot_exp_episode_series(
                frac_series,
                "Fraction of steps above tau by episode (mean over runs)",
                "frac_steps_above_tau",
                os.path.join(compare_dir, "exp_frac_steps_above_tau_by_episode.png"),
            )
            if args.threshold is not None and np.isfinite(args.threshold):
                plot_time_to_tau_cdf(
                    episode_records,
                    group_key,
                    float(args.threshold),
                    os.path.join(compare_dir, "exp_time_to_tau_cdf.png"),
                    param_name,
                    param_scale,
                )
                plot_reach_tau_rate(
                    episode_records,
                    group_key,
                    float(args.threshold),
                    os.path.join(compare_dir, "exp_reach_tau_rate.png"),
                    param_name,
                    param_scale,
                )
            plot_gov_outcomes_by_exp(
                gov_by_exp,
                os.path.join(compare_dir, "exp_gov_outcomes.png"),
            )
        if step_events:
            warn_missing_tail(episodes_seen, episodes_ended)
            (
                g_series,
                g_obs_series,
                s_series,
                _alive_series,
                _progress_series,
                _penalty_series,
                _total_series,
                gap_series,
                dormant_series,
                _time_to_threshold,
                _gov_outcomes,
                _tradeoff_points,
                _pareto_points,
            ) = collect_exp_compare(step_events, include_exp_names, args.threshold)
            plot_exp_compare_series(
                g_series,
                "G_true over time (per-step max over lineages; median over runs; q10–q90)",
                "G_true",
                os.path.join(compare_dir, "exp_compare_G_t.png"),
                clamp_min=0.0,
                clamp_max=1.0,
            )
            plot_exp_compare_series(
                s_series,
                "S_eff over time (per-step mean over lineages; median over runs; q10–q90)",
                "S_eff",
                os.path.join(compare_dir, "exp_compare_S_t.png"),
                clamp_min=0.0,
            )
            plot_exp_compare_series(
                gap_series,
                "gap_k over time (per-step mean over lineages; median over runs; q10–q90)",
                "gap_k_mean",
                os.path.join(compare_dir, "exp_compare_gap_k.png"),
                clamp_min=0.0,
            )
            plot_exp_compare_series(
                dormant_series,
                "Dormant stock mass over time (per-step mean; median over runs; q10–q90)",
                "dormant_mass_total",
                os.path.join(compare_dir, "exp_dormant_stock_t.png"),
                clamp_min=0.0,
            )
            plot_exp_gap_true_vs_observed(
                g_series,
                g_obs_series,
                "G_true vs observed over time (median over runs)",
                "max_gap_ratio",
                os.path.join(compare_dir, "exp_gap_true_vs_observed_t.png"),
                clamp_min=0.0,
                clamp_max=1.0,
            )
        return
    if not args.logs and not args.log_paths and not args.suite_dir and not args.run_dir:
        parser.error("Please provide --suite_dir or --log_paths.")
    if not args.logs:
        parser.error("either --logs or --log_paths is required")
    out_dir = args.out_dir or os.path.join("runs", "fig")
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
        if plot_enabled("reward_g", args.debug_plots):
            plot_reward_and_g(
                step_end,
                data["episode_seed"],
                out_path(f"l{lid}_reward_g"),
                f"Reward and G_true (lineage {lid})",
            )

        if plot_enabled("reward_hist", args.debug_plots):
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
        lambda e: snapshot_s_eff(e.get("snapshot", {})),
    )
    last_steps_stock, last_stock = summarize_last_episode_scalar(
        stock_events, lambda e: snapshot_s_eff(e.get("snapshot", {}))
    )
    plot_series(
        steps,
        mean_stock_total,
        std_stock_total,
        last_steps_stock,
        last_stock,
        "Total effective stock",
        "S_eff",
        out_path("stock_total"),
    )
    plot_stock_and_g(stock_events, data["step_end"], data["episode_seed"], out_path("stock_total_g"))
    steps, mean_cap_max, std_cap_max = summarize_scalar_series(
        stock_events,
        data["episode_seed"],
        lambda e: float(np.max(e["snapshot"].get("cap_k_global", [])))
        if e.get("snapshot", {}).get("cap_k_global") is not None
        else None,
    )
    last_steps_cap, last_cap = summarize_last_episode_scalar(
        stock_events,
        lambda e: float(np.max(e["snapshot"].get("cap_k_global", [])))
        if e.get("snapshot", {}).get("cap_k_global") is not None
        else None,
    )
    plot_series(
        steps,
        mean_cap_max,
        std_cap_max,
        last_steps_cap,
        last_cap,
        "cap_k_global max",
        "cap_k_global max",
        out_path("cap_k_max"),
    )
    if plot_enabled("agents_reward_corr", args.debug_plots):
        plot_agents_reward_correlation(data["step_end"], out_path("agents_reward_corr"))

    steps, mean_stock_vec, std_stock_vec = summarize_vector_series(
        stock_events, data["episode_seed"], lambda e: snapshot_s_eff_vec(e.get("snapshot", {}))
    )
    plot_element_series(
        steps,
        mean_stock_vec,
        std_stock_vec,
        "Element S_eff over time",
        "S_eff",
        out_path("stock_elements"),
    )
    # omit stock histograms per request

    for lid in lineage_ids:
        step_end = filter_by_lineage(data["step_end"], lid)
        steps, mean_control_vec, std_control_vec = summarize_vector_series(
            step_end, data["episode_seed"], lambda e: np.array(e["snapshot"].get("totals", []), dtype=np.float32)
        )
        if plot_enabled("control_elements", args.debug_plots):
            plot_element_series(
                steps,
                mean_control_vec,
                std_control_vec,
                f"Element control totals over time (lineage {lid})",
                "control_totals",
                out_path(f"l{lid}_control_elements"),
            )
        if plot_enabled("control_elements_hist", args.debug_plots):
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
        if plot_enabled("latent_total", args.debug_plots):
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
        if plot_enabled("latent_total_hist", args.debug_plots):
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
        if plot_enabled("latent_elements", args.debug_plots):
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
        if plot_enabled("counts", args.debug_plots):
            plot_counts(action_applied, args.window, out_path(f"l{lid}_counts"))
        plot_action_mix(action_applied, args.window, out_path(f"l{lid}_action_mix"))
        if plot_enabled("action_hist", args.debug_plots):
            plot_action_hist_fallback(
                action_applied,
                actor_decision,
                out_path(f"l{lid}_action_hist"),
            )

        auth_events = filter_by_lineage(data["authorization"], lid)
        prop_meta = {pid: meta for pid, meta in proposal_meta.items() if meta.get("lineage_id") == lid}
        if auth_events:
            if plot_enabled("auth", args.debug_plots):
                plot_authorization(auth_events, out_path(f"l{lid}_auth"))
            plot_governance_counts(auth_events, prop_meta, args.window, out_path(f"l{lid}_gov_total"))
            if plot_enabled("gov_veto", args.debug_plots):
                plot_governance_veto_rate(auth_events, prop_meta, args.window, out_path(f"l{lid}_gov_veto"))
            if plot_enabled("gov_committee", args.debug_plots):
                plot_governance_committee_rate(
                    auth_events, prop_meta, args.window, out_path(f"l{lid}_gov_committee")
                )
            if plot_enabled("gov_authorized", args.debug_plots):
                plot_governance_authorized_rate(
                    auth_events, prop_meta, args.window, out_path(f"l{lid}_gov_authorized")
                )
            if plot_enabled("gov_approval_hist", args.debug_plots):
                plot_governance_hist(auth_events, out_path(f"l{lid}_gov_approval_hist"))
            plot_committee_outcomes(auth_events, out_path(f"l{lid}_committee_outcomes"))

        vote_events = filter_by_lineage(data["vote_cast"], lid)
        if plot_enabled("votes", args.debug_plots):
            plot_votes(vote_events, args.window, out_path(f"l{lid}_votes"))

        plot_agent_trajectories(step_end, data["episode_seed"], out_path(f"l{lid}_agents"))
        if plot_enabled("agents_hist", args.debug_plots):
            alive_vals = [e["snapshot"].get("alive", 0) for e in step_end]
            plot_hist(
                alive_vals,
                f"Alive agents histogram (lineage {lid})",
                "alive agents",
                out_path(f"l{lid}_agents_hist"),
            )
        plot_g_and_stock(step_end, data["episode_seed"], stock_events, out_path(f"l{lid}_g_stock"))
        if plot_enabled("reward_stock_window", args.debug_plots):
            plot_reward_stock_window(step_end, stock_events, args.window, out_path(f"l{lid}_reward_stock_window"))
        if plot_enabled("entrenchments", args.debug_plots):
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
