import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from threading import Thread
from typing import Dict, List, Optional, Tuple

import numpy as np


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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


def sanitize_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_-." else "_" for ch in value.strip())
    return cleaned or "run"


def format_value(value: float) -> str:
    text = f"{value:.6g}"
    return sanitize_name(text)


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return parts if parts else None


def detect_git_commit(root: str) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""
    return result.stdout.strip()


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dump_simple_yaml(data: dict, path: str) -> None:
    def _dump(obj, indent: int, lines: list) -> None:
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
    lines = []
    _dump(data, 0, lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def build_override_dict(param_name: str, value: float) -> dict:
    keys = [k for k in param_name.split(".") if k]
    override = current = {}
    for key in keys[:-1]:
        current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return override


def stable_hash_seed(
    master_seed: int,
    exp_name: str,
    param_name: str,
    param_value: float,
    run_idx: int,
) -> int:
    payload = f"{master_seed}|{exp_name}|{param_name}|{param_value:.10g}|{run_idx}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % (2**31)


def write_json(path: str, payload) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def parse_step_metrics(path: str) -> List[Dict]:
    step_events = []
    if not os.path.exists(path):
        return step_events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "step_metrics":
                step_events.append(event)
    step_events.sort(
        key=lambda e: (
            e.get("episode_idx", e.get("episode_id", 0)),
            e.get("t", e.get("step", 0)),
        )
    )
    return step_events


def parse_episode_summaries(path: str) -> List[Dict]:
    summaries = []
    if not os.path.exists(path):
        return summaries
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
                summaries.append(event)
    return summaries


def parse_param_value_from_path(path: str, value_map: Dict[str, float]) -> float:
    parts = path.split(os.sep)
    for part in parts:
        if part.startswith("value="):
            key = part.split("=", 1)[1]
            if key in value_map:
                return value_map[key]
            try:
                return float(key)
            except ValueError:
                return float("nan")
    return float("nan")


def summarize_log(path: str, tau: float) -> Tuple[Dict, List[float]]:
    summaries = parse_episode_summaries(path)
    if summaries:
        final_s_vals = [float(e.get("final_S_eff_episode", float("nan"))) for e in summaries]
        max_g_vals = [
            float(e.get("max_G_true_episode", e.get("max_C_episode", float("nan")))) for e in summaries
        ]
        final_alive_vals = [float(e.get("final_alive_agents_episode", float("nan"))) for e in summaries]
        final_s_by_episode = [float(v) for v in final_s_vals]
        final_s_mean = float(np.nanmean(final_s_vals)) if final_s_vals else float("nan")
        final_s_median = float(np.nanmedian(final_s_vals)) if final_s_vals else float("nan")
        final_s_last = final_s_vals[-1] if final_s_vals else float("nan")
        max_g_max = float(np.nanmax(max_g_vals)) if max_g_vals else float("nan")
        max_g_mean = float(np.nanmean(max_g_vals)) if max_g_vals else float("nan")
        survival_tau = (
            float(np.mean([1.0 if np.isfinite(v) and v < tau else 0.0 for v in max_g_vals]))
            if max_g_vals
            else float("nan")
        )
        final_alive_mean = float(np.nanmean(final_alive_vals)) if final_alive_vals else float("nan")
        return (
            {
                "final_S_mean": final_s_mean,
                "final_S_median": final_s_median,
                "final_S_last": final_s_last,
                "max_G_true_max": max_g_max,
                "max_G_true_mean": max_g_mean,
                "survival_tau_epfrac": survival_tau,
                "time_to_tau_mean": float("nan"),
                "time_to_tau_censored_rate": float("nan"),
                "final_alive_mean": final_alive_mean,
                "episodes": len(final_s_vals),
                "tau": tau,
            },
            final_s_by_episode,
        )
    step_events = parse_step_metrics(path)
    if not step_events:
        return {}, []
    episodes: Dict[int, Dict] = {}
    for event in step_events:
        episode_idx = event.get("episode_idx", event.get("episode_id", 0))
        if episode_idx is None:
            episode_idx = 0
        episode_idx = int(episode_idx)
        t = int(event.get("t", event.get("step", 0)))
        record = episodes.setdefault(
            episode_idx,
            {
                "last_t": -1,
                "final_s_vals": [],
                "final_alive_vals": [],
                "max_g_by_lineage": {},
                "time_to_tau": None,
            },
        )
        lineage_entries = event.get("lineages", [])
        if isinstance(lineage_entries, dict):
            lineage_entries = list(lineage_entries.values())
        g_vals = []
        for entry in lineage_entries:
            lid = entry.get("lineage_id")
            g_true = float(entry.get("G_true", 0.0))
            g_vals.append(g_true)
            if lid is not None:
                prev = record["max_g_by_lineage"].get(lid, 0.0)
                record["max_g_by_lineage"][lid] = max(prev, g_true)
        if g_vals and record["time_to_tau"] is None and float(np.mean(g_vals)) > tau:
            record["time_to_tau"] = t
        if t >= record["last_t"]:
            record["last_t"] = t
            record["final_s_vals"] = [
                float(entry.get("S_eff_post", entry.get("S_eff", 0.0))) for entry in lineage_entries
            ]
            record["final_alive_vals"] = [float(entry.get("agent_count_alive", 0.0)) for entry in lineage_entries]
    episode_ids = sorted(episodes.keys())
    final_s_by_episode = []
    final_s_vals = []
    max_g_vals = []
    final_alive_vals = []
    time_to_tau_vals = []
    censored = 0
    for eid in episode_ids:
        record = episodes[eid]
        final_s_e = float(np.sum(record["final_s_vals"])) if record["final_s_vals"] else float("nan")
        final_s_by_episode.append(final_s_e)
        final_s_vals.append(final_s_e)
        max_g_e = float(np.max(list(record["max_g_by_lineage"].values()))) if record["max_g_by_lineage"] else float("nan")
        max_g_vals.append(max_g_e)
        final_alive_e = float(np.mean(record["final_alive_vals"])) if record["final_alive_vals"] else float("nan")
        final_alive_vals.append(final_alive_e)
        if record["time_to_tau"] is None:
            censored += 1
        else:
            time_to_tau_vals.append(float(record["time_to_tau"]))
    final_s_mean = float(np.nanmean(final_s_vals)) if final_s_vals else float("nan")
    final_s_median = float(np.nanmedian(final_s_vals)) if final_s_vals else float("nan")
    final_s_last = final_s_vals[-1] if final_s_vals else float("nan")
    max_g_max = float(np.nanmax(max_g_vals)) if max_g_vals else float("nan")
    max_g_mean = float(np.nanmean(max_g_vals)) if max_g_vals else float("nan")
    survival_tau = float(np.mean([1.0 if np.isfinite(v) and v < tau else 0.0 for v in max_g_vals])) if max_g_vals else float("nan")
    time_to_tau_mean = float(np.mean(time_to_tau_vals)) if time_to_tau_vals else float("nan")
    time_to_tau_censored_rate = float(censored) / float(len(episode_ids)) if episode_ids else float("nan")
    final_alive_mean = float(np.nanmean(final_alive_vals)) if final_alive_vals else float("nan")
    return (
        {
            "final_S_mean": final_s_mean,
            "final_S_median": final_s_median,
            "final_S_last": final_s_last,
            "max_G_true_max": max_g_max,
            "max_G_true_mean": max_g_mean,
            "survival_tau_epfrac": survival_tau,
            "time_to_tau_mean": time_to_tau_mean,
            "time_to_tau_censored_rate": time_to_tau_censored_rate,
            "final_alive_mean": final_alive_mean,
            "episodes": len(episode_ids),
            "tau": tau,
        },
        final_s_by_episode,
    )


def aggregate_by_param(rows: List[Dict]) -> List[Dict]:
    buckets: Dict[str, List[Dict]] = {}
    for row in rows:
        key = str(row["param_value"])
        buckets.setdefault(key, []).append(row)
    summary = []
    for key, items in sorted(buckets.items(), key=lambda kv: float(kv[0])):
        def _mean_std(values: List[float]) -> Tuple[float, float]:
            arr = np.array(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return float("nan"), float("nan")
            return float(np.mean(arr)), float(np.std(arr))

        final_s_vals = [it.get("final_S_mean", float("nan")) for it in items]
        max_g_vals = [it.get("max_G_true_max", float("nan")) for it in items]
        alive_vals = [it.get("final_alive_mean", float("nan")) for it in items]
        time_vals = [it.get("time_to_tau_mean", float("nan")) for it in items]
        surv_vals = [it.get("survival_tau_epfrac", float("nan")) for it in items]
        final_s_mean, final_s_std = _mean_std(final_s_vals)
        max_g_mean, max_g_std = _mean_std(max_g_vals)
        alive_mean, alive_std = _mean_std(alive_vals)
        time_mean, time_std = _mean_std(time_vals)
        survival_mean, survival_std = _mean_std(surv_vals)
        summary.append(
            {
                "param_value": float(key),
                "n": len(items),
                "final_s_mean": final_s_mean,
                "final_s_std": final_s_std,
                "max_g_mean": max_g_mean,
                "max_g_std": max_g_std,
                "survival_rate_tau": survival_mean,
                "survival_rate_tau_std": survival_std,
                "time_to_tau_mean": time_mean,
                "time_to_tau_std": time_std,
                "final_alive_agents_mean": alive_mean,
                "final_alive_agents_std": alive_std,
            }
        )
    return summary


def plot_sweep(rows: List[Dict], out_dir: str, final_s_by_episode: Dict[float, List[List[float]]]) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    summary = aggregate_by_param(rows)
    xs = [row["param_value"] for row in summary]
    plt.figure()
    plt.errorbar(xs, [r["final_s_mean"] for r in summary], yerr=[r["final_s_std"] for r in summary], fmt="-o")
    plt.xlabel("param_value")
    plt.ylabel("final_S_mean (mean ± std across runs)")
    plt.title("Final S vs parameter")
    plt.savefig(os.path.join(out_dir, "param_vs_final_s.png"))
    plt.close()

    plt.figure()
    plt.errorbar(xs, [r["max_g_mean"] for r in summary], yerr=[r["max_g_std"] for r in summary], fmt="-o")
    plt.xlabel("param_value")
    plt.ylabel("max_G_true_max (mean ± std across runs)")
    plt.title("Max G_true vs parameter")
    plt.savefig(os.path.join(out_dir, "param_vs_max_g.png"))
    plt.close()

    plt.figure()
    plt.plot(xs, [r["survival_rate_tau"] for r in summary], marker="o")
    plt.xlabel("param_value")
    plt.ylabel("survival_rate_tau")
    plt.title("Survival rate vs parameter")
    plt.savefig(os.path.join(out_dir, "param_vs_survival_rate.png"))
    plt.close()

    plt.figure()
    plt.errorbar(xs, [r["time_to_tau_mean"] for r in summary], yerr=[r["time_to_tau_std"] for r in summary], fmt="-o")
    plt.xlabel("param_value")
    plt.ylabel("time_to_tau (mean ± std)")
    plt.title("Time to tau vs parameter")
    plt.savefig(os.path.join(out_dir, "param_vs_time_to_tau.png"))
    plt.close()

    if final_s_by_episode:
        plt.figure()
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for idx, param_value in enumerate(sorted(final_s_by_episode.keys())):
            series_list = final_s_by_episode[param_value]
            if not series_list:
                continue
            color = colors[idx % len(colors)] if colors else None
            max_len = max(len(s) for s in series_list)
            mean_vals = []
            for i in range(max_len):
                vals = [s[i] for s in series_list if i < len(s) and np.isfinite(s[i])]
                mean_vals.append(float(np.mean(vals)) if vals else float("nan"))
            for series in series_list:
                xs = list(range(len(series)))
                plt.plot(xs, series, color=color, alpha=0.25, linewidth=1.0)
            plt.plot(list(range(len(mean_vals))), mean_vals, color=color, linewidth=2.0, label=str(param_value))
        plt.xlabel("episode")
        plt.ylabel("final_S per episode")
        plt.title("Final S by episode")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "final_S_by_episode.png"))
        plt.close()


def plot_tradeoff_pareto_by_param(
    rows: List[Dict],
    out_dir: str,
    tau: Optional[float],
    param_name: str,
    param_scale: str,
) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt
    summary = aggregate_by_param(rows)
    points = []
    for entry in summary:
        param = float(entry.get("param_value", float("nan")))
        x_mean = float(entry.get("max_g_mean", float("nan")))
        y_mean = float(entry.get("final_s_mean", float("nan")))
        x_std = float(entry.get("max_g_std", 0.0))
        y_std = float(entry.get("final_s_std", 0.0))
        if not np.isfinite(param) or not np.isfinite(x_mean) or not np.isfinite(y_mean):
            continue
        points.append((param, x_mean, y_mean, x_std, y_std))
    if not points:
        return
    pareto = []
    best_y = -float("inf")
    for param, x_mean, y_mean, _, _ in sorted(points, key=lambda item: item[1]):
        if y_mean > best_y + 1e-12:
            pareto.append((param, x_mean, y_mean))
            best_y = y_mean
    plt.figure()
    for idx, (param, x_mean, y_mean, x_std, y_std) in enumerate(points):
        plt.errorbar(
            [x_mean],
            [y_mean],
            xerr=[x_std],
            yerr=[y_std],
            fmt="o",
            color="tab:blue",
            alpha=0.7,
        )
    if pareto:
        fx = [p[1] for p in pareto]
        fy = [p[2] for p in pareto]
        plt.plot(fx, fy, linewidth=2.0, color="tab:orange", label="Pareto frontier")
        y_vals = [p[2] for p in pareto]
        y_range = max(y_vals) - min(y_vals) if y_vals else 0.0
        prefix = "λ" if param_name in {"lambda_penalty", "lambda_gap"} else param_name
        for idx, (param, x_mean, y_mean) in enumerate(pareto):
            label = f"{prefix}={param:g}" if param_scale == "log" else f"{prefix}={param:g}"
            dy = (0.02 * y_range) if y_range > 0 else 0.02 * max(1.0, abs(y_mean))
            offset = dy if idx % 2 == 0 else -dy
            plt.annotate(
                label,
                (x_mean, y_mean),
                xytext=(x_mean, y_mean + offset),
                textcoords="data",
                fontsize=8,
                arrowprops={"arrowstyle": "-", "lw": 0.5},
            )
    if tau is not None and np.isfinite(tau):
        plt.axvline(tau, linestyle="--", color="gray", linewidth=1.0)
    plt.title("Pareto frontier by param (x=risk, y=progress; mean over runs)")
    plt.xlabel("max_G_true (mean over runs)")
    plt.ylabel("final_S_eff (mean over runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tradeoff_pareto_by_param.png"))
    plt.close()


def add_to_zip(zipf: zipfile.ZipFile, path: str, base_dir: str) -> bool:
    if not os.path.exists(path):
        return False
    arcname = os.path.relpath(path, base_dir)
    zipf.write(path, arcname)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", type=str, default=os.path.join("runs", "sweeps"))
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", choices=["train", "simulate"], default="train")
    parser.add_argument("--exp_name", type=str, default="baseline")
    parser.add_argument("--param_name", type=str, required=True)
    parser.add_argument("--param_min", type=float, required=True)
    parser.add_argument("--param_max", type=float, required=True)
    parser.add_argument("--param_n", type=int, required=True)
    parser.add_argument("--param_scale", choices=["linear", "log"], default="log")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--master_seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no_reset", action="store_true")
    parser.add_argument("--log_mode", choices=["full", "metrics_only", "none"], default=None)
    parser.add_argument("--log_stride", type=int, default=None)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--bundle_small", action="store_true")
    parser.add_argument("--no_bundle_small", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args, extra = parser.parse_known_args()

    if args.param_name.startswith("--"):
        parser.error("param_name should not start with '--'")
    if args.seeds and args.runs is not None:
        parser.error("choose either --seeds or --runs")
    if not args.seeds and args.runs is None:
        parser.error("provide --seeds or --runs with --master_seed")
    if args.reset and args.no_reset:
        parser.error("choose either --reset or --no_reset")
    if args.param_scale == "log" and (args.param_min <= 0 or args.param_max <= 0):
        parser.error("--param_min and --param_max must be > 0 for log scale")
    if args.analyze and not args.bundle_small and not args.no_bundle_small:
        args.bundle_small = True

    if args.suite_name:
        suite_name = args.suite_name
    else:
        timestamp = local_timestamp()
        suite_name = f"{timestamp}_{sanitize_name(args.exp_name)}_seed{args.master_seed}"
    suite_root = os.path.join(args.suite_dir, sanitize_name(suite_name))
    if os.path.exists(suite_root) and args.overwrite:
        shutil.rmtree(suite_root)
    os.makedirs(suite_root, exist_ok=True)
    suite_started_utc = utc_now()
    suite_start_time = time.time()
    if args.param_scale == "linear":
        values = np.linspace(args.param_min, args.param_max, args.param_n).tolist()
    else:
        values = np.geomspace(args.param_min, args.param_max, args.param_n).tolist()
    write_json(os.path.join(suite_root, "param_values.json"), values)

    seed_list = parse_csv_list(args.seeds)
    git_commit = detect_git_commit(os.getcwd())
    manifest_path = os.path.join(suite_root, "sweep_manifest.jsonl")
    config_sha256 = ""
    if os.path.exists(args.config):
        config_sha256 = hash_file(args.config)

    total_runs = 0
    if seed_list is not None:
        total_runs = len(values) * len(seed_list)
    else:
        total_runs = len(values) * int(args.runs)
    completed_runs = 0
    total_runtime = 0.0
    for value_idx, value in enumerate(values):
        value_label = f"value={format_value(value)}"
        group_dir = os.path.join(suite_root, args.param_name, value_label, args.exp_name)
        os.makedirs(group_dir, exist_ok=True)
        if seed_list is not None:
            seeds = [int(s) for s in seed_list]
        else:
            seeds = [
                stable_hash_seed(args.master_seed, args.exp_name, args.param_name, value, i)
                for i in range(int(args.runs))
            ]
        for run_idx, seed in enumerate(seeds):
            run_dir = os.path.join(group_dir, f"run_{run_idx:03d}")
            os.makedirs(run_dir, exist_ok=True)
            log_path = os.path.join(run_dir, "logs.jsonl")
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            stdout_path = os.path.join(run_dir, "stdout.txt")
            stderr_path = os.path.join(run_dir, "stderr.txt")
            config_arg = args.config
            override_path = ""
            if "." in args.param_name:
                override_path = os.path.join(run_dir, "config_override.yaml")
                override_dict = build_override_dict(args.param_name, value)
                dump_simple_yaml(override_dict, override_path)
                config_arg = f"{args.config},{override_path}"
            cmd = [
                sys.executable,
                "-m",
                "hops.sim",
                "--config",
                config_arg,
                "--mode",
                args.mode,
                "--exp_name",
                args.exp_name,
                "--seed",
                str(seed),
                "--run_dir",
                run_dir,
                "--bundle",
                "false",
                "--log",
                log_path,
                "--checkpoint_dir",
                checkpoint_dir,
                "--param_name",
                args.param_name,
                "--param_value",
                str(value),
                f"--{args.param_name}",
                str(value),
            ]
            if args.episodes is not None:
                cmd.extend(["--episodes", str(args.episodes)])
            if args.steps is not None:
                cmd.extend(["--steps", str(args.steps)])
            if args.reset or not args.no_reset:
                cmd.append("--reset")
            if args.log_mode is not None:
                cmd.extend(["--log_mode", args.log_mode])
            if args.log_stride is not None:
                cmd.extend(["--log_stride", str(args.log_stride)])
            if extra:
                cmd.extend(extra)
            started = utc_now()
            start_time = time.time()
            with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(
                stderr_path, "w", encoding="utf-8"
            ) as stderr_f:
                if args.stream:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                    )
                    assert proc.stdout is not None
                    assert proc.stderr is not None
                    def _drain(src, dst_file, dst_stream):
                        for line in src:
                            dst_file.write(line)
                            dst_stream.write(line)
                            dst_stream.flush()

                    t_out = Thread(target=_drain, args=(proc.stdout, stdout_f, sys.stdout), daemon=True)
                    t_err = Thread(target=_drain, args=(proc.stderr, stderr_f, sys.stderr), daemon=True)
                    t_out.start()
                    t_err.start()
                    return_code = int(proc.wait())
                    t_out.join(timeout=1.0)
                    t_err.join(timeout=1.0)
                else:
                    result = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f)
                    return_code = result.returncode
            duration = time.time() - start_time
            ended = utc_now()
            status = "ok" if return_code == 0 else "failed"
            config_hash = ""
            config_path = os.path.join(run_dir, "config_effective.json")
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    config_hash = hashlib.sha256(f.read()).hexdigest()
            row = {
                "suite_name": suite_name,
                "suite_dir": suite_root,
                "exp_name": args.exp_name,
                "mode": args.mode,
                "param_name": args.param_name,
                "param_value": value,
                "param_scale": args.param_scale,
                "param_min": args.param_min,
                "param_max": args.param_max,
                "param_n": args.param_n,
                "seed": seed,
                "run_idx": run_idx,
                "run_dir": run_dir,
                "log_path": log_path,
                "git_commit": git_commit,
                "config_path": args.config,
                "config_sha256": config_sha256,
                "config_override": override_path,
                "timestamp_start": started,
                "timestamp_end": ended,
                "duration_sec": round(duration, 3),
                "config_hash": config_hash,
                "exit_code": return_code,
                "status": status,
            }
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            completed_runs += 1
            total_runtime += duration
            if args.stream:
                avg = total_runtime / float(max(1, completed_runs))
                remaining = max(0, total_runs - completed_runs)
                eta = format_eta(avg * remaining)
                print(
                    f"sweep progress {completed_runs}/{total_runs} eta={eta}",
                    file=sys.stderr,
                )

    suite_ended_utc = utc_now()
    suite_duration = time.time() - suite_start_time
    sweep_meta = {
        "suite_name": suite_name,
        "suite_dir": suite_root,
        "started_utc": suite_started_utc,
        "ended_utc": suite_ended_utc,
        "duration_sec": round(suite_duration, 3),
        "total_runs": total_runs,
        "completed_runs": completed_runs,
    }
    write_json(os.path.join(suite_root, "sweep_meta.json"), sweep_meta)

    log_paths = sorted(glob.glob(os.path.join(suite_root, "**", "logs.jsonl"), recursive=True))
    value_map = {format_value(v): v for v in values}
    metric_rows = []
    final_s_by_episode: Dict[float, List[List[float]]] = {}
    for path in log_paths:
        param_value = parse_param_value_from_path(path, value_map)
        metrics, final_s_episode = summarize_log(path, args.tau)
        if not metrics:
            continue
        metric_rows.append(
            {
                "param_value": float(param_value),
                "final_S_mean": metrics.get("final_S_mean", float("nan")),
                "final_S_median": metrics.get("final_S_median", float("nan")),
                "final_S_last": metrics.get("final_S_last", float("nan")),
                "max_G_true_max": metrics.get("max_G_true_max", float("nan")),
                "max_G_true_mean": metrics.get("max_G_true_mean", float("nan")),
                "survival_tau_epfrac": metrics.get("survival_tau_epfrac", float("nan")),
                "time_to_tau_mean": metrics.get("time_to_tau_mean", float("nan")),
                "time_to_tau_censored_rate": metrics.get("time_to_tau_censored_rate", float("nan")),
                "final_alive_mean": metrics.get("final_alive_mean", float("nan")),
                "episodes": metrics.get("episodes", 0),
                "tau": metrics.get("tau", args.tau),
            }
        )
        final_s_by_episode.setdefault(float(param_value), []).append(final_s_episode)

    summary = metric_rows
    summary_csv = os.path.join(suite_root, "sweep_summary.csv")
    summary_json = os.path.join(suite_root, "sweep_summary.json")
    if summary:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        write_json(summary_json, summary)

    if args.analyze and log_paths:
        out_dir = os.path.join(suite_root, "analysis")
        os.makedirs(out_dir, exist_ok=True)
        analyze_cmd = [
            sys.executable,
            "-m",
            "hops.analyze",
            "--log_paths",
            *log_paths,
            "--out_dir",
            out_dir,
        ]
        subprocess.run(analyze_cmd, check=False)
        sweep_dir = os.path.join(suite_root, "analysis", "sweep_compare")
        os.makedirs(sweep_dir, exist_ok=True)
        plot_sweep(summary, sweep_dir, final_s_by_episode)
        plot_tradeoff_pareto_by_param(summary, sweep_dir, args.tau, args.param_name, args.param_scale)
    elif args.analyze and not log_paths:
        print("warning: no logs found for analysis", file=sys.stderr)

    if args.bundle_small:
        bundle_path = os.path.join(suite_root, "bundle_small.zip")
        include = []
        summary_csv = os.path.join(suite_root, "sweep_summary.csv")
        summary_json = os.path.join(suite_root, "sweep_summary.json")
        manifest = os.path.join(suite_root, "sweep_manifest.jsonl")
        include.extend([summary_csv, summary_json, manifest])
        sweep_plots = glob.glob(os.path.join(suite_root, "analysis", "sweep_compare", "*.png"))
        exp_plots = glob.glob(os.path.join(suite_root, "analysis", "exp_compare", "*.png"))
        include.extend(sorted(sweep_plots))
        include.extend(sorted(exp_plots))
        missing = []
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for path in include:
                if not add_to_zip(zipf, path, suite_root):
                    missing.append(path)
        if missing:
            print(f"bundle_small: skipped {len(missing)} missing files", file=sys.stderr)
    if args.zip:
        archive_path = suite_root + ".zip"
        if os.path.exists(archive_path):
            os.remove(archive_path)
        shutil.make_archive(suite_root, "zip", suite_root)


if __name__ == "__main__":
    main()
