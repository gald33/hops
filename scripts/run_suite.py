import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time

from threading import Thread
from typing import List
from tqdm import tqdm

def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value.strip())
    return cleaned or "run"


def parse_csv(value: str, cast=str):
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [cast(p) for p in parts]


def stable_hash_seed(master_seed: int, exp_name: str, idx: int) -> int:
    payload = f"{master_seed}|{exp_name}|{idx}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % (2**31)


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


def available_experiments() -> List[str]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from hops.sim import EXPERIMENTS
    except Exception:
        return []
    return sorted(EXPERIMENTS.keys())


def ensure_clean_suite_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        if not overwrite:
            raise SystemExit(f"suite_dir already exists: {path} (use --overwrite to replace)")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def flag_conflict(extra_args, flags) -> str:
    for flag in flags:
        if any(arg == flag or arg.startswith(flag + "=") for arg in extra_args):
            return flag
    return ""


def write_manifest(path: str, payload: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def write_runs_summary_csv(path: str, rows: list) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_sim_cmd(args, exp_name: str, seed: int, run_dir: str, log_path: str, checkpoint_dir: str):
    cmd = [
        sys.executable,
        "-m",
        "hops.sim",
        "--config",
        args.config,
        "--mode",
        args.mode,
        "--exp_name",
        exp_name,
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
    ]
    if args.episodes is not None:
        cmd.extend(["--episodes", str(args.episodes)])
    if args.steps is not None:
        cmd.extend(["--steps", str(args.steps)])
    if args.horizon_T is not None:
        cmd.extend(["--horizon_T", str(args.horizon_T)])
    if args.mode == "simulate" and args.simulate_epsilon is not None:
        cmd.extend(["--simulate_epsilon", str(args.simulate_epsilon)])
    if args.log_mode is not None:
        cmd.extend(["--log_mode", args.log_mode])
    if args.log_stride is not None:
        cmd.extend(["--log_stride", str(args.log_stride)])
    if args.reset or not args.no_reset:
        cmd.append("--reset")
    if args.extra_args_list:
        cmd.extend(args.extra_args_list)
    if args.no_progress:
        cmd.append("--no_progress")
    elif args.progress:
        cmd.extend(["--progress", "true"])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_dir", type=str, default=os.path.join("runs", "suites"))
    parser.add_argument("--suite_name", type=str, default=None)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", choices=["train", "simulate"], default="train")
    parser.add_argument("--exp_names", type=str, default="baseline")
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--master_seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--horizon_T", type=int, default=None)
    parser.add_argument("--simulate_epsilon", type=float, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no_reset", action="store_true")
    parser.add_argument("--log_mode", choices=["full", "metrics_only", "none"], default=None)
    parser.add_argument("--log_stride", type=int, default=None)
    parser.add_argument("--extra_args", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--analyze_window", type=int, default=10)
    parser.add_argument("--progress", type=str, default=None)
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    if args.reset and args.no_reset:
        parser.error("choose either --reset or --no_reset")
    if args.seeds and args.num_runs is not None:
        parser.error("choose either --seeds or --num_runs")
    no_progress_env = os.getenv("HOPS_NO_PROGRESS", "").strip().lower()
    env_disables_progress = no_progress_env in {"1", "true", "t", "yes", "y"}
    if args.no_progress:
        args.progress = False
    elif args.progress is None:
        args.progress = sys.stderr.isatty() and not env_disables_progress
    else:
        args.progress = args.progress.strip().lower() not in {"0", "false", "f", "no", "n"}

    exp_names = parse_csv(args.exp_names, str) or ["baseline"]
    exp_names = [name.strip() for name in exp_names if name.strip()]
    if not exp_names:
        parser.error("exp_names must contain at least one name")
    known_experiments = available_experiments()
    if not known_experiments:
        parser.error("could not load experiment list for validation (run from repo root or install the package)")
    unknown = [name for name in exp_names if name not in known_experiments]
    if unknown:
        known_list = ", ".join(known_experiments)
        bad_list = ", ".join(unknown)
        parser.error(f"unknown exp_name(s): {bad_list}. available: {known_list}")

    if args.seeds:
        seeds = parse_csv(args.seeds, int)
    else:
        if args.num_runs is None:
            parser.error("provide --seeds or --num_runs with --master_seed")
        seeds = None

    args.extra_args_list = shlex.split(args.extra_args) if args.extra_args else []
    conflict = flag_conflict(
        args.extra_args_list,
        {
            "--config",
            "--log",
            "--checkpoint_dir",
            "--exp_name",
            "--seed",
            "--run_dir",
            "--bundle",
            "--mode",
            "--reset",
            "--no_reset",
            "--analyze",
            "--analyze_window",
            "--progress",
            "--no_progress",
        },
    )
    if conflict:
        parser.error(f"--extra_args must not include {conflict} (set it via suite runner flags)")

    if args.suite_name:
        suite_name = args.suite_name
    else:
        timestamp = local_timestamp()
        exp_joined = sanitize_name("-".join(exp_names))
        suite_name = f"{timestamp}_{exp_joined}_seed{args.master_seed}"
    suite_name = sanitize_name(suite_name)
    final_suite_dir = os.path.join(args.suite_dir, suite_name)
    ensure_clean_suite_dir(final_suite_dir, args.overwrite)

    manifest_path = os.path.join(final_suite_dir, "manifest.json")
    suite_started_utc = utc_now()
    suite_start_time = time.time()
    manifest = {
        "suite_name": suite_name,
        "created_utc": suite_started_utc,
        "mode": args.mode,
        "git_commit": detect_git_commit(os.getcwd()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "command_line": sys.argv,
        "common_args": {
            "config": args.config,
            "config_sha256": hash_file(args.config) if os.path.exists(args.config) else "",
            "episodes": args.episodes,
            "steps": args.steps,
            "horizon_T": args.horizon_T,
            "simulate_epsilon": args.simulate_epsilon,
            "log_mode": args.log_mode,
            "log_stride": args.log_stride,
            "reset": args.reset,
            "no_reset": args.no_reset,
            "extra_args": args.extra_args_list,
            "zip": args.zip,
            "analyze": args.analyze,
            "analyze_window": args.analyze_window,
        },
        "runs": [],
    }
    write_manifest(manifest_path, manifest)

    successes = 0
    failures = 0
    aborted = False
    total_runs = 0
    if seeds is not None:
        total_runs = len(exp_names) * len(seeds)
    else:
        total_runs = len(exp_names) * int(args.num_runs)
    suite_bar = tqdm(
        total=total_runs,
        desc=f"suite {suite_name}",
        unit="run",
        disable=not args.progress,
        file=sys.stderr,
        dynamic_ncols=True,
    )
    for exp_name in exp_names:
        safe_exp = sanitize_name(exp_name)
        exp_dir = os.path.join(final_suite_dir, safe_exp)
        os.makedirs(exp_dir, exist_ok=True)
        if seeds is None:
            exp_seeds = [stable_hash_seed(args.master_seed, exp_name, i) for i in range(args.num_runs)]
        else:
            exp_seeds = list(seeds)
        for run_idx, seed in enumerate(exp_seeds):
            run_dir = os.path.join(exp_dir, f"run_{run_idx:03d}")
            os.makedirs(run_dir, exist_ok=True)
            log_path = os.path.join(run_dir, "logs.jsonl")
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            stdout_path = os.path.join(run_dir, "stdout.txt")
            stderr_path = os.path.join(run_dir, "stderr.txt")
            started = utc_now()
            start_time = time.time()
            cmd = build_sim_cmd(args, exp_name, seed, run_dir, log_path, checkpoint_dir)
            return_code = 0
            status = "ok"
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
                if return_code != 0:
                    status = "failed"
                    aborted = True
            duration = time.time() - start_time
            ended = utc_now()
            if status == "ok":
                successes += 1
            else:
                failures += 1
            if suite_bar is not None:
                suite_bar.update(1)
                suite_bar.set_postfix_str(
                    f"{exp_name} run={run_idx} seed={seed} status={status}"
                )

            config_hash = ""
            config_path = os.path.join(run_dir, "config_effective.json")
            if os.path.exists(config_path):
                with open(config_path, "rb") as f:
                    config_hash = hashlib.sha256(f.read()).hexdigest()

            run_record = {
                "exp_name": exp_name,
                "run_idx": run_idx,
                "seed": int(seed),
                "run_dir": os.path.relpath(run_dir, final_suite_dir),
                "log_path": os.path.relpath(log_path, final_suite_dir),
                "checkpoint_dir": os.path.relpath(checkpoint_dir, final_suite_dir),
                "status": status,
                "return_code": return_code,
                "started_utc": started,
                "ended_utc": ended,
                "duration_sec": round(duration, 3),
            }
            if config_hash:
                run_record["config_hash"] = config_hash
            manifest["runs"].append(run_record)
            write_manifest(manifest_path, manifest)
            if args.analyze and os.path.exists(log_path):
                reports_dir = os.path.join(run_dir, "reports")
                os.makedirs(reports_dir, exist_ok=True)
                analyze_cmd = [
                    sys.executable,
                    "-m",
                    "hops.analyze",
                    "--logs",
                    log_path,
                    "--window",
                    str(args.analyze_window),
                    "--out_dir",
                    reports_dir,
                ]
                subprocess.run(analyze_cmd, check=False)
            if aborted:
                break
        if aborted:
            break

    if suite_bar is not None:
        suite_bar.close()
    if args.analyze:
        suite_cmd = [
            sys.executable,
            "-m",
            "hops.analyze",
            "--suite_dir",
            final_suite_dir,
            "--window",
            str(args.analyze_window),
        ]
        subprocess.run(suite_cmd, check=False)

    suite_ended_utc = utc_now()
    suite_duration = time.time() - suite_start_time
    manifest["ended_utc"] = suite_ended_utc
    manifest["duration_sec"] = round(suite_duration, 3)
    manifest["aborted_on_failure"] = aborted
    write_manifest(manifest_path, manifest)

    summary_rows = []
    for run in manifest.get("runs", []):
        summary_rows.append(
            {
                "exp_name": run.get("exp_name"),
                "run_idx": run.get("run_idx"),
                "seed": run.get("seed"),
                "status": run.get("status"),
                "return_code": run.get("return_code"),
                "duration_sec": run.get("duration_sec"),
                "started_utc": run.get("started_utc"),
                "ended_utc": run.get("ended_utc"),
                "run_dir": run.get("run_dir"),
                "log_path": run.get("log_path"),
            }
        )
    summary_csv_path = os.path.join(final_suite_dir, "suite_summary.csv")
    summary_json_path = os.path.join(final_suite_dir, "suite_summary.json")
    write_runs_summary_csv(summary_csv_path, summary_rows)
    write_manifest(
        summary_json_path,
        {
            "suite_name": suite_name,
            "suite_dir": final_suite_dir,
            "started_utc": suite_started_utc,
            "ended_utc": suite_ended_utc,
            "duration_sec": round(suite_duration, 3),
            "successes": successes,
            "failures": failures,
            "aborted_on_failure": aborted,
            "runs": summary_rows,
        },
    )

    print(f"suite complete: {successes} ok, {failures} failed")
    print(f"suite_dir: {final_suite_dir}")
    if aborted:
        raise SystemExit("suite aborted after first failure")
    if args.zip:
        archive_path = final_suite_dir + ".zip"
        if os.path.exists(archive_path):
            if args.overwrite:
                os.remove(archive_path)
            else:
                raise SystemExit(f"zip already exists: {archive_path} (use --overwrite to replace)")
        shutil.make_archive(final_suite_dir, "zip", final_suite_dir)
        print(f"zip: {archive_path}")


if __name__ == "__main__":
    main()
