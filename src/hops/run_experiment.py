import argparse
import datetime
import os
import shutil
import subprocess
import sys
import uuid


def sanitize_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value.strip())
    return cleaned or "run"


def auto_run_dir(exp_name: str, seed: int) -> str:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    shortid = uuid.uuid4().hex[:8]
    safe_name = sanitize_name(exp_name)
    return os.path.join("runs", f"{timestamp}_{safe_name}_seed{seed}_{shortid}")


def run_cmd(cmd):
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no_reset", action="store_true")
    parser.add_argument("--simulate_epsilon", type=float, default=0.0)
    parser.add_argument("--no_simulate", action="store_true")
    parser.add_argument("--no_analyze", action="store_true")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir or auto_run_dir(args.exp_name, args.seed)
    train_cmd = [
        sys.executable,
        "-m",
        "hops.sim",
        "--mode",
        "train",
        "--exp_name",
        args.exp_name,
        "--seed",
        str(args.seed),
        "--run_dir",
        run_dir,
        "--bundle",
        "true",
    ]
    if args.reset and args.no_reset:
        parser.error("choose either --reset or --no_reset")
    if args.reset or not args.no_reset:
        train_cmd.append("--reset")
    run_cmd(train_cmd)

    if not args.no_simulate:
        sim_log = os.path.join(run_dir, "logs", "simulate.jsonl")
        sim_cmd = [
            sys.executable,
            "-m",
            "hops.sim",
            "--mode",
            "simulate",
            "--exp_name",
            args.exp_name,
            "--seed",
            str(args.seed),
            "--run_dir",
            run_dir,
            "--bundle",
            "true",
            "--simulate_epsilon",
            str(args.simulate_epsilon),
            "--log",
            sim_log,
        ]
        run_cmd(sim_cmd)

    if not args.no_analyze:
        analyze_cmd = [
            sys.executable,
            "-m",
            "hops.analyze",
            "--run_dir",
            run_dir,
        ]
        run_cmd(analyze_cmd)
    if args.zip:
        shutil.make_archive(run_dir, "zip", run_dir)


if __name__ == "__main__":
    main()
