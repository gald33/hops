import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from hops.registry import (
    git_info,
    utc_now_iso,
    write_package_record,
    write_registry_records,
)

from hops.analysis_contract import PLOT_REGISTRY, resolve_plot_id


def _find_repo_root(start: str) -> Optional[str]:
    current = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(current, "pyproject.toml")) or os.path.exists(
            os.path.join(current, ".git")
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _resolve_yaml_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    repo_root = _find_repo_root(os.path.dirname(__file__))
    if repo_root:
        candidate = os.path.join(repo_root, path)
        if os.path.exists(candidate):
            return candidate
    return path


def _resolve_package_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    repo_root = _find_repo_root(os.path.dirname(__file__))
    if repo_root:
        candidate = os.path.join(repo_root, path)
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(path)


def load_simple_yaml(path: str) -> Dict:
    path = _resolve_yaml_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"YAML file not found: {path} (run from repo root or use an absolute path)"
        )
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            return loaded if isinstance(loaded, dict) else {"value": loaded}
    except Exception:
        return _parse_minimal_yaml(path)


def _parse_minimal_yaml(path: str) -> Dict:
    def parse_scalar(value: str):
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip() for p in inner.split(",")]
            return [parse_scalar(p) for p in parts]
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value.strip('"').strip("'")

    data: Dict = {}
    stack: List[Tuple[int, object, Optional[str]]] = [(0, data, None)]
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            content = line.strip()
            while stack and indent < stack[-1][0]:
                stack.pop()
            if not stack:
                stack = [(0, data, None)]
            parent, pending_key = stack[-1][1], stack[-1][2]
            if pending_key and indent > stack[-1][0]:
                if content.startswith("- "):
                    new_container: object = []
                else:
                    new_container = {}
                assert isinstance(parent, dict)
                parent[pending_key] = new_container
                stack[-1] = (stack[-1][0], parent, None)
                stack.append((indent, new_container, None))
                parent, pending_key = new_container, None
            if content.startswith("- "):
                item_str = content[2:].strip()
                if not isinstance(parent, list):
                    raise ValueError(f"list item without list context: {line}")
                if not item_str:
                    item = {}
                    parent.append(item)
                    stack.append((indent + 2, item, None))
                    continue
                if ":" in item_str:
                    key, rest = item_str.split(":", 1)
                    key = key.strip()
                    val = rest.strip()
                    item = {key: parse_scalar(val)} if val else {key: None}
                    parent.append(item)
                    if val == "":
                        stack.append((indent + 2, item, key))
                    else:
                        stack.append((indent + 2, item, None))
                    continue
                parent.append(parse_scalar(item_str))
                continue
            if ":" not in content:
                raise ValueError(f"invalid line: {line}")
            key, rest = content.split(":", 1)
            key = key.strip()
            val = rest.strip()
            if not isinstance(parent, dict):
                raise ValueError(f"mapping entry without dict context: {line}")
            if val == "":
                parent[key] = None
                stack[-1] = (stack[-1][0], parent, key)
            else:
                parent[key] = parse_scalar(val)
    return data


def dump_simple_yaml(data: Dict, path: str) -> None:
    def _dump(obj, indent: int) -> List[str]:
        lines = []
        pad = " " * indent
        for key, val in obj.items():
            if isinstance(val, dict):
                lines.append(f"{pad}{key}:")
                lines.extend(_dump(val, indent + 2))
            else:
                if isinstance(val, bool):
                    val_str = "true" if val else "false"
                else:
                    val_str = str(val)
                lines.append(f"{pad}{key}: {val_str}")
        return lines

    content = "\n".join(_dump(data, 0)) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def set_cfg(cfg: Dict, path: Tuple[str, ...], value) -> None:
    node = cfg
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def set_cfg_from_dot(cfg: Dict, dotted: str, value) -> None:
    path = tuple(part.strip() for part in dotted.split(".") if part.strip())
    if not path:
        return
    set_cfg(cfg, path, value)


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_clean_dir(path: str) -> bool:
    if os.path.exists(path):
        shutil.rmtree(path)
        return True
    return False


def _coerce_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [value]


def _resolve_path(base_dir: str, path: str, fallback_dir: Optional[str] = None) -> str:
    if os.path.isabs(path):
        return path
    candidate = os.path.abspath(os.path.join(base_dir, path))
    if os.path.exists(candidate):
        return candidate
    if fallback_dir:
        fallback = os.path.abspath(os.path.join(fallback_dir, path))
        if os.path.exists(fallback):
            return fallback
    return candidate


@dataclass
class JobResult:
    job_name: str
    job_kind: str
    job_root: str
    suite_dir: str
    suite_name: str
    command: List[str]
    stdout_path: str
    stderr_path: str
    status: str
    return_code: int
    duration_sec: float
    episode_points_path: Optional[str]
    config_resolved_path: str
    config_resolved_hash: str
    warnings: List[str]
    analysis: Dict


def build_job_config(
    base_config_path: str,
    job: Dict,
    job_root: str,
    job_name: str,
    write: bool = True,
) -> Tuple[str, str]:
    cfg = load_simple_yaml(base_config_path)
    if "episodes" in job:
        set_cfg(cfg, ("training", "episodes"), int(job["episodes"]))
    if "steps" in job:
        set_cfg(cfg, ("training", "steps_per_episode"), int(job["steps"]))
    if "log_mode" in job:
        set_cfg(cfg, ("logging", "log_mode"), str(job["log_mode"]))
    if "log_stride" in job:
        set_cfg(cfg, ("logging", "log_stride"), int(job["log_stride"]))
    overrides = job.get("config_overrides", {})
    if isinstance(overrides, dict):
        for dotted, value in overrides.items():
            set_cfg_from_dot(cfg, dotted, value)
    resolved_path = os.path.join(job_root, "config_resolved.yaml")
    if write:
        dump_simple_yaml(cfg, resolved_path)
        return resolved_path, hash_file(resolved_path)
    tmp_path = os.path.join(job_root, "config_resolved.yaml")
    content = json.dumps(cfg, sort_keys=True)
    return tmp_path, hashlib.sha256(content.encode("utf-8")).hexdigest()


def run_subprocess(
    cmd: List[str],
    stdout_path: str,
    stderr_path: str,
    stream: bool,
    cwd: Optional[str],
    env: Optional[Dict[str, str]] = None,
) -> int:
    with open(stdout_path, "w", encoding="utf-8") as stdout_f, open(
        stderr_path, "w", encoding="utf-8"
    ) as stderr_f:
        if stream:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
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

            from threading import Thread

            t_out = Thread(target=_drain, args=(proc.stdout, stdout_f, sys.stdout), daemon=True)
            t_err = Thread(target=_drain, args=(proc.stderr, stderr_f, sys.stderr), daemon=True)
            t_out.start()
            t_err.start()
            code = int(proc.wait())
            t_out.join(timeout=1.0)
            t_err.join(timeout=1.0)
            return code
        result = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f, cwd=cwd, env=env)
        return result.returncode


def build_suite_cmd(job: Dict, suite_dir: str, suite_name: str, config_path: str, stream: bool) -> List[str]:
    cmd = [
        sys.executable,
        os.path.join("scripts", "run_suite.py"),
        "--suite_dir",
        suite_dir,
        "--suite_name",
        suite_name,
        "--config",
        config_path,
        "--mode",
        str(job.get("mode", "train")),
        "--exp_names",
        ",".join(_coerce_list(job.get("exp_names", ["baseline"]))),
    ]
    seeds = _coerce_list(job.get("seeds"))
    if seeds:
        cmd.extend(["--seeds", ",".join(str(s) for s in seeds)])
    elif job.get("num_runs") is not None:
        cmd.extend(["--num_runs", str(job["num_runs"])])
        cmd.extend(["--master_seed", str(job.get("master_seed", 0))])
    if job.get("episodes") is not None:
        cmd.extend(["--episodes", str(job["episodes"])])
    if job.get("steps") is not None:
        cmd.extend(["--steps", str(job["steps"])])
    if job.get("horizon_T") is not None:
        cmd.extend(["--horizon_T", str(job["horizon_T"])])
    if job.get("simulate_epsilon") is not None:
        cmd.extend(["--simulate_epsilon", str(job["simulate_epsilon"])])
    if job.get("log_mode") is not None:
        cmd.extend(["--log_mode", str(job["log_mode"])])
    if job.get("log_stride") is not None:
        cmd.extend(["--log_stride", str(job["log_stride"])])
    if job.get("reset", False):
        cmd.append("--reset")
    else:
        cmd.append("--no_reset")
    if job.get("analyze", True):
        cmd.append("--analyze")
    if job.get("zip", False):
        cmd.append("--zip")
    if stream or job.get("stream", False):
        cmd.append("--stream")
    cmd.append("--overwrite")
    extra_args = job.get("extra_args", "")
    if extra_args:
        cmd.extend(["--extra_args", str(extra_args)])
    return cmd


def build_sweep_cmd(job: Dict, suite_dir: str, suite_name: str, config_path: str, stream: bool) -> List[str]:
    cmd = [
        sys.executable,
        os.path.join("scripts", "sweep_param.py"),
        "--suite_dir",
        suite_dir,
        "--suite_name",
        suite_name,
        "--config",
        config_path,
        "--mode",
        str(job.get("mode", "train")),
        "--exp_name",
        str(job.get("exp_name", "baseline")),
        "--param_name",
        str(job["param_name"]),
        "--param_min",
        str(job["param_min"]),
        "--param_max",
        str(job["param_max"]),
        "--param_n",
        str(job["param_n"]),
        "--param_scale",
        str(job.get("param_scale", "log")),
    ]
    seeds = _coerce_list(job.get("seeds"))
    if seeds:
        cmd.extend(["--seeds", ",".join(str(s) for s in seeds)])
    elif job.get("runs") is not None:
        cmd.extend(["--runs", str(job["runs"])])
        cmd.extend(["--master_seed", str(job.get("master_seed", 0))])
    if job.get("episodes") is not None:
        cmd.extend(["--episodes", str(job["episodes"])])
    if job.get("steps") is not None:
        cmd.extend(["--steps", str(job["steps"])])
    if job.get("log_mode") is not None:
        cmd.extend(["--log_mode", str(job["log_mode"])])
    if job.get("log_stride") is not None:
        cmd.extend(["--log_stride", str(job["log_stride"])])
    if job.get("tau") is not None:
        cmd.extend(["--tau", str(job["tau"])])
    if job.get("reset", False):
        cmd.append("--reset")
    else:
        cmd.append("--no_reset")
    if job.get("analyze", True):
        cmd.append("--analyze")
    if job.get("zip", False):
        cmd.append("--zip")
    if stream or job.get("stream", False):
        cmd.append("--stream")
    cmd.append("--overwrite")
    return cmd


def parse_package_config(path: str) -> Dict:
    cfg = load_simple_yaml(path)
    if "package" not in cfg:
        cfg["package"] = {}
    return cfg


def run_package(
    package_path: str,
    stream: bool,
    zip_bundle: bool,
    dry_run: bool,
    allow_dirty: bool,
    contract_mode: Optional[str] = None,
    evaluate_claims_flag: bool = False,
    no_claims: bool = False,
) -> int:
    package_path = _resolve_package_path(package_path)
    repo_root = _find_repo_root(os.path.dirname(__file__)) or os.getcwd()
    if repo_root and os.path.isdir(repo_root):
        os.chdir(repo_root)
    base_dir = os.path.dirname(package_path)
    pkg_cfg = parse_package_config(package_path)
    package_meta = pkg_cfg.get("package", {})
    package_name = package_meta.get("name") or pkg_cfg.get("package_name")
    app_version = package_meta.get("app_version") or pkg_cfg.get("app_version")
    analysis_defaults = package_meta.get("analysis", {})
    claims_defaults = package_meta.get("claims", {})
    if contract_mode is None:
        contract_mode = "contract" if analysis_defaults else "legacy"
    if not package_name:
        raise SystemExit("package.name is required")
    if not app_version:
        raise SystemExit("package.app_version is required")
    jobs = pkg_cfg.get("jobs", [])
    if not jobs:
        raise SystemExit("package config has no jobs")
    commit, dirty = git_info(repo_root)
    if dirty and not allow_dirty:
        raise SystemExit("git worktree is dirty; re-run with --allow_dirty to proceed")
    package_dir = os.path.join(repo_root, "runs", "packages", package_name)
    os.makedirs(package_dir, exist_ok=True)
    manifest_path = os.path.join(package_dir, "package_manifest.json")
    package_started = utc_now_iso()
    package_config_hash = hash_file(package_path)
    manifest = {
        "package_name": package_name,
        "app_version": app_version,
        "package_path": package_path,
        "package_config_hash": package_config_hash,
        "created_utc": package_started,
        "git_commit": commit,
        "dirty_worktree": dirty,
        "jobs": [],
        "warnings": [],
    }
    job_results: List[JobResult] = []
    job_rows: List[Dict] = []
    package_warnings: List[str] = []
    for job in jobs:
        job_name = job.get("name")
        job_kind = job.get("kind")
        if not job_name or not job_kind:
            raise SystemExit("each job requires name and kind")
        job_root = os.path.join(package_dir, job_name)
        mode = str(job.get("mode", "train"))
        job_analysis = dict(analysis_defaults)
        job_override = job.get("analysis", {})
        if isinstance(job_override, dict):
            job_analysis.update(job_override)
        requested_reset = bool(job.get("reset", False))
        forced_reset = mode == "train"
        job_warnings: List[str] = []
        if forced_reset and not requested_reset:
            job_warnings.append("reset forced for training")
        if forced_reset:
            job["reset"] = True
        else:
            job["reset"] = requested_reset if "reset" in job else True
        deleted = False
        if not dry_run and job["reset"]:
            deleted = ensure_clean_dir(job_root)
        if deleted:
            job_warnings.append("removed existing output dir for clean reset")
        if not dry_run:
            os.makedirs(job_root, exist_ok=True)
        base_config = job.get("config", package_meta.get("config", "config.yaml"))
        base_config_path = _resolve_path(base_dir, base_config, fallback_dir=repo_root)
        config_root = os.path.join(package_dir, "_configs", job_name)
        if not dry_run:
            os.makedirs(config_root, exist_ok=True)
        resolved_config_path, resolved_hash = build_job_config(
            base_config_path,
            job,
            config_root,
            job_name,
            write=not dry_run,
        )
        suite_dir = os.path.dirname(job_root)
        suite_name = job_name
        if job_kind == "suite":
            if "analyze" not in job:
                job["analyze"] = True
            cmd = build_suite_cmd(job, suite_dir, suite_name, resolved_config_path, stream)
        elif job_kind == "sweep":
            required = ["param_name", "param_min", "param_max", "param_n"]
            missing = [name for name in required if name not in job]
            if missing:
                raise SystemExit(f"{job_name}: missing required sweep fields: {', '.join(missing)}")
            if "analyze" not in job:
                job["analyze"] = True
            cmd = build_sweep_cmd(job, suite_dir, suite_name, resolved_config_path, stream)
        else:
            raise SystemExit(f"unknown job kind: {job_kind}")
        stdout_path = os.path.join(job_root, "package_stdout.txt")
        stderr_path = os.path.join(job_root, "package_stderr.txt")
        job_started = time.time()
        return_code = 0
        status = "skipped"
        if dry_run:
            print("dry_run:", " ".join(cmd))
            status = "dry_run"
        else:
            env = os.environ.copy()
            src_path = os.path.join(repo_root, "src")
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
            return_code = run_subprocess(cmd, stdout_path, stderr_path, stream, cwd=repo_root, env=env)
            status = "ok" if return_code == 0 else "failed"
        duration = time.time() - job_started
        episode_points_path = None
        if not dry_run:
            from hops.package_analyze import find_episode_points, read_episode_points

            episode_points_path = find_episode_points(job_root)
            if episode_points_path:
                rows = read_episode_points(episode_points_path)
                for row in rows:
                    row["package_name"] = package_name
                    row["job_name"] = job_name
                    job_rows.append(row)
        job_results.append(
            JobResult(
                job_name=job_name,
                job_kind=job_kind,
                job_root=job_root,
                suite_dir=suite_dir,
                suite_name=suite_name,
                command=cmd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                status=status,
                return_code=return_code,
                duration_sec=round(duration, 3),
                episode_points_path=episode_points_path,
                config_resolved_path=resolved_config_path,
                config_resolved_hash=resolved_hash,
                warnings=list(job_warnings),
                analysis=job_analysis,
            )
        )
        package_warnings.extend([f"{job_name}: {w}" for w in job_warnings])
        manifest["jobs"].append(
            {
                "job_name": job_name,
                "kind": job_kind,
                "mode": mode,
                "suite_dir": suite_dir,
                "suite_name": suite_name,
                "job_root": job_root,
                "command": cmd,
                "status": status,
                "return_code": return_code,
                "duration_sec": round(duration, 3),
                "episode_points_path": episode_points_path,
                "config_resolved_path": resolved_config_path,
                "config_resolved_hash": resolved_hash,
                "warnings": list(job_warnings),
                "analysis": job_analysis,
            }
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        if status != "ok" and not dry_run:
            break
    manifest["ended_utc"] = utc_now_iso()
    manifest["warnings"].extend(package_warnings)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    if not dry_run:
        output_dir = analysis_defaults.get("output_dir")
        if output_dir:
            plots_dir = output_dir if os.path.isabs(output_dir) else os.path.join(package_dir, output_dir)
        else:
            plots_dir = os.path.join(package_dir, "analysis", "package_compare")
        from hops.package_analyze import (
            build_analysis_manifest,
            update_parquet_registry,
            write_package_outputs,
        )

        outputs = {"produced": {}}
        analysis_manifest = {}
        analysis_error = None
        try:
            always_include = analysis_defaults.get("always_include", [])
            plot_includes = [
                p
                for p in always_include
                if p in PLOT_REGISTRY or resolve_plot_id(p) in PLOT_REGISTRY
            ]
            requested_plots = analysis_defaults.get("plots") or []
            effective_plots = list(dict.fromkeys(list(requested_plots) + plot_includes))
            outputs = write_package_outputs(
                package_dir,
                package_name,
                job_rows,
                plots_dir,
                zip_bundle,
                requested_plots=effective_plots,
                requested_tables=analysis_defaults.get("tables"),
                contract_mode=contract_mode or "legacy",
            )
            manifest["analysis_outputs"] = outputs
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
        except Exception as exc:
            analysis_error = exc
            outputs = {"produced": {}, "error": str(exc)}
        finally:
            analysis_manifest = build_analysis_manifest(
                package_dir=package_dir,
                package_name=package_name,
                app_version=app_version,
                config_hash=package_config_hash,
                git_commit=commit,
                is_dirty=dirty,
                analysis_config=analysis_defaults,
                job_results=[job.__dict__ for job in job_results],
                package_outputs=outputs,
            )
        if analysis_error:
            raise analysis_error
        if analysis_manifest.get("exit_status"):
            errors = analysis_manifest.get("errors", [])
            summary = errors[0] if errors else {"reason": "unknown"}
            raise SystemExit(2)
        if job_rows:
            parquet_path = os.path.join(repo_root, "runs", "_registry", "episode_points.parquet")
            warn = update_parquet_registry(job_rows, parquet_path)
            if warn:
                manifest.setdefault("warnings", []).append(warn)
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, sort_keys=True)
        claims_manifest = None
        should_run_claims = (bool(claims_defaults) or evaluate_claims_flag) and not no_claims
        if should_run_claims:
            from hops.claims_eval import evaluate_claims
            from hops.claims_schema import ClaimsSchemaError, load_claims_block

            claims_config = load_claims_block(claims_defaults, base_dir)
            job_claims: List[Dict] = []
            for job in jobs:
                job_claims_block = job.get("claims")
                if not job_claims_block:
                    continue
                job_cfg = load_claims_block(job_claims_block, base_dir)
                for claim in job_cfg.get("claims", []):
                    if claim.get("scope") and claim.get("scope") != job.get("kind"):
                        raise SystemExit(
                            f"job {job.get('name')} claim {claim.get('id')} scope must match job kind {job.get('kind')}"
                        )
                    data_block = claim.get("data", {}) or {}
                    data_block.setdefault("job_name", job.get("name"))
                    claim["data"] = data_block
                    job_claims.append(claim)
            claims_config["claims"].extend(job_claims)
            try:
                claims_manifest = evaluate_claims(
                    package_dir=package_dir,
                    package_name=package_name,
                    app_version=app_version,
                    config_hash=package_config_hash,
                    git_commit=commit or "",
                    is_dirty=dirty,
                    claims_config=claims_config,
                    job_results=[job.__dict__ for job in job_results],
                )
            except ClaimsSchemaError as exc:
                raise SystemExit(str(exc))
            manifest["claims_outputs"] = {
                "output_dir": claims_manifest.get("output_dir") if claims_manifest else None
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
            if claims_manifest and claims_manifest.get("exit_status"):
                raise SystemExit(2)
        if zip_bundle:
            bundle_path = os.path.join(package_dir, "package_bundle_small.zip")
            if os.path.exists(bundle_path):
                import zipfile

                extra_paths = [
                    os.path.join(package_dir, "analysis", "analysis_manifest.json"),
                    os.path.join(package_dir, "analysis", "claims", "claims_manifest.json"),
                    os.path.join(package_dir, "analysis", "claims", "claims_report.md"),
                ]
                with zipfile.ZipFile(bundle_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
                    for path in extra_paths:
                        if os.path.exists(path):
                            zf.write(path, arcname=os.path.relpath(path, package_dir))
    if not dry_run:
        run_records = build_run_records(job_results, job_rows, package_name, app_version, commit, dirty)
        write_registry_records(run_records)
        package_record = {
            "package_name": package_name,
            "app_version": app_version,
            "package_dir": package_dir,
            "package_path": package_path,
            "git_commit": commit,
            "dirty_worktree": dirty,
            "started_utc": package_started,
            "ended_utc": manifest.get("ended_utc", utc_now_iso()),
            "jobs": len(job_results),
        }
        write_package_record(package_record)
    return 0


def build_run_records(
    job_results: List[JobResult],
    job_rows: List[Dict],
    package_name: str,
    app_version: str,
    commit: str,
    dirty: Optional[bool],
) -> List[Dict]:
    metrics_by_key = build_metrics_by_key(job_rows)
    records = []
    for job in job_results:
        if job.status not in {"ok", "failed"}:
            continue
        resolved_cfg = {}
        if job.config_resolved_path and os.path.exists(job.config_resolved_path):
            resolved_cfg = load_simple_yaml(job.config_resolved_path)
        cfg_episodes = None
        cfg_steps = None
        training_cfg = resolved_cfg.get("training", {}) if isinstance(resolved_cfg, dict) else {}
        if isinstance(training_cfg, dict):
            cfg_episodes = training_cfg.get("episodes")
            cfg_steps = training_cfg.get("steps_per_episode")
        if job.job_kind == "suite":
            manifest_path = os.path.join(job.job_root, "manifest.json")
            if not os.path.exists(manifest_path):
                continue
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            common = manifest.get("common_args", {})
            for run in manifest.get("runs", []):
                exp_name = run.get("exp_name", "")
                seed = run.get("seed")
                metrics = metrics_by_key.get((job.job_name, exp_name, seed, ""))
                records.append(
                    {
                        "run_id": str(uuid.uuid4()),
                        "timestamp": run.get("ended_utc", utc_now_iso()),
                        "package_name": package_name,
                        "job_name": job.job_name,
                        "job_kind": job.job_kind,
                        "exp_name": exp_name,
                        "app_version": app_version,
                        "git_commit": commit,
                        "dirty_worktree": dirty,
                        "seed": seed,
                        "episodes": common.get("episodes", cfg_episodes),
                        "steps": common.get("steps", cfg_steps),
                        "config_resolved_hash": job.config_resolved_hash,
                        "config_resolved_path": job.config_resolved_path,
                        "run_dir": run.get("run_dir"),
                        "log_path": run.get("log_path"),
                        "analysis_dir": os.path.join(job.job_root, "analysis"),
                        "bundle_path": job.job_root + ".zip",
                        "status": run.get("status"),
                        "return_code": run.get("return_code"),
                        "metrics": metrics or {},
                    }
                )
        elif job.job_kind == "sweep":
            manifest_path = os.path.join(job.job_root, "sweep_manifest.jsonl")
            if not os.path.exists(manifest_path):
                continue
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    exp_name = row.get("exp_name", "")
                    seed = row.get("seed")
                    param_value = row.get("param_value")
                    metrics = metrics_by_key.get((job.job_name, exp_name, seed, str(param_value)))
                    config_override = row.get("config_override", "")
                    config_override_hash = hash_file(config_override) if config_override and os.path.exists(config_override) else ""
                    records.append(
                        {
                            "run_id": str(uuid.uuid4()),
                            "timestamp": row.get("timestamp_end", utc_now_iso()),
                            "package_name": package_name,
                            "job_name": job.job_name,
                            "job_kind": job.job_kind,
                            "exp_name": exp_name,
                            "app_version": app_version,
                            "git_commit": commit,
                            "dirty_worktree": dirty,
                            "seed": seed,
                            "episodes": row.get("episodes", cfg_episodes),
                            "steps": row.get("steps", cfg_steps),
                            "param_name": row.get("param_name"),
                            "param_value": param_value,
                            "config_resolved_hash": job.config_resolved_hash,
                            "config_resolved_path": job.config_resolved_path,
                            "config_override_path": config_override,
                            "config_override_hash": config_override_hash,
                            "run_dir": row.get("run_dir"),
                            "log_path": row.get("log_path"),
                            "analysis_dir": os.path.join(job.job_root, "analysis"),
                            "bundle_path": job.job_root + ".zip",
                            "status": row.get("status"),
                            "return_code": row.get("exit_code"),
                            "metrics": metrics or {},
                        }
                    )
    return records


def build_metrics_by_key(job_rows: List[Dict]) -> Dict[Tuple[str, str, Optional[int], str], Dict]:
    groups = defaultdict(list)
    for row in job_rows:
        job_name = row.get("job_name", "")
        exp_name = row.get("exp_name", "")
        seed = row.get("seed", None)
        param_value = row.get("param_value", "")
        key = (job_name, exp_name, seed, str(param_value))
        groups[key].append(row)
    metrics = {}
    for key, items in groups.items():
        welfare = [_safe_float(r.get("final_welfare")) for r in items]
        welfare = [v for v in welfare if math.isfinite(v)]
        risk = [_safe_float(r.get("max_gap_true_ratio")) for r in items]
        risk = [v for v in risk if math.isfinite(v)]
        t_tau = [_safe_float(r.get("time_to_tau")) for r in items]
        t_tau = [v for v in t_tau if math.isfinite(v)]
        censored = [str(r.get("censored", "")).lower() in {"1", "true", "t"} for r in items]
        metrics[key] = {
            "final_welfare_mean": float(sum(welfare) / len(welfare)) if welfare else float("nan"),
            "max_gap_true_ratio_max": max(risk) if risk else float("nan"),
            "time_to_tau_mean": float(sum(t_tau) / len(t_tau)) if t_tau else float("nan"),
            "censored_rate": float(sum(censored) / len(censored)) if censored else float("nan"),
        }
    return metrics


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        if value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=False)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--zip", action="store_true", default=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_dirty", action="store_true")
    parser.add_argument("--list_plots", action="store_true")
    parser.add_argument("--list_tables", action="store_true")
    parser.add_argument("--list_claims", action="store_true")
    parser.add_argument("--contract_mode", choices=["legacy", "contract"])
    parser.add_argument("--evaluate_claims", action="store_true")
    parser.add_argument("--no_claims", action="store_true")
    args = parser.parse_args()
    if args.list_plots:
        from hops.analysis_contract import describe_plot_registry

        info = describe_plot_registry()
        print("plot_id | filename | scopes | description")
        for plot_id, entry in info.items():
            print(
                f"{plot_id} | {entry.get('filename','')} | {entry.get('scopes','')} | {entry.get('description','')}"
            )
        return
    if args.list_tables:
        from hops.analysis_contract import describe_table_registry

        info = describe_table_registry()
        print("table_id | filename | scopes | description")
        for table_id, entry in info.items():
            print(
                f"{table_id} | {entry.get('filename','')} | {entry.get('scopes','')} | {entry.get('description','')}"
            )
        return
    if args.list_claims:
        from hops.claims_contract import describe_claim_registry

        info = describe_claim_registry()
        print("claim_id | scopes | description")
        for claim_id, entry in info.items():
            print(f"{claim_id} | {entry.get('scopes','')} | {entry.get('description','')}")
        return
    if not args.package:
        raise SystemExit("--package is required unless a --list_* flag is used")
    raise SystemExit(
        run_package(
            package_path=args.package,
            stream=args.stream,
            zip_bundle=args.zip,
            dry_run=args.dry_run,
            allow_dirty=args.allow_dirty,
            contract_mode=args.contract_mode,
            evaluate_claims_flag=args.evaluate_claims,
            no_claims=args.no_claims,
        )
    )
