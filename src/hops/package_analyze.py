import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hops.analysis_contract import PLOT_REGISTRY, TABLE_REGISTRY, resolve_plot_id
from hops.registry import utc_now_iso


def find_episode_points(job_root: str) -> Optional[str]:
    preferred = [
        os.path.join(job_root, "analysis", "exp_compare", "episode_points.csv"),
        os.path.join(job_root, "analysis", "sweep_compare", "episode_points.csv"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path
    matches = []
    for root, _dirs, files in os.walk(job_root):
        if "episode_points.csv" in files:
            matches.append(os.path.join(root, "episode_points.csv"))
    if not matches:
        return None
    matches.sort()
    return matches[0]


def _coerce_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        if value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    val = str(value).strip().lower()
    return val in {"1", "true", "t", "yes", "y"}


def read_episode_points(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_groups(rows: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (
            row.get("job_name", ""),
            row.get("exp_name", ""),
            row.get("param_name", ""),
            row.get("param_value", ""),
        )
        groups[key].append(row)
    summaries = []
    for (job_name, exp_name, param_name, param_value), items in groups.items():
        welfare_vals = [_coerce_float(r.get("final_welfare")) for r in items]
        welfare_vals = [v for v in welfare_vals if math.isfinite(v)]
        risk_vals = [_coerce_float(r.get("max_gap_true_ratio")) for r in items]
        risk_vals = [v for v in risk_vals if math.isfinite(v)]
        obs_vals = [_coerce_float(r.get("max_gap_observed_ratio")) for r in items]
        obs_vals = [v for v in obs_vals if math.isfinite(v)]
        t_tau_vals = [_coerce_float(r.get("time_to_tau")) for r in items]
        t_tau_vals = [v for v in t_tau_vals if math.isfinite(v)]
        divergence_vals = [_coerce_float(r.get("divergence")) for r in items]
        divergence_vals = [v for v in divergence_vals if math.isfinite(v)]
        censored = [_coerce_bool(r.get("censored")) for r in items]
        censored_rate = float(np.mean(censored)) if censored else float("nan")
        summaries.append(
            {
                "job_name": job_name,
                "exp_name": exp_name,
                "param_name": param_name or "",
                "param_value": param_value,
                "episodes": len(items),
                "final_welfare_mean": float(np.mean(welfare_vals)) if welfare_vals else float("nan"),
                "final_welfare_median": float(np.median(welfare_vals)) if welfare_vals else float("nan"),
                "max_gap_true_ratio_max": float(np.max(risk_vals)) if risk_vals else float("nan"),
                "max_gap_observed_ratio_max": float(np.max(obs_vals)) if obs_vals else float("nan"),
                "time_to_tau_mean": float(np.mean(t_tau_vals)) if t_tau_vals else float("nan"),
                "censored_rate": censored_rate,
                "divergence_mean": float(np.mean(divergence_vals)) if divergence_vals else float("nan"),
                "survival_tau_epfrac": 1.0 - censored_rate if math.isfinite(censored_rate) else float("nan"),
            }
        )
    return summaries


def _group_by_key(rows: List[Dict], key: str) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return grouped


def plot_tradeoff_scatter(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    exp_names = sorted({r.get("exp_name", "") for r in rows})
    job_names = sorted({r.get("job_name", "") for r in rows})
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    plt.figure()
    for exp_idx, exp_name in enumerate(exp_names):
        for job_idx, job_name in enumerate(job_names):
            subset = [r for r in rows if r.get("exp_name") == exp_name and r.get("job_name") == job_name]
            if not subset:
                continue
            xs = [_coerce_float(r.get("max_gap_true_ratio")) for r in subset]
            ys = [_coerce_float(r.get("final_welfare")) for r in subset]
            xs = [x for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
            ys = [y for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
            if not xs:
                continue
            color = colors[exp_idx % len(colors)] if colors else None
            marker = markers[job_idx % len(markers)]
            label = f"{exp_name} / {job_name}"
            plt.scatter(xs, ys, alpha=0.45, s=22, marker=marker, color=color, edgecolors="none", label=label)
    plt.title("Package tradeoff scatter (episode-level)")
    plt.xlabel("max_gap_true_ratio (episode)")
    plt.ylabel("final_welfare (log1p S_eff)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    pts = sorted(points, key=lambda item: item[0])
    frontier = []
    best_y = -float("inf")
    for x, y in pts:
        if y > best_y + 1e-12:
            frontier.append((x, y))
            best_y = y
    return frontier


def plot_package_frontiers(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    grouped = defaultdict(list)
    for row in rows:
        job_name = row.get("job_name", "")
        exp_name = row.get("exp_name", "")
        risk = _coerce_float(row.get("max_gap_true_ratio"))
        welfare = _coerce_float(row.get("final_welfare"))
        if not (math.isfinite(risk) and math.isfinite(welfare)):
            continue
        grouped[(job_name, exp_name)].append((risk, welfare))
    if not grouped:
        return
    plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, ((job_name, exp_name), points) in enumerate(sorted(grouped.items())):
        frontier = pareto_frontier(points)
        if not frontier:
            continue
        xs = [p[0] for p in frontier]
        ys = [p[1] for p in frontier]
        color = colors[idx % len(colors)] if colors else None
        plt.plot(xs, ys, linewidth=2.0, color=color, label=f"{exp_name} / {job_name}")
    plt.title("Package Pareto frontiers (min risk, max welfare)")
    plt.xlabel("max_gap_true_ratio (episode)")
    plt.ylabel("final_welfare (log1p S_eff)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_tradeoff_by_param(rows: List[Dict], path: str) -> None:
    params = [r for r in rows if r.get("param_value") not in (None, "")]
    if not params:
        return
    grouped = defaultdict(list)
    for row in params:
        val = row.get("param_value", "")
        grouped[str(val)].append(row)
    xs = []
    ys = []
    labels = []
    def _sort_key(item):
        val = _coerce_float(item[0])
        return (0 if math.isfinite(val) else 1, val)

    for val, items in sorted(grouped.items(), key=_sort_key):
        risk_vals = [_coerce_float(r.get("max_gap_true_ratio")) for r in items]
        welfare_vals = [_coerce_float(r.get("final_welfare")) for r in items]
        risk_vals = [v for v in risk_vals if math.isfinite(v)]
        welfare_vals = [v for v in welfare_vals if math.isfinite(v)]
        if not risk_vals or not welfare_vals:
            continue
        xs.append(float(np.max(risk_vals)))
        ys.append(float(np.mean(welfare_vals)))
        labels.append(val)
    if not xs:
        return
    plt.figure()
    plt.scatter(xs, ys, s=40, color="tab:blue")
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(f"{float(label):.3g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    frontier = pareto_frontier(list(zip(xs, ys)))
    if frontier:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        plt.plot(fx, fy, linewidth=2.0, color="black")
    plt.title("Tradeoff by parameter (aggregated per value)")
    plt.xlabel("max_gap_true_ratio_max")
    plt.ylabel("final_welfare_mean")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def update_parquet_registry(rows: List[Dict], path: str) -> Optional[str]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return "pyarrow not available; skipping parquet update"
    table = pa.Table.from_pylist(rows)
    if os.path.exists(path):
        existing = pq.read_table(path)
        table = pa.concat_tables([existing, table], promote=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)
    return None


def write_package_outputs(
    package_dir: str,
    package_name: str,
    job_rows: List[Dict],
    plots_dir: str,
    zip_bundle: bool,
    requested_plots: Optional[List[str]] = None,
    requested_tables: Optional[List[str]] = None,
    contract_mode: str = "legacy",
) -> Dict[str, str]:
    os.makedirs(plots_dir, exist_ok=True)
    combined_path = os.path.join(package_dir, "combined_episode_points.csv")
    summary_path = os.path.join(package_dir, "package_summary.csv")
    summary_json_path = os.path.join(package_dir, "package_summary.json")
    base_fields = [
        "package_name",
        "job_name",
        "exp_name",
        "run_id",
        "seed",
        "episode_idx",
        "param_name",
        "param_value",
    ]
    extra_fields = sorted({k for row in job_rows for k in row.keys() if k not in base_fields})
    combined_fields = base_fields + extra_fields
    produced = {}
    tables = set(requested_tables or ["combined_episode_points_csv", "package_summary_csv", "package_summary_json"])
    write_csv(combined_path, job_rows, combined_fields)
    produced["combined_episode_points_csv"] = combined_path
    summary_rows = summarize_groups(job_rows)
    summary_fields = [
        "job_name",
        "exp_name",
        "param_name",
        "param_value",
        "episodes",
        "final_welfare_mean",
        "final_welfare_median",
        "max_gap_true_ratio_max",
        "max_gap_observed_ratio_max",
        "time_to_tau_mean",
        "censored_rate",
        "divergence_mean",
        "survival_tau_epfrac",
    ]
    rows = summary_rows if summary_rows else []
    write_csv(summary_path, rows, summary_fields)
    produced["package_summary_csv"] = summary_path
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, sort_keys=True)
    produced["package_summary_json"] = summary_json_path
    if contract_mode not in {"legacy", "contract"}:
        contract_mode = "legacy"
    if contract_mode == "legacy":
        plot_ids = [pid for pid, spec in PLOT_REGISTRY.items() if "package_compare" in spec["scopes"]]
    else:
        plot_ids = requested_plots or []
    plots = set(resolve_plot_id(p) for p in plot_ids)
    if "package_tradeoff_scatter" in plots:
        plot_tradeoff_scatter(job_rows, os.path.join(plots_dir, "package_tradeoff_scatter.png"))
        produced["package_tradeoff_scatter"] = os.path.join(plots_dir, "package_tradeoff_scatter.png")
    if "package_pareto_frontier" in plots:
        plot_package_frontiers(job_rows, os.path.join(plots_dir, "package_pareto_frontier.png"))
        produced["package_pareto_frontier"] = os.path.join(plots_dir, "package_pareto_frontier.png")
    if "package_tradeoff_pareto_by_param" in plots:
        plot_tradeoff_by_param(job_rows, os.path.join(plots_dir, "package_tradeoff_pareto_by_param.png"))
        produced["package_tradeoff_pareto_by_param"] = os.path.join(
            plots_dir, "package_tradeoff_pareto_by_param.png"
        )
    bundle_path = ""
    if zip_bundle:
        import zipfile

        bundle_path = os.path.join(package_dir, "package_bundle_small.zip")
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            manifest_path = os.path.join(package_dir, "package_manifest.json")
            for path in [
                combined_path,
                summary_path,
                summary_json_path,
                manifest_path,
            ]:
                if os.path.exists(path):
                    zf.write(path, arcname=os.path.relpath(path, package_dir))
            for plot_id in [
                "package_tradeoff_scatter",
                "package_pareto_frontier",
                "package_tradeoff_pareto_by_param",
            ]:
                filename = PLOT_REGISTRY.get(plot_id, {}).get("files", {}).get("package_compare")
                if not filename:
                    continue
                path = os.path.join(plots_dir, filename)
                if os.path.exists(path):
                    zf.write(path, arcname=os.path.relpath(path, package_dir))
    return {
        "combined_episode_points_csv": combined_path,
        "package_summary_csv": summary_path,
        "package_summary_json": summary_json_path,
        "plots_dir": plots_dir,
        "package_bundle_small": bundle_path,
        "produced": produced,
    }


def build_analysis_manifest(
    package_dir: str,
    package_name: str,
    app_version: str,
    config_hash: str,
    git_commit: str,
    is_dirty: Optional[bool],
    analysis_config: Dict,
    job_results: List[Dict],
    package_outputs: Dict,
) -> Dict:
    requested_plots = analysis_config.get("plots") or list(PLOT_REGISTRY.keys())
    always_include = analysis_config.get("always_include", [])
    for item in always_include:
        if item in PLOT_REGISTRY and item not in requested_plots:
            requested_plots.append(item)
    always_tables = [
        "analysis_manifest_json",
        "package_manifest_json",
        "combined_episode_points_csv",
        "package_summary_csv",
        "package_summary_json",
    ]
    requested_tables = analysis_config.get("tables") or []
    always_include = analysis_config.get("always_include", [])
    table_includes = [item for item in always_include if item in TABLE_REGISTRY]
    requested_tables = list(dict.fromkeys(always_tables + list(requested_tables) + table_includes))
    strict = bool(analysis_config.get("strict", False))
    manifest = {
        "package_name": package_name,
        "app_version": app_version,
        "created_at": utc_now_iso(),
        "code_version": {
            "git_sha": git_commit or "unknown",
            "is_dirty": is_dirty if is_dirty is not None else "unknown",
        },
        "config_hash": config_hash,
        "requested": {
            "plots": requested_plots,
            "tables": requested_tables,
            "primary_metrics": analysis_config.get("primary_metrics", []),
            "risk": analysis_config.get("risk", {}),
            "welfare": analysis_config.get("welfare", {}),
            "strict": strict,
        },
        "produced": {
            "plots": [],
            "tables": [],
            "jobs": {},
        },
        "missing": {
            "package": [],
            "jobs": {},
        },
        "warnings": [],
        "errors": [],
        "strict": strict,
        "exit_status": 0,
    }
    if package_outputs.get("error"):
        manifest["errors"].append({"reason": "analysis_exception", "detail": package_outputs.get("error")})
    for plot_id in requested_plots:
        resolved = resolve_plot_id(plot_id)
        spec = PLOT_REGISTRY.get(resolved)
        scope = "package_compare"
        if not spec:
            manifest["missing"]["package"].append({"plot_id": plot_id, "reason": "unknown_plot_id"})
            manifest["produced"]["plots"].append(
                {
                    "plot_id": plot_id,
                    "path": "",
                    "status": "failed",
                    "reason": "unknown_plot_id",
                    "scope": scope,
                }
            )
            if strict:
                manifest["errors"].append({"plot_id": plot_id, "reason": "unknown_plot_id"})
            else:
                manifest["warnings"].append({"plot_id": plot_id, "reason": "unknown_plot_id"})
            continue
        if scope not in spec["scopes"]:
            manifest["missing"]["package"].append({"plot_id": plot_id, "reason": "not_applicable"})
            manifest["produced"]["plots"].append(
                {
                    "plot_id": plot_id,
                    "path": "",
                    "status": "skipped",
                    "reason": "not_applicable",
                    "scope": scope,
                }
            )
            if strict:
                manifest["errors"].append({"plot_id": plot_id, "reason": "not_applicable"})
            else:
                manifest["warnings"].append({"plot_id": plot_id, "reason": "not_applicable"})
            continue
        filename = spec.get("files", {}).get("package_compare", "")
        path = package_outputs.get("produced", {}).get(resolved) or (
            os.path.join(package_dir, "analysis", "package_compare", filename) if filename else ""
        )
        if path and os.path.exists(path):
            manifest["produced"]["plots"].append(
                {"plot_id": plot_id, "path": path, "status": "ok", "scope": scope}
            )
        else:
            manifest["produced"]["plots"].append(
                {
                    "plot_id": plot_id,
                    "path": path or "",
                    "status": "failed",
                    "reason": "not_generated",
                    "scope": scope,
                }
            )
            if strict:
                manifest["errors"].append({"plot_id": plot_id, "reason": "not_generated"})
            else:
                manifest["warnings"].append({"plot_id": plot_id, "reason": "not_generated"})
    for table_id in requested_tables:
        spec = TABLE_REGISTRY.get(table_id)
        if not spec:
            manifest["missing"]["package"].append({"table_id": table_id, "reason": "unknown_table_id"})
            manifest["produced"]["tables"].append(
                {"table_id": table_id, "path": "", "status": "failed", "reason": "unknown_table_id"}
            )
            if strict:
                manifest["errors"].append({"table_id": table_id, "reason": "unknown_table_id"})
            else:
                manifest["warnings"].append({"table_id": table_id, "reason": "unknown_table_id"})
            continue
        if "package" not in spec["scopes"]:
            manifest["missing"]["package"].append({"table_id": table_id, "reason": "not_applicable"})
            manifest["produced"]["tables"].append(
                {"table_id": table_id, "path": "", "status": "skipped", "reason": "not_applicable"}
            )
            if strict:
                manifest["errors"].append({"table_id": table_id, "reason": "not_applicable"})
            else:
                manifest["warnings"].append({"table_id": table_id, "reason": "not_applicable"})
            continue
        path = os.path.join(package_dir, spec["filename"])
        if table_id == "analysis_manifest_json":
            status = "ok"
        else:
            status = "ok" if os.path.exists(path) else "failed"
        entry = {"table_id": table_id, "path": path, "status": status}
        if status != "ok":
            entry["reason"] = "not_generated"
            if strict:
                manifest["errors"].append({"table_id": table_id, "reason": "not_generated"})
            else:
                manifest["warnings"].append({"table_id": table_id, "reason": "not_generated"})
        manifest["produced"]["tables"].append(entry)
    for job in job_results:
        job_name = job.get("job_name", "")
        job_kind = job.get("job_kind", "")
        job_root = job.get("job_root", "")
        job_analysis = job.get("analysis", {})
        job_plots = job_analysis.get("plots")
        if job_plots is None:
            job_plots = []
        job_tables = job_analysis.get("tables")
        if job_tables is None:
            job_tables = []
        job_strict = bool(job_analysis.get("strict", strict))
        produced = {"plots": [], "tables": []}
        missing = []
        for plot_id in job_plots:
            resolved = resolve_plot_id(plot_id)
            spec = PLOT_REGISTRY.get(resolved)
            scope = "suite_exp_compare" if job_kind == "suite" else "sweep_compare"
            if not spec:
                missing.append({"plot_id": plot_id, "reason": "unknown_plot_id"})
                produced["plots"].append(
                    {
                        "plot_id": plot_id,
                        "path": "",
                        "status": "failed",
                        "reason": "unknown_plot_id",
                        "scope": scope,
                    }
                )
                continue
            if scope not in spec["scopes"]:
                missing.append({"plot_id": plot_id, "reason": "not_applicable"})
                produced["plots"].append(
                    {
                        "plot_id": plot_id,
                        "path": "",
                        "status": "skipped",
                        "reason": "not_applicable",
                        "scope": scope,
                    }
                )
                continue
            filename = spec.get("files", {}).get(scope, "")
            rel_path = os.path.join("analysis", "exp_compare", filename)
            if scope == "sweep_compare":
                rel_path = os.path.join("analysis", "sweep_compare", filename)
            full_path = os.path.join(job_root, rel_path)
            if os.path.exists(full_path):
                produced["plots"].append(
                    {"plot_id": plot_id, "path": full_path, "status": "ok", "scope": scope}
                )
            else:
                produced["plots"].append(
                    {
                        "plot_id": plot_id,
                        "path": full_path,
                        "status": "failed",
                        "reason": "not_found",
                        "scope": scope,
                    }
                )
                if job_strict:
                    missing.append({"plot_id": plot_id, "reason": "not_found"})
        for table_id in job_tables:
            spec = TABLE_REGISTRY.get(table_id)
            if not spec:
                missing.append({"table_id": table_id, "reason": "unknown_table_id"})
                produced["tables"].append(
                    {"table_id": table_id, "path": "", "status": "failed", "reason": "unknown_table_id"}
                )
                continue
            if job_kind == "suite":
                if table_id in {"suite_episode_points_csv", "episode_points_csv"}:
                    ep_path = find_episode_points(job_root)
                    if ep_path:
                        produced["tables"].append({"table_id": table_id, "path": ep_path, "status": "ok"})
                    else:
                        produced["tables"].append(
                            {"table_id": table_id, "path": "", "status": "failed", "reason": "not_found"}
                        )
                        if job_strict:
                            missing.append({"table_id": table_id, "reason": "not_found"})
                else:
                    missing.append({"table_id": table_id, "reason": "not_applicable"})
                    produced["tables"].append(
                        {"table_id": table_id, "path": "", "status": "skipped", "reason": "not_applicable"}
                    )
            elif job_kind == "sweep":
                if table_id in {"episode_points_csv"}:
                    ep_path = find_episode_points(job_root)
                    if ep_path:
                        produced["tables"].append({"table_id": table_id, "path": ep_path, "status": "ok"})
                    else:
                        produced["tables"].append(
                            {"table_id": table_id, "path": "", "status": "failed", "reason": "not_found"}
                        )
                        if job_strict:
                            missing.append({"table_id": table_id, "reason": "not_found"})
                elif table_id == "sweep_summary_csv":
                    path = os.path.join(job_root, "sweep_summary.csv")
                    if os.path.exists(path):
                        produced["tables"].append({"table_id": table_id, "path": path, "status": "ok"})
                    else:
                        produced["tables"].append(
                            {"table_id": table_id, "path": path, "status": "failed", "reason": "not_found"}
                        )
                        if job_strict:
                            missing.append({"table_id": table_id, "reason": "not_found"})
                elif table_id == "sweep_summary_json":
                    path = os.path.join(job_root, "sweep_summary.json")
                    if os.path.exists(path):
                        produced["tables"].append({"table_id": table_id, "path": path, "status": "ok"})
                    else:
                        produced["tables"].append(
                            {"table_id": table_id, "path": path, "status": "failed", "reason": "not_found"}
                        )
                        if job_strict:
                            missing.append({"table_id": table_id, "reason": "not_found"})
                else:
                    missing.append({"table_id": table_id, "reason": "not_applicable"})
                    produced["tables"].append(
                        {"table_id": table_id, "path": "", "status": "skipped", "reason": "not_applicable"}
                    )
            else:
                missing.append({"table_id": table_id, "reason": "not_applicable"})
                produced["tables"].append(
                    {"table_id": table_id, "path": "", "status": "skipped", "reason": "not_applicable"}
                )
        manifest["produced"]["jobs"][job_name] = produced
        manifest["missing"]["jobs"][job_name] = missing
        if job_strict and missing:
            manifest["errors"].append({"job_name": job_name, "reason": "strict_missing_artifacts"})
        elif missing:
            manifest["warnings"].append({"job_name": job_name, "reason": "missing_artifacts"})
        job_manifest = {
            "job_name": job_name,
            "job_kind": job_kind,
            "requested": {"plots": job_plots, "tables": job_tables, "strict": job_strict},
            "produced": produced,
            "missing": missing,
        }
        job_manifest_path = os.path.join(job_root, "analysis", "analysis_manifest.json")
        os.makedirs(os.path.dirname(job_manifest_path), exist_ok=True)
        with open(job_manifest_path, "w", encoding="utf-8") as f:
            json.dump(job_manifest, f, indent=2, sort_keys=True)
    if manifest["errors"]:
        manifest["exit_status"] = 1
    manifest_path = os.path.join(package_dir, "analysis", "analysis_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest
