import json
import os
from typing import Dict, List, Optional, Tuple

from hops.analysis_contract import PLOT_REGISTRY, TABLE_REGISTRY, resolve_plot_id
from hops.claims_contract import resolve_claim_id
from hops.claims_data import (
    aggregate,
    aggregate_by_param,
    aggregate_by_seed,
    apply_filter,
    filter_implies_reached_tau,
    find_episode_points,
    load_data_source,
)
from hops.claims_stats import bootstrap_ci, paired_seed_bootstrap, spearman_trend
from hops.registry import utc_now_iso


TIME_TO_TAU_METRICS = {"time_to_tau"}


def _metric_values(rows: List[Dict], metric: str) -> List[float]:
    values = []
    for row in rows:
        val = row.get(metric)
        if isinstance(val, (int, float)):
            values.append(float(val))
    return values


def _compare(value: float, op: str, target: float) -> bool:
    if op == ">=":
        return value >= target
    if op == ">":
        return value > target
    if op == "<=":
        return value <= target
    if op == "<":
        return value < target
    if op == "==":
        return value == target
    if op == "!=" :
        return value != target
    return False


def _evidence_tables(scope: str, package_dir: str, job_root: Optional[str], tables: List[str]) -> Tuple[List[str], List[str]]:
    missing = []
    present = []
    for table_id in tables:
        spec = TABLE_REGISTRY.get(table_id)
        if not spec:
            missing.append(table_id)
            continue
        filename = spec.get("filename")
        if scope == "package":
            path = os.path.join(package_dir, filename)
        else:
            if not job_root:
                missing.append(table_id)
                continue
            if table_id in {"suite_episode_points_csv", "episode_points_csv"}:
                path = find_episode_points(job_root) if job_root else None
            elif table_id in {"sweep_summary_csv", "sweep_summary_json"}:
                path = os.path.join(job_root, filename)
            else:
                path = os.path.join(job_root, filename)
        if path and os.path.exists(path):
            present.append(table_id)
        else:
            missing.append(table_id)
    return present, missing


def _evidence_plots(scope: str, package_dir: str, job_root: Optional[str], plots: List[str]) -> Tuple[List[str], List[str]]:
    missing = []
    present = []
    for plot_id in plots:
        resolved = resolve_plot_id(plot_id)
        spec = PLOT_REGISTRY.get(resolved)
        if not spec:
            missing.append(plot_id)
            continue
        files = spec.get("files", {})
        path = None
        if scope == "package":
            filename = files.get("package_compare")
            if filename:
                path = os.path.join(package_dir, "analysis", "package_compare", filename)
        elif scope == "suite":
            filename = files.get("suite_exp_compare")
            if filename and job_root:
                path = os.path.join(job_root, "analysis", "exp_compare", filename)
        elif scope == "sweep":
            filename = files.get("sweep_compare")
            if filename and job_root:
                path = os.path.join(job_root, "analysis", "sweep_compare", filename)
        if path and os.path.exists(path):
            present.append(plot_id)
        else:
            missing.append(plot_id)
    return present, missing


def _evaluate_single_metric(
    rows: List[Dict], metric: Dict, defaults: Dict, data_filter: Optional[str]
) -> Dict:
    name = metric.get("name")
    agg = metric.get("agg", "mean")
    op = metric.get("op", ">=")
    target = float(metric.get("value", 0.0))
    handle_censoring = metric.get("handle_censoring") or defaults.get("handle_censoring")
    if name in TIME_TO_TAU_METRICS and handle_censoring == "separate":
        if not filter_implies_reached_tau(data_filter):
            return {
                "metric": name,
                "status": "inconclusive",
                "notes": "time_to_tau requires reached_tau filter under handle_censoring=separate",
            }
    values = _metric_values(rows, name)
    values = [v for v in values if v is not None]
    if not values:
        return {"metric": name, "status": "inconclusive", "notes": "no data"}
    summary_value = aggregate(values, agg)
    lo, mid, hi = bootstrap_ci(
        values,
        rng_seed=int(metric.get("rng_seed", defaults.get("rng_seed", 0))),
        n_boot=int(metric.get("n_boot", defaults.get("n_boot", 1000))),
        ci=float(metric.get("ci", defaults.get("ci", 0.95))),
    )
    passed = _compare(summary_value, op, target)
    return {
        "metric": name,
        "status": "passed" if passed else "failed",
        "value": summary_value,
        "ci": [lo, mid, hi],
        "op": op,
        "target": target,
    }


def _evaluate_pair_metric(
    rows: List[Dict],
    metric: Dict,
    defaults: Dict,
    base_filter: Optional[str],
    target: Dict,
) -> Dict:
    name = metric.get("name")
    agg = metric.get("agg", "mean")
    effect = metric.get("effect", target.get("effect", "delta"))
    op = metric.get("op", target.get("op", ">="))
    target_value = float(metric.get("value", target.get("value", 0.0)))
    paired_by = target.get("paired_by", "seed")
    treat_filter = target.get("treatment", {}).get("filter")
    control_filter = target.get("control", {}).get("filter")
    treat_rows = apply_filter(rows, base_filter)
    treat_rows = apply_filter(treat_rows, treat_filter)
    control_rows = apply_filter(rows, base_filter)
    control_rows = apply_filter(control_rows, control_filter)
    if paired_by != "seed":
        return {"metric": name, "status": "inconclusive", "notes": "paired_by must be seed"}
    treat_by_seed = aggregate_by_seed(treat_rows, name, agg)
    control_by_seed = aggregate_by_seed(control_rows, name, agg)
    pairs = []
    for seed, treat_val in treat_by_seed.items():
        if seed in control_by_seed:
            pairs.append((treat_val, control_by_seed[seed]))
    if not pairs:
        return {"metric": name, "status": "inconclusive", "notes": "no paired seeds"}
    lo, mid, hi = paired_seed_bootstrap(
        pairs,
        rng_seed=int(metric.get("rng_seed", defaults.get("rng_seed", 0))),
        n_boot=int(metric.get("n_boot", defaults.get("n_boot", 1000))),
        ci=float(metric.get("ci", defaults.get("ci", 0.95))),
        effect=effect,
    )
    passed = _compare(mid, op, target_value)
    return {
        "metric": name,
        "status": "passed" if passed else "failed",
        "effect": effect,
        "value": mid,
        "ci": [lo, mid, hi],
        "op": op,
        "target": target_value,
        "n_pairs": len(pairs),
    }


def _evaluate_trend_metric(rows: List[Dict], metric: Dict, defaults: Dict, target: Dict) -> Dict:
    name = metric.get("name")
    agg = metric.get("agg", "mean")
    op = metric.get("op", target.get("op", ">="))
    target_value = float(metric.get("value", target.get("value", 0.0)))
    param_name = target.get("param")
    require_pvalue = bool(target.get("require_pvalue", False))
    pvalue_threshold = float(target.get("pvalue_threshold", 0.05))
    param_values = aggregate_by_param(rows, name, agg)
    if not param_values:
        return {"metric": name, "status": "inconclusive", "notes": "no param data"}
    xs = sorted(param_values.keys())
    ys = [param_values[x] for x in xs]
    rho, pvalue = spearman_trend(xs, ys)
    if require_pvalue and pvalue is None:
        return {"metric": name, "status": "inconclusive", "notes": "pvalue unavailable"}
    passed = _compare(rho, op, target_value)
    if require_pvalue and pvalue is not None and pvalue > pvalue_threshold:
        passed = False
    result = {
        "metric": name,
        "status": "passed" if passed else "failed",
        "rho": rho,
        "pvalue": pvalue,
        "op": op,
        "target": target_value,
        "param": param_name,
    }
    return result


def evaluate_claim(
    claim: Dict,
    package_dir: str,
    job_root: Optional[str],
    defaults: Dict,
) -> Dict:
    claim_defaults = dict(defaults)
    for key in ("handle_censoring", "rng_seed", "n_boot", "ci", "plots_required"):
        if claim.get(key) is not None:
            claim_defaults[key] = claim.get(key)
    claim_id = resolve_claim_id(claim.get("id"))
    scope = claim.get("scope")
    target = claim.get("target", {}) or {}
    data = claim.get("data", {}) or {}
    data_filter = data.get("filter")
    source = data.get("source")
    if not source:
        if scope == "package":
            source = "combined_episode_points"
        elif scope == "suite":
            source = "suite_episode_points"
        elif scope == "sweep":
            source = "sweep_summary"
    rows = load_data_source(package_dir, job_root, source)
    rows = apply_filter(rows, data_filter)
    metrics = claim.get("metrics") or []
    target_kind = target.get("kind", "single")
    metric_results = []
    warnings: List[str] = []
    errors: List[str] = []
    evidence = claim.get("evidence", {}) or {}
    tables_required = evidence.get("tables", []) or []
    plots_required = evidence.get("plots", []) or []
    plots_hard = bool(evidence.get("plots_required", claim_defaults.get("plots_required", False)))
    tables_present, tables_missing = _evidence_tables(scope, package_dir, job_root, tables_required)
    plots_present, plots_missing = _evidence_plots(scope, package_dir, job_root, plots_required)
    if tables_missing:
        errors.append(f"missing tables: {', '.join(tables_missing)}")
        return {
            "claim_id": claim_id,
            "scope": scope,
            "status": "skipped",
            "metrics": [],
            "warnings": warnings,
            "errors": errors,
            "evidence": {"tables_present": tables_present, "tables_missing": tables_missing},
            "data_summary": {"rows": len(rows)},
        }
    if plots_missing:
        msg = f"missing plots: {', '.join(plots_missing)}"
        if plots_hard:
            errors.append(msg)
            return {
                "claim_id": claim_id,
                "scope": scope,
                "status": "skipped",
                "metrics": [],
                "warnings": warnings,
                "errors": errors,
                "evidence": {"plots_present": plots_present, "plots_missing": plots_missing},
                "data_summary": {"rows": len(rows)},
            }
        warnings.append(msg)
    if not metrics:
        return {
            "claim_id": claim_id,
            "scope": scope,
            "status": "inconclusive",
            "metrics": [],
            "warnings": warnings,
            "errors": errors,
            "evidence": {"tables_present": tables_present, "tables_missing": tables_missing},
            "data_summary": {"rows": len(rows)},
        }
    if target_kind == "pair":
        for metric in metrics:
            metric_results.append(_evaluate_pair_metric(rows, metric, claim_defaults, data_filter, target))
    elif target_kind == "trend":
        for metric in metrics:
            metric_results.append(_evaluate_trend_metric(rows, metric, claim_defaults, target))
    else:
        for metric in metrics:
            metric_results.append(_evaluate_single_metric(rows, metric, claim_defaults, data_filter))
    status = "supported"
    for result in metric_results:
        if result.get("status") == "failed":
            status = "rejected"
            break
        if result.get("status") in {"inconclusive", "skipped"}:
            status = "inconclusive"
    return {
        "claim_id": claim_id,
        "scope": scope,
        "status": status,
        "metrics": metric_results,
        "warnings": warnings,
        "errors": errors,
        "evidence": {
            "tables_present": tables_present,
            "tables_missing": tables_missing,
            "plots_present": plots_present,
            "plots_missing": plots_missing,
        },
        "data_summary": {"rows": len(rows)},
    }


def evaluate_claims(
    package_dir: str,
    package_name: str,
    app_version: str,
    config_hash: str,
    git_commit: str,
    is_dirty: Optional[bool],
    claims_config: Dict,
    job_results: List[Dict],
) -> Dict:
    defaults = claims_config.get("defaults", {}) or {}
    strict = bool(claims_config.get("strict", False))
    fail_on = claims_config.get("fail_on") or ["rejected"]
    if isinstance(fail_on, str):
        fail_on = [fail_on]
    if not isinstance(fail_on, list):
        fail_on = ["rejected"]
    output_dir = claims_config.get("output_dir") or os.path.join("analysis", "claims")
    output_path = output_dir if os.path.isabs(output_dir) else os.path.join(package_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    manifest_path = os.path.join(output_path, "claims_manifest.json")
    report_path = os.path.join(output_path, "claims_report.md")
    results: List[Dict] = []
    warnings: List[str] = []
    errors: List[str] = []
    exit_status = 0
    try:
        for claim in claims_config.get("claims", []) or []:
            scope = claim.get("scope")
            if scope not in {"package", "suite", "sweep"}:
                results.append(
                    {
                        "claim_id": resolve_claim_id(claim.get("id")),
                        "scope": scope,
                        "status": "skipped",
                        "metrics": [],
                        "warnings": [],
                        "errors": [f"unknown scope: {scope}"],
                    }
                )
                continue
            job_name = claim.get("data", {}).get("job_name")
            job_names = claim.get("data", {}).get("job_names")
            if isinstance(job_names, str):
                job_names = [job_names]
            if scope == "package":
                result = evaluate_claim(claim, package_dir, None, defaults)
                results.append(result)
            else:
                targets = []
                for job in job_results:
                    if scope == "suite" and job.get("job_kind") != "suite":
                        continue
                    if scope == "sweep" and job.get("job_kind") != "sweep":
                        continue
                    if job_name and job.get("job_name") != job_name:
                        continue
                    if job_names and job.get("job_name") not in job_names:
                        continue
                    targets.append(job)
                if not targets:
                    results.append(
                        {
                            "claim_id": resolve_claim_id(claim.get("id")),
                            "scope": scope,
                            "status": "skipped",
                            "metrics": [],
                            "warnings": [],
                            "errors": ["no matching jobs"],
                        }
                    )
                for job in targets:
                    result = evaluate_claim(claim, package_dir, job.get("job_root"), defaults)
                    result["job_name"] = job.get("job_name")
                    results.append(result)
        strict_fail = any(result.get("status") in fail_on for result in results)
        if strict and strict_fail:
            exit_status = 2
            errors.append("strict claims failed")
    except Exception as exc:
        errors.append(str(exc))
        exit_status = 2 if strict else 0
    requested_ids = []
    for claim in claims_config.get("claims", []) or []:
        claim_id = claim.get("id")
        if claim_id:
            requested_ids.append(resolve_claim_id(claim_id))
    manifest = {
        "package_name": package_name,
        "app_version": app_version,
        "created_at": utc_now_iso(),
        "code_version": {
            "git_sha": git_commit or "unknown",
            "is_dirty": is_dirty if is_dirty is not None else "unknown",
        },
        "config_hash": config_hash,
        "requested": {"claims": requested_ids},
        "results": results,
        "warnings": warnings,
        "errors": errors,
        "strict": strict,
        "exit_status": exit_status,
        "output_dir": output_path,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    _write_report(report_path, manifest)
    return manifest


def _write_report(path: str, manifest: Dict) -> None:
    lines = ["# Claims Report", "", f"Generated: {manifest.get('created_at')}", ""]
    lines.append("| Claim | Status | Scope | Job | Notes |")
    lines.append("| --- | --- | --- | --- | --- |")
    for result in manifest.get("results", []):
        claim_id = result.get("claim_id")
        status = result.get("status")
        scope = result.get("scope")
        job = result.get("job_name", "")
        notes = "; ".join(result.get("errors", []) + result.get("warnings", []))
        lines.append(f"| {claim_id} | {status} | {scope} | {job} | {notes} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
