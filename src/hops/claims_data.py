import ast
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple


METRIC_ALIASES = {
    "final_W_true": "final_welfare",
    "final_welfare_e": "final_welfare",
    "welfare_e": "final_welfare_raw",
    "final_S_eff": "final_S_eff",
    "max_G_true": "max_gap_true_ratio",
    "max_G_true_episode": "max_gap_true_ratio",
    "risk_e": "max_gap_true_ratio",
    "T_tau": "time_to_tau",
    "time_to_tau_mean": "time_to_tau",
    "gap_divergence": "gap_divergence_episode",
    "divergence": "gap_divergence_episode",
    "divergence_e": "gap_divergence_episode",
}

NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE]-?\d+)?$")


def _coerce_value(value):
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    val = str(value).strip()
    if val == "":
        return None
    low = val.lower()
    if low in {"true", "false"}:
        return low == "true"
    if NUMERIC_RE.match(val):
        try:
            return float(val)
        except ValueError:
            return value
    return value


def load_csv(path: str) -> List[Dict]:
    import csv

    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: _coerce_value(v) for k, v in row.items()})
    return rows


def load_json(path: str):
    import json

    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonicalize_rows(rows: Iterable[Dict]) -> List[Dict]:
    canonical = []
    for row in rows:
        normalized = {k: _coerce_value(v) for k, v in row.items()}
        for alias, canon in METRIC_ALIASES.items():
            if canon not in normalized and alias in normalized:
                normalized[canon] = normalized.get(alias)
        final_s = normalized.get("final_S_eff")
        if normalized.get("final_welfare") is None and isinstance(final_s, (int, float)):
            normalized["final_welfare"] = float(math.log1p(max(0.0, final_s)))
        if normalized.get("max_gap_true_ratio") is None:
            max_g = normalized.get("max_G_true")
            if isinstance(max_g, (int, float)):
                normalized["max_gap_true_ratio"] = float(max_g)
        if normalized.get("time_to_tau") is None:
            t_tau = normalized.get("T_tau")
            if isinstance(t_tau, (int, float)):
                normalized["time_to_tau"] = float(t_tau)
        if normalized.get("censored") is None and "reached_tau" in normalized:
            reached = normalized.get("reached_tau")
            if isinstance(reached, bool):
                normalized["censored"] = not reached
        if normalized.get("reached_tau") is None and "censored" in normalized:
            censored = normalized.get("censored")
            if isinstance(censored, bool):
                normalized["reached_tau"] = not censored
        if normalized.get("reached_tau") is None:
            t_tau = normalized.get("time_to_tau")
            horizon = normalized.get("horizon")
            if isinstance(t_tau, (int, float)):
                if isinstance(horizon, (int, float)):
                    normalized["reached_tau"] = t_tau < horizon
                else:
                    normalized["reached_tau"] = math.isfinite(t_tau)
        if normalized.get("dormant_mass_end") is None:
            for alt in ("dormant_mass_t_end", "dormant_stock_end", "dormant_mass"):
                if alt in normalized and isinstance(normalized.get(alt), (int, float)):
                    normalized["dormant_mass_end"] = normalized.get(alt)
                    break
        canonical.append(normalized)
    return canonical


ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.Compare,
    ast.Name,
    ast.Constant,
    ast.UnaryOp,
    ast.Load,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
)


def _parse_filter(expr: str) -> ast.AST:
    normalized = re.sub(r"\btrue\b", "True", expr, flags=re.IGNORECASE)
    normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bnull\b", "None", normalized, flags=re.IGNORECASE)
    tree = ast.parse(normalized, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError("unsupported filter expression")
    return tree


def _eval_node(node: ast.AST, row: Dict):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, row)
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, row) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(vals)
        if isinstance(node.op, ast.Or):
            return any(vals)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_eval_node(node.operand, row))
    if isinstance(node, ast.Name):
        return row.get(node.id)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, row)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_node(comp, row)
            if left is None or right is None:
                return False
            if isinstance(op, ast.Eq) and not (left == right):
                return False
            if isinstance(op, ast.NotEq) and not (left != right):
                return False
            if isinstance(op, ast.Gt) and not (left > right):
                return False
            if isinstance(op, ast.GtE) and not (left >= right):
                return False
            if isinstance(op, ast.Lt) and not (left < right):
                return False
            if isinstance(op, ast.LtE) and not (left <= right):
                return False
            left = right
        return True
    return False


def apply_filter(rows: Iterable[Dict], expr: Optional[str]) -> List[Dict]:
    if not expr:
        return list(rows)
    tree = _parse_filter(expr)
    filtered = []
    for row in rows:
        if _eval_node(tree, row):
            filtered.append(row)
    return filtered


def filter_implies_reached_tau(expr: Optional[str]) -> bool:
    if not expr:
        return False
    try:
        tree = _parse_filter(expr)
    except Exception:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            if node.left.id != "reached_tau":
                continue
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, ast.Eq) and isinstance(comp, ast.Constant) and comp.value is True:
                    return True
    return False


def group_by(rows: Iterable[Dict], key: str) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        group_key = str(row.get(key, ""))
        grouped.setdefault(group_key, []).append(row)
    return grouped


def aggregate(values: List[float], agg: str) -> float:
    if not values:
        return float("nan")
    if agg == "mean":
        return float(sum(values) / len(values))
    if agg == "median":
        values_sorted = sorted(values)
        mid = len(values_sorted) // 2
        if len(values_sorted) % 2 == 0:
            return float((values_sorted[mid - 1] + values_sorted[mid]) / 2.0)
        return float(values_sorted[mid])
    if agg == "max":
        return float(max(values))
    if agg == "min":
        return float(min(values))
    return float(sum(values) / len(values))


def aggregate_by_seed(rows: Iterable[Dict], metric: str, agg: str) -> Dict[str, float]:
    groups = group_by(rows, "seed")
    seed_values: Dict[str, float] = {}
    for seed, items in groups.items():
        vals = [item.get(metric) for item in items]
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        if vals:
            seed_values[seed] = aggregate(vals, agg)
    return seed_values


def aggregate_by_param(rows: Iterable[Dict], metric: str, agg: str) -> Dict[float, float]:
    groups = group_by(rows, "param_value")
    param_values: Dict[float, float] = {}
    for param, items in groups.items():
        vals = [item.get(metric) for item in items]
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        if vals:
            try:
                param_float = float(param)
            except (TypeError, ValueError):
                continue
            param_values[param_float] = aggregate(vals, agg)
    return param_values


def load_data_source(package_dir: str, job_root: Optional[str], source: str) -> List[Dict]:
    if source == "combined_episode_points":
        path = os.path.join(package_dir, "combined_episode_points.csv")
        return canonicalize_rows(load_csv(path))
    if source in {"package_summary", "package_summary_csv"}:
        path = os.path.join(package_dir, "package_summary.csv")
        return canonicalize_rows(load_csv(path))
    if source in {"package_summary_json"}:
        path = os.path.join(package_dir, "package_summary.json")
        return canonicalize_rows(load_json(path) or [])
    if source in {"suite_episode_points", "episode_points"} and job_root:
        path = find_episode_points(job_root)
        return canonicalize_rows(load_csv(path)) if path else []
    if source == "sweep_summary" and job_root:
        path = os.path.join(job_root, "sweep_summary.csv")
        return canonicalize_rows(load_csv(path))
    if source == "sweep_summary_json" and job_root:
        path = os.path.join(job_root, "sweep_summary.json")
        return canonicalize_rows(load_json(path) or [])
    return []


def find_episode_points(job_root: str) -> Optional[str]:
    candidates = [
        os.path.join(job_root, "analysis", "exp_compare", "episode_points.csv"),
        os.path.join(job_root, "analysis", "sweep_compare", "episode_points.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    for root, _dirs, files in os.walk(job_root):
        if "episode_points.csv" in files:
            return os.path.join(root, "episode_points.csv")
    return None


def summarize_rows(rows: Iterable[Dict]) -> Dict[str, int]:
    rows_list = list(rows)
    seeds = {row.get("seed") for row in rows_list if row.get("seed") not in (None, "")}
    return {
        "rows": len(rows_list),
        "seeds": len(seeds),
    }


def split_groups(rows: Iterable[Dict], filter_expr: Optional[str]) -> Tuple[List[Dict], List[str]]:
    try:
        filtered = apply_filter(rows, filter_expr)
        return filtered, []
    except Exception as exc:
        return [], [str(exc)]
