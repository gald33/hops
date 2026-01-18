import os
from typing import Dict, List, Optional, Set


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


ALLOWED_TOP_KEYS = {
    "version",
    "defaults",
    "claims",
    "include",
    "strict",
    "fail_on",
    "output_dir",
    "always_include",
}
ALLOWED_DEFAULTS_KEYS = {
    "scope",
    "rng_seed",
    "n_boot",
    "ci",
    "handle_censoring",
    "data",
    "target",
    "metrics",
    "evidence",
    "plots_required",
}
ALLOWED_CLAIM_KEYS = {
    "id",
    "scope",
    "description",
    "target",
    "data",
    "metrics",
    "evidence",
    "strict",
    "fail_on",
    "handle_censoring",
    "rng_seed",
    "n_boot",
    "ci",
    "plots_required",
}
ALLOWED_TARGET_KEYS = {
    "kind",
    "treatment",
    "control",
    "effect",
    "op",
    "value",
    "param",
    "pvalue_threshold",
    "require_pvalue",
    "paired_by",
}
ALLOWED_DATA_KEYS = {
    "source",
    "filter",
    "unit",
    "job_name",
    "job_names",
}
ALLOWED_METRIC_KEYS = {
    "name",
    "agg",
    "op",
    "value",
    "effect",
    "n_boot",
    "ci",
    "rng_seed",
}
ALLOWED_EVIDENCE_KEYS = {"tables", "plots", "plots_required"}


class ClaimsSchemaError(ValueError):
    pass


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
    stack = [(0, data, None)]
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
                    raise ClaimsSchemaError(f"list item without list context: {line}")
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
                raise ClaimsSchemaError(f"invalid line: {line}")
            key, rest = content.split(":", 1)
            key = key.strip()
            val = rest.strip()
            if not isinstance(parent, dict):
                raise ClaimsSchemaError(f"mapping entry without dict context: {line}")
            if val == "":
                parent[key] = None
                stack[-1] = (stack[-1][0], parent, key)
            else:
                parent[key] = parse_scalar(val)
    return data


def load_yaml(path: str) -> Dict:
    if yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                return loaded
            return {"value": loaded}
    return _parse_minimal_yaml(path)


def _validate_keys(obj: Dict, allowed: Set[str], path: str) -> None:
    extra = set(obj.keys()) - allowed
    if extra:
        raise ClaimsSchemaError(f"unknown keys at {path}: {', '.join(sorted(extra))}")


def _deep_merge(base: Dict, extra: Dict) -> Dict:
    merged = dict(base)
    for key, val in extra.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _apply_defaults(claim: Dict, defaults: Dict) -> Dict:
    merged = dict(claim)
    for key, val in defaults.items():
        if key not in ALLOWED_CLAIM_KEYS:
            continue
        if key not in merged or merged[key] is None:
            merged[key] = val
        elif isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(val, merged[key])
    return merged


def _normalize_claims(claims: List[Dict]) -> List[Dict]:
    normalized = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise ClaimsSchemaError("claim entries must be mappings")
        _validate_keys(claim, ALLOWED_CLAIM_KEYS, "claim")
        claim_id = claim.get("id")
        if not claim_id or not isinstance(claim_id, str):
            raise ClaimsSchemaError("claim.id is required and must be a string")
        scope = claim.get("scope")
        if not scope or not isinstance(scope, str):
            raise ClaimsSchemaError(f"claim {claim_id} scope is required")
        target = claim.get("target", {})
        if target is not None:
            if not isinstance(target, dict):
                raise ClaimsSchemaError(f"claim {claim_id} target must be a mapping")
            _validate_keys(target, ALLOWED_TARGET_KEYS, f"claim {claim_id}.target")
        data = claim.get("data", {})
        if data is not None:
            if not isinstance(data, dict):
                raise ClaimsSchemaError(f"claim {claim_id} data must be a mapping")
            _validate_keys(data, ALLOWED_DATA_KEYS, f"claim {claim_id}.data")
        metrics = claim.get("metrics")
        if metrics is not None:
            if not isinstance(metrics, list):
                raise ClaimsSchemaError(f"claim {claim_id} metrics must be a list")
            for metric in metrics:
                if not isinstance(metric, dict):
                    raise ClaimsSchemaError(f"claim {claim_id} metric entries must be mappings")
                _validate_keys(metric, ALLOWED_METRIC_KEYS, f"claim {claim_id}.metrics")
                if not metric.get("name"):
                    raise ClaimsSchemaError(f"claim {claim_id} metric.name is required")
        evidence = claim.get("evidence", {})
        if evidence is not None:
            if not isinstance(evidence, dict):
                raise ClaimsSchemaError(f"claim {claim_id} evidence must be a mapping")
            _validate_keys(evidence, ALLOWED_EVIDENCE_KEYS, f"claim {claim_id}.evidence")
        normalized.append(claim)
    return normalized


def _merge_includes(block: Dict, base_dir: str, seen: Optional[Set[str]] = None) -> Dict:
    merged_defaults: Dict = {}
    merged_claims: List[Dict] = []
    merged_always: List[str] = []
    seen = seen or set()
    includes = block.get("include", []) or []
    if isinstance(includes, str):
        includes = [includes]
    for inc in includes:
        path = inc
        tried: List[str] = []
        if not os.path.isabs(path):
            base_path = os.path.abspath(os.path.join(base_dir, path))
            tried.append(base_path)
            if os.path.exists(base_path):
                path = base_path
            else:
                cwd_path = os.path.abspath(os.path.join(os.getcwd(), path))
                tried.append(cwd_path)
                path = cwd_path
        else:
            path = os.path.abspath(path)
            tried.append(path)
        if not os.path.exists(path):
            raise ClaimsSchemaError(
                f"claims include not found: {inc} (tried: {', '.join(tried)})"
            )
        if path in seen:
            continue
        seen.add(path)
        inc_cfg = load_yaml(path)
        _validate_keys(inc_cfg, ALLOWED_TOP_KEYS, "claims include")
        inc_defaults = inc_cfg.get("defaults", {}) or {}
        if not isinstance(inc_defaults, dict):
            raise ClaimsSchemaError(f"defaults in {path} must be a mapping")
        merged_defaults = _deep_merge(merged_defaults, inc_defaults)
        inc_claims = inc_cfg.get("claims", []) or []
        if not isinstance(inc_claims, list):
            raise ClaimsSchemaError(f"claims in {path} must be a list")
        merged_claims.extend(inc_claims)
        inc_always = inc_cfg.get("always_include", []) or []
        if isinstance(inc_always, str):
            inc_always = [inc_always]
        if not isinstance(inc_always, list):
            raise ClaimsSchemaError(f"always_include in {path} must be a list")
        merged_always.extend(str(item) for item in inc_always)
        nested = _merge_includes(inc_cfg, os.path.dirname(path), seen)
        merged_defaults = _deep_merge(merged_defaults, nested.get("defaults", {}))
        merged_claims.extend(nested.get("claims", []))
        merged_always.extend(nested.get("always_include", []))
    return {"defaults": merged_defaults, "claims": merged_claims, "always_include": merged_always}


def load_claims_block(block: Optional[Dict], base_dir: str) -> Dict:
    if not block:
        return {"defaults": {}, "claims": [], "always_include": []}
    if not isinstance(block, dict):
        raise ClaimsSchemaError("claims block must be a mapping")
    _validate_keys(block, ALLOWED_TOP_KEYS, "claims block")
    merged = _merge_includes(block, base_dir)
    defaults = merged.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ClaimsSchemaError("defaults must be a mapping")
    _validate_keys(defaults, ALLOWED_DEFAULTS_KEYS, "claims.defaults")
    defaults = _deep_merge(defaults, block.get("defaults", {}) or {})
    claims = merged.get("claims", [])
    claims.extend(block.get("claims", []) or [])
    always_include = merged.get("always_include", [])
    block_always = block.get("always_include", []) or []
    if isinstance(block_always, str):
        block_always = [block_always]
    if not isinstance(block_always, list):
        raise ClaimsSchemaError("always_include must be a list")
    always_include = list(dict.fromkeys([str(item) for item in always_include + block_always]))
    claims = _normalize_claims(claims)
    deduped: List[Dict] = []
    seen_ids: Set[str] = set()
    for claim in claims:
        claim_id = claim.get("id")
        if claim_id in seen_ids:
            raise ClaimsSchemaError(f"duplicate claim id: {claim_id}")
        seen_ids.add(claim_id)
        deduped.append(_apply_defaults(claim, defaults))
    deduped = _normalize_claims(deduped)
    return {
        "version": block.get("version"),
        "defaults": defaults,
        "claims": deduped,
        "strict": bool(block.get("strict", False)),
        "fail_on": block.get("fail_on"),
        "output_dir": block.get("output_dir"),
        "always_include": always_include,
    }
