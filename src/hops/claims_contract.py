from typing import Dict


CLAIM_REGISTRY = {
    "C001_risk_bounded": {
        "scopes": ["package"],
        "description": "Episode-level risk stays within [0,1] bounds.",
    },
    "C002_time_to_tau_defined": {
        "scopes": ["package"],
        "description": "Time-to-tau is reported for reached episodes.",
    },
}

CLAIM_ALIASES = {}


def resolve_claim_id(claim_id: str) -> str:
    return CLAIM_ALIASES.get(claim_id, claim_id)


def describe_claim_registry() -> Dict[str, Dict[str, str]]:
    info = {}
    for claim_id, spec in sorted(CLAIM_REGISTRY.items()):
        info[claim_id] = {
            "scopes": ",".join(spec.get("scopes", [])),
            "description": spec.get("description", ""),
        }
    return info
