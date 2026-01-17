import os
from typing import Dict


PLOT_REGISTRY = {
    "package_tradeoff_scatter": {
        "files": {"package_compare": "package_tradeoff_scatter.png"},
        "scopes": ["package_compare"],
        "description": "Package-level episode scatter (risk vs welfare).",
    },
    "package_pareto_frontier": {
        "files": {"package_compare": "package_pareto_frontier.png"},
        "scopes": ["package_compare"],
        "description": "Package-level Pareto frontier.",
    },
    "package_tradeoff_pareto_by_param": {
        "files": {"package_compare": "package_tradeoff_pareto_by_param.png"},
        "scopes": ["package_compare"],
        "description": "Package-level param Pareto frontier.",
    },
    "exp_tradeoff_pareto": {
        "files": {"suite_exp_compare": "exp_tradeoff_pareto.png"},
        "scopes": ["suite_exp_compare"],
        "description": "Suite episode-level tradeoff Pareto.",
    },
    "exp_tradeoff_scatter": {
        "files": {"suite_exp_compare": "exp_tradeoff_scatter.png"},
        "scopes": ["suite_exp_compare"],
        "description": "Suite episode-level tradeoff scatter.",
    },
    "exp_pareto_frontier": {
        "files": {"suite_exp_compare": "exp_pareto_frontier.png"},
        "scopes": ["suite_exp_compare"],
        "description": "Suite per-episode Pareto frontier.",
    },
    "exp_time_to_tau_cdf": {
        "files": {"suite_exp_compare": "exp_time_to_tau_cdf.png"},
        "scopes": ["suite_exp_compare"],
        "description": "Suite time-to-tau ECDF.",
    },
    "exp_reach_tau_rate": {
        "files": {"suite_exp_compare": "exp_reach_tau_rate.png"},
        "scopes": ["suite_exp_compare"],
        "description": "Suite reach-tau rate by exp.",
    },
    "exp_gap_true_vs_observed_t": {
        "files": {"suite_exp_compare": "exp_gap_true_vs_observed_t.png"},
        "scopes": ["suite_exp_compare"],
        "description": "True vs observed gap over time.",
    },
    "exp_dormant_stock_t": {
        "files": {"suite_exp_compare": "exp_dormant_stock_t.png"},
        "scopes": ["suite_exp_compare"],
        "description": "Dormant stock mass over time.",
    },
    "exp_gov_outcomes": {
        "files": {"suite_exp_compare": "exp_gov_outcomes.png"},
        "scopes": ["suite_exp_compare"],
        "description": "Governance outcomes by exp.",
    },
    "sweep_time_to_tau": {
        "files": {"sweep_compare": "sweep_time_to_tau.png"},
        "scopes": ["sweep_compare"],
        "description": "Sweep time-to-tau vs param.",
    },
    "sweep_tradeoff_scatter_colored": {
        "files": {"sweep_compare": "sweep_tradeoff_scatter_colored.png"},
        "scopes": ["sweep_compare"],
        "description": "Sweep tradeoff scatter colored by param.",
    },
    "sweep_gap_divergence": {
        "files": {"sweep_compare": "sweep_gap_divergence.png"},
        "scopes": ["sweep_compare"],
        "description": "Sweep true vs observed gap divergence.",
    },
    "sweep_tradeoff_pareto_by_param": {
        "files": {"sweep_compare": "tradeoff_pareto_by_param.png"},
        "scopes": ["sweep_compare"],
        "description": "Sweep param Pareto frontier.",
    },
}

PLOT_ALIASES = {
    "tradeoff_scatter": "package_tradeoff_scatter",
    "tradeoff_pareto": "package_pareto_frontier",
    "tradeoff_pareto_by_param": "package_tradeoff_pareto_by_param",
    "time_to_tau_cdf": "exp_time_to_tau_cdf",
    "reach_tau_rate": "exp_reach_tau_rate",
    "gap_true_vs_observed_t": "exp_gap_true_vs_observed_t",
    "dormant_stock_t": "exp_dormant_stock_t",
    "gov_outcomes": "exp_gov_outcomes",
    "time_to_tau": "sweep_time_to_tau",
    "gap_divergence": "sweep_gap_divergence",
    "sweep_tradeoff_pareto_by_param": "sweep_tradeoff_pareto_by_param",
}

TABLE_REGISTRY = {
    "package_manifest_json": {
        "filename": "package_manifest.json",
        "scopes": ["package"],
        "description": "Package run manifest.",
    },
    "combined_episode_points_csv": {
        "filename": "combined_episode_points.csv",
        "scopes": ["package"],
        "description": "Combined episode points across jobs.",
    },
    "package_summary_csv": {
        "filename": "package_summary.csv",
        "scopes": ["package"],
        "description": "Package summary metrics.",
    },
    "package_summary_json": {
        "filename": "package_summary.json",
        "scopes": ["package"],
        "description": "Package summary metrics (JSON).",
    },
    "analysis_manifest_json": {
        "filename": "analysis_manifest.json",
        "scopes": ["package", "job"],
        "description": "Analysis manifest of requested/produced artifacts.",
    },
    "episode_points_csv": {
        "filename": "episode_points.csv",
        "scopes": ["suite_job", "sweep_job"],
        "description": "Episode points for a job (suite or sweep).",
    },
    "suite_episode_points_csv": {
        "filename": "episode_points.csv",
        "scopes": ["suite_job"],
        "description": "Suite episode points (per job).",
    },
    "sweep_summary_csv": {
        "filename": "sweep_summary.csv",
        "scopes": ["sweep_job"],
        "description": "Sweep summary CSV (per job).",
    },
    "sweep_summary_json": {
        "filename": "sweep_summary.json",
        "scopes": ["sweep_job"],
        "description": "Sweep summary JSON (per job).",
    },
}


def resolve_plot_id(plot_id: str) -> str:
    return PLOT_ALIASES.get(plot_id, plot_id)


def describe_plot_registry() -> Dict[str, Dict[str, str]]:
    info = {}
    for plot_id, spec in sorted(PLOT_REGISTRY.items()):
        info[plot_id] = {
            "filename": ",".join(f"{k}:{v}" for k, v in spec["files"].items()),
            "scopes": ",".join(spec["scopes"]),
            "description": spec["description"],
        }
    return info


def describe_table_registry() -> Dict[str, Dict[str, str]]:
    info = {}
    for table_id, spec in sorted(TABLE_REGISTRY.items()):
        info[table_id] = {
            "filename": spec["filename"],
            "scopes": ",".join(spec["scopes"]),
            "description": spec["description"],
        }
    return info
