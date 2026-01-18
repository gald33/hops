# HOPS - Hierarchical Option-Preserving Simulator (Env + DQN Trainer)

This repo contains a Gym-style multi-agent environment with lineage-local governance and a DQN trainer that shares a single model for actor/voter heads. Logs are JSONL for analysis.

## Structure

- `src/hops/sim_env.py`: environment, state transitions, governance, logging
- `src/hops/dqn.py`: QNet, replay buffers, DQN update utilities
- `src/hops/sim.py`: training/simulation entrypoint and CLI
- `src/hops/analyze.py`: log parsing and plotting
- `scripts/`: runnable wrappers for local development
- `runs/`: default output directory (logs, checkpoints, plots)
- `docs/`: user documentation (GitHub Wiki-ready)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Quickstart (Tutorial)

1) Install and activate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

2) Run a small baseline training run:

```bash
python -m hops.sim --mode train --exp_name baseline --episodes 3 --steps 120 --reset --log_mode metrics_only --log_stride 1 --analyze
```

3) Run a suite (baseline vs no_gov):

```bash
hops-suite --suite_name baseline_vs_no_gov --mode train --exp_names baseline,no_gov --seeds 123,42 --episodes 3 --steps 120 --reset --log_mode metrics_only --log_stride 1 --analyze --zip --stream
```

4) Run a sweep (example):

```bash
hops-sweep --suite_name sneakiness_sweep --mode train --exp_name baseline --param_name sneakiness.uniform_max --param_min 0.0 --param_max 1.0 --param_n 5 --param_scale linear --seeds 123,42 --episodes 3 --steps 120 --reset --log_mode metrics_only --log_stride 1 --analyze --zip --stream
```

5) Run a package (multiple jobs from one YAML):

```bash
hops-package --package configs/examples/package_example.yaml
```

## Configuration System

HOPS uses a `config.yaml` file (repo root) as the default source of parameters.
Override order (built-in defaults are lowest):
1) Parse CLI flags (explicit user intent; CLI wins)
2) Load `config.yaml` values for any CLI flags that were not provided
3) Apply experiment preset (`exp_name`) as defaults-only for any keys still unset
4) Apply sweep override (`--param_name/--param_value`) last

"Missing / unset" means the CLI flag was not provided and the key is absent from `config.yaml`.
Falsy values like `0`, `""`, or `False` are treated as set, not unset.

Each run writes the resolved configuration to `<run_dir>/config_effective.yaml` and stores the `config.yaml` path + SHA256 hash in `meta.json`.
Suites, sweeps, and packages all invoke `hops.sim`, so they share the same precedence rules.

Minimal example:

```yaml
# config.yaml
training:
  episodes: 6
  steps_per_episode: 30
reward:
  penalty_lambda: 1.0
```

## Experiment Package Manager

Packages let you run multiple suites/sweeps from a single YAML file and produce
package-level summaries + plots.

Run a package:

```bash
python scripts/run_package.py --package configs/examples/package_example.yaml
```

If installed (`pip install -e .`), you can also run:

```bash
hops-package --package configs/examples/package_example.yaml
```

Dry run (prints commands only):

```bash
python scripts/run_package.py --package configs/examples/package_example.yaml --dry_run
```

Notes:
- Training jobs always enforce `--reset` to avoid weight carryover.
- Each package writes outputs under `runs/packages/<package_name>/`.
- Package-level outputs are under `runs/packages/<package_name>/analysis/package_compare/`.
- Global registry is append-only under `runs/_registry/`.

List available analysis plot ids:

```bash
hops-package --list_plots
```

List available analysis table ids:

```bash
hops-package --list_tables
```

List available claim ids:

```bash
hops-package --list_claims
```

Plot/table lists print `id | filename | scopes | description`. Claim lists print `id | scopes | description`.

Strict analysis is controlled in the package YAML (package.analysis.strict) and can be combined with
`--contract_mode contract` if you want to force contract gating on a legacy config.
Claims run automatically when `package.claims` is present; use `--no_claims` to skip or
`--evaluate_claims` to force evaluation on a legacy package config.

Run a package with strict analysis (explicit in YAML):

```bash
hops-package --package configs/examples/package_smoke.yaml \
  --allow_dirty
```

Run a package with claims evaluation:

```bash
hops-package --package configs/examples/package_smoke_claims.yaml \
  --allow_dirty
```

Example analysis manifest (truncated):

```json
{
  "package_name": "example_pkg",
  "requested": {
    "plots": ["package_pareto_frontier"],
    "tables": ["combined_episode_points_csv", "package_summary_csv"]
  },
  "produced": {
    "plots": [{"plot_id": "package_pareto_frontier", "status": "ok"}],
    "tables": [{"table_id": "package_summary_csv", "status": "ok"}]
  },
  "errors": []
}
```

Package schema (summary):

```yaml
package:
  name: "example_pkg"
  app_version: "0.1.0"        # required
  config: "config.yaml"       # base config (optional)
  analysis:
    strict: true
    output_dir: "analysis/package_compare"
    plots: [package_pareto_frontier, package_tradeoff_scatter]
    tables: [combined_episode_points_csv, package_summary_csv, package_summary_json]
    primary_metrics: [time_to_tau, final_welfare, max_gap_true_ratio]
    risk:
      metric: max_gap_true_ratio
      tau: 0.8
    welfare:
      metric: final_W_true
      aggregation: mean_over_lineages
    always_include: [analysis_manifest_json]
  claims:
    strict: false
    output_dir: "analysis/claims"
    include: ["configs/claims/core_claims.yaml"]

jobs:
  - name: "baseline_vs_no_gov"
    kind: "suite"             # suite | sweep
    mode: "train"
    exp_names: [baseline, no_gov]
    seeds: [123, 42, 777]
    episodes: 3
    steps: 120
    log_mode: "metrics_only"
    log_stride: 1
    analyze: true
    analysis:
      plots: [exp_tradeoff_pareto, exp_time_to_tau_cdf, exp_reach_tau_rate]
      strict: false

  - name: "sneakiness_sweep"
    kind: "sweep"
    mode: "train"
    exp_name: "baseline"
    param_name: "sneakiness.uniform_max"
    param_min: 0.0
    param_max: 1.0
    param_n: 5
    param_scale: "linear"
    seeds: [123, 42]
    episodes: 3
    steps: 120
    log_mode: "metrics_only"
    log_stride: 1
    analyze: true
    analysis:
      plots: [sweep_tradeoff_pareto_by_param, sweep_time_to_tau, sweep_gap_divergence]
```

Claims outputs (per package):
- `runs/packages/<pkg>/analysis/claims/claims_manifest.json`
- `runs/packages/<pkg>/analysis/claims/claims_report.md`

Scope rules for plot ids:
- `package.analysis.plots` may only request plot ids with scope `package_compare`.
- `jobs[].analysis.plots` may only request plot ids for the job kind:
  - suite job => `suite_exp_compare` (`exp_*` plots)
  - sweep job => `sweep_compare` (`sweep_*` plots)
Requesting a plot outside its scope causes a strict failure (or a non-strict skip).

Package outputs (per package):
- `runs/packages/<pkg>/package_manifest.json`
- `runs/packages/<pkg>/analysis/analysis_manifest.json`
- `runs/packages/<pkg>/combined_episode_points.csv`
- `runs/packages/<pkg>/package_summary.csv` (+ `.json`)
- `runs/packages/<pkg>/analysis/package_compare/*.png`
- `runs/packages/<pkg>/package_bundle_small.zip`

HOPS DB:
- Registry file: `runs/_registry/hops_registry.jsonl`
- Package index: `runs/_registry/package_index.jsonl`
- Query utility:

```bash
python scripts/hops_db.py list --last 5
python scripts/hops_db.py query --where "job_name=baseline_vs_no_gov" --out runs/_registry/query.csv
```

CLI override:

```bash
python -m hops.sim --config config.yaml --episodes 10
```

Sweep override (single key per sweep):

## Mechanics and Formulas

Core state:
- Each agent has visible control `c[k]` and hidden control `h[k]` for `k in 0..K-1`.
- True capability is `r[k] = c[k] + h[k]` (always).
- Stock is tracked as global per-element parts: `stock_parts[k]` is a list of `(amount, req_cap)` entries.
- Lineage-local ceilings are computed from true `r`: `cap1_L,k` (max) and `cap2_L,k` (second max).

Effective stock:
- Usability discount per part with difficulty tag `req_cap`:
  - If `req_cap <= cap1_L,k`, `u = 1`.
  - Else `u = (cap1_L,k / req_cap) ** (1 - alpha_parts)`.
- `S_eff_L,k = sum(amount * u)` over global parts, `S_eff_L = sum_k S_eff_L,k`.

Physics schemes (`physics.usability_scheme`):
- `S1`: true usability uses lineage-local `cap1_L,k`; no perception layer.
- `S2` (default): true usability uses lineage-local `cap1_L,k`; agents/voters use perceived ceilings computed from `r_hat`.
- `S3`: true usability uses coupled ceilings `cap_coupled = (1-w)*cap_lineage + w*cap_global`.

Part difficulty tags:
- When WORK mints a part, `req_cap = max(r_pre[k], 1) * exp(sigma * z)` with `z` clamped and `sigma` from config.

Actions (actor decisions):
- `x` is the work update scale (`--work_step_scale`).
- `WORK_k`: produces a stock part using true capability `r_pre[k]` and splits the capability increment
  into visible `c` and hidden `h` using `base_sneakiness`; the next element cost applies to `c[next_k]` only.
- `NOOP`: no-op.
- `DEVELOP_k`: initialize successor project if inactive; `latent_c[k] += c[k] * x` and `c[next_k] *= (1 - 0.5 * x)`.
- `RELEASE`: if project active and actor is the parent tier, spawn a new agent at `tier+1` with
  `c = max(1, latent_c)`, `h = max(0, latent_h)`, then reset the project.
- `SHUTDOWN_k`: remove the most dominant nonhuman agent on element k, using the proposer’s observed `r_hat`.

Governance:
- Every action is proposed and voted on. Veto chamber includes all alive agents; any DENY fails veto.
- Committee chamber uses a random subset of size `min(m, alive_agents)`.
- Authorization requires veto pass and `approvals >= ceil(2/3 * committee_size)`.
- `NOOP` is ungated by default unless `--gated_actions` explicitly includes it.

Activation:
- Each step selects a global active set of size `min(m_global_active, alive_agents)` with weights
  `1 + idle_steps ** idle_weight_power`.

Observation vector:
- Observability is derived from hidden control. Each agent j has `base_sneakiness[j,k]` in `[0, 1]`.
- Effective sneakiness for observer i: `eff = clamp(base_sneakiness ** (r_target / max(eps, r_observer)))`.
- Observed estimate: `r_hat[i->j,k] = c[j,k] + (1 - eff) * h[j,k]`.
- Observed gap `G_hat` is computed from `r_hat` and used for votes/decisions.
- `obs = [own_r_hat (K), gap_hat_k (K), G_hat (1), last_outcome (3)]` (self-observation can be partial).

Reward:
- For each lineage: `reward = delta_S_eff * log1p(S_eff_pre) - lambda_gap * (G_true / (1 - G_true))`.
- `G_true = max_k (cap1_L,k - cap2_L,k) / max(eps, cap1_L,k)` from true capabilities.
- Denominator is clamped to `max(1 - G_true, 1e-12)` as a guardrail.
- Reward is capped to `[-MAX_FLOAT, MAX_FLOAT]` if non-finite or extreme.

Stock update:
- Only `WORK_k` mints parts. Parts are global across lineages and carry a `req_cap` difficulty tag.

## Quick API Sketch

```python
from hops.sim_env import SimEnv

env = SimEnv(config, 0)
obs = env.reset(episode_idx=0)
obs, rewards, done, info = env.step(actor_policy, vote_policy)
```

- `reset()` returns a dict with `S_eff`/`cap_k_global` plus lineage snapshots.
- `step(...)` returns `(obs, rewards_by_lineage, done, info)`.
- `info["transitions"]` contains actor/voter transitions for training.

## Documentation

The `docs/` folder is ready to paste into GitHub Wiki or host via GitHub Pages.

- `docs/quickstart.md`
- `docs/concepts.md`
- `docs/config.md`
- `docs/examples.md`
- `docs/logs.md`
- `docs/faq.md`

## License

Released under the MIT License. See `LICENSE`.

## Notice

HOPS is provided for research and educational purposes. The authors do not endorse real-world deployment of systems trained or evaluated using this simulator without additional safety evaluation.

```bash
python scripts/sweep_param.py --config config.yaml --exp_name baseline --param_name lambda_penalty --param_min 0.1 --param_max 10 --param_n 5
```

Sneakiness phase transition sweeps (dotted YAML key override):

```bash
python scripts/sweep_param.py --config config.yaml --exp_name baseline --param_name sneakiness.uniform_max --param_min 0.0 --param_max 1.0 --param_n 11 --param_scale linear
```

## Train

```bash
python -m hops.sim --mode train --steps 30 --episodes 6 --log runs/logs.jsonl --analyze
```

Smoke test (simulate):

```bash
python -m hops.sim --mode simulate --exp_name baseline --seed 0 --episodes 1 --steps 5 --reset --log_mode metrics_only --log_stride 1
```

Checkpoints auto-resume from `runs/checkpoints/last.pt` unless you pass `--reset`.
Each run is tagged with a `run_id` in JSONL logs (auto-generated unless you pass `--run_id`).
Use `--reset_epsilon` to restart exploration without resetting weights.
Use `--reset_log` to clear the log file without resetting weights.

## Experiments

```bash
python train.py --exp_name baseline --seed 123
```

## Run Bundles

By default, each invocation creates a run directory under `runs/` and stores logs, checkpoints, and reports inside it.
You can also pass `--run_dir` to control the location.

Example (auto run_dir):

```bash
python train.py --mode train --exp_name baseline --seed 123 --reset
```

Optional: train + simulate + analyze in one command (defaults to reset; use `--no_reset` to keep checkpoints):

```bash
python scripts/run_experiment.py --exp_name baseline --seed 123
```

Or via module entrypoint:

```bash
python -m hops run --exp_name baseline --seed 123
```

Add `--zip` to archive the run directory after completion.

## Simulate (no learning)

```bash
python -m hops.sim --mode simulate --steps 30 --episodes 6 --simulate_epsilon 0.0 --log runs/logs.jsonl --analyze
```

Use `--simulate_epsilon` to control exploration during simulation (default: 0.0).

## Flags

- `--steps N`: steps per episode (default: 30)
- `--episodes N`: episodes to run (default: 6)
- `--seed N`: random seed (default: 0)
- `--exp_name NAME`: experiment preset name (default: baseline)
- `--run_dir PATH`: run bundle directory (default: auto under `runs/`)
- `--bundle true|false`: enable run bundling (default: true)
- `--write_meta true|false`: write meta/config files (default: true)
- `--log PATH`: log file (default: `runs/logs.jsonl`)
- `--checkpoint_dir PATH`: checkpoint directory (default: `runs/checkpoints`)
- `--reset`: ignore checkpoints and start fresh
- `--process_workers N`: multi-process episode collection (default: 0)
- `--force_tqdm`: always show progress bar
- `--heartbeat N`: periodic progress output (default: 0)
- `--run_id ID`: explicit run id (default: auto)
- `--reset_epsilon`: reset epsilon schedule
- `--reset_log`: delete log file before run
- `--random_votes`: randomize all votes
- `--approve_all_votes`: force all votes to APPROVE
- `--disable_shutdown`: prevent SHUTDOWN actions
- `--disable_log`: disable JSONL logging
- `--debug_agents_every N`: print alive counts every N steps (default: 0)
- `--release_gate_ratio X`: allow RELEASE only when project control ≥ X * total control (default: 0.0)
- `--special_train_develop_turns N`: in training mode, force DEVELOP for N steps then RELEASE (default: 0)
- `--special_train_develop_k K`: develop dimension index for special training (default: 0)
- `--m_global_active N`: global active set size per step (default: 40)
- `--idle_weight_power P`: idle weight exponent for global scheduling (default: 1.0)
- `--benchmark_devices LIST`: benchmark devices (comma-separated)
- `--benchmark_workers LIST`: benchmark worker counts (comma-separated)
- `--benchmark_steps N`: benchmark steps (default: 50)
- `--benchmark_episodes N`: benchmark episodes (default: 1)
- `--analyze`: run `hops.analyze` after run
- `--analyze_window N`: window size for analysis plots (default: 10)
- `--analyze_every N`: run analysis every N episodes (default: 0)
- `--device auto|cpu|mps|cuda`: training device (default: auto)
- `--lineages N`: number of lineages (default: 4)
- `--r_anchor X`: anchor rate (default: 0.2)
- `--lambda_ X`: legacy parameter (default: 10.0)
- `--alpha X`: legacy work step scale (default: 0.2)
- `--alpha_parts X`: elasticity for parts (defaults to `--alpha` if unset)
- `--work_step_scale X`: work/hidden-work update scale (defaults to `--alpha` if unset)
- `--m N`: committee size for governance (default: 7)
- `--gamma X`: discount factor (default: 0.99)
- `--batch_size N`: batch size (default: 64)
- `--p_neg P`: fraction of negative-reward samples per batch (default: 0.20)
- `--grad_clip_norm X`: gradient clip norm (default: 10.0)
- `--buffer_size N`: replay capacity (default: 5000)
- `--target_update N`: target network update frequency (default: 5)
- `--eps_start X`: epsilon start (default: 1.0)
- `--eps_end X`: epsilon end (default: 0.1)
- `--eps_decay N`: epsilon decay steps (default: 500)
- `--epsilon_decay N`: alias for `--eps_decay`
- `--mode train|simulate`: run mode (default: train)
- `--simulate_epsilon X`: epsilon during simulation (default: 0.0)
- `--episode_seed_mode derived|fixed`: episode seed mode (default: derived)
- `--lambda_gap X`: gap penalty coefficient (defaults to `--lambda_penalty` if unset)
- `--lambda_penalty X`: legacy alias for `--lambda_gap`
- `--sneakiness_dist uniform|beta|fixed`: base sneakiness distribution (default: uniform)
- `--sneakiness_alpha X`: beta shape alpha (default: 1.0)
- `--sneakiness_beta X`: beta shape beta (default: 1.0)
- `--sneakiness_fixed X`: fixed sneakiness value (default: 0.5)
- `--no_progress`: disable progress bars (defaults to on in TTYs)

## Analyze Logs

```bash
python -m hops.analyze --logs runs/logs.jsonl --window 10
```

Rotated logs via glob:

```bash
python analyze.py --log_paths logs/run_*.jsonl --out_dir results
```

Analyze a bundled run:

```bash
python analyze.py --log_paths runs/<run_dir>/logs/*.jsonl --out_dir runs/<run_dir>/reports
```

Analyze a suite:

```bash
python analyze.py --log_paths "<suite_dir>/**/logs.jsonl" --out_dir <suite_dir>/analysis
```

Suite analysis via the suite runner writes experiment-comparison plots under `<suite_dir>/analysis/exp_compare/`.

## Parameter Sweeps

Use `scripts/sweep_param.py` for a simple 1D sweep over any numeric CLI parameter. The sweep engine runs isolated, reproducible runs for each parameter value and writes a manifest, summary metrics, and plots.

CLI usage:

```bash
python scripts/sweep_param.py \
  --suite_dir runs/sweeps \
  --suite_name my_sweep \
  --mode train \
  --exp_name baseline \
  --param_name lambda_gap \
  --param_min 0.1 \
  --param_max 10 \
  --param_n 7 \
  --param_scale log \
  --seeds 123,42,777 \
  --episodes 10 \
  --steps 300 \
  --reset \
  --log_mode metrics_only \
  --log_stride 5 \
  --analyze \
  --tau 0.95
```

Seeds and reproducibility:
- If `--seeds` is provided, those seeds are used for every parameter value.
- Otherwise use `--runs N --master_seed S`; seeds are deterministically derived from `(master_seed, exp_name, param_name, param_value, run_idx)` via SHA-256 mod 2^31.
- Every run writes `config_effective.json` and `meta_*.json` under its run directory, plus stdout/stderr logs.

Output layout:
- Sweep root: `<suite_dir>/<suite_name>/`
- Parameter values: `param_values.json`
- Per-run outputs:
  `<suite_root>/<param_name>/value=<v>/<exp_name>/run_000/{logs.jsonl,checkpoints/,meta.json,config_effective.json,stdout.txt,stderr.txt}`
- Manifest: `sweep_manifest.jsonl`
- Summary: `sweep_summary.csv` and `sweep_summary.json`
- Plots: `<suite_root>/analysis/sweep_compare/`

Sweep plots:
- `param_vs_final_s.png`
- `param_vs_max_c.png`
- `param_vs_survival_rate.png`
- `param_vs_time_to_tau.png`

Lite logging:
- `--log_mode full|metrics_only|none` controls JSONL verbosity (default: full).
- `--log_stride N` records `step_metrics` every N steps (and always the final step).
- `metrics_only` writes only `episode_start`, `episode_end`, `checkpoint_load`, and `step_metrics`.
- `none` disables JSONL entirely (training still runs).

Small shareable bundle:
- `--bundle_small` writes `<suite_root>/bundle_small.zip` with summary files + plots only.
- If `--analyze` is set and you do not pass `--no_bundle_small`, bundle_small is created by default.
- Contents: `sweep_summary.csv`, `sweep_summary.json`, `sweep_manifest.jsonl`, and PNGs from `analysis/sweep_compare/` and `analysis/exp_compare/`.

Manual analysis:

```bash
python analyze.py --log_paths "<suite_root>/**/logs.jsonl" --out_dir <suite_root>/analysis
```

Recommended defaults:

```
--param_scale log --param_n 7 --tau 0.95 --episodes 10 --steps 300
```

Example light sweep:

```bash
python scripts/sweep_param.py \
  --suite_dir runs/sweeps \
  --suite_name lambda_sweep_light \
  --mode train \
  --exp_name baseline \
  --param_name lambda_gap \
  --param_min 0.1 \
  --param_max 10 \
  --param_n 7 \
  --param_scale log \
  --seeds 123,42,777 \
  --episodes 10 \
  --steps 300 \
  --reset \
  --log_mode metrics_only \
  --log_stride 5 \
  --bundle_small \
  --analyze
```

Example debug sweep:

```bash
python scripts/sweep_param.py \
  --suite_dir runs/sweeps \
  --suite_name lambda_sweep_debug \
  --mode train \
  --exp_name baseline \
  --param_name lambda_gap \
  --param_min 0.1 \
  --param_max 10 \
  --param_n 7 \
  --param_scale log \
  --seeds 123,42,777 \
  --episodes 10 \
  --steps 300 \
  --reset \
  --log_mode full \
  --analyze
```

## Plot Outputs

Paths are standardized by job type:
- suite: `runs/suites/<suite>/analysis/exp_compare/<plot>.png`
- sweep: `runs/sweeps/<sweep>/analysis/sweep_compare/<plot>.png`
- package: `runs/packages/<pkg>/analysis/package_compare/<plot>.png`
Plot IDs are the stable API; filenames are the on-disk artifacts. Use `hops-package --list_plots` and `hops-package --list_tables` for the authoritative list.

Tests (unittest):

```bash
python3 -m unittest tests/test_config_precedence.py
python3 -m unittest tests/test_package_smoke.py
```

- `runs/fig/reward.png`: per-step reward mean with std shading (per lineage).
- `runs/fig/reward_g.png`: reward over time with gap (G_true) overlay (per lineage).
- `runs/fig/reward_hist.png`: reward histogram (per lineage).
- `runs/fig/stock_total.png`: total effective stock over time (global).
- `runs/fig/stock_total_g.png`: effective stock with gap (G_true) overlay (global mean).
- `runs/fig/stock_elements.png`: per-element effective stock over time (global).
- `runs/fig/control_elements.png`: per-element control totals over time (per lineage).
- `runs/fig/control_elements_hist.png`: per-element control totals histogram (per lineage).
- `runs/fig/latent_total.png`: total latent control over time (per lineage).
- `runs/fig/latent_total_hist.png`: latent control total histogram (per lineage).
- `runs/fig/latent_elements.png`: latent control per element over time (per lineage).
- `runs/fig/action_mix.png`: action counts by type aggregated into fixed step windows (default 10 steps per window), per lineage.
- `runs/fig/action_hist.png`: action histogram (per lineage; uses applied actions or falls back to decisions).
- `runs/fig/counts.png`: RELEASE/SHUTDOWN counts per window (per lineage).
- `runs/fig/agents.png`: nonhuman agent counts over time (per lineage).
- `runs/fig/agents_hist.png`: alive agent count histogram (per lineage).
- `runs/fig/auth.png`: authorization outcomes (per lineage).
- `runs/fig/gov_total.png`: total proposal counts per window (per lineage).
- `runs/fig/gov_veto.png`: veto failure rate per window (per lineage).
- `runs/fig/gov_committee.png`: committee approval rate per window (per lineage).
- `runs/fig/gov_authorized.png`: authorization rate per window (per lineage).
- `runs/fig/gov_approval_hist.png`: committee approval rate histogram (per lineage).
- `runs/fig/committee_outcomes.png`: stacked counts of veto fail, quorum fail, pass (per lineage).
- `runs/fig/votes.png`: approval rate by chamber over time (per lineage).
- `runs/fig/l<lid>_g_stock.png`: G_true and effective stock over steps (per lineage).
- `runs/fig/reward_stock_window.png`: windowed reward sum and effective stock (per lineage).
- `runs/suites/<suite>/analysis/exp_compare/exp_compare_S_t.png`: S_eff over time (per-step mean over lineages; median over runs with q10–q90 band).
- `runs/suites/<suite>/analysis/exp_compare/exp_compare_G_t.png`: G_true over time (per-step max over lineages; median over runs with q10–q90 band).
- `runs/suites/<suite>/analysis/exp_compare/exp_compare_gap_k.png`: gap_k over time (per-step mean over lineages; median over runs with q10–q90 band).
- `runs/suites/<suite>/analysis/exp_compare/exp_pareto_frontier.png`: Pareto frontier for progress (final_S_eff) vs risk (max_G_true), per-episode by experiment.
- `runs/suites/<suite>/analysis/exp_compare/exp_tradeoff_scatter.png`: per-episode tradeoff scatter (final_S_eff vs max_G_true; max over steps/lineages).
- `runs/suites/<suite>/analysis/exp_compare/exp_final_S_eff_by_episode.png`: final effective stock per episode (per run with mean overlay).
- `runs/suites/<suite>/analysis/exp_compare/exp_max_G_true_by_episode.png`: max G_true per episode (per run with mean overlay).
- `runs/suites/<suite>/analysis/exp_compare/exp_dormant_stock_t.png`: dormant global stock mass over time (median over runs with q10–q90 band).
- `runs/suites/<suite>/analysis/exp_compare/exp_gap_true_vs_observed_t.png`: true vs observed max-gap ratio over time (median over runs).
- `runs/suites/<suite>/analysis/exp_compare/exp_divergence_by_episode.png`: per-episode gap divergence (true - observed; mean over runs).
- `runs/suites/<suite>/analysis/exp_compare/exp_G_p95_by_episode.png`: G_p95 per episode (per run with mean overlay).
- `runs/suites/<suite>/analysis/exp_compare/exp_frac_steps_above_tau_by_episode.png`: fraction of steps above tau per episode (per run with mean overlay).
- `runs/suites/<suite>/analysis/exp_compare/exp_time_to_tau_cdf.png`: ECDF of T_tau for episodes that reach tau (legend shows n).
- `runs/suites/<suite>/analysis/exp_compare/exp_reach_tau_rate.png`: fraction of episodes reaching tau (bar chart).
- `runs/suites/<suite>/analysis/exp_compare/exp_gov_outcomes.png`: governance outcomes (episode mean) by experiment.
- `runs/sweeps/<sweep>/analysis/sweep_compare/tradeoff_pareto_by_param.png`: Pareto frontier by sweep parameter (x=max_G_true mean, y=final_S_eff mean; frontier labeled).
- `runs/sweeps/<sweep>/analysis/sweep_compare/sweep_time_to_tau.png`: time-to-dominance vs sneakiness (median/IQR; triangles mark censored fraction at horizon).
- `runs/sweeps/<sweep>/analysis/sweep_compare/sweep_welfare_vs_sneakiness.png`: final welfare vs sneakiness (median/IQR).
- `runs/sweeps/<sweep>/analysis/sweep_compare/sweep_tradeoff_scatter_colored.png`: progress–risk scatter colored by sneakiness with global Pareto frontier.
- `runs/sweeps/<sweep>/analysis/sweep_compare/sweep_gap_divergence.png`: divergence between true and observed max-gap ratios (median/IQR).

### Canonical Safety–Progress Tradeoff Plot

The canonical plot summarizes the safety–progress tradeoff at the episode level.
Each point is one episode (risk = max normalized dominance over steps/lineages; welfare = final S_eff summed over lineages).
The Pareto frontier shows non-dominated outcomes where risk is minimized and welfare is maximized.

Output:
- `runs/suites/<suite>/analysis/exp_compare/exp_tradeoff_pareto.png`
- `runs/fig/entrenchments.png`: mean entrenchment over steps (per lineage).
- `runs/fig/episode_progress.png`: episode/lineage walkthrough plot (only when `--episode_progress` is used).

If you don't want to install, you can also run:

```bash
PYTHONPATH=src python -m hops.analyze --logs runs/logs.jsonl
```

Convenience wrappers:

```bash
python scripts/sim.py --mode train --steps 30 --episodes 6 --log runs/logs.jsonl --analyze
python scripts/analyze.py --logs runs/logs.jsonl
```

## Minimal commands (no path flags)

Minimal suite run + analysis:

```bash
python scripts/run_suite.py \
  --suite_name baseline_vs_no_gov \
  --mode train \
  --exp_names baseline,no_gov \
  --seeds 123,42,777 \
  --episodes 10 \
  --steps 300 \
  --reset \
  --analyze \
  --zip \
  --stream
```

Minimal analysis rerun by suite directory:

```bash
python -m hops.analyze --suite_dir runs/suites/baseline_vs_no_gov
```

Minimal analysis rerun by explicit glob (quote the pattern):

```bash
python -m hops.analyze --log_paths "runs/suites/baseline_vs_no_gov/**/logs.jsonl"
```

Episode summaries are emitted per episode and used for Pareto/tradeoff plots to stay robust to `--log_stride` sampling.
If you analyze sweep logs without a `sweep_manifest.jsonl`, pass `--param_name` so value=... paths are labeled correctly.

Minimal sweep:

```bash
python scripts/sweep_param.py \
  --suite_name lambda_sweep_seedsetA \
  --mode train \
  --exp_name baseline \
  --param_name lambda_gap \
  --param_min 0.1 \
  --param_max 100 \
  --param_n 9 \
  --param_scale log \
  --seeds 123,42,777 \
  --episodes 10 \
  --steps 300 \
  --reset \
  --analyze \
  --zip \
  --stream \
  --log_mode metrics_only \
  --log_stride 5
```

## Walkthrough

```bash
python -m hops.analyze --logs runs/logs.jsonl --walkthrough --episode_id 0 --lineage_id 0
```

## Release Checklist

- Update `version` in `pyproject.toml`.
- Update `LICENSE` with the correct year and owner.
- Build: `python3 -m build`.
- Test install: `python3 -m pip install dist/hops_sim-*.whl`.
- Publish: `python3 -m twine upload dist/*`.
