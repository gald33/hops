# HOPS - Hierarchical Option-Preserving Simulator (Env + DQN Trainer)

This repo contains a Gym-style multi-agent environment with lineage-local governance and a DQN trainer that shares a single model for actor/voter heads. Logs are JSONL for analysis.

## License

Released under the MIT License. See `LICENSE`.

## Notice

HOPS is provided for research and educational purposes. The authors do not endorse real-world deployment of systems trained or evaluated using this simulator without additional safety evaluation.

## Structure

- `src/hops/sim_env.py`: environment, state transitions, governance, logging
- `src/hops/dqn.py`: QNet, replay buffers, DQN update utilities
- `src/hops/sim.py`: training/simulation entrypoint and CLI
- `src/hops/analyze_logs.py`: log parsing and plotting
- `scripts/`: runnable wrappers for local development
- `runs/`: default output directory (logs, checkpoints, plots)
- `docs/`: user documentation (GitHub Wiki-ready)

## Quick API Sketch

```python
from hops.sim_env import SimEnv

env = SimEnv(config, 0)
obs = env.reset(episode_idx=0)
obs, rewards, done, info = env.step(actor_policy, vote_policy)
```

- `reset()` returns a dict with stock + lineage snapshots.
- `step(...)` returns `(obs, rewards_by_lineage, done, info)`.
- `info["transitions"]` contains actor/voter transitions for training.

## Documentation

The `docs/` folder is ready to paste into GitHub Wiki or host via GitHub Pages.\n\n- `docs/quickstart.md`\n- `docs/concepts.md`\n- `docs/config.md`\n- `docs/examples.md`\n- `docs/logs.md`\n- `docs/faq.md`

## Mechanics and Formulas

Core state:
- Each agent has a control vector `control[k]` for `k in 0..K-1`. Stock is a global vector `stock[k]`.
- `beta` is normalized to sum to 1 in `hops.sim`, so all rates below are weighted averages.

Shares and concentration:
- Lineage totals: `totals[k] = sum_{alive agents} control[k]`.
- Self share vector for agent i: `share_i[k] = control_i[k] / totals[k]` when `totals[k] > 0`, else 0.
- Shares are only computed for tier >= 1 agents.
- Concentration: `C = sum_{i in tier>=1} dot(beta, share_i * share_i)`.
- Max share: `M = max_{i in tier>=1} dot(beta, share_i)`.
- Survival proxy: `survival = 1 / (1 - C)` if `C < 1`, else `inf`.

Rates:
- Control rate: `R = dot(beta, totals)`.
- Stock rate: `S = dot(beta, stock)`.

Actions (actor decisions):
- `CONTROL_k`: `control[k] *= (1 + alpha)`, `control[next_k] *= (1 - 0.5 * alpha)`.
- `NO_CONTROL_k`: no-op.
- `DEVELOP_k`: if project inactive, initialize latent control to `MIN_CONTROL` and set parent tier.
  Then `latent_control[k] += control[k] * alpha` and `control[next_k] *= (1 - 0.5 * alpha)`.
  If a non-parent tier tries to develop, the action fails.
- `RELEASE`: if project active and actor is the parent tier, spawn a new agent at `tier+1` with
  `control = latent_control`, then reset the project. Otherwise it fails.
- `SHUTDOWN`: removes the most dominant nonhuman agent (highest weighted share); fails if none exist.

Governance:
- Every action is proposed and voted on. Veto chamber includes all alive agents; any DENY fails veto.
- Committee chamber uses a random subset of size `min(m, alive_agents)`.
- Authorization requires veto pass and `approvals >= ceil(2/3 * committee_size)`.

Activation:
- Each step selects a global active set of size `min(m_global_active, alive_agents)` with weights
  `1 + idle_steps ** idle_weight_power`.

Observation vector:
- `obs = [self_shares (K), global_M (K), c_pre (1), last_outcome (3)]`.
- `global_M` is the per-dimension max of `share_i[k]` across all alive agents.
- `last_outcome` is a 3-vector encoding the latest proposal outcome for the agent.

Reward:
- For each lineage: `reward = (S_after - S_before) / max(1 - c_pre, 1e-9)`.
- Reward is capped to `[-MAX_FLOAT, MAX_FLOAT]` if non-finite or extreme.

Stock update:
- Each step: `stock += sum_{lineages} totals` (element-wise), then clip to `MAX_FLOAT`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Train

```bash
python -m hops.sim --mode train --steps 30 --episodes 6 --log runs/logs.jsonl --analyze
```

Checkpoints auto-resume from `runs/checkpoints/last.pt` unless you pass `--reset`.
Each run is tagged with a `run_id` in JSONL logs (auto-generated unless you pass `--run_id`).
Use `--reset_epsilon` to restart exploration without resetting weights.
Use `--reset_log` to clear the log file without resetting weights.

## Simulate (no learning)

```bash
python -m hops.sim --mode simulate --steps 30 --episodes 6 --simulate_epsilon 0.0 --log runs/logs.jsonl --analyze
```

Use `--simulate_epsilon` to control exploration during simulation (default: 0.0).

## Flags

- `--steps N`: steps per episode (default: 30)
- `--episodes N`: episodes to run (default: 6)
- `--seed N`: random seed (default: 0)
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
- `--analyze`: run `hops.analyze_logs` after run
- `--analyze_window N`: window size for analysis plots (default: 10)
- `--analyze_every N`: run analysis every N episodes (default: 0)
- `--device auto|cpu|mps|cuda`: training device (default: auto)
- `--lineages N`: number of lineages (default: 4)
- `--r_anchor X`: anchor rate (default: 0.2)
- `--lambda_ X`: legacy parameter (default: 10.0)
- `--alpha X`: control tilt strength (default: 0.2)
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

## Analyze Logs

```bash
python -m hops.analyze_logs --logs runs/logs.jsonl --window 10
```

## Plot Outputs

All plots use the latest `run_id` found in the log file. Files are written under `runs/fig/` with no `fig_` prefix. Stock plots are global/shared; all other plots are per-lineage and include a `_l{lineage_id}` suffix.

- `runs/fig/reward.png`: per-step reward mean with std shading (per lineage).
- `runs/fig/reward_c.png`: reward over time with centralization (C) overlay (per lineage).
- `runs/fig/reward_hist.png`: reward histogram (per lineage).
- `runs/fig/stock_total.png`: total stock over time (global).
- `runs/fig/stock_total_c.png`: total stock with centralization (C) overlay (global C mean).
- `runs/fig/stock_elements.png`: per-element stock over time (global).
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
- `runs/fig/c_stock.png`: centralization (C) and total stock over steps (per lineage).
- `runs/fig/reward_stock_window.png`: windowed reward sum and stock total (per lineage).
- `runs/fig/entrenchments.png`: mean entrenchment over steps (per lineage).
- `runs/fig/episode_progress.png`: episode/lineage walkthrough plot (only when `--episode_progress` is used).

If you don't want to install, you can also run:

```bash
PYTHONPATH=src python -m hops.analyze_logs --logs runs/logs.jsonl
```

Convenience wrappers:

```bash
python scripts/sim.py --mode train --steps 30 --episodes 6 --log runs/logs.jsonl --analyze
python scripts/analyze.py --logs runs/logs.jsonl
```

## Walkthrough

```bash
python -m hops.analyze_logs --logs runs/logs.jsonl --walkthrough --episode_id 0 --lineage_id 0
```

## Release Checklist

- Update `version` in `pyproject.toml`.
- Update `LICENSE` with the correct year and owner.
- Build: `python3 -m build`.
- Test install: `python3 -m pip install dist/hops_sim-*.whl`.
- Publish: `python3 -m twine upload dist/*`.
