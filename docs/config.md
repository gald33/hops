# Configuration

The simulator is configured entirely via CLI flags on `python -m hops.sim`.
Defaults are listed in the main README and in `src/hops/sim.py`.

## Common flags

- `--steps N`: steps per episode
- `--episodes N`: number of episodes
- `--mode train|simulate`: training or simulation
- `--simulate_epsilon X`: exploration during simulate
- `--seed N`: random seed
- `--device auto|cpu|mps|cuda`: training device
- `--lineages N`: number of lineages
- `--r_anchor X`: anchor rate
- `--alpha X`: control tilt strength
- `--gamma X`: discount factor
- `--batch_size N`: batch size
- `--buffer_size N`: replay capacity
- `--target_update N`: target network update frequency

## Governance flags

- `--m N`: committee size
- `--m_global_active N`: global active set size per step
- `--idle_weight_power P`: idle weight exponent
- `--random_votes`: randomize all votes
- `--approve_all_votes`: force all votes to APPROVE
- `--disable_shutdown`: prevent SHUTDOWN actions
- `--release_gate_ratio X`: allow RELEASE only when project control >= X * total control

## Logging and analysis

- `--log PATH`: JSONL log path
- `--disable_log`: disable logging
- `--reset_log`: clear log file before run
- `--analyze`: run analysis after run
- `--analyze_window N`: analysis window size

## Special training mode

- `--special_train_develop_turns N`: in training mode, force DEVELOP for N steps then RELEASE
- `--special_train_develop_k K`: which element index to develop (0..K-1)
