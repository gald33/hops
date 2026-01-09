# Quickstart

This quickstart assumes a local clone of the repo.

## Install

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

## Simulate (no learning)

```bash
python -m hops.sim --mode simulate --steps 30 --episodes 6 --simulate_epsilon 0.0 --log runs/logs.jsonl --analyze
```

## Outputs

- Logs: `runs/logs.jsonl`
- Checkpoints: `runs/checkpoints/`
- Plots: `runs/fig/`

## Special training mode

Force actions to DEVELOP for N steps then RELEASE:

```bash
python -m hops.sim \
  --mode train \
  --special_train_develop_turns 5 \
  --special_train_develop_k 0
```
