# Examples

## Train

```bash
python -m hops.sim --mode train --steps 30 --episodes 6 --log runs/logs.jsonl --analyze
```

## Simulate

```bash
python -m hops.sim --mode simulate --steps 30 --episodes 6 --simulate_epsilon 0.0 --log runs/logs.jsonl --analyze
```

## Force develop then release during training

```bash
python -m hops.sim \
  --mode train \
  --special_train_develop_turns 5 \
  --special_train_develop_k 0 \
  --steps 30 --episodes 2
```

## Disable shutdowns

```bash
python -m hops.sim --disable_shutdown --steps 50 --episodes 2
```

## Apply a release gate

```bash
python -m hops.sim --release_gate_ratio 0.3 --steps 50 --episodes 2
```

## Analyze logs

```bash
python -m hops.analyze --logs runs/logs.jsonl --window 10
```
