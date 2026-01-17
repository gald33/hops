# FAQ

## Why are rewards NaN or huge?
Rewards depend on `delta_S_eff`, a gap-based danger penalty (`G_true`), and a `log(1 + S_eff_pre)` factor. If effective stock or control saturates to large values, rewards can blow up. The simulator caps values at `MAX_FLOAT` but you can still see extreme magnitudes in logs.

## Why is RELEASE failing?
RELEASE fails if there is no active project or the acting agent is not the project parent tier. It can also be blocked by `--release_gate_ratio`.

## Why are votes always approving/denying?
Check `--approve_all_votes` and `--random_votes`. If neither is set, votes use the current policy.

## Why did the run stop early?
The run ends if the step cap is reached or if state values hit the saturation cap (`cap_reached`).

## Where are the plots?
When `--analyze` is set, plots are written under `runs/fig/`.
