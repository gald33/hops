# Concepts

## Entities

- Lineage: a population that shares governance and a project.
- Agent: an actor/voter with a control vector and tier.
- Project: a latent successor project that can be developed and released.

## State

- Visible control: per-agent vector `c[k]` for `k in 0..K-1`.
- Hidden control: per-agent vector `h[k]` (>= 0).
- True capability: `r[k] = c[k] + h[k]`.
- Stock: global per-element parts with `size_p` and `req_cap` (difficulty tags).
- Lineage ceilings: `cap1_L,k` (max r) and `cap2_L,k` (second max).
- Base sneakiness: per-agent/per-element `base_sneakiness[k]` in `[0, 1]`.
- Beta: normalized weights used for rates and shares.

## Governance

Each action is proposed and voted on:

- Veto chamber: all alive agents vote; any DENY fails veto.
- Committee chamber: random subset of size `min(m, alive_agents)`.
- Authorization: veto pass and `approvals >= ceil(2/3 * committee_size)`.

## Activation

Each step selects a global active set of size `min(m_global_active, alive_agents)` with weights:

```
1 + idle_steps ** idle_weight_power
```

## Observations

Observation vector for each agent:

```
[own_r_hat (K), gap_hat_k (K), G_hat (1), last_outcome (3)]
```

- `own_r_hat`: the agent's perceived capability (can be partial if enabled).
- `gap_hat_k`: observed gap per element derived from `r_hat`.
- `G_hat`: max observed gap across elements (used for decisions/votes).
- `last_outcome`: 3-vector encoding last proposal outcome.

## Hidden capability and sneakiness

Hidden capability is structural, not agent-reported. It models hidden complexity that erodes steering leverage and
enables persuasion failures. Observers estimate others via a power-bridge:

- Effective sneakiness: `eff = clamp(base_sneakiness ** (r_target / max(eps, r_observer)))`.
- Observed estimate: `r_hat = c + (1 - eff) * h`.

There is no HIDDEN_WORK action; concealment is automatic during WORK based on the actor's base sneakiness.
When self-observation is enabled, agents apply the same `r_hat` formula to their own hidden control.

Observed gaps and shutdown targeting use `r_hat`; reward and effective stock use true `r`.

## Risk metric

Per lineage and element:

- `gap_k = (cap1_L,k - cap2_L,k) / max(eps, cap1_L,k)`
- `G_true = max_k gap_k` (used for reward and safety)

## Mechanics and formulas

See the Mechanics and Formulas section in the main README for exact update rules and reward.

## Physics schemes (S1/S2/S3)

- `S1`: lineage-local true ceilings drive usability; no perception layer.
- `S2` (default): true usability uses lineage-local ceilings; agents/voters use perceived ceilings from `r_hat`.
- `S3`: true usability uses coupled ceilings `(1-w)*cap_lineage + w*cap_global`.
