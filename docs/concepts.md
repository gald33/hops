# Concepts

## Entities

- Lineage: a population that shares governance and a project.
- Agent: an actor/voter with a control vector and tier.
- Project: a latent successor project that can be developed and released.

## State

- Control: per-agent vector `control[k]` for `k in 0..K-1`.
- Stock: global vector `stock[k]` shared across lineages.
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
[self_shares (K), global_M (K), c_pre (1), last_outcome (3)]
```

- `self_shares`: agent's share of lineage totals per dimension.
- `global_M`: per-dimension max self-share across all alive agents.
- `c_pre`: lineage concentration before actions.
- `last_outcome`: 3-vector encoding last proposal outcome.

## Mechanics and formulas

See the Mechanics and Formulas section in the main README for exact update rules and reward.
