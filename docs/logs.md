# Logs

Logs are JSONL events in `runs/logs.jsonl`. Each line is one event.
All events include `run_id`, `episode_id`, `episode_idx`, and `episode_seed`.

## Core event types

- `episode_start`: config snapshot and initial agents.
- `step_start`: lineage snapshot at step start.
- `active_set`: which agents are active this step.
- `actor_decision`: chosen action and observation.
- `proposal_created`: governance proposal for an action.
- `vote_cast`: individual vote records.
- `authorization_result`: aggregate voting outcome.
- `action_applied`: applied or rejected action with details.
- `step_end`: lineage snapshot and reward.
- `cap_reached`: saturation event (float caps reached).
- `episode_end`: total returns and runtime.

## Common fields

- `lineage_id`, `agent_id`, `tier`
- `action`, `action_name`
- `obs`: observation vector
- `reward`: scalar reward for the step (lineage-level)
- `snapshot`: lineage summary (stock, totals, control, concentration, etc.)

## Tips

- Use `hops.analyze_logs` to generate plots and summaries.
- Rewards are per-lineage per-step; a single reward value is used for all transitions from that lineage and step.
