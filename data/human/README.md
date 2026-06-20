# Human judgments (planned)

This directory is reserved for human forced-choice judgments, which will
provide a human baseline for the model results.

## Schema

One CSV file per collection round, for example `round1.csv`, with a header
row and these columns:

| column | type | description |
|---|---|---|
| item_id | string | matches `item_id` in `data/items.jsonl` |
| participant_id | string | anonymised participant code |
| choice | string | `A` or `B`, the continuation the participant preferred |

Participants see context, target, and both continuations in randomised
order, and pick the continuation that follows more naturally. Congruence is
computed by joining on `item_id` and comparing `choice` to `forced`, exactly
as for models.

No human data has been collected yet.
