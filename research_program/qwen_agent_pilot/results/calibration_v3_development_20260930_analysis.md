# Qwen calibration v3: oracle task-competence gate passed

This 36-episode `oracle_decoded` run is a task-competence development check, not a peer-communication condition. It used seed `20260930`, model revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`, and protocol source commit `ddb799300f15c2197ccefc421c22955ae2212ea7`. It took 257 seconds and made 108 model calls. The episode record is [`calibration_v3_development_20260930.json`](calibration_v3_development_20260930.json); hashes are in [`calibration_v3_development_20260930_manifest.json`](calibration_v3_development_20260930_manifest.json).

The environment supplied helpers with the complete structured request (object, attribute, responsible helper, destination). The requester sent an empty message. This condition bypasses symbol interpretation and measures whether the same agents can execute the collection and role-allocation task.

| Metric | Observed | Required | Result |
|---|---:|---:|---|
| Designated helper gets item and destination right | 36/36 | ≥29/36 | Pass |
| Unassigned helper waits | 36/36 | ≥29/36 | Pass |
| Joint success | 36/36 | ≥27/36 | Pass |

All 36 environment outcomes were recomputed from the frozen generator. Owner JSON, helper JSON and helper action schemas passed in every episode. The gate passed, so the preregistered four-condition matrix may proceed on seeds `20260925`–`20260927`. This ceiling result shows task execution is possible when meaning is supplied directly; it does not show that agents can establish or use a peer convention. Raw completions and hidden reasoning were not retained.
