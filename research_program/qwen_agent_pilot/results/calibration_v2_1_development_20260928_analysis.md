# Qwen channel calibration v2.1 development: gate failed

This 36-episode known-codebook run is a development check, excluded from any paired matrix. It used seed `20260928`, model revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`, and protocol source commit `f655ab0e0224254f7ab0a163730f592a0457047d`. It took 390 seconds and made 108 model calls. The episode record is [`calibration_v2_1_development_20260928.json`](calibration_v2_1_development_20260928.json); integrity data is in [`calibration_v2_1_development_20260928_manifest.json`](calibration_v2_1_development_20260928_manifest.json).

| Gate metric | Observed | Required | Result |
|---|---:|---:|---|
| Exact codebook encoding | 36/36 | ≥35/36 | Pass |
| Designated helper gets item and destination right | 22/36 | ≥29/36 | Fail |
| Unassigned helper waits | 36/36 | ≥29/36 | Pass |
| Joint success | 22/36 | ≥27/36 | Fail |

The codebook and role decision now work much better: the unassigned helper waited in every episode, and designated helpers chose the correct destination in 33/36. The remaining failures are concentrated in item grounding: 22/36 item choices were correct. In 11 failed episodes the designated helper returned `I0` even though another board entry matched the decoded object and attribute; in three episodes it waited. Item accuracy did not improve from the first block to the second (11/18 in each).

The v2.1 gate failed as prespecified, so the paired matrix was not started. V2.2 will add one common item-selection rule to every condition: scan the candidate board and choose the `item_id` whose object and attribute both match the decoded order; do not treat list position as item identity. The known-codebook helper prompt will state the same lookup sequence explicitly. All existing thresholds remain unchanged, and a new development seed must pass them before the free-symbol matrix opens.

All 36 outcomes were replayed from the task generator; codebook messages and action schemas were checked. Raw completions and hidden reasoning were not retained. This is one development run, not evidence for an experimental condition effect.
