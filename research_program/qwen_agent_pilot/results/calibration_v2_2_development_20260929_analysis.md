# Qwen channel calibration v2.2 development: gate failed

This 36-episode known-codebook run is a development check, excluded from any paired matrix. It used seed `20260929`, model revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`, and protocol source commit `e337fc3ede3eb79d68b53d1438a9183baf8ea613`. It took 446 seconds and made 108 model calls. The episode record is [`calibration_v2_2_development_20260929.json`](calibration_v2_2_development_20260929.json); hashes are in [`calibration_v2_2_development_20260929_manifest.json`](calibration_v2_2_development_20260929_manifest.json).

| Gate metric | Observed | Required | Result |
|---|---:|---:|---|
| Exact codebook encoding | 36/36 | ≥35/36 | Pass |
| Designated helper gets item and destination right | 22/36 | ≥29/36 | Fail |
| Unassigned helper waits | 35/36 | ≥29/36 | Pass |
| Joint success | 21/36 | ≥27/36 | Fail |

The common object–attribute-to-item rule improved execution for agents that acted: 22 of 23 designated actions selected the correct board item, and all 23 selected the correct destination. The designated agents were highly uneven in this one run: A acted correctly in 11/12 assigned rounds, C in 11/12, and B in 0/12. Thirteen assigned rounds had no action; one helper action failed its item/action schema. This is a single development seed and does not establish a persistent agent-identity effect.

The prespecified gate failed and the paired matrix was not started. Further prompt tuning on this seed would risk overfitting. V3 will add a separate task-competence upper bound in which the environment supplies helpers with the decoded order directly. If that control also fails, the collection task is not ready for communication claims. If it passes, the paired matrix can compare blank channel, oracle-decoded order, shared codebook, and free symbols; the oracle condition is clearly labeled as environment supervision, not peer language.

All 36 outcomes were replayed, all codebook encodings were checked, all 72 helper JSON objects parsed, and 71/72 helper actions passed the action schema. Raw completions and hidden reasoning were not retained. This run is descriptive only.
