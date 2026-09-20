# Matched channel calibration v2.2: explicit board grounding

Protocol and decision gates are frozen before the v2.2 development run. The v1 matrix and v2/v2.1 development runs remain separate records. No paired matrix has yet been run under the revised task prompts.

## Reason for this revision

V2.1 achieved 36/36 correct codebook encodings and 36/36 waits by unassigned helpers, but designated helpers selected the right item in only 22/36 rounds. They chose the correct destination in 33/36. In 11 failed rounds, a helper returned item `I0` although another candidate had the decoded object and attribute; three other rounds ended in waiting. Item performance was flat between blocks. The model therefore needs a clearer mapping from decoded object/attribute values to the candidate board's item identifier.

V2.2 adds the same board-selection instruction in all three conditions: inspect the public entries and return the `item_id` whose `object` and `attribute` both match the known target; copy the identifier instead of inferring it from list position. The known-codebook condition adds the same lookup step after decoding the row. The one-worker rule remains identical across conditions.

## Development gate

Run one 36-episode known-codebook condition with new seed `20260929`. This run is development only and excluded from the paired matrix. Proceed only if all four thresholds pass:

- exact sender encoding: at least 35/36;
- designated helper gets both item and destination right: at least 29/36;
- unassigned helper waits: at least 29/36;
- joint success: at least 27/36.

If any threshold fails, do not run the free-symbol matrix. Revise and record another development version before collecting paired channel comparisons. Keep all four thresholds fixed.

## Paired screening matrix

If the development gate passes, compare blank channel, known shared codebook, and free symbols using new paired seeds `20260925`, `20260926`, and `20260927`. Each seed-condition cell uses the same balanced 36 episodes and fresh agent histories. Rotate condition order across seeds and pair decoding seeds within a seed. Model revision is `16daa4818c54ce5f5436f929d52542eb65bbed9d`; temperature 0.35; maximum completion length 120 tokens.

Report per-seed joint success, designated item accuracy, destination accuracy, and unassigned-helper waiting. For free symbols, report valid rate, distinct messages, sender–meaning repetition, cross-sender agreement, and message mutual information with sender and meaning. A constant message across all senders is collapse, not semantic agreement. Three seeds are sufficient only for a screening read; do not make a significance claim.

## Records

Keep all development records separate from the paired matrix and keep v1 results separate from v2.2. Retain episode-level inputs, messages, actions and outcomes, but not raw completions or hidden reasoning. Model weights remain outside Git.
