# Matched channel calibration v3: task-competence upper bound

Protocol and decision gate frozen before the v3 development run. V1 and the v2, v2.1 and v2.2 development runs remain separate records; no paired matrix has yet been run under the revised task prompts.

## Why add an oracle condition

The v2 development gates exposed different bottlenecks. V2 underperformed on designated-helper activation; v2.1 fixed waiting but missed target-item selection; v2.2 produced 22 correct board selections among 23 designated actions, yet one helper context did not act in any of its 12 assigned episodes. These single-seed checks do not establish persistent causes. Further prompt tuning on the same checks risks overfitting.

V3 adds an environment-provided decoded-order condition. Helpers receive the complete structured request directly from the environment. This is a task-competence upper bound, not peer communication and not a language-emergence condition. It retains the same board, one-worker policy, item matching rule, action format and feedback. If the agents cannot perform in this condition, the task is not ready to support claims about communication.

## Development gate

Run 36 episodes in the `oracle_decoded` condition using seed `20260930`. This run is excluded from the formal matrix. Proceed only if all three thresholds pass:

- designated helper gets both item and destination right in at least 29/36 episodes;
- unassigned helper waits in at least 29/36 episodes;
- joint success is at least 27/36 episodes.

If any gate fails, do not run the channel matrix. Simplify the task or action protocol and freeze another development version first. Do not relax the thresholds in response to this run.

## Paired screening matrix

If the oracle gate passes, compare four conditions:

- **Blank:** no requester content reaches helpers.
- **Oracle decoded:** the environment supplies the complete request to helpers. This estimates task competence with message interpretation removed.
- **Known codebook:** requester and helpers receive the same fixed mapping between six arbitrary six-character strings and complete orders.
- **Free symbols:** requesters send 4–8 characters from `@#%&+=?~`; helpers receive no mapping.

Use fresh contexts and the same balanced 36-episode schedule within each condition for paired seeds `20260925`, `20260926`, and `20260927`. Rotate condition order across seeds, and pair decoding seeds within each seed. This is 432 episodes and 1,296 model calls. Use temperature 0.35, maximum completion length 120 tokens, and model revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`.

Report per-seed joint success, designated item accuracy, destination accuracy, and unassigned-helper waiting for all conditions. For the known-codebook condition, report exact encoding accuracy. For free symbols, report valid rate, distinct messages, sender–meaning repetition, cross-sender agreement, and mutual information with sender and meaning. A constant message across all senders counts as collapse, not a shared semantic mapping. Compare free symbols with blank for the effect of interaction and with oracle-decoded for the task-competence gap. Three seeds support screening only; do not make a significance claim.

## Record handling

Keep the oracle development run separate from the paired matrix. Keep all v1/v2 development records separate from v3. Retain episode-level inputs, messages, actions and outcomes, but not raw model completions or hidden reasoning. Model weights remain outside Git.
