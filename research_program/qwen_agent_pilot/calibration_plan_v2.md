# Matched channel calibration v2: explicit action allocation

Protocol and decision gates frozen before the development run. This remains a screening study, not a confirmatory test of language emergence.

## Why the task changed

In v1, the known-codebook condition achieved 108/108 exact sender encodings and 89/108 correct designated-helper item-plus-destination actions. Yet the unassigned helper waited only 12/108 times, with 0/36 waits in the first two seeds. The prompt allowed waiting but did not state that only the designated helper may act. Both helpers often selected the same correct target. V1 therefore mixed message use with an underspecified action-allocation rule.

V2 adds the same explicit rule to every condition: each order has one responsible helper; only that helper acts; the other waits; if an agent cannot confidently tell that it is responsible, it waits. No condition gets extra semantic information from this rule. The known-codebook arm still supplies the complete mapping; the free-symbol arm still requires a mapping to form through interaction; the blank arm still transmits nothing.

## Development gate

Run one 36-episode known-codebook development condition with seed `20260924`. This seed is excluded from the formal screen. Begin the new paired matrix only if the development run reaches all four thresholds:

- exact sender encoding: at least 35/36;
- designated helper gets both item and destination right: at least 29/36;
- unassigned helper waits: at least 29/36;
- joint success: at least 27/36.

If any threshold fails, revise the task prompt and record another development version before running free-symbol comparisons. Do not count development episodes in the formal matrix.

## Paired screening matrix

If the gate passes, compare blank channel, known shared codebook, and free symbols with fresh contexts and the same balanced 36-episode schedule per condition. Use new paired seeds `20260925`, `20260926`, and `20260927`, temperature 0.35, maximum completion length 120 tokens, and the pinned model revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`. Rotate condition order across seeds and pair decoding seeds within each seed. Write a checkpoint after every seed-condition run.

Primary outcomes remain joint success, designated item accuracy, destination accuracy, and unassigned-helper waiting. Message measures are interpreted only in the free-symbol condition and reported per seed; cross-agent agreement is not counted as semantic agreement when every sender uses one constant string. No significance claim is made from three seeds.

## Record handling

Keep the v1 run and its analysis separate. The v2 development result is labeled as development and excluded from the paired matrix. Retain episode-level inputs, sent symbols, actions and outcomes, but not raw model completions or hidden reasoning. Model weights remain outside Git.
