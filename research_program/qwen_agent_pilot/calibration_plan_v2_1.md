# Matched channel calibration v2.1: explicit codebook use

Protocol and decision gates are frozen before the v2.1 development run. The v1 matrix and failed v2 development run remain separate records. No paired free-symbol matrix has been run under either revised prompt.

## Reason for this revision

The v2 development run (seed `20260924`) passed exact codebook encoding (36/36) and unassigned-helper waiting (36/36), but designated-helper item-plus-destination success was only 14/36. Inspection found a prompt contradiction: the known-codebook helper still read “You do not know the private order” before receiving a full codebook. V2.1 removes that sentence and tells the helper to match the received code, compare the decoded `responsible_helper` with its own identity, and act only when the row names it. A nonmatching or missing row means wait.

The common one-worker rule remains identical in all conditions: exactly one helper is responsible; only that helper acts; the other waits; if responsibility is unclear, wait. The codebook condition contains no extra semantic facts in v2.1; the change clarifies how to use the already-provided mapping.

## Development gate

Run one 36-episode known-codebook condition with new seed `20260928`. This run is a development check and is excluded from the paired matrix. Proceed only if all four thresholds pass:

- exact sender encoding: at least 35/36;
- designated helper gets both item and destination right: at least 29/36;
- unassigned helper waits: at least 29/36;
- joint success: at least 27/36.

If any threshold fails, do not run the free-symbol matrix. Revise the task and record another development version first. Do not relax the thresholds in response to this run.

## Paired screening matrix

If the development gate passes, use new paired seeds `20260925`, `20260926`, and `20260927` across blank channel, known shared codebook and free symbols. Each seed-condition cell replays the same 36 balanced episodes, with fresh agent histories. Rotate condition order across seeds and pair model sampling seeds within each seed. Use temperature 0.35, maximum completion length 120 tokens, and model revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`.

Report per-seed joint success, designated item accuracy, destination accuracy and unassigned waiting. For free symbols, also report valid rate, distinct messages, within-sender repetition, cross-sender agreement, and message mutual information with sender and meaning. A constant message across all senders is collapse, not a semantic convention. This small matrix remains exploratory and supports no significance claim.

## Records

Keep v1, v2 development, and v2.1 development results separate. Store episode-level inputs, sent strings, actions and outcomes, but no raw model completions or hidden reasoning. Model weights remain outside Git.
