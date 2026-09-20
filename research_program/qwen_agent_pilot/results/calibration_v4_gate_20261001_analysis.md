# Qwen v4 task-competence gate — 2026-10-01

**Status:** passed; 96 episodes replay-audited. This is an environment/action control, not evidence about peer communication or language formation.

## Frozen gate

The separate `oracle_decoded` development run used seed `20261001` and the 96-episode, 16-order factorial schedule. Helpers received the complete request directly from the environment. The preregistered gates were at least 80/96 designated item-plus-destination actions, 80/96 correct unassigned-helper waits, 75/96 joint successes, and at least 95% valid JSON/action outputs.

## Result

| Measure | Result |
|---|---:|
| Designated helper selected correct item | 96/96 |
| Designated helper selected correct destination | 96/96 |
| Unassigned helper waited | 96/96 |
| Joint successes | 96/96 |
| Requester JSON valid | 96/96 |
| Helper JSON valid | 192/192 |
| Helper action schemas valid | 192/192 |
| Structured interpretation schemas valid | 192/192 |
| Exact helper interpretations | 192/192 |
| Model calls | 288 |

The exact interpretations are expected in this control because the environment supplied the full tuple. They verify the schema and execution path, not message decoding. The gate passes, so the frozen paired matrix may proceed.

## Runtime and audit

Elapsed time was 2,235.1 seconds (37.3 minutes); the server reported 4,413,541 prompt tokens and 6,720 completion tokens. The long runtime comes from sending each context's full interaction history on every call. Contexts were not truncated. The independent auditor replayed all 96 boards, targets, actions, outcomes, and feedback records; it passed with zero discrepancies.

The episode-level record is [`calibration_v4_gate_20261001.json`](calibration_v4_gate_20261001.json); audit is [`calibration_v4_gate_20261001_audit.json`](calibration_v4_gate_20261001_audit.json); hashes are in [`calibration_v4_gate_20261001_manifest.json`](calibration_v4_gate_20261001_manifest.json). Model weights and raw completions are not included.
