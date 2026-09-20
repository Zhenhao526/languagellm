# Qwen channel calibration v2 development: gate failed

This 36-episode run is a development check, excluded from any paired condition matrix. It used seed `20260924`, the known shared codebook, model revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`, and protocol source commit `37b71197bd03f2590d1072f05905b9f5efab5eec`. It took 344 seconds and made 108 model calls. The episode record is [`calibration_v2_development_20260924.json`](calibration_v2_development_20260924.json); file integrity data is in [`calibration_v2_development_20260924_manifest.json`](calibration_v2_development_20260924_manifest.json).

| Gate metric | Observed | Required | Result |
|---|---:|---:|---|
| Exact codebook encoding | 36/36 | ≥35/36 | Pass |
| Designated helper gets item and destination right | 14/36 | ≥29/36 | Fail |
| Unassigned helper waits | 36/36 | ≥29/36 | Pass |
| Joint success | 14/36 | ≥27/36 | Fail |

The explicit one-worker rule eliminated duplicate actions, but designated helpers often also waited. The v2 helper prompt still inherited the sentence “You do not know the private order” from the v1 prompt, then appended the codebook below it. This conflicts with the instruction to act on a decoded assignment and likely encouraged excessive abstention. The data show the consequence: all 36 unassigned helpers waited, but the designated helper acted correctly in only 14 rounds.

The v2 development gate therefore failed as prespecified; the free-symbol matrix was not started. The next development version removes the contradictory sentence in the known-codebook condition and gives a short, explicit lookup procedure: match the received code, compare `responsible_helper` with your identity, act only on a match. The shared allocation rule remains identical across conditions. This change must pass on a new development seed before any formal paired matrix begins.

The outcome replay, message-code mapping, and helper action schemas were audited. No raw completions or hidden reasoning were retained. This single development run is not evidence for a condition effect or language formation.
