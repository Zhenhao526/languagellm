# Qwen v4: information allocation and outcome feedback

**Status:** protocol frozen; implementation tests pass; model runs have not started.

## Why this experiment

The v3 free-symbol arm had no designated-helper actions, although the direct-order control succeeded and the known-codebook arm succeeded on most episodes. That leaves two plausible bottlenecks entangled: the receiver has to infer both the requested payload and which teammate should act, and receives only a scalar team outcome after each episode.

Recent work makes a broad claim that LLM agents can form artificial languages or develop novel conventions too close to existing work. This experiment is a mechanism screen: it asks how information about *who should act* and the informativeness of environmental feedback change shared symbol–meaning conventions in a triadic task. It does not test language origins, open-ended grammar, or compositional generalization.

## Task and factors

Three persistent contexts for the same pinned Qwen3.5-9B-8bit model rotate through requester and helper roles. A requester privately receives a four-part order: object, attribute, responsible helper, and destination. It broadcasts one opaque 4–8-character string from `@#%&+=?~`. Both helpers see the same four-item public board and must decide whether to act, which item to select, and where to deliver it. One helper is responsible; the other should wait. Natural-language messages between agents are unavailable.

The meaning space is a complete `2 × 2 × 2 × 2` crossing: two objects, two attributes, two helper positions, and two destinations. Each run contains 96 episodes. In each of two blocks, every requester sees each of the 16 orders once. The schedule and boards are paired across conditions within a seed.

The main factors are:

| Condition | Who should act | Outcome feedback after acting |
|---|---|---|
| `hidden_scalar` | Helpers must infer the selected partner from the signal and history | Team success and scalar reward |
| `hidden_component` | Helpers must infer the selected partner from the signal and history | Public per-helper action-allocation, item, and destination correctness |
| `public_partner_scalar` | The environment privately tells each helper whether it is assigned; the signal should encode only object, attribute, and destination | Team success and scalar reward |
| `public_partner_component` | The environment privately tells each helper whether it is assigned; the signal should encode only object, attribute, and destination | Public per-helper action-allocation, item, and destination correctness |

Two controls are included in the formal matrix:

- `known_codebook` supplies all agents with one fixed 16-entry mapping from orders to opaque strings. It checks whether the action task and the receiver's code lookup work when the convention is supplied.
- `oracle_decoded` gives helpers the complete order directly from the environment, without a peer message. It checks task execution with interpretation removed.

The component feedback reports whether each helper's act/wait choice was appropriate, and whether an action selected the right item and destination. It does not state the hidden object, attribute, selected partner, or destination directly. It is visible only after the current message and actions, so it cannot replace the communication channel during an episode.

## Measurements and limits

The main behavioral measure is each helper's structured best interpretation of the message, recorded before its action field. The environment stores this report but does not show it to the other agents or reward it directly. This is the same measurement prompt in every condition and may itself help the model organize a guess; interpret results as convention learning under this instrumentation.

We will report per-seed and per-block task success, exact tuple and field-wise interpretation accuracy, action-allocation accuracy, message validity and diversity, exact repeatability, cross-requester agreement, and descriptive mutual information between messages and requester, payload, partner position, and complete meaning. The message–partner association is especially diagnostic in the hidden-partner conditions; in public-partner conditions, a stable payload code should remain constant across the two partner variants.

All 16 combinations appear during exposure. Therefore this matrix cannot show zero-shot recombination or compositional grammar. It is a screening test of whether interaction produces a shared convention and which information demand or feedback condition supports it. If a condition yields a stable convention, the next confirmatory study should freeze the sender mapping, withhold combinations, and test fresh receivers and partner turnover. No language-emergence claim will be made from symbol diversity, mutual information, or task reward alone.

## Frozen run settings

- Development-only task gate: seed `20261001`, 96 episodes in `oracle_decoded`.
- Proceed to the paired matrix only if, in that gate, designated item-plus-destination accuracy is at least 80/96, unassigned helpers wait at least 80/96, joint success is at least 75/96, and at least 95% of JSON/action outputs parse.
- Formal paired seeds: `20261002`, `20261003`, and `20261004`; the development seed is excluded.
- Formal conditions: all six conditions in `calibration_v4.py`, 96 episodes each. Rotate condition order by seed; reset all agent histories at each seed–condition run.
- Model: local `mlx-community/Qwen3.5-9B-8bit`, revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`; temperature `0.35`; completion limit `160` tokens.
- Preserve only messages, structured interpretations, actions, feedback, outcomes, and usage counts. Do not retain raw model completions or hidden reasoning.
- The three formal seeds are screening-level replication. Analyze seeds as runs; do not treat episodes within a persistent-agent run as independent samples or make population-level significance claims.

## Reproduction

First start the local server using the command in `README.md`, then run the oracle gate:

```sh
PYTHONPATH=. /Users/xia/.venvs/qwen35-mlx/bin/python \
  -m research_program.qwen_agent_pilot.calibration_v4 \
  --model /Users/xia/Models/Qwen3.5-9B-8bit \
  --seed 20261001 --condition oracle_decoded \
  --out research_program/qwen_agent_pilot/results/calibration_v4_gate_20261001.json
```

If the gate passes, run the frozen paired matrix:

```sh
PYTHONPATH=. /Users/xia/.venvs/qwen35-mlx/bin/python \
  -m research_program.qwen_agent_pilot.calibration_v4 \
  --model /Users/xia/Models/Qwen3.5-9B-8bit \
  --seed 20261002 --seed 20261003 --seed 20261004 \
  --out research_program/qwen_agent_pilot/results/calibration_v4_20261002-04.json
```

Pass `--resume` with the same output path after an interruption. Each seed–condition result is checkpointed atomically. The oracle gate and formal matrix remain separate records.
