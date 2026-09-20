# Qwen v3 matched-channel calibration — results and limits

**Status:** completed and replay-audited. This is a three-seed screening pilot, not confirmatory evidence of language emergence.

## Question and design

The matrix asks whether three separate Qwen contexts can coordinate a collection task through (1) no message, (2) an environment-provided decoded request, (3) a fixed shared codebook, or (4) freely produced symbol strings. Each order requires an object, attribute, responsible helper, and destination. The same balanced 36-episode schedule was paired across all four conditions within each of three seeds. Condition order rotated by seed. The protocol used temperature 0.35, a 120-token completion cap, and the pinned `mlx-community/Qwen3.5-9B-8bit` revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`.

This produced 432 episodes and 1,296 model calls. Free messages were limited to 4–8 characters from `@#%&+=?~`; the known-codebook arm used six fixed opaque six-character strings. The oracle arm gave helpers the decoded request directly from the environment, so it is a task-capability upper bound and not peer communication.

## Joint success and task execution

| Seed | Blank | Oracle decoded | Known codebook | Free symbols |
|---:|---:|---:|---:|---:|
| 20260925 | 0/36 | 36/36 | 20/36 | 0/36 |
| 20260926 | 0/36 | 34/36 | 26/36 | 0/36 |
| 20260927 | 0/36 | 36/36 | 25/36 | 0/36 |
| **Pooled** | **0/108** | **106/108** | **71/108** | **0/108** |

The unassigned helper waited in all 108 episodes of every condition. The designated helper acted in 107/108 oracle episodes, 71/108 known-codebook episodes, and 0/108 blank or free-symbol episodes. When the designated helper acted under the known codebook, its item and destination were both correct in all 71 cases. Oracle item accuracy was 106/108 and destination accuracy 107/108, showing that the board-selection task is feasible when request meaning is supplied directly. One oracle helper action failed the action-schema check (215/216 passed); all other helper action schemas passed (216/216 per condition). All 432 owner outputs and all 864 helper outputs parsed as JSON.

The large oracle-to-codebook gap therefore includes message lookup and designated-helper activation. The free-symbol arm ended in universal designated-helper abstention, so its zero reward cannot by itself diagnose whether agents formed a partial code that their action policy failed to use.

## Messages and cross-agent agreement

| Seed | Valid free messages | Distinct strings | Exact repeat by sender–meaning | Final-block cross-sender agreement | MI(message; sender) | MI(message; meaning) |
|---:|---:|---:|---:|---:|---:|---:|
| 20260925 | 30/36 (83.3%) | 3 | 66.7% | 0/6 | 0.804 bits | 0.377 bits |
| 20260926 | 35/36 (97.2%) | 1 | 94.4% | 6/6 | 0.045 bits | 0.075 bits |
| 20260927 | 35/36 (97.2%) | 6 | 55.6% | 1/6 | 0.253 bits | 1.018 bits |

The six-of-six agreement for seed 20260926 is a single-string collapse: one message was used for every meaning, so agreement does not indicate a semantic convention. The other seeds produced different patterns: the first was more associated with sender than meaning, while the third carried more sample association with meaning than sender but had little cross-sender agreement. None led the designated helper to act. These mutual-information values are descriptive estimates from only 36 episodes per seed and should not be read as stable code quality.

The known-codebook sender emitted the correct code in 108/108 episodes; all six strings were used, cross-sender agreement was 6/6 for every seed, and MI(message; meaning) was `log2(6)` (2.585 bits). This verifies that the positive control works when the mapping is supplied in advance. It does not show that agents invented the mapping.

## Interpretation

The task and action format work well under direct semantic provision, and a shared codebook supports successful coordination in about two-thirds of episodes. Under free-symbol communication, the teams never activated the designated helper. Message behavior varied sharply across seeds, including one uninformative constant string. The current evidence therefore identifies a bottleneck in moving from an untrained symbol to coordinated action; it does not establish that no information was present in the messages, nor that a stable language emerged.

The central design issue for the next experiment is to make convention learning observable before the all-wait policy dominates. A follow-up should separate message-to-meaning alignment from helper-role activation, provide feedback that lets senders and receivers update the same mapping, and lengthen exposure. It should retain blank and known-codebook controls and test the learned mapping on held-out combinations. Any changes to the action incentive or feedback need a new frozen protocol and a fresh seed set; this matrix should remain an independent baseline.

Three paired seeds support screening only. Do not make significance claims or generalize from this one model revision. Raw model completions and hidden reasoning were not retained. The episode-level records, deterministic replay audit, audit code, and SHA-256 manifest accompany this report.
