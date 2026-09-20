# Experiment log

## 2026-09-20 — Qwen three-context feasibility pilot

**Status:** completed first local task run; 36 episodes archived and audited. This is exploratory progress; the broader research goal remains active.

**Question.** Can three separately maintained contexts of one local Qwen model use a constrained symbol string and environmental feedback to coordinate a collection task that requires expressing an object, an attribute, a responsible partner, and a destination?

**Frozen pilot protocol (`balanced_six_meaning_two_block_v2`).** The three requester identities are crossed with six fixed meaning tuples in two independently shuffled blocks (36 episodes). Every requester therefore sees each tuple twice, once in each block. Each episode has one private requester order, four candidate items, two helpers who choose independently, and a joint reward that requires the designated helper to select and deliver the exact item while the other helper waits. Only 4–8 character strings from `@#%&+=?~` are passed as requester messages. Invalid strings become an empty message; the runner does not repair them. Each context retains its own history. Raw generations and hidden reasoning are not saved.

**Why the initial 18-episode draft was replaced before any model run.** Its schedule repeatedly paired each meaning with the same requester, confounding meaning with sender identity. The balanced 36-episode schedule gives each requester every meaning in both blocks and preserves one within-context repetition for early adaptation.

**Implementation checks completed.** Unit tests cover the balanced schedule, unique target item, message validation and suppression, malformed helper output, exact destination matching, and the requirement that the unassigned helper wait. A mocked 36-episode runner test verifies all 108 role calls and the summary metrics. The `mlx-vlm` server CLI is available in the pinned environment. These checks validate the task runner only; no model behavior has yet been observed.

**Model/runtime target.** `mlx-community/Qwen3.5-9B-8bit`, pinned to Hub revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`, served locally through `mlx-vlm==0.7.1` on loopback. Both local shards match the expected sizes and SHA-256 values in `model_lock.json`. Weights stay outside the Git repository. The initial Xet transport stalled and returned expired temporary URLs; the successful transfer used the standard Hub HTTP backend with the same pinned revision.

**Single-turn smoke check.** The server loaded the local checkpoint and returned HTTP 200 from `/v1/models`. Two fresh role contexts then completed API calls: requester JSON parsed, but its message failed the symbol-channel validator; the helper JSON/action schema passed. No raw response was saved. This confirms the runner must retain invalid-message suppression rather than silently correcting strings. The smoke calls are excluded from the 36-episode task dataset, which starts with fresh histories and seed `20260920`.

**Completed run.** Source commit used for the run: `848d72a09bb6a7954b46bd96ecaab873ccf66e00`. The 36-episode run completed 108 model calls in 210.7 seconds. The main run had 36/36 valid symbol messages and 72/72 valid helper actions, but only 1/36 joint successes. It emitted two distinct strings. Exact within-requester/meaning repeats were 15/18; final cross-requester agreement was 0/6. Empirical mutual information was 0.625 bit between message and requester identity and 0.021 bit between message and meaning ID. In the last block each requester used one string for all six meanings. This is consistent with a sender-specific default code, not a shared semantic mapping. Item accuracy was 8/36, destination accuracy 8/36, both correct for the designated helper 5/36, and the unassigned helper waited 8/36. The outcome and per-field reconstruction audit passed; see `results/pilot_20260920_analysis.md`, the episode record, and `results/pilot_20260920_manifest.json`.

**Interpretation.** The result does not show language emergence or an effect of interaction: it is one model, one seed, 36 episodes and has no no-message or known-code control. The one success in block 1 is too little to identify learning. Sparse feedback, two exposures per requester–meaning pair, task competence and pretraining remain plausible explanations, not established causes.

**Record-quality improvement for later runs.** After this run, the runner was extended to store the public candidate board and target item ID and to report item and destination correctness separately. The archived v2 result was not rewritten; the current runner emits v3 records. The analysis reconstructs the v2 board from the fixed seed and verifies the stored joint outcomes.

**Storage check.** The verified 8-bit checkpoint remains at `/Users/xia/Models/Qwen3.5-9B-8bit`, outside Git. The Hugging Face cache contains no second copy of these weight shards. Two zero-byte partial-download files left by the failed Xet attempt were removed; the model and Python environment are retained because the next experiments need them.

**Limits fixed in advance.** This is a one-seed, one-model, 36-episode feasibility run. The six meanings are a small set of holistic tuples, not a factorial design, so it cannot establish compositional grammar. A shared pretrained language model, common English task instructions, and punctuation symbols with pretrained associations also prevent treating the result as de novo language emergence. There is no condition comparison in this pilot, so it cannot estimate effects of environment or communication hyperparameters. A successful run will only establish that the local interaction loop works and provide exploratory behavior for designing the next controlled study.

**Primary pilot readouts.** Protocol-valid message rate; joint success by block and requester; within-requester/meaning exact message repeatability; and cross-requester agreement on the final message for the same meaning. These are descriptive pilot measures, not confirmatory tests.

## 2026-09-20 — matched channel calibration v1

**Status.** Completed and pushed to `main`; protocol/runner commit `99e8d90d`, archived data commit `edda3612`, sync receipt `55fa0614`. Full protocol is `calibration_plan.md`; results and replay audit are in `results/calibration_20260920_analysis.md` and `results/calibration_20260920_manifest.json`.

**Design.** Three paired seeds (`20260921`–`20260923`) crossed with blank channel, known shared codebook and free symbols; 36 episodes per cell, 324 total and 972 model calls. All 324 schedules matched across conditions and all 324 outcomes replayed from the task generator. No raw completions or hidden reasoning were retained.

**Result.** Joint success was blank 5/108, known codebook 11/108, and free symbols 2/108. The known-codebook sender selected the exact code in 108/108 rounds; designated item-plus-destination accuracy was 89/108. The other helper waited in only 12/108 rounds, including 0/36 for each of the first two codebook seeds. Free-symbol message–meaning mutual information varied by seed (0.989, 0.679, 0.075 bits); the last seed used one string for all meanings, so its 6/6 cross-sender agreement is message collapse, not evidence of a semantic convention.

**Diagnosis.** The v1 helper prompt permitted waiting but did not clearly state that only the designated helper should act. Both helpers often chose the same correct item. Since the known-codebook condition also failed the wait component, joint reward confounds message decoding with an underspecified action-allocation rule. The results are diagnostic and do not support a claim of language emergence.

**Next step.** V2 adds the same explicit single-worker rule to all conditions. A known-codebook development run with a held-out seed must meet predeclared encoding, target-action, waiting and joint-success thresholds before the new paired matrix proceeds. V1 remains archived and will not be pooled with V2.

## 2026-09-20 — v2 known-codebook development gate

**Status.** Completed development-only run with seed `20260924`; gate failed, so the paired v2 matrix was not started. Protocol source commit: `37b71197bd03f2590d1072f05905b9f5efab5eec`. Result, analysis and SHA-256 manifest are in `results/calibration_v2_development_20260924*`.

**Observed.** The sender encoded all 36 orders correctly. The unassigned helper waited in 36/36 episodes, but the designated helper completed the item-plus-destination action in only 14/36; joint success was 14/36. Thus the global one-worker instruction fixed duplicate actions but caused substantial designated-helper abstention.

**Prompt diagnosis.** The known-codebook helper still inherited “You do not know the private order” immediately before receiving a full codebook. This conflicts with the lookup task and plausibly explains the extra abstention. The four prespecified gates were encoding ≥35/36, designated action ≥29/36, waiting ≥29/36 and joint success ≥27/36; encoding and waiting passed, designated action and joint success failed.

**Next version.** V2.1 removes that contradictory sentence in the known-codebook arm and gives an exact lookup procedure. It retains the common one-worker rule, the same gate thresholds and formal paired seeds `20260925`–`20260927`. A new development seed (`20260928`) must pass all gates before any free-symbol matrix is launched. This failed run remains a separate development artifact and is excluded from later comparisons.

## 2026-09-20 — v2.1 protocol frozen

**Status.** The development gate revealed a prompt contradiction, so the formal matrix remains closed. V2.1 removes “You do not know the private order” only from the known-codebook helper prompt and adds an explicit three-step code lookup/identity/action procedure. The generic one-worker policy and all four v2 gate thresholds are unchanged. The next development seed is `20260928`; the formal seeds `20260925`–`20260927` remain unused and reserved. See `calibration_plan_v2_1.md`.

## 2026-09-20 — v2.1 known-codebook development gate

**Status.** Completed development-only run with seed `20260928`; gate failed, so no paired matrix was launched. Protocol source commit: `f655ab0e0224254f7ab0a163730f592a0457047d`. Result, analysis and SHA-256 manifest are archived under `results/calibration_v2_1_development_20260928*`.

**Observed.** Sender encoding was 36/36 and unassigned-helper waiting was 36/36. The designated helper selected the right destination in 33/36 but the right item in only 22/36; 11 wrong item actions selected `I0` despite another board entry matching object and attribute, and three rounds ended in waiting. Item accuracy was the same in both blocks, 11/18.

**Next version.** V2.2 adds a common rule in every condition to scan the public board and return the `item_id` whose object and attribute match the decoded order. The codebook condition gets the same exact field-matching procedure alongside its existing role lookup. Thresholds stay fixed; a new development seed (`20260929`) must pass before using the still-unused formal seeds `20260925`–`20260927`.

## 2026-09-20 — v2.2 known-codebook development gate

**Status.** Completed development-only run with seed `20260929`; gate failed and the paired matrix was not started. Protocol source commit: `e337fc3ede3eb79d68b53d1438a9183baf8ea613`. Data, analysis and SHA-256 manifest are in `results/calibration_v2_2_development_20260929*`.

**Observed.** Sender encoding was 36/36, unassigned-helper waiting 35/36, designated item accuracy 22/36, destination accuracy 23/36, and joint success 21/36. Among 23 designated actions, 22 selected the correct board item and all 23 selected the correct destination; 13 designated rounds had no action. A acted correctly in 11/12 assigned rounds, C in 11/12, and B in 0/12 for this seed. This single run cannot establish an identity effect.

**Interpretation.** The shared board-match prompt appears to have fixed most item selection when agents act. The remaining gate failure is unstable designated-helper activation. Repeatedly tuning prompts on one seed would overfit the control and obscure whether the model understands roles, the codebook, or the task.

**Next version.** V3 adds an environment-provided decoded-order upper bound, clearly separated from peer communication. It tests whether the three-context model can perform the collection task when message interpretation is removed. If this upper-bound condition passes, run the paired blank/oracle/known-codebook/free-symbol matrix on the reserved seeds `20260925`–`20260927`; otherwise simplify the task before any language comparison. The v2.2 development record remains separate.

## 2026-09-20 — v2.2 item-grounding protocol frozen

V2.2 adds one common candidate-board rule to all conditions: scan the board for the entry whose `object` and `attribute` fields both match the decoded order, then copy that entry's `item_id`. The codebook helper prompt repeats this as an explicit lookup step. The one-worker rule and four development thresholds remain unchanged. Development seed `20260929` is the only next run; formal seeds `20260925`–`20260927` remain unused. If any gate fails, do not launch the paired matrix.

## 2026-09-20 — v2.2 known-codebook development gate

**Status.** Completed development-only run with seed `20260929`; gate failed and the paired matrix was not started. Protocol source commit: `e337fc3ede3eb79d68b53d1438a9183baf8ea613`. Result, analysis and SHA-256 manifest are in `results/calibration_v2_2_development_20260929*`.

**Observed.** Sender encoding was 36/36, unassigned-helper waiting 35/36, designated item accuracy 22/36, destination accuracy 23/36, and joint success 21/36. Among 23 designated actions, 22 selected the correct board item and all 23 selected the correct destination. A acted correctly in 11/12 assigned rounds, C in 11/12, and B in 0/12 for this seed. This single run cannot establish a persistent identity effect.

**Interpretation.** The shared board-match prompt appears to have fixed most item selection when agents act. The remaining gate failure is unstable designated-helper activation; further prompt tuning on one seed would risk overfitting.

## 2026-09-20 — v3 oracle task-control protocol frozen

V3 adds an environment-supplied decoded-order upper bound, separate from peer communication, to test task competence with message interpretation removed. The formal matrix retains blank, oracle-decoded, known-codebook and free-symbol conditions. Development seed `20260930` must pass designated-action, unassigned-wait and joint-success thresholds before any paired matrix. If it passes, the unused paired seeds remain `20260925`–`20260927` (four arms, 432 episodes). See `calibration_plan_v3.md`.

## 2026-09-20 — v3 oracle task-competence gate

**Status.** Passed. Seed `20260930`, protocol source commit `ddb799300f15c2197ccefc421c22955ae2212ea7`; result, analysis and SHA-256 manifest are in `results/calibration_v3_development_20260930*`.

**Observed.** With the complete request supplied directly by the environment, designated item-plus-destination accuracy, unassigned-helper waiting, and joint success were all 36/36. All schemas passed and all outcomes replayed. This supports using the task for channel comparisons; it does not test peer language.

**Next step.** The preregistered four-arm, three-seed matrix on `20260925`–`20260927` is complete and replay-audited. The oracle upper bound remains separate from peer-channel evidence; see the v3 matrix record below.

## 2026-09-20 — v3 paired channel calibration matrix

**Status.** Completed 12 seed-condition runs, 432 episodes and 1,296 model calls. The protocol and runner source commit was `e78b4baa707e67402f5e2276f400fe1e690ac04e`. Results, analysis, replay audit and SHA-256 manifest are in `results/calibration_v3_20260920*`.

**Paired joint success.** Across seeds `20260925`–`20260927`, blank was 0/108, environment-oracle decoded was 106/108, known codebook was 71/108, and free symbols was 0/108. Per-seed counts were blank 0/36, 0/36, 0/36; oracle 36/36, 34/36, 36/36; codebook 20/36, 26/36, 25/36; and free symbols 0/36 in all three seeds. The unassigned helper waited in 108/108 episodes in each condition.

**Task execution and channel behavior.** The designated helper acted in 107/108 oracle and 71/108 codebook episodes, and in none of the blank or free-symbol episodes. Every designated action in the codebook arm selected the correct item and destination. The oracle produced 106/108 correct items and 107/108 correct destinations. Codebook encoding was exact in 108/108 rounds. Free-symbol messages were valid in 100/108 rounds; message–meaning mutual information was 0.377, 0.075 and 1.018 bits by seed. The middle seed used a single string for every meaning; its 6/6 final-block cross-sender agreement is message collapse, not a shared semantic code. No free-symbol team succeeded.

**Audit.** `audit_calibration_v3.py` replayed all 432 outcomes, recomputed all 12 run summaries, and verified 324 paired schedule comparisons. The audit passed. All owner/helper outputs parsed as JSON; 863/864 helper action schemas passed, with the sole failure in the oracle condition. Raw completions and hidden reasoning were not retained.

**Interpretation and limit.** Direct environment provision confirms the task can be completed. The shared codebook supports successful behavior when decoded, while the free-symbol condition never activates the designated helper. This leaves an activation/grounding bottleneck and prevents a behavioral claim about what free messages meant. Three seeds support screening only; this does not demonstrate language emergence or establish a reliable effect size.

**Next design decision.** Separate symbol-to-meaning alignment from helper activation, increase exposure, and evaluate transfer to held-out meaning combinations. Any change to action incentives or feedback requires a new frozen protocol and fresh seeds; keep this matrix as its own baseline.

## 2026-09-20 — literature-driven v4 protocol freeze

**Literature update.** The closest new prior art is GlossoGen (arXiv:2609.01491v1, submitted 2026-09-01), which already studies language change in a complex two-role LLM cooperation task, communication pressure, postmortem convention discussion, productive structure, and newcomer transmission. An ICLR 2026 under-review paper also directly claims artificial language in LLM referential games, using an engineered semantic-retrieval speaker and structural-proximity listener. Therefore this project will not claim novelty for “LLMs form symbols,” complex tasks, pressure, compositionality, or transmission alone. The focused prior-art note is `research_program/llm_language_emergence_prior_art_2026-09.md`.

**Question.** In a three-agent task with two interchangeable helpers, how does the need to signal *who should act*, alongside object/property/destination content, interact with the informativeness of environmental feedback to shape a shared symbol–meaning convention?

**Frozen protocol.** V4 crosses a full `2 × 2 × 2 × 2` meaning space (16 orders) with hidden vs. individually revealed helper assignment and scalar vs. component-wise post-action feedback. Each run contains 96 episodes: three requesters each see all 16 orders once in each of two blocks. Messages remain 4–8 characters from `@#%&+=?~`; contexts share no natural-language communication. Known-codebook and direct-order conditions are controls. The receiver outputs a private structured interpretation alongside its action; this is an instrumentation step common to all conditions and may itself help organize a guess.

**Measurement and interpretation boundary.** V4 records receiver reports by semantic field, task actions, shared message forms, repeatability, cross-requester agreement, and mutual information with payload, partner position, and requester. This stage does not hold out combinations, so it cannot establish compositionality or zero-shot language generalization. Results will be called convention-learning diagnostics, not de novo language origin. Formal seeds are `20261002`–`20261004`; the separate 96-episode direct-order task gate uses `20261001` and is required before the matrix.

**Implementation checks.** The v4 environment, runner, replay auditor, and plan have been added. The schedule, scoring, role-visibility prompt, feedback distinction, parser failure handling, and mock-record replay audit are covered by six unit tests. These tests passed before the model run began.

## 2026-09-20 — v4 oracle task-competence gate

**Status.** Passed. Seed `20261001`; source commit `305f89260d9243d97686cd8917f131fd4b560400`; 96 episodes and 288 model calls. Data, independent replay audit, gate interpretation, and hashes are in `results/calibration_v4_gate_20261001*`.

**Observed.** Designated item-plus-destination actions, non-designated waits, and joint successes were each 96/96. Requester JSON was valid 96/96; helper JSON, interpretation schemas, and action schemas were each valid 192/192. All four structured interpretation fields matched the directly supplied order in 192/192 helper reports. This is an oracle task control; those exact interpretations are not message-decoding evidence.

**Audit and runtime.** Independent replay of all 96 boards, targets, actions, outcomes, and feedback records passed. The run took 2,235.1 seconds and consumed 4.41 million prompt tokens because complete interaction histories are sent at every turn. No raw completions or hidden reasoning were retained.

**Decision.** The preregistered thresholds pass. Proceed with paired conditions `hidden_scalar`, `hidden_component`, `public_partner_scalar`, `public_partner_component`, `known_codebook`, and `oracle_decoded` on fresh seeds `20261002`–`20261004`. This is still a mechanism screening matrix; a positive result would require held-out composition and receiver-transfer tests before any language-formation claim.

## 2026-09-20 — v4 formal matrix: first hidden-partner, scalar-feedback cell

**Status.** One formal cell has completed: `hidden_scalar`, seed `20261002`, 96 episodes and 288 model calls. The frozen protocol source is `305f89260d9243d97686cd8917f131fd4b560400`. At the time of this entry, `hidden_component` for the same seed was still running. This is an interim record, not the full matrix result.

**Observed.** The cell produced 3/96 joint successes, 7/96 correct designated items, 13/96 correct destinations, and 64/96 unassigned-helper waits. Helpers' structured interpretations were exact in 0/192 reports. Field accuracy was object 63/192 (0.328), attribute 61/192 (0.318), responsible helper 0/192, and destination 60/192 (0.313). The owner emitted valid messages in 95/96 episodes; all 95 were the same six-symbol string, `%&+=@~`. Message entropy and empirical mutual information with meaning, payload, partner position, and requester were all 0. The one-string consistency is message collapse, not a shared convention.

**Block pattern.** Joint success was 1/48 in block 0 and 2/48 in block 1; exact helper interpretations stayed 0/96 in each block. This cell gives no evidence of within-cell semantic learning.

**Audit and limit.** Independent replay passed for all 96 episodes with zero errors. The audited one-cell checkpoint SHA-256 was `170510c9d171c92a12743aa6d46df44d3caa6384d13c463d6f755c6e229adf6c`. This is one seed in one factorial arm; it cannot identify an effect of feedback granularity. The paired `hidden_component` cell and the remaining frozen conditions must complete before interpreting the factorial contrasts. The partial episode file remains a local checkpoint and is not yet the final archived matrix.

## 2026-09-20 — v4 interim paired checkpoint: hidden-partner feedback contrast

**Status.** The first two of 18 seed-condition runs are complete for seed `20261002`: `hidden_scalar` and `hidden_component`. An independent replay auditor checked all 192 episodes across both cells and passed with zero errors. The two-cell checkpoint SHA-256 at this point was `30f3dffd0010be108d1ca4ddefc9ae96ad44aa77914f05e0b5218b8387b45a6c`. The runner has started `public_partner_scalar` for the same seed; the checkpoint remains local and is still in progress.

**Matched descriptive contrast.** Joint success was 3/96 under scalar feedback and 4/96 under component-wise feedback. Designated-item accuracy was 7/96 versus 16/96; destination accuracy 13/96 versus 33/96; unassigned-helper waiting 64/96 versus 32/96. These differences are descriptive for one seed only. Both conditions emitted the same single valid message, `%&+=@~`, in 95/96 episodes; message entropy and mutual information with meaning, payload, partner position, and requester were zero. Both had zero exact helper interpretations out of 192, including zero responsible-helper interpretations. The additional outcome detail therefore does not appear to have broken message collapse or enabled a shared mapping in this seed.

**Interpretation boundary.** The task score moved slightly, but the communication channel carried no measured semantic information. The one-seed difference cannot establish that component feedback improves coordination, and it does not constitute language formation. The public-partner cell now isolates whether making the intended helper identifiable changes this pattern while keeping scalar feedback fixed; complete all preregistered seeds before inference.

## 2026-09-20 — v4 interim checkpoint: public-partner scalar-feedback cell

**Status.** Three of the 18 formal seed-condition runs are now complete for seed `20261002` (288 episodes, 864 model calls). The third cell, `public_partner_scalar`, completed in 3,137.7 seconds. The independent replay auditor passed all 288 episodes with zero errors. The three-cell checkpoint SHA-256 was `04ed4c704cdf4b1bd077377a78bc690df93f3d7d0a6301d7c4d8089d2740ce46`. The runner has started `public_partner_component`; the matrix remains incomplete and its episode file remains a local checkpoint.

**Public-partner scalar result.** Joint success, designated-item correctness, designated-destination correctness, and exact helper interpretation were all 0 (the exact-interpretation denominator was 192); the unassigned helper waited in 96/96 episodes. All 96 messages were protocol-valid, but 95 used one form and one used another. Message entropy was 0.084 bits; empirical mutual information was 0.043 bits with meaning, 0.032 with payload, 0.010 with partner position, and 0.017 with requester. These tiny values are consistent with the 95:1 message skew and do not show a useful shared mapping.

**Cross-arm reading.** Making the intended helper public did not rescue the scalar-feedback cell: it still had no successful task execution or exact interpretation. Combined with the two hidden-partner cells, this indicates that partner visibility alone is not sufficient under this prompt/task setup. It does not identify whether the bottleneck is symbol grounding, insufficient exposure, task-policy abstention, or model capability. The next paired cell keeps partner visibility public and changes only scalar versus component-wise feedback. All conclusions remain single-seed diagnostics pending the remaining conditions and seeds.

## 2026-09-20 — user-requested pause of the v4 formal matrix

**Status.** The experiment runner and local MLX model server are stopped. The preserved matrix checkpoint contains three completed cells for seed `20261002`—`hidden_scalar`, `hidden_component`, and `public_partner_scalar`—for 288 episodes and 864 model calls. Its SHA-256 is `04ed4c704cdf4b1bd077377a78bc690df93f3d7d0a6301d7c4d8089d2740ce46`. The aggregate replay audit passed all 288 episodes with zero errors; the checkpoint and audit are archived with this pause record.

**Interrupted cell.** The runner had begun `public_partner_component` for seed `20261002`; its last confirmed progress output was 60/96 episodes. By the time the pause was applied, the runner process had already exited. The active cell is not present in the atomic checkpoint and no partial episode records were written. Its exact final progress is therefore unknown and none of it is counted. On resumption, the runner will restart this whole cell from episode 0, then skip the three completed cells.

**Resume commands.** Start the local server, then run the matrix command with `--resume` from the repository root. The checkpoint configuration is unchanged.

    /Users/xia/.venvs/qwen35-mlx/bin/python -m mlx_vlm.server --host 127.0.0.1 --port 8080 --model /Users/xia/Models/Qwen3.5-9B-8bit

    PYTHONPATH=. /Users/xia/.venvs/qwen35-mlx/bin/python -m research_program.qwen_agent_pilot.calibration_v4 --model /Users/xia/Models/Qwen3.5-9B-8bit --seed 20261002 --seed 20261003 --seed 20261004 --out research_program/qwen_agent_pilot/results/calibration_v4_20261002-04.json --resume
