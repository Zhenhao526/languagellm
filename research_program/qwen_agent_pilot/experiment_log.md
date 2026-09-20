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
