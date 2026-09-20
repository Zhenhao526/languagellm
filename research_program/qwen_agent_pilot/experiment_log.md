# Experiment log

## 2026-09-20 — Qwen three-context feasibility pilot

**Status:** protocol implemented; model weights downloading; no model episode has been run.

**Question.** Can three separately maintained contexts of one local Qwen model use a constrained symbol string and environmental feedback to coordinate a collection task that requires expressing an object, an attribute, a responsible partner, and a destination?

**Frozen pilot protocol (`balanced_six_meaning_two_block_v2`).** The three requester identities are crossed with six fixed meaning tuples in two independently shuffled blocks (36 episodes). Every requester therefore sees each tuple twice, once in each block. Each episode has one private requester order, four candidate items, two helpers who choose independently, and a joint reward that requires the designated helper to select and deliver the exact item while the other helper waits. Only 4–8 character strings from `@#%&+=?~` are passed as requester messages. Invalid strings become an empty message; the runner does not repair them. Each context retains its own history. Raw generations and hidden reasoning are not saved.

**Why the initial 18-episode draft was replaced before any model run.** Its schedule repeatedly paired each meaning with the same requester, confounding meaning with sender identity. The balanced 36-episode schedule gives each requester every meaning in both blocks and preserves one within-context repetition for early adaptation.

**Implementation checks completed.** Unit tests cover the balanced schedule, unique target item, message validation and suppression, malformed helper output, exact destination matching, and the requirement that the unassigned helper wait. A mocked 36-episode runner test verifies all 108 role calls and the summary metrics. The `mlx-vlm` server CLI is available in the pinned environment. These checks validate the task runner only; no model behavior has yet been observed.

**Model/runtime target.** `mlx-community/Qwen3.5-9B-8bit`, pinned to Hub revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`, served locally through `mlx-vlm==0.7.1` on loopback. Weights stay outside the Git repository. The initial Xet transport stalled and returned expired temporary URLs; the current download uses the standard Hub HTTP backend with the same pinned revision.

**Limits fixed in advance.** This is a one-seed, one-model, 36-episode feasibility run. The six meanings are a small set of holistic tuples, not a factorial design, so it cannot establish compositional grammar. A shared pretrained language model, common English task instructions, and punctuation symbols with pretrained associations also prevent treating the result as de novo language emergence. There is no condition comparison in this pilot, so it cannot estimate effects of environment or communication hyperparameters. A successful run will only establish that the local interaction loop works and provide exploratory behavior for designing the next controlled study.

**Primary pilot readouts.** Protocol-valid message rate; joint success by block and requester; within-requester/meaning exact message repeatability; and cross-requester agreement on the final message for the same meaning. These are descriptive pilot measures, not confirmatory tests.
