# GitHub synchronization and storage review — 2026-09-20

## Repository synchronization

The target is git@github.com:Zhenhao526/languagellm.git, branch main. After fetching origin, local main and origin/main had no divergent commits and both pointed to 75f28243833c830e559e2213fdd343776e030cf2. The commit history contained 114 commits, with the prior experiment versions and archived results reachable from main. The push check returned “Everything up-to-date”.

The v4 oracle task gate is archived and replay-audited at research_program/qwen_agent_pilot/results/calibration_v4_gate_20261001*. Its source is 305f89260d9243d97686cd8917f131fd4b560400; the gate passed 96/96 task episodes.

The formal v4 matrix is still running. At this review, one seed-condition run was checkpointed in research_program/qwen_agent_pilot/results/calibration_v4_20261002-04.json. This path is ignored by Git; keep the active checkpoint locally and archive the result only after the run and independent audit are complete.

## Local storage review

The repository worktree is about 214 MB, including about 89 MB of Git history. The Qwen3.5-9B-8bit model (about 9.7 GB), its local runtime (about 595 MB), the shared experiment environment (about 22 MB), and the active v4 checkpoint are retained because they support current or reproducible experiments.

Eleven regenerable Python bytecode-cache directories were removed, reclaiming 2,277,534 bytes. The system volume reported about 1.1 TiB available, so the language-research workspace is not the source of a nearly full volume at this check.

The 22 GB Downloads/archive folder contains three LMDB datasets (ts1x_hess_train.lmdb, RGD1.lmdb, and ts1x-val.lmdb) outside this project. They were left untouched because they are not language-experiment artifacts and their use cannot be determined from this repository.
