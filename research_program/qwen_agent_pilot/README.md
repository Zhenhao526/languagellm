# Three-context local Qwen agent pilot

This pilot moves the validated collection-and-selection task from tabular policies to three independent Qwen contexts on one local model server. Each round one agent privately receives an order containing an object, an attribute, a responsible partner and a destination. It broadcasts only a 4–8 character string from `@#%&+=?~`; the other two agents see the candidate collection board and choose whether to act, which item to collect and where to deliver it. Roles rotate across agents.

The pilot uses no processing stage, road event, shared image, object index as a semantic label, or natural-language inter-agent channel. The world board is structured text, and only the public candidate inventory is shared. Each agent has a separate message history. Actions and scalar team feedback are applied by the environment, not written by another agent. Six fixed meaning tuples are crossed with all three requester identities in each of two seed-shuffled blocks (36 episodes), so each requester sends each meaning twice.

The first run is a feasibility and prompt-compliance test, not evidence of language emergence. Qwen already has strong linguistic and symbolic priors; a successful convention may reflect those priors and shared instructions. The run must therefore be interpreted alongside the non-linguistic policy results and later baseline/ablation conditions.

## First pilot result

The 2026-09-20 run completed 36 balanced episodes in 210.7 seconds. All 36 requester messages passed the symbol constraint and all 72 helper actions parsed, but joint success was only 1/36. The model used just two distinct strings; in the final block each requester sent one constant string across all six meanings, and the three senders did not share one message per meaning. This is a protocol-compliance result, not evidence that a semantic code formed. The episode audit and limits are in [`results/pilot_20260920_analysis.md`](results/pilot_20260920_analysis.md); the compact episode record and SHA-256 receipt are [`results/pilot_20260920.json`](results/pilot_20260920.json) and [`results/pilot_20260920_manifest.json`](results/pilot_20260920_manifest.json).

The first prompt-only run cannot separate weak communication from task competence, sparse feedback or insufficient exposure. The next step is a matched calibration of an empty channel, a known shared codebook and free symbol communication before increasing repetitions or varying ecological factors.

## Matched channel calibration

The v1 screening protocol is preserved in [`calibration_plan.md`](calibration_plan.md), and its runner version is available at Git commit `99e8d90d`. The v1 matrix paired the same schedule and decoding seeds across three conditions: closed channel, a fixed shared codebook, and free symbols. It used three seeds (324 episodes and 972 model calls) and saved an atomic checkpoint after every seed-condition run.

The v1 matrix is complete. Qwen encoded the supplied codebook correctly, but the task prompt did not clearly tell non-designated helpers to wait; that rule was followed inconsistently, including in the positive-control condition. Free-symbol message–meaning association varied sharply by seed, with one seed collapsing to a single string. See the [audited v1 result and interpretation](results/calibration_20260920_analysis.md), the [episode-level data](results/calibration_20260920.json), and the [integrity manifest](results/calibration_20260920_manifest.json).

V2, v2.1, and v2.2 development results are preserved [here](results/calibration_v2_development_20260924_analysis.md), [here](results/calibration_v2_1_development_20260928_analysis.md), and [here](results/calibration_v2_2_development_20260929_analysis.md). None passed its prespecified task gate. V3 added an environment-provided decoded-order upper bound to test task competence separately from message interpretation. This is an external task-control condition, not peer communication. The frozen protocol is [`calibration_plan_v3.md`](calibration_plan_v3.md).

First run the 36-episode oracle task-competence gate with seed `20260930`:

```sh
PYTHONPATH=. /Users/xia/.venvs/qwen35-mlx/bin/python \
  -m research_program.qwen_agent_pilot.calibration_oracle_dev \
  --model /Users/xia/Models/Qwen3.5-9B-8bit \
  --seed 20260930 \
  --out research_program/qwen_agent_pilot/results/calibration_v3_development_20260930.json
```

The v3 oracle gate passed: 36/36 designated actions, 36/36 unassigned-helper waits, and 36/36 joint successes. The result is a task-control development run, not part of the matrix; see its [analysis](results/calibration_v3_development_20260930_analysis.md) and [integrity manifest](results/calibration_v3_development_20260930_manifest.json).

The oracle gate passed. The paired four-condition matrix is now complete: 432 episodes and 1,296 model calls on seeds `20260925`–`20260927`. Joint success was 0/108 with a blank channel, 106/108 with the environment-provided oracle, 71/108 with the known codebook, and 0/108 with free symbols. The designated helper never acted in the free-symbol arm, so its zero reward does not by itself tell whether any message carried partial meaning. The six-of-six agreement in one free-symbol seed came from a single constant string. See the [analysis and seed-level results](results/calibration_v3_20260920_analysis.md), the [episode records](results/calibration_v3_20260920.json), the [replay audit](results/calibration_v3_20260920_audit.json), and the [integrity manifest](results/calibration_v3_20260920_manifest.json).

To reproduce or resume the matrix with the same protocol and output path:

```sh
PYTHONPATH=. /Users/xia/.venvs/qwen35-mlx/bin/python \
  -m research_program.qwen_agent_pilot.calibration \
  --model /Users/xia/Models/Qwen3.5-9B-8bit \
  --out research_program/qwen_agent_pilot/results/calibration_v3_20260920.json
```

Pass `--resume` with the same output path to continue an interrupted matrix. Each completed seed-condition run is checkpointed atomically. Replay the finished record and verify its paired schedule with:

```sh
PYTHONPATH=. /Users/xia/.venvs/qwen35-mlx/bin/python \
  -m research_program.qwen_agent_pilot.audit_calibration_v3 \
  --results research_program/qwen_agent_pilot/results/calibration_v3_20260920.json \
  --out research_program/qwen_agent_pilot/results/calibration_v3_20260920_audit.json
```

## Reproduce locally

The pinned Apple Silicon environment is recorded in [`requirements.macos-arm64.lock`](requirements.macos-arm64.lock), and the model revision plus official expected shard hashes are in [`model_lock.json`](model_lock.json). Model weights are stored at `/Users/xia/Models/Qwen3.5-9B-8bit` and are not part of this repository. Start the API server bound to loopback:

```sh
/Users/xia/.venvs/qwen35-mlx/bin/python -m mlx_vlm.server \
  --host 127.0.0.1 --port 8080 \
  --model /Users/xia/Models/Qwen3.5-9B-8bit
```

Then, from the repository root, run the balanced pilot:

```sh
PYTHONPATH=. /Users/xia/.venvs/qwen35-mlx/bin/python \
  -m research_program.qwen_agent_pilot.pilot \
  --base-url http://127.0.0.1:8080/v1 \
  --model /Users/xia/Models/Qwen3.5-9B-8bit \
  --out research_program/qwen_agent_pilot/results/pilot_20260920.json
```

The output contains episode-level goals, messages, actions and outcomes, but no raw model completions or hidden reasoning. The pilot runner tests can be run without loading the model with:

```sh
PYTHONPATH=. /Users/xia/.venvs/qwen35-mlx/bin/python \
  -m research_program.qwen_agent_pilot.tests.test_pilot
```
