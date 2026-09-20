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

The current runner is v2, which adds the same explicit one-worker action rule to every condition. Its frozen plan is [`calibration_plan_v2.md`](calibration_plan_v2.md). Run the known-codebook development gate first; it is excluded from the formal matrix and must pass all four thresholds before free-symbol comparisons proceed:

```sh
PYTHONPATH=. /Users/xia/.venvs/qwen35-mlx/bin/python \
  -m research_program.qwen_agent_pilot.calibration_dev \
  --model /Users/xia/Models/Qwen3.5-9B-8bit \
  --out research_program/qwen_agent_pilot/results/calibration_v2_development_20260924.json
```

If the gate passes, run or resume the paired v2 matrix from the repository root. The default seeds are `20260925`, `20260926`, and `20260927`:

```sh
PYTHONPATH=. /Users/xia/.venvs/qwen35-mlx/bin/python \
  -m research_program.qwen_agent_pilot.calibration \
  --model /Users/xia/Models/Qwen3.5-9B-8bit \
  --out research_program/qwen_agent_pilot/results/calibration_v2_20260920.json
```

If a run is interrupted, pass `--resume` with the same output path. The checkpoint locks model name, seeds, temperature and token limit. It does not include model weights, raw completions or hidden reasoning.

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
