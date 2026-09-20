# Three-context local Qwen agent pilot

This pilot moves the validated collection-and-selection task from tabular policies to three independent Qwen contexts on one local model server. Each round one agent privately receives an order containing an object, an attribute, a responsible partner and a destination. It broadcasts only a 4–8 character string from `@#%&+=?~`; the other two agents see the candidate collection board and choose whether to act, which item to collect and where to deliver it. Roles rotate across agents.

The pilot uses no processing stage, road event, shared image, object index as a semantic label, or natural-language inter-agent channel. The world board is structured text, and only the public candidate inventory is shared. Each agent has a separate message history. Actions and scalar team feedback are applied by the environment, not written by another agent. Six fixed meaning tuples are crossed with all three requester identities in each of two seed-shuffled blocks (36 episodes), so each requester sends each meaning twice.

The first run is a feasibility and prompt-compliance test, not evidence of language emergence. Qwen already has strong linguistic and symbolic priors; a successful convention may reflect those priors and shared instructions. The run must therefore be interpreted alongside the non-linguistic policy results and later baseline/ablation conditions.

## Reproduce locally

The pinned Apple Silicon environment is recorded in [`requirements.macos-arm64.lock`](requirements.macos-arm64.lock). Model weights are stored at `/Users/xia/Models/Qwen3.5-9B-8bit` and are not part of this repository. Start the API server bound to loopback:

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
