# Qwen v3 matched-channel calibration archive

- Repository: `git@github.com:Zhenhao526/languagellm.git`, branch `main`.
- Frozen protocol and runner source commit: `e78b4baa707e67402f5e2276f400fe1e690ac04e`.
- The paired matrix contains 12 seed-condition runs: 432 episodes and 1,296 model calls across blank, environment-oracle, known-codebook and free-symbol conditions.
- Independent replay audit passed for all 432 outcomes, all 12 summaries and 324 paired schedule comparisons.
- Episode data, audit output, analysis, and SHA-256 manifest are archived under `research_program/qwen_agent_pilot/results/calibration_v3_20260920*`.
- The report marks the matrix as screening-only. The oracle condition is an environment-provided task control, not peer communication.
- Raw model completions and hidden reasoning were not retained. Model weights remain outside Git at `/Users/xia/Models/Qwen3.5-9B-8bit`.
