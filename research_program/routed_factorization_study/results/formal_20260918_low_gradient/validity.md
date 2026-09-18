# Validity

- `prepared.json`, `plan.json`, and `freeze.json` freeze the older source snapshot and configuration.
- The source snapshot is retained because this run predates the routed-study source commit.
- `audit.json` reports `status: passed` and `max_abs_replay_error: 0.0`.
- `results.json` is the compact endpoint payload; raw training trees were audited before deletion.
