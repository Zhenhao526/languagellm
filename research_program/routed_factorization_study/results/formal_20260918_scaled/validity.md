# Validity

- `prepared.json`, `plan.json`, and `freeze.json` freeze the design and source hashes.
- `source_commit.txt` records the source commit used for the formal run.
- `audit.json` reports `status: passed` and `max_abs_replay_error: 0.0`.
- `results.json` is the compact endpoint payload; raw training trees were audited before deletion.
