# Shared routing formal archive

This compact archive contains the 9-seed, four-architecture shared-routing role-symmetry matrix. It has 144 parent runs and 432 child runs at 3,000 updates. `tied_routed` shares one learned slot-to-attribute routing matrix between sender and receiver while keeping their factor tables separate; `routed` learns independent routing matrices.

The combined replay audit is `passed`: 432,000 parent log rows, 1,296,000 child log rows, 576/1,728 checkpoints, all architecture/visibility/population/support paired streams, and maximum replay error 0.0. Per-shard audit receipts are under `part_audits/`; their `partial` labels indicate three-seed shards and are resolved by `combined_audit.json`.

The formal aligned/hidden held-out-combination means are factorized 0.948 (8/9 functional), tied_routed 0.552 (4/9), routed 0.050 (0/9), and holistic 0.028 (0/9). Seed-level contrasts and route mismatch are in `seed_analysis.json` and `seed_analysis.md`.

The frozen execution source commit was `5e4fd39dcc27ad74758d02d38e24ebe943c9e413`. Raw execution trees, logs and checkpoints are intentionally excluded from this archive and are removed only after the manifest and audits are verified.
