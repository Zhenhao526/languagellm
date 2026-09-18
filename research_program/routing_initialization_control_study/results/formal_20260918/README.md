# Routing initialization control formal archive (2026-09-18)

This compact archive records the nine-seed control for the shared-routing result.
The `routed` arm uses independent random route initialization and updates,
`sync_routed` uses elementwise-identical initial route matrices but updates the
two roles independently, and `tied_routed` keeps one shared route parameter
during learning. All arms retain separate atomic factor tables.

The frozen aligned/hidden matrix has 54 parent runs and 81 child runs, each
trained for 3,000 updates. The primary endpoint is natural return on the
held-out-combination support. `results.json` is the compact endpoint payload;
`aggregate.*` and `seed_analysis.*` contain seed summaries; `combined_audit.json`
and `part_audits/` contain the independent replay checks.

Raw training logs and checkpoints were used for audit and then removed after
this archive was verified. The frozen source snapshot is included for the
code that generated the runs; `seed_analysis.py` is included as a reporting
helper and is not part of the runner source hash list.
