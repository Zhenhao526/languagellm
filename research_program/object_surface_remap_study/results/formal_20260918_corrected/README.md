# Formal archive: corrected binary object-surface remapping

This archive contains the corrected 9-seed matrix: 18 parent runs and 216 child runs. The all-goal evaluation stream groups four consecutive goal values under each partner, so the `permuted` control swaps messages across distinct goals. It compares capacity-matched `mono4` and staged `dual2`, `joint_history` and `slot_local` reception, identity versus swap mappings, full versus leave-one-goal-out support, and live versus silent channels.

`aggregate.json`, `aggregate.md`, `compact_results.json`, and `parent_shift.json` retain endpoint metrics, learning gains, codebooks, contrasts, and the incumbent parent identity-to-swap shift without training trajectories. `audit.json` records independent replay of all training logs, checkpoints, and paired streams. Raw execution trees were removed after hash verification; the full per-update result is intentionally not retained locally because the compact archive is the reproducibility record used in this repository.

The earlier invalid-control diagnostic is preserved at `../formal_20260918/` with its validity note.
