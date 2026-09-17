# Repetition-pressure study

This package tests whether reusing the same action meaning across multiple consequence windows changes the structure and noise tolerance of learned shared symbols. `unique4` exposes four distinct balanced Boolean meanings once each, `repeat2` repeats two independent meanings twice each, and `shared4` repeats one hidden parity in all four stages. The three forms (`dual2`, `triple2`, `atomic8`), paired channel noise, rotating hidden partners and three adaptation modes are held constant.

The preregistered candidate criterion is natural return ≥ 0.60 plus minimum code distance ≥ 2; for `shared4`, the two goals sharing a parity must also have identical codewords. No teacher codebook, language model or semantic label is supplied. The formal matrix has 81 parent runs and 729 child runs.

## Formal result

The balanced formal matrix is archived in [`results/formal_20260918`](results/formal_20260918/). All three shards passed independent replay audit: 81 parent runs, 729 child runs, 243,000 parent training-log rows, 2,187,000 child training-log rows, 1,458,000 paired-noise trajectory rows, and maximum replay error 0.

For `coadapt` (nine seeds per cell), natural return for `unique4`/`repeat2` stayed below the 0.60 functional threshold in every noise arm. `shared4` reached 0.667 at p=0; `triple2` reached 0.637 at p=0.10, with 8/9 error-correcting candidates. No `unique4` or `repeat2` endpoint met the candidate criterion. The task-paired `repeat2−unique4` effect was small and inconsistent, while `shared4−unique4` was positive in most matched cells.

The `shared4` result is a controlled boundary case: four stages expose only two hidden parity classes, so high return does not demonstrate an open-ended language. It shows that repeated consequences can make a reusable code robust under this restricted meaning space. The superseded unbalanced pilot is retained only as [`results/invalid_20260917/unbalanced_unique4_partial_20260917/receipt.json`](results/invalid_20260917/unbalanced_unique4_partial_20260917/receipt.json); its statistics were never analyzed.
