# Repetition-pressure study

This package tests whether reusing the same action meaning across multiple consequence windows changes the structure and noise tolerance of learned shared symbols. `unique4` exposes four distinct Boolean meanings once each, `repeat2` repeats two independent meanings twice each, and `shared4` repeats one hidden parity in all four stages. The three forms (`dual2`, `triple2`, `atomic8`), paired channel noise, rotating hidden partners and three adaptation modes are held constant.

The preregistered candidate criterion is natural return ≥ 0.60 plus minimum code distance ≥ 2; for `shared4`, the two goals sharing a parity must also have identical codewords. No teacher codebook, language model or semantic label is supplied.
