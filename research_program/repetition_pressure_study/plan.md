# Repetition-pressure curve

The formal matrix varies only the reuse pattern of task meanings while keeping four action stages, two private binary factors, capacity, partner rotation, channel noise and training budget fixed. `unique4` uses `[a, b, a xor b, not(a xor b)]`, `repeat2` uses `[a, a, b, b]`, and `shared4` uses `[a xor b]` in all four stages. All four `unique4` functions are balanced over the four goals. The task factor is crossed with `dual2`, `triple2`, `atomic8`, `scratch`, `worker_only`, `coadapt`, and noise p=0, 0.10, 0.25.

A functional endpoint has natural return at least 0.60; an error-correcting candidate additionally requires minimum code distance at least 2. For `shared4`, candidate codewords must be equal within parity classes; for the full-goal tasks, the minimum is over all four goal codewords.
