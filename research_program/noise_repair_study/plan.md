# Noisy transmission and protocol repair

This study starts from the nine parent seeds that passed the frozen staged dual2 compositionality rule. Worker 0 is replaced with a fresh receiver and trained under independent binary slot-flip noise at rates 0%, 10%, 25% and 40%.

Two adaptation regimes are crossed with two receiver representations:

- `worker_only` freezes the sender and workers 1–3. It tests whether a new receiver can recover the existing protocol under noisy delivery.
- `coadapt` updates the hidden sender and new worker together while workers 1–3 remain frozen. It tests whether sender–receiver repair can recover performance by renegotiating the code.
- `joint_history` indexes the complete received message history; `slot_local` indexes the current staged slot for worker 0. Incumbent workers always keep their parent joint-history representation.

Every condition shares the same per-update worlds, goals, partners, message uniforms, action uniforms and noise uniforms within a seed. Evaluation reports the new worker and frozen incumbent workers separately under clean, training-noise, silent, permuted and recombined messages. A noisy endpoint is composable when its natural return is at least 0.60 and its absolute recombined-minus-natural gap is at most 0.02.

The comparison separates maintenance from repair: worker-only degradation measures whether the old code survives noise; coadapt improvement accompanied by incumbent degradation indicates a renegotiated code rather than faithful transmission. Parent checkpoints and raw execution trees are temporary audit inputs; the repository archive keeps only frozen plans, compressed results and audit receipts.
