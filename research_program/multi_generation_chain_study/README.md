# Three-generation protocol chain

This study extends the staged, factorized `dual2` parent protocol to a sequential three-event chain. Every seed has a held-out target combination during child training. Generation 1 replaces worker 0, generation 2 replaces the hidden sender, and generation 3 replaces worker 0 again. The final parameters of each event initialize the next event.

`joint_history` gives the replaced worker the complete received-message history; `slot_local` gives it only the current staged slot. Each representation has paired `live` and `silent` branches with shared worlds, goals, partners, message uniforms and action uniforms. Evaluation reports held-out natural return, silent and permuted controls, slot recombination, semantic success over local layouts, sender codebook fidelity, and composability transitions.

The compact archive contains no raw training logs or checkpoints. The full execution tree and parent checkpoints were used for the independent replay audit and removed after the archive receipt was written.
