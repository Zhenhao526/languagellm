# Binary object-surface remapping

This study asks whether a fresh worker can reuse a parent protocol when the
visible surface labels of objects are stably swapped. The latent goals and
semantic scenes remain paired; only the surface-to-semantic mapping changes.

- `dual2`: two binary staged slots, four complete message states;
- `mono4`: one four-valued token, the capacity-matched atomic control;
- `joint_history` versus `slot_local`: accumulated versus stage-local receipt;
- identity versus stable surface-label swap, full versus leave-one-goal-out
  support, and live versus silent communication.

The parent learns on all four binary goal combinations. A new worker replaces
worker 0 with the sender frozen and learns from reward. The primary endpoint is
leave-one-out live held-out return; permutation and silent controls check that a
surface remap result is mediated by the message rather than a fixed action
shortcut.

The all-goal evaluation stream groups four consecutive goal values under the
same partner. This is required for the permutation control to exchange messages
across distinct goals, and a regression test checks that the control is not a
no-op. The first 9-seed run is retained as a replay-audited diagnostic archive
because it used a synchronized partner/goal cycle that made this control
ineffective. The replacement run will be archived separately after the corrected
stream passes audit.

This is a finite tabular protocol-transfer experiment. It measures the
conditions for local repair and transfer; it does not claim that the agents
have acquired human language.
