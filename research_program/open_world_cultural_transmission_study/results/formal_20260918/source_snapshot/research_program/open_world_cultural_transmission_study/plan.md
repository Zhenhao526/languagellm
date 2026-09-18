# Open-world cultural transmission plan

## Question

When an extension for a new attribute value has formed, can a replacement worker learn it from an incumbent and use it to recover an unseen double-new meaning? This separates cultural transmission from parameter inheritance and from fresh–fresh coadaptation.

## Frozen design

Parents train for 3,000 updates on all nine meanings formed by values 0, 1 and 2. The expansion child then sees value 3, learns only single-new meanings with a fresh sender and receiver together, and is frozen. The message remains two slots of four symbols, so the expanded 4x4 meaning space is capacity matched to the message-state space.

- `holistic`: complete-meaning rows;
- `factorized`: fixed slot-to-attribute factor tables;
- `tied_routed`: role-specific factor tables with a shared learned routing matrix;
- expansion support `expanded_single_pair`: role 2 on single-new meanings and `(3,3)` excluded;
- transfer supports `expanded_single_sender` (fresh sender only), `expanded_single_receiver` (fresh receiver only), `expanded_single_pair` (both fresh), and `expanded_single_none` (frozen replacement control).

Old meanings retain the rotating role schedule. Every transfer condition has the same scene, target, partner, and sampling streams; only the fresh-role schedule differs. The incumbent is copied from the expansion child checkpoint and never updated during transfer.

## Predictions and interpretation

1. Factorized and tied-routed expansion children should recover single-new meanings and the held-out double-new meaning more often than holistic children.
2. If a replacement sender or receiver can recover the double-new meaning after single-new interaction with a frozen incumbent, the result supports cultural transmission of atomic coordinates rather than direct parameter inheritance.
3. Fresh–fresh transfer is an upper-bound coadaptation control, while `expanded_single_none` should fail. Natural and donor-recombined double-new messages should agree when the result is genuinely compositional.
4. Holistic transfer should be seed-sensitive or fail on the double-new endpoint because it has no shared coordinate across meanings.

This is a tabular mechanism test. It does not establish human-like language, open vocabulary, or cultural evolution by itself.

## Audit

Every update records stream hashes and parameter hashes. Expansion and transfer replay audits verify role schedules, no double-new leakage, checkpoint hashes, frozen incumbent hashes, and paired streams before raw trees are removed.
