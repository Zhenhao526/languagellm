# Open-world cultural transmission study

This study asks whether a single-new extension that has already formed can be learned by a replacement worker and then support an unseen double-new meaning. Parents first train on values `0,1,2` only. An expansion child learns the six single-new meanings with a fresh sender and receiver together; `(3,3)` is absent from that support. The resulting child checkpoint is then frozen as the incumbent culture.

The transfer stage replaces the sender, the receiver, both roles, or neither role (`expanded_single_sender`, `expanded_single_receiver`, `expanded_single_pair`, `expanded_single_none`). Only the fresh policy is updated; the incumbent checkpoint is replay-checked as constant. All transfer conditions train on old meanings plus single-new meanings and are evaluated on the held-out double-new meaning.

The policies are `holistic`, fixed `factorized`, and `tied_routed`. Primary endpoints are new-single recovery, new-double recovery, natural versus recombined double-new return, and paired transfer contrasts. This is a mechanism test of cultural learning and compositional reconstruction, not a claim that a tabular agent has human language.
