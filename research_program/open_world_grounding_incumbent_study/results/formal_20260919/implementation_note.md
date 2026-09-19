# Implementation note

This formal follow-up adds receiver initialization to the open-world grounding-transfer protocol. The parent learns the old 3×3 ontology. Expansion children share single-new meanings involving value 3 and leave (3,3) out, then freeze. In transfer, an incumbent receiver is copied from the frozen expansion policy while sender parameters remain fresh; the fresh receiver sees either identity labels or the stable reverse bijection [3,2,1,0]. The `expanded_single_receiver` arm assigns the new-value evaluation and feedback role to the fresh receiver while the sender remains incumbent.

The fixed factorized policy indexes attribute slots separately; the holistic policy indexes complete meanings. The reverse intervention is a tabular label permutation, not an image or learned visual encoder.
