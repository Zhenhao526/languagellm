# Open-world grounding transfer plan

## Question

When a new value has entered a culturally transmitted protocol, does zero-shot composition survive a stable change in the surface labels perceived by a replacement receiver? This separates transfer of atomic message coordinates from transfer of the receiver's semantic grounding.

## Frozen stages

1. Parents train on the 3x3 old world (`0,1,2`) with two four-symbol message slots.
2. An expansion child sees single-new meanings with a fresh sender and receiver together. `(3,3)` is absent. The child checkpoint is frozen as the incumbent culture.
3. A fresh sender, receiver, both roles, or neither role is replaced. For transfer episodes, incumbent receivers read canonical scene meanings. Fresh receivers read either canonical labels (`identity`) or `reverse=[3,2,1,0]` surface labels.

The reverse mapping is applied to scene attributes only; target semantics, sender goals, messages and action uniforms remain paired. Thus the intervention changes perceptual grounding while retaining a bijective object code.

## Predictions

- Fixed factorized transfer should preserve double-new recombination under identity, as in the preceding study.
- A reverse surface should impose an initial receiver-transfer cost. With single-new feedback, a factorized receiver may relearn atomic surface mappings and recover `(3,3)`; holistic transfer should remain poor and `tied_routed` should be sensitive to routing and direction.
- The 0-update control should fail under both mappings.
- If reverse-minus-identity is small after training while donor-recombined messages remain functional, the result supports separation of message-coordinate transmission from receiver grounding. If the reverse arm loses double-new recombination despite single-new recovery, the two mechanisms interact and must be modeled jointly.

## Controls and audit

Mapping arms share semantic scenes, goals, partners, message uniforms and action uniforms. Training logs additionally hash both canonical and fresh-receiver scene streams. Replay checks role schedules, no `(3,3)` leakage, parameter hashes, frozen incumbent hashes, mapping-specific scenes and paired streams.
