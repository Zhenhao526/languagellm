# Open-world incumbent grounding plan

## Question

When a culturally transmitted protocol meets a stable perceptual surface change, does an incumbent receiver lose its grounding, and can single-new feedback restore the held-out double-new composition?

## Conditions

- Parent: values `0,1,2`; expansion child: single-new value `3`; `(3,3)` absent.
- Transfer support: fresh sender, fresh receiver, both, or 0-update replacement.
- Surface mapping: `identity` or `reverse=[3,2,1,0]` for fresh receiver scenes.
- Receiver initialization: `fresh` random receiver; `incumbent_receiver` copies only the frozen incumbent receiver parameters. The sender stays fresh in both cases.
- Architectures: holistic and fixed factorized, so receiver-copying does not share a routing parameter with the sender.

The key endpoint is held-out double-new natural return and donor-recombined return. Initial-to-final gain measures surface repair. `incumbent_receiver/reverse` is the direct grounding-collapse condition; `fresh/reverse` controls ordinary learning under the same perceptual code.

## Predictions

Fixed factorized should preserve or recover double-new recombination after an incumbent receiver sees a stable reverse surface, while holistic should fail because the new joint meaning has no shared coordinate. The incumbent receiver should show a larger initial reverse-minus-identity loss than the fresh receiver, followed by a feedback-dependent repair gain. The 0-update arm and no-feedback double-new endpoint rule out parameter copying alone.

## Audit

Every training row hashes canonical and fresh-receiver scenes. Replay checks no `(3,3)` leakage, parameter hashes, copied receiver parameters, frozen incumbent hashes, support/mapping/init paired streams and checkpoint hashes.
