# Validity and limits

The formal matrix contains 36 parent runs, 18 expansion children and 288 transfer runs across nine seeds. It crosses two architectures, two surface mappings, two receiver initializations and four role supports. Six independent replay shards cover all stages; maximum absolute replay error is 0.0, and the incumbent parameter hashes remain fixed. The `(3,3)` meaning is excluded from every transfer training stream.

An additional checkpoint comparison checked all receiver parameter arrays for each incumbent-receiver transfer initialization against its frozen expansion checkpoint. All 144 checks passed; the sender arrays remained freshly initialized. The audit is retained in `receiver_copy_audit.json`.

The primary receiver-only transfer endpoint measures the held-out double-new meaning after zero updates and after 3,000 single-new feedback updates. A supplemental paired Student t analysis uses nine seed-level contrasts and is retained separately from the original frozen analysis. No training code or endpoint definition changed after the formal run.

The object surface is a deterministic permutation of tabular attribute labels. This study does not include visual perception, open-ended meanings, linguistic agents, or human participants. It supports a narrow mechanism claim about receiver grounding and factorized transfer, not a claim that agents generated human language.
