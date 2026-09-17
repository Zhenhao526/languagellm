"""Frozen design for single-slot causal transfer on matched policies.

The source policies and heldout cases are inherited from the final matched
full-combination control.  Each intervention replaces only one first-window
message slot at the selected sender, or all four slots as a full-packet
reference.  The remaining receiver packet is left untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

from research_program.triadic_compositional_matched_control_semantic_transfer_probe import design as matched_design

ROOT = Path(__file__).resolve().parents[1]
SEEDS = matched_design.SEEDS
ARMS = matched_design.ARMS
SCHEDULES = matched_design.SCHEDULES
LIVES = matched_design.LIVES
CONDITIONS = matched_design.CONDITIONS
AXES = matched_design.case_design.AXES
CHUNK_SIZE = matched_design.CHUNK_SIZE
SHAM_ROWS_PER_SENDER = matched_design.SHAM_ROWS_PER_SENDER
MATCHED_RUN = ROOT / "triadic_compositional_matched_control_semantic_transfer_probe/results/matched_control_005"

MASKS = tuple(
    [(f"slot_{slot}", (slot,)) for slot in range(4)]
    + [("full", (0, 1, 2, 3))]
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def make_prepared():
    require(MATCHED_RUN.is_dir(), f"Missing matched-control authority: {MATCHED_RUN}")
    matched = json.loads((MATCHED_RUN / "prepared.json").read_text(encoding="utf8"))
    require(matched["schema"] == "triadic_compositional_matched_control_semantic_transfer_prepared_v1", "Unexpected matched source")
    require(matched["cases"]["case_count"] == 3456, "Unexpected heldout case count")
    return dict(
        schema="triadic_compositional_slot_transfer_prepared_v1",
        seeds=list(SEEDS), arms=list(ARMS), schedules=list(SCHEDULES), lives=["live", "silent"],
        conditions=list(CONDITIONS), axes=list(AXES), masks=[dict(name=n, slots=list(s)) for n, s in MASKS],
        heldout_spec=matched["heldout_spec"], cases=matched["cases"], sources=matched["sources"],
        matched_run=str(MATCHED_RUN), matched_prepared_sha256=None,
        scientific_question="Does a heldout aligned message move the receiver through a specific first-window slot, or only as an undifferentiated packet?",
        primary="aligned-minus-placebo plan transfer for each single slot, with full-packet reference",
        secondary="partner transfer, physical execution, Q, conditional Q, action-change rate and same-packet sham replay",
        intervention="For each mask, replace only the selected first-window token positions in the receiver's selected sender packet with the aligned or placebo donor tokens; recompute W2 and actions causally.",
        placebo="Same cyclic sender×axis×background donor mapping as matched_control_005; only selected slots are copied, so all masks share the same source/placebo case pairing.",
        controls="silent is a closed-channel alias; full is the pre-existing whole-packet reference; seen/all arms share matched parent and random streams.",
        no_training_updates=True,
        claim_boundary="A slot-selective response is a frozen-policy causal diagnostic, not evidence for lexical meaning, compositional syntax or language origin.",
    )


if __name__ == "__main__":
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))

