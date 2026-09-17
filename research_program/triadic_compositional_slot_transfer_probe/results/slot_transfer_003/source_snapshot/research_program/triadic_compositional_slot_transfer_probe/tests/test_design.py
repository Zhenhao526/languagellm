"""Structural checks for the frozen single-slot design."""
from research_program.triadic_compositional_slot_transfer_probe import design


def test_masks_and_policy_grid():
    assert [name for name, _ in design.MASKS] == ["slot_0", "slot_1", "slot_2", "slot_3", "full"]
    assert len(design.make_prepared()["sources"]) == 64


def test_masks_are_disjoint_single_slots_and_full():
    masks = dict(design.MASKS)
    assert all(len(masks[f"slot_{i}"]) == 1 for i in range(4))
    assert masks["full"] == (0, 1, 2, 3)

