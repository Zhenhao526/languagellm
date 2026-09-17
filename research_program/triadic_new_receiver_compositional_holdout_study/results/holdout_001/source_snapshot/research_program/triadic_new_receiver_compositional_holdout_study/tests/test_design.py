from __future__ import annotations

from .. import design


def test_split_is_joint_and_marginally_complete():
    split = design.split_needs()
    assert len(split["train"]) == 1248
    assert len(split["heldout"]) == 312
    assert all(len({row[a] for row in split["train"]}) == 24 for a in range(3))
    assert all(len({row[a] for row in split["heldout"]}) == 24 for a in range(3))
    assert len(set(map(tuple, split["train"])) & set(map(tuple, split["heldout"]))) == 0


def test_condition_grid():
    assert len(design.CONDITIONS) == 4
    for condition in design.CONDITIONS:
        schedule, arm, live = design.parse_condition(condition)
        assert schedule in design.SCHEDULES
        assert arm == design.ARM
        assert live is condition.endswith("_live")
