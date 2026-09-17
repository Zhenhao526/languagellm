import numpy as np

from research_program.triadic_multi_legal_neutral_action_study import design


def test_condition_grid_and_parser():
    assert len(design.CONDITIONS) == 8
    parsed = [design.parse_condition(c) for c in design.CONDITIONS]
    assert {p[0] for p in parsed} == set(design.SCHEDULES)
    assert {p[1] for p in parsed} == set(design.CANCEL_REWARDS)
    assert {p[4] for p in parsed} == {False, True}


def test_multi_support_and_neutral_action():
    needs = design.multi_needs()
    assert len(needs) == 1452
    assert [sum(design.pair_index(row) == p for row in needs) for p in range(3)] == [484, 484, 484]
    prepared = design.make_prepared()
    assert prepared['explicit_neutral_action']['index'] == 17
    assert all(spec['world_count'] == 1452 * len(spec['layouts']) * 6 for spec in prepared['partitions'].values())

