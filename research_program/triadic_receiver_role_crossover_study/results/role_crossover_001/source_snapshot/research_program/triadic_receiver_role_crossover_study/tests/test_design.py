from research_program.triadic_receiver_role_crossover_study import design


def test_role_grid():
    assert design.ROLES == ("A", "B", "C")
    assert len(design.CONDITIONS) == 12
    assert len(design.SEEDS) * len(design.CONDITIONS) == 96


def test_condition_parser():
    for condition in design.CONDITIONS:
        role, schedule, live = design.parse_condition(condition)
        assert role in design.ROLES
        assert schedule in design.SCHEDULES
        assert isinstance(live, bool)
