from research_program.triadic_factorized_neutral_altpartner_direction_probe import design


def test_cases_are_balanced_and_complete():
    prepared = design.make_prepared()
    assert prepared['partition']['world_count'] == 56160
    assert prepared['cases']['case_count'] == 76032
    assert sorted(design.neighbors(design.all_needs()).keys())[:0] == []


def test_condition_parse():
    assert design.parse_condition(design.CONDITIONS[0]) == ('static', True)
    assert design.parse_condition(design.CONDITIONS[-1]) == ('rematched', False)
