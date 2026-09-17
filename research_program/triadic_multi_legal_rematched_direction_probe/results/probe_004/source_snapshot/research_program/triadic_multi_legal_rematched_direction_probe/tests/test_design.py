from research_program.triadic_multi_legal_rematched_direction_probe import design


def test_case_grid_and_neighbor_support():
    prepared = design.make_prepared()
    assert len(prepared['seeds']) == 16
    assert len(prepared['conditions']) == 4
    assert prepared['cases']['case_count'] == 151632
    assert prepared['partition']['world_count'] == 50544
    assert design.parse_condition(design.CONDITIONS[0]) == ('static', True)
    assert design.parse_condition(design.CONDITIONS[-1]) == ('rematched', False)
