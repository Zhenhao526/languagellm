from research_program.triadic_receiver_role_crossover_confirmatory_semantic_transfer_probe import design


def test_cases_and_placebo_are_frozen_shape():
    spec = design.source_spec()
    cases = design.make_cases(spec)
    assert cases["case_count"] == 76032
    assert len(cases["placebo_donor_case_indices"]) == cases["case_count"]
    assert all(index != donor for index, donor in enumerate(cases["placebo_donor_case_indices"]))


def test_policy_grid():
    assert len(design.SEEDS) * len(design.CONDITIONS) == 96
    assert set(design.ROLES) == {"A", "B", "C"}
    assert set(design.AXES) == {"kind", "length", "destination"}
