from research_program.triadic_compositional_holdout_semantic_transfer_probe import design


def test_heldout_cases():
    cases = design.make_cases(design.source_prepared())
    assert cases["case_count"] == 3456
    assert cases["strata_count"] == 216
    assert all(i != j for i, j in enumerate(cases["placebo_donor_case_indices"]))


def test_policy_grid():
    assert len(design.SEEDS) * len(design.CONDITIONS) == 32
