from research_program.triadic_factorized_neutral_altpartner_fi_capacity_pilot import design


def test_fi_pilot_is_independent_and_paired():
    prepared = design.make_prepared()
    assert prepared['information'] == 'FI'
    assert prepared['seeds'] == list(range(66721, 66729))
    assert len(prepared['runs']) == 32
    assert all(row['information'] == 'FI' for row in prepared['runs'])
