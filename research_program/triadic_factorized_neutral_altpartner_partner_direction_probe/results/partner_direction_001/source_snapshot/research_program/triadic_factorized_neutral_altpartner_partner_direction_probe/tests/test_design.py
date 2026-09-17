from research_program.triadic_factorized_neutral_altpartner_partner_direction_probe import design
from research_program.triadic_action_dependency_study import environment


def test_composite_cases_change_partner_set():
    prepared = design.make_prepared()
    assert prepared['partition']['world_count'] == 1560 * 6 * 6
    assert prepared['cases']['case_count'] == 3456 * 6 * 6
    assert set(prepared['cases']['axis']) == set(design.AXES)
    needs = [tuple(x) for x in prepared['partition']['needs']]
    lookup = {x: i for i, x in enumerate(needs)}
    for ri, di in zip(prepared['cases']['receiver_state_indices'][:1000], prepared['cases']['donor_state_indices'][:1000]):
        nl, no = 6, 6
        rr = needs[(ri // no) // nl]
        dr = needs[(di // no) // nl]
        rp = {tuple(p[:2]) for p in environment.full_success_plans(rr)}
        dp = {tuple(p[:2]) for p in environment.full_success_plans(dr)}
        assert rp != dp


def test_all_composite_axes_have_expected_counts():
    prepared = design.make_prepared()
    counts = {axis: prepared['cases']['axis'].count(axis) for axis in design.AXES}
    assert counts == {'kind_length': 1536 * 36, 'kind_destination': 960 * 36,
                      'length_destination': 960 * 36}
