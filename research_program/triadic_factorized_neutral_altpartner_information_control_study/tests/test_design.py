import numpy as np

from research_program.triadic_action_dependency_study import dataset
from research_program.triadic_factorized_neutral_altpartner_information_control_study import design, remap, runner, kernel


def test_grid_and_alternative_partner_support():
    prepared = design.make_prepared()
    assert len(prepared['runs']) == 64
    assert prepared['need_count'] == 1560
    assert prepared['pair_set_counts'] == {'AB_AC': 520, 'AB_BC': 520, 'AC_BC': 520}
    assert prepared['legal_plan_multiplicity'] == 2
    assert prepared['legal_pair_multiplicity'] == 2
    assert all(len(prepared['partitions'][part]['needs']) == 1560 for part in design.PARTS)
    assert design.parse_condition(design.CONDITIONS[0]) == ('static', 'altpair_factorized', 'strict', 'FI', True)
    assert design.parse_condition(design.CONDITIONS[-1]) == ('rematched', 'altpair_factorized', 'strict', 'FI', False)


def test_remap_preserves_two_full_plans_and_public_view():
    spec = design.make_prepared()['partitions']['train']
    arrays = dataset.make_arrays(spec, information='FI')
    x, rewards, states = arrays['x_FI'][:6], arrays['rewards'][:6], arrays['packed_states'][:6]
    pids = np.arange(6, dtype=np.int8)
    out_x, out_rewards, out_states = remap.remap_batch(x, rewards, states, pids)
    assert np.allclose(out_x[:, :, 21:50], x[:, :, 21:50])
    assert np.array_equal(out_states[:, 3:], states[:, 3:])
    assert np.array_equal(out_states[:, :3], np.asarray([states[i, list(remap.PERMS[i])] for i in range(6)]))
    assert np.all((out_rewards == 1).sum(axis=1) == 2)
    assert np.all(np.isin(out_rewards, (0., .5, 1.)))
    assert np.all(out_x[:, :, 53] == 1)


def test_factorized_kernel_shapes_and_neutral_head():
    spec = design.make_prepared()['partitions']['train']
    arrays = runner.make_arrays(spec)
    networks = runner.make_networks(66701)
    ix = np.arange(4)
    uniforms = runner.core.draw_uniforms(runner.core.make_message_rngs(66701), len(ix))
    gradients, row = kernel.training_gradients(networks, arrays['x_FI'][ix], arrays['rewards'][ix], True, uniforms, 1)
    assert len(gradients) == 9
    assert gradients[2]['W3'].shape == (64, 18)
    assert row['mean_engagement_probability'] >= 0
