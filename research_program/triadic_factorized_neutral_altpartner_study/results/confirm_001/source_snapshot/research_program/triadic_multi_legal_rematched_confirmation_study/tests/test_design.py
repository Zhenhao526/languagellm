import numpy as np

from research_program.triadic_action_dependency_study import dataset
from research_program.triadic_multi_legal_rematched_confirmation_study import design, remap, runner


def test_grid_and_two_legal_support():
    prepared = design.make_prepared()
    assert len(prepared['runs']) == 64
    assert prepared['need_count'] == 1452
    assert prepared['pair_counts'] == [484, 484, 484]
    assert prepared['legal_plan_multiplicity'] == 2
    assert all(len(prepared['partitions'][part]['needs']) == 1452 for part in design.PARTS)
    assert design.parse_condition(design.CONDITIONS[0]) == ('static', 'partial_multi', 'strict', True)
    assert design.parse_condition(design.CONDITIONS[-1]) == ('rematched', 'partial_multi', 'strict', False)


def test_remap_preserves_multi_legal_structure_and_public_view():
    spec = design.make_prepared()['partitions']['train']
    arrays = dataset.make_arrays(spec, information='PL')
    x, rewards, states = arrays['x_PL'][:6], arrays['rewards'][:6], arrays['packed_states'][:6]
    pids = np.arange(6, dtype=np.int8)
    out_x, out_rewards, out_states = remap.remap_batch(x, rewards, states, pids)
    assert np.allclose(out_x[:, :, 21:50], x[:, :, 21:50])
    assert np.array_equal(out_states[:, 3:], states[:, 3:])
    assert np.array_equal(out_states[:, :3], np.asarray([states[i, list(remap.PERMS[i])] for i in range(6)]))
    assert np.all((out_rewards == 1).sum(axis=1) == 2)
    assert np.all(np.isin(out_rewards, (0., .5, 1.)))
    for row in range(6):
        for actor in range(3):
            assert out_x[row, actor, 50 + actor] == 1


def test_array_builder_rejects_no_hidden_information():
    spec = design.make_prepared()['partitions']['train']
    arrays = runner.make_arrays(spec)
    assert arrays['x_PL'].shape[1:] == (3, 54)
    assert np.all(arrays['x_PL'][:, :, 53] == 0)
