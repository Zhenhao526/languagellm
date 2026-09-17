import numpy as np

from research_program.triadic_action_dependency_study import dataset
from research_program.triadic_rematched_partner_confirmation_study import design, remap


def test_grid_and_pair_rotation():
    prepared = design.make_prepared()
    assert len(prepared['runs']) == 64
    assert [design.heldout_pair_for_seed(s) for s in design.SEEDS[:3]] == [0, 1, 2]
    assert all(len(prepared['seed_partitions'][str(s)]['train']['needs']) == 3584 for s in design.SEEDS)


def test_remap_preserves_public_features_and_relabels_reward_columns():
    spec = design.build_seed_partitions(design.SEEDS[0])['train']
    arrays = dataset.make_arrays(spec, information='PL')
    x = arrays['x_PL'][:6]; rewards = arrays['rewards'][:6]; states = arrays['packed_states'][:6]
    pids = np.arange(6, dtype=np.int8)
    out_x, out_rewards, out_states = remap.remap_batch(x, rewards, states, pids)
    assert np.allclose(out_x[:, :, 21:50], x[:, :, 21:50])
    assert np.array_equal(out_states[:, 3:], states[:, 3:])
    assert np.array_equal(out_states[:, :3], np.asarray([states[i, list(remap.PERMS[i])] for i in range(6)]))
    for row, pidx in enumerate(pids):
        assert np.array_equal(out_rewards[row], rewards[row, remap.PLAN_MAPS[pidx]])
        for actor in range(3):
            assert out_x[row, actor, 50 + actor] == 1
