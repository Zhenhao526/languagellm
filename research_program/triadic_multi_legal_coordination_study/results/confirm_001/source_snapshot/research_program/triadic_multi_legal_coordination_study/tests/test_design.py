import numpy as np

from research_program.triadic_multi_legal_coordination_study import design, kernel
from research_program.triadic_action_dependency_study import dataset


def test_multi_legal_domain():
    prepared = design.make_prepared()
    assert len(prepared['runs']) == 32
    assert prepared['need_count'] == 1452
    assert prepared['pair_counts'] == [484, 484, 484]
    assert all(len(prepared['partitions']['train']['needs']) == 1452 for _ in [0])


def test_multi_kernel_accepts_two_full_plans():
    spec = design.make_prepared()['partitions']['train']
    arrays = dataset.make_arrays(spec, information='PL')
    rewards = arrays['rewards'][:8]
    assert np.all((rewards == 1).sum(axis=1) == 2)
    logits = np.zeros((8, 3, 17), dtype=np.float64)
    terms = kernel.objective_terms(logits, rewards)
    assert np.isfinite(terms['log_J']).all()
    assert np.all((terms['full_success_probability'] > 0))
