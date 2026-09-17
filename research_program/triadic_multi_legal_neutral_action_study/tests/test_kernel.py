import numpy as np

from research_program.triadic_multi_legal_neutral_action_study import kernel


def test_cancel_event_has_finite_exact_objective_and_zero_sum_logit_gradient():
    logits = np.zeros((3, 3, 18), dtype=np.float64)
    rewards = np.zeros((3, 24), dtype=np.float64)
    rewards[:, :2] = 1.0
    terms = kernel.objective_terms(logits, rewards, 0.1)
    assert terms['J'].shape == (3,)
    assert np.all(np.isfinite(terms['log_J']))
    assert terms['posterior_weights'].shape == (3, 25)
    assert np.allclose(terms['log_J_logit_gradient'].sum(axis=-1), 0.0, atol=1e-12)
    assert np.all(terms['cancel_protocol_probability'] > 0)

