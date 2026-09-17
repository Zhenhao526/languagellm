import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_content_response_study import intervene


def test_diagonal_intervention_reconstructs_natural_live_rollout():
    networks = core.make_networks(91317)
    rng = np.random.default_rng(8)
    x = rng.normal(size=(23, 3, 54))
    natural = core.rollout(networks, x, True)
    senders = np.arange(23, dtype=np.int8) % 3
    donor = natural["messages"][np.arange(len(senders)), 0, senders, :]
    probe = intervene.first_window_intervene(networks, x, natural["messages"], senders, donor)
    expected_p, _ = core.base.policy_distribution(natural["action_logits"])
    assert np.array_equal(probe["messages"], natural["messages"])
    assert np.array_equal(probe["action_indices"], np.argmax(expected_p, axis=-1))
    assert np.array_equal(probe["action_probabilities"], expected_p)
    assert probe["neural_forward_samples"] == 6 * len(x)


def test_route_replaces_only_cross_sender_slots():
    first = np.zeros((2, 3, 4), dtype=np.int8)
    donor = np.ones((2, 4), dtype=np.int8)
    sender = np.array([0, 2], dtype=np.int8)
    route = intervene.replace_outward(first, sender, donor, True)
    natural = intervene.route_window(first, True)
    assert np.array_equal(route[0, 0], natural[0, 0])
    assert np.array_equal(route[1, 2], natural[1, 2])
    assert not np.array_equal(route[0, 1, :32], natural[0, 1, :32])
