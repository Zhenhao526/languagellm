from __future__ import annotations

import numpy as np

from research_program.routing_initialization_control_study import design, environment, policy, runner


def main():
    assert design.parse_child_condition("tied_routed_aligned_hidden_heldout_combo") == (
        "tied_routed", "aligned", "hidden", "heldout_combo"
    )
    ep = design.episode_stream(design.SEEDS[0], design.BATCH_SIZE, support="full", update=1)
    for architecture in design.ARCHITECTURES:
        params = policy.make_policy(9001, architecture)
        probs, _ = policy.sender_arrays(params, ep["goal_meaning"], ep["partner_id"], "hidden")
        assert probs.shape == (design.BATCH_SIZE, 2, 4)
        assert np.allclose(probs.sum(-1), 1.0)
        tr = environment.rollout([params, params], None, ep, architecture, "aligned", "hidden")
        assert tr["receiver_probs"].shape == (design.BATCH_SIZE, design.SCENE_SIZE)
        assert np.allclose(tr["receiver_probs"].sum(-1), 1.0)
        grad = runner.parent_gradient(params, ep, tr)
        assert all(np.all(np.isfinite(value)) for value in grad.values())
        assert policy.routing_probabilities(params, side="sender").shape == (2, 2)
        assert policy.routing_probabilities(params, side="receiver").shape == (2, 2)
    sync = policy.make_policy(9002, "sync_routed")
    assert np.allclose(sync["sender_route_hidden"], sync["receiver_route_hidden"])
    state = design.message_index(np.array([[0, 1], [-1, -1]], dtype=np.int64))
    assert np.array_equal(design.decode_message_state(state), np.array([[0, 1], [-1, -1]]))
    for architecture in design.ARCHITECTURES:
        params = policy.make_policy(9003, architecture)
        assert environment.recombination_messages([params, params], params, ep, architecture, "aligned", "hidden").shape == (design.BATCH_SIZE, 2)
    print("routing_initialization_control_study tests passed")


if __name__ == "__main__":
    main()
