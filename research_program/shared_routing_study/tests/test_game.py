from __future__ import annotations

import numpy as np

from research_program.shared_routing_study import design, environment, policy, runner


def main():
    assert design.parse_child_condition("tied_routed_conflict_hidden_heldout_combo") == (
        "tied_routed", "conflict", "hidden", "heldout_combo"
    )
    assert design.parse_child_condition("routed_conflict_hidden_heldout_combo") == (
        "routed", "conflict", "hidden", "heldout_combo"
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
        if architecture in ("routed", "tied_routed"):
            assert policy.routing_probabilities(params, side="sender").shape == (2, 2)
            assert policy.routing_probabilities(params, side="receiver").shape == (2, 2)
    routed = policy.make_policy(9002, "routed")
    state = design.message_index(np.array([[0, 1], [-1, -1]], dtype=np.int64))
    assert np.array_equal(design.decode_message_state(state), np.array([[0, 1], [-1, -1]]))
    assert environment.recombination_messages([routed, routed], routed, ep, "routed", "aligned", "hidden").shape == (design.BATCH_SIZE, 2)
    tied = policy.make_policy(9003, "tied_routed")
    assert environment.recombination_messages([tied, tied], tied, ep, "tied_routed", "aligned", "hidden").shape == (design.BATCH_SIZE, 2)
    print("shared_routing_study tests passed")


if __name__ == "__main__":
    main()
