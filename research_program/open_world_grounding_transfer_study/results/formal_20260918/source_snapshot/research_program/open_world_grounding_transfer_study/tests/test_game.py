from __future__ import annotations

import numpy as np

from research_program.open_world_grounding_transfer_study import design, environment, policy, runner


def main():
    assert design.parse_child_condition("tied_routed_aligned_hidden_expanded_single_pair") == (
        "tied_routed", "aligned", "hidden", "expanded_single_pair"
    )
    assert design.parse_transfer_condition("factorized_aligned_hidden_reverse_expanded_single_sender") == (
        "factorized", "aligned", "hidden", "reverse", "expanded_single_sender"
    )
    old = design.episode_stream(design.SEEDS[0], design.BATCH_SIZE, support="old_world", update=1)
    pair = design.episode_stream(design.SEEDS[0], design.BATCH_SIZE,
                                 support="expanded_single_pair", update=1)
    assert np.all(np.isin(old["goal_meaning"], design.OLD_MEANING_INDICES))
    assert not np.any(pair["goal_meaning"] == design.NEW_VALUE * design.VALUES + design.NEW_VALUE)
    reverse = design.episode_stream(design.SEEDS[0], design.BATCH_SIZE,
                                    support="expanded_single_pair", update=1,
                                    surface_mapping="reverse")
    assert np.any(reverse["fresh_scene_meanings"] != reverse["scene_meanings"])
    assert np.array_equal(design.object_surface_permutation("reverse"), np.array([3, 2, 1, 0]))
    eval_ep = design.balanced_eval_stream(
        design.SEEDS[0], 32, "new_single", "expanded_single_pair", design.heldout_combo(design.SEEDS[0])
    )
    for scene, target in zip(eval_ep["scene_meanings"], eval_ep["goal_meaning"]):
        attrs = design.attrs_from_meaning(target)
        scene_attrs = design.attrs_from_meaning(scene)
        assert np.any((scene_attrs[:, 0] == attrs[0]) & (scene != target))
        assert np.any((scene_attrs[:, 1] == attrs[1]) & (scene != target))
    new_mask = design.is_new_meaning(pair["goal_meaning"])
    assert np.all(pair["fresh_role"][new_mask] == 2)
    sender_ep = design.episode_stream(design.SEEDS[0], design.BATCH_SIZE,
                                      support="expanded_single_sender", update=1)
    receiver_ep = design.episode_stream(design.SEEDS[0], design.BATCH_SIZE,
                                        support="expanded_single_receiver", update=1)
    sender_mask = design.is_new_meaning(sender_ep["goal_meaning"])
    receiver_mask = design.is_new_meaning(receiver_ep["goal_meaning"])
    assert np.all(sender_ep["fresh_role"][sender_mask] == 0)
    assert np.all(receiver_ep["fresh_role"][receiver_mask] == 1)
    assert not np.any(sender_ep["goal_meaning"] == 15)
    assert not np.any(receiver_ep["goal_meaning"] == 15)
    for architecture in design.ARCHITECTURES:
        params = policy.make_policy(9001, architecture)
        probs, _ = policy.sender_arrays(params, old["goal_meaning"], old["partner_id"], "hidden")
        assert probs.shape == (design.BATCH_SIZE, design.MESSAGE_LENGTH, design.ALPHABET_SIZE)
        assert np.allclose(probs.sum(-1), 1.0)
        tr = environment.rollout([params, params], None, old, architecture, "aligned", "hidden")
        assert tr["receiver_probs"].shape == (design.BATCH_SIZE, design.SCENE_SIZE)
        assert np.allclose(tr["receiver_probs"].sum(-1), 1.0)
        grad = runner.parent_gradient(params, old, tr)
        assert all(np.all(np.isfinite(value)) for value in grad.values())
        child = policy.make_policy(9002, architecture)
        tr2 = environment.rollout([params, params], child, pair, architecture, "aligned", "hidden")
        assert np.all(tr2["sender_fresh"][new_mask])
        assert np.all(tr2["receiver_fresh"][new_mask])
        grad2 = runner.child_gradient(child, pair, tr2, "hidden")
        assert all(np.all(np.isfinite(value)) for value in grad2.values())
        if architecture == "tied_routed":
            assert policy.routing_probabilities(params, side="sender").shape == (design.MESSAGE_LENGTH, design.ATTRIBUTES)
    state = design.message_index(np.array([[0, 1], [-1, -1]], dtype=np.int64))
    assert np.array_equal(design.decode_message_state(state), np.array([[0, 1], [-1, -1]]))
    print("open_world_grounding_transfer_study tests passed")


if __name__ == "__main__":
    main()
