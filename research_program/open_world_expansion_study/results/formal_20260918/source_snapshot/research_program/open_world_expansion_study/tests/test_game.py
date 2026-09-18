from __future__ import annotations

import numpy as np

from research_program.open_world_expansion_study import design, environment, policy, runner


def main():
    assert design.parse_child_condition("tied_routed_aligned_hidden_expanded_pair") == (
        "tied_routed", "aligned", "hidden", "expanded_pair"
    )
    old = design.episode_stream(design.SEEDS[0], design.BATCH_SIZE, support="old_world", update=1)
    pair = design.episode_stream(design.SEEDS[0], design.BATCH_SIZE, support="expanded_pair", update=1)
    single_pair = design.episode_stream(design.SEEDS[0], design.BATCH_SIZE,
                                        support="expanded_single_pair", update=1)
    assert np.all(np.isin(old["goal_meaning"], design.OLD_MEANING_INDICES))
    assert not np.any(single_pair["goal_meaning"] == design.NEW_VALUE * design.VALUES + design.NEW_VALUE)
    eval_ep = design.balanced_eval_stream(
        design.SEEDS[0], 32, "new_single", "expanded_pair", design.heldout_combo(design.SEEDS[0])
    )
    for scene, target in zip(eval_ep["scene_meanings"], eval_ep["goal_meaning"]):
        attrs = design.attrs_from_meaning(target)
        scene_attrs = design.attrs_from_meaning(scene)
        assert np.any((scene_attrs[:, 0] == attrs[0]) & (scene != target))
        assert np.any((scene_attrs[:, 1] == attrs[1]) & (scene != target))
    new_mask = design.is_new_meaning(pair["goal_meaning"])
    assert np.all(pair["fresh_role"][new_mask] == 2)
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
    print("open_world_expansion_study tests passed")


if __name__ == "__main__":
    main()
