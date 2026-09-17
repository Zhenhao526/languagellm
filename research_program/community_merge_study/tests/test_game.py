from __future__ import annotations

import numpy as np

from research_program.community_merge_study import design, environment, policy, runner


def main():
    assert len(design.CHILD_CONDITIONS) == 48
    assert design.parse_child_condition("conflict_visible_coadapt_heldout_combo_alternating") == ("conflict", "visible", "coadapt", "heldout_combo", "alternating")
    seed = design.SEEDS[0]
    aligned = design.episode_stream(seed, 512, support="full", role="alternating", update=1)
    combo = design.episode_stream(seed, 512, support="heldout_combo", role="alternating", update=1)
    value = design.episode_stream(seed, 512, support="heldout_value", role="alternating", update=1)
    assert np.array_equal(aligned["scene_meanings"], combo["scene_meanings"])
    assert np.array_equal(aligned["scene_meanings"], value["scene_meanings"])
    assert not np.any(combo["goal_meaning"] == design.heldout_combo(seed))
    assert not np.any(value["goal_meaning"] // design.VALUES == design.heldout_value(seed))
    assert np.any(aligned["fresh_role"] == 0) and np.any(aligned["fresh_role"] == 1)
    cp = policy.make_policy(seed)
    fresh = policy.make_policy(seed + 1, visible=True)
    tr = environment.rollout([cp, cp], fresh, aligned, "conflict", "visible", "alternating", sample=True)
    assert tr["sender_probs"].shape == (512, design.MESSAGE_LENGTH, design.ALPHABET_SIZE)
    assert tr["receiver_probs"].shape == (512, design.SCENE_SIZE)
    assert np.allclose(tr["receiver_probs"].sum(axis=1), 1.0)
    grad_fresh, grad_comm = runner.child_gradients([cp, cp], fresh, aligned, tr, "visible", "coadapt")
    assert grad_fresh["sender_visible"].shape == fresh["sender_visible"].shape
    assert len(grad_comm) == design.COMMUNITIES
    parent_ep = dict(aligned, community_id=np.zeros(512, dtype=np.int8), partner_id=np.zeros(512, dtype=np.int8))
    parent_tr = environment.rollout([cp], None, parent_ep, "aligned", "hidden", "alternating", sample=False)
    assert parent_tr["actions"].shape == (512,)
    recombined = environment.recombination_messages([cp, cp], fresh, aligned, "conflict", "visible")
    assert recombined.shape == (512, design.MESSAGE_LENGTH)
    print("community_merge_study tests passed")


if __name__ == "__main__":
    main()
