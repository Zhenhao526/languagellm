from __future__ import annotations

import numpy as np

from research_program.ecological_factorization_study import design, environment, policy


def main():
    assert len(design.GOAL_PAIRS) == 9
    assert len(set(design.HOLISTIC_TARGET_MAP)) == 9
    assert design.parse_parent_condition("tri3_factorized") == ("tri3", "factorized")
    assert design.parse_child_condition("slot_local_tri3_factorized_leave_one_out_live") == ("slot_local", "tri3", "factorized", "leave_one_out", "live")
    ep = design.episode_stream(78101, 4096, task="factorized", support="leave_one_out", heldout=2, update=1)
    assert not np.any(design.goal_index(ep["goal"]) == 2)
    assert np.all(np.sort(ep["site_type"], axis=-1) == np.arange(design.OBJECT_TYPES))
    p = policy.make_policy(7, "tri3", "slot_local")
    tr = environment.rollout(p, ep, "tri3", "factorized", "slot_local", "live", sample=False, partner_filter=0)
    assert tr["actions"].shape == (len(ep["goal"]), design.HORIZON)
    assert all(np.asarray(state).max() <= design.ALPHABET_SIZES["tri3"] for state in tr["state"] if state is not None)
    override = environment.recombination_override(p, ep, "tri3", "factorized", basis="raw")
    assert override.shape == (len(ep["goal"]), 2)
    print("ecological_factorization_study tests passed")


if __name__ == "__main__":
    main()
