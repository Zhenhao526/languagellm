"""Fast invariants for representation-controlled staged decoding."""
from __future__ import annotations

import numpy as np

from research_program.action_dependent_signaling_study import design as base
from research_program.receiver_representation_study import design, environment


def main():
    assert design.parse_condition("joint_history_leave_one_out_live") == ("joint_history", "leave_one_out", "live")
    ep = design.episode_stream(76101, 2048, support="leave_one_out", update=1)
    assert not np.any(base.goal_index(ep["goal"]) == design.heldout_goal(76101))
    ev = design.balanced_eval_stream(950001, 4096, "heldout", 2)
    assert np.all(base.goal_index(ev["goal"]) == 2)
    from research_program.action_dependent_signaling_study import policy
    p = policy.make_policy(12, "dual2")
    tr = environment.rollout_slot_local(p, ep, "live", sample=False, partner_filter=0)
    assert tr["actions"].shape == (len(ep["goal"]), base.HORIZON)
    assert all(np.asarray(state).max() <= 2 for state in tr["state"] if state is not None)
    print("receiver_representation_study tests passed")


if __name__ == "__main__":
    main()
