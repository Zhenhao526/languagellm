"""Fast invariants for the held-out composition design."""
from __future__ import annotations

import numpy as np

from research_program.action_dependent_signaling_study import design as base
from research_program.heldout_composition_study import design


def main():
    assert design.CONDITIONS == ("full_live", "full_silent", "leave_one_out_live", "leave_one_out_silent")
    assert [design.heldout_goal(seed) for seed in design.SEEDS[:8]] == [0, 1, 2, 3, 0, 1, 2, 3]
    ep = design.episode_stream(76101, 4096, support="leave_one_out", update=1)
    assert not np.any(base.goal_index(ep["goal"]) == design.heldout_goal(76101))
    ev = design.balanced_eval_stream(940001, 4096, "heldout", 2)
    assert np.all(base.goal_index(ev["goal"]) == 2)
    print("heldout_composition_study tests passed")


if __name__ == "__main__":
    main()
