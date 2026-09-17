from __future__ import annotations

import numpy as np
from research_program.partner_exposure_study import design, environment, policy


def main():
    assert len(design.PARENT_CONDITIONS) == 2
    assert len(design.CHILD_CONDITIONS) == 16
    assert design.parse_child_condition("heterogeneous_visible_coadapt_leave_one_out") == ("heterogeneous", "visible", "coadapt", "leave_one_out")
    seed = design.SEEDS[0]
    hom = design.episode_stream(seed, 256, population_mode="homogeneous", update=1)
    het = design.episode_stream(seed, 256, population_mode="heterogeneous", update=1)
    for name in ("site_semantic", "goal", "partner_id", "message_uniforms", "action_uniforms"):
        assert np.array_equal(hom[name], het[name]), name
    odd = np.isin(hom["partner_id"], [1, 3])
    assert np.any(hom["site_surface"][odd] != het["site_surface"][odd])
    assert np.array_equal(hom["site_surface"][~odd], het["site_surface"][~odd])
    loo = design.episode_stream(seed, 256, support="leave_one_out", heldout=design.heldout_goal(seed), update=1)
    assert not np.any(design.goal_index(loo["goal"]) == design.heldout_goal(seed))
    params = policy.make_policy(seed)
    hidden = environment.rollout(params, hom, "hidden", sample=False)
    visible = environment.rollout(params, hom, "visible", sample=False)
    assert hidden["actions"].shape == visible["actions"].shape == (256, design.HORIZON)
    assert len(hidden["message_probs"]) == design.MESSAGE_LENGTH
    grad = __import__("research_program.partner_exposure_study.runner", fromlist=["gradient"]).gradient(params, hom, hidden, "hidden")
    assert grad["sender_logits_hidden"].shape == params["sender_logits_hidden"].shape
    assert grad["sender_logits_visible"].shape == params["sender_logits_visible"].shape
    assert grad["worker_logits"].shape == params["worker_logits"].shape
    eval_ep = design.balanced_eval_stream(123, design.WORKERS * 32, "heldout", 0, "heterogeneous")
    assert np.bincount(eval_ep["partner_id"], minlength=design.WORKERS).tolist() == [32, 32, 32, 32]
    assert environment.recombination_override(params, eval_ep, "visible").shape == (128, design.MESSAGE_LENGTH)
    print("partner_exposure_study tests passed")


if __name__ == "__main__":
    main()
