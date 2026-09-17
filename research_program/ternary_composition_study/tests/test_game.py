from __future__ import annotations

import numpy as np

from research_program.ternary_composition_study import design, environment, policy, runner


def test_design_and_stream():
    assert design.parse_condition("tri3_factorized_staged_live") == ("tri3", "factorized", "staged", "live")
    assert design.parse_condition("mono9_entangled_simultaneous_silent") == ("mono9", "entangled", "simultaneous", "silent")
    a = design.episode_stream(77101, 16, "tri3", "factorized", update=4)
    b = design.episode_stream(77101, 16, "tri3", "factorized", update=4)
    for key in ("site_type", "goal", "partner_id", "message_uniforms", "action_uniforms"):
        assert np.array_equal(a[key], b[key])
    assert np.all(np.sort(a["site_type"], axis=-1) == np.arange(3))


def test_rollout_and_recombination():
    p = policy.make_policy(17, "tri3")
    ep = design.episode_stream(77101, 32, "tri3", "factorized", update=1)
    tr = environment.rollout(p, ep, "tri3", "factorized", "staged", "live", sample=False, partner_filter=0)
    assert tr["actions"].shape == (32, design.HORIZON)
    assert runner.recombination_override(p, ep, "tri3", "factorized").shape == (32, 2)
