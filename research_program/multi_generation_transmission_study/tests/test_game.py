from __future__ import annotations

import numpy as np

from research_program.action_dependent_signaling_study import policy
from research_program.multi_generation_transmission_study import design, environment, runner


def test_condition_and_stage_role():
    c = "worker_then_sender_g1live_g2silent"
    assert design.parse_condition(c) == ("worker_then_sender", "live", "silent")
    assert design.stage_condition(c, 1) == ("worker_then_sender", "live")
    assert design.stage_condition(c, 2) == ("worker_then_sender", "silent")
    assert runner.stage_role("sender_then_worker", 1) == "sender"
    assert runner.stage_role("sender_then_worker", 2) == "worker"


def test_slot_local_override_and_stream_pairing():
    p = policy.make_policy(123, "dual2")
    ep = design.episode_stream(76103, 1, 16, update=7)
    override = np.zeros((16, 2), dtype=np.int64)
    tr = environment.rollout_slot_local(p, ep, "live", sample=False, partner_filter=design.TARGET_WORKER, message_override=override)
    assert tr["selected_messages"].shape == (16, 2)
    assert np.array_equal(tr["selected_messages"], override)
    ep2 = design.episode_stream(76103, 1, 16, update=7)
    for key in ("site_type", "goal", "partner_id", "message_uniforms", "action_uniforms"):
        assert np.array_equal(ep[key], ep2[key])
