from __future__ import annotations

import numpy as np

from research_program.action_dependent_signaling_study import policy
from research_program.noise_repair_study import design, environment, runner


def test_conditions_and_noise_pairing():
    assert design.parse_condition("worker_only_slot_local_p25") == ("worker_only", "slot_local", 0.25, "p25")
    assert design.parse_condition("coadapt_joint_history_p40") == ("coadapt", "joint_history", 0.4, "p40")
    a = design.episode_stream(76103, 32, update=7)
    b = design.episode_stream(76103, 32, update=7)
    for key in ("site_type", "goal", "partner_id", "message_uniforms", "action_uniforms", "noise_uniforms"):
        assert np.array_equal(a[key], b[key])


def test_noisy_delivery_changes_only_non_null_tokens():
    params = policy.make_policy(123, "dual2")
    ep = design.episode_stream(76103, 16, update=3)
    tr = environment.rollout(params, ep, "slot_local", "live", 1.0, sample=False, partner_filter=design.TARGET_WORKER)
    selected = tr["selected_messages"]
    delivered = tr["delivered_messages"]
    assert np.all((delivered == (1 - selected)) | (delivered == design.NULL_MESSAGE))
    assert tr["actions"].shape == (16, design.HORIZON)


def test_stage_role_and_codebook():
    assert runner.sender_codebook(policy.make_policy(12, "dual2"))
