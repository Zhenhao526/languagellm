from __future__ import annotations

import numpy as np

from research_program.shared_redundancy_study import design, environment, policy


def test_forms_and_stream_are_paired():
    assert design.parse_condition("triple2_worker_only_p10") == ("triple2", "worker_only", 0.10, "p10")
    a = design.episode_stream(78101, 16, "triple2", update=3)
    b = design.episode_stream(78101, 16, "triple2", update=3)
    for key in ("site_type", "goal", "partner_id", "message_uniforms", "action_uniforms", "noise_uniforms"):
        assert np.array_equal(a[key], b[key])


def test_noisy_rollout_shape():
    params = policy.make_policy(17, "dual2")
    episode = design.episode_stream(78101, 16, "dual2", update=1)
    trajectory = environment.rollout(params, episode, "dual2", "live", 0.10, sample=False)
    assert trajectory["actions"].shape == (16, design.HORIZON)


if __name__ == "__main__":
    test_forms_and_stream_are_paired()
    test_noisy_rollout_shape()
    print("shared_redundancy_study tests passed")
