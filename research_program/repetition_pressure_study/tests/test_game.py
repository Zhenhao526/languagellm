from __future__ import annotations

import numpy as np

from research_program.repetition_pressure_study import design, environment, policy


def test_conditions_and_stream_are_paired():
    assert design.parse_condition("repeat2_triple2_worker_only_p10") == ("repeat2", "triple2", "worker_only", 0.10, "p10")
    a = design.episode_stream(80101, 16, "triple2", "repeat2", update=3)
    b = design.episode_stream(80101, 16, "triple2", "repeat2", update=3)
    for key in ("site_type", "goal", "target_bits", "partner_id", "message_uniforms", "action_uniforms", "noise_uniforms"):
        assert np.array_equal(a[key], b[key])
    assert a["target_bits"].shape == (16, 4)


def test_task_meanings_are_balanced():
    goal = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    unique = design.target_bits(goal, "unique4")
    repeat = design.target_bits(goal, "repeat2")
    shared = design.target_bits(goal, "shared4")
    assert np.unique(unique, axis=0).shape[0] == 4
    assert np.all(unique.mean(axis=0) == 0.5)
    assert np.array_equal(repeat[:, :2], repeat[:, [0, 0]])
    assert np.array_equal(repeat[:, 2:], repeat[:, [2, 2]])
    assert np.array_equal(shared[:, 0], shared[:, 1])
    assert np.array_equal(shared[:, 0], np.bitwise_xor(goal[:, 0], goal[:, 1]))


def test_noisy_rollout_shape():
    params = policy.make_policy(17, "dual2")
    episode = design.episode_stream(80101, 16, "dual2", "shared4", update=1)
    trajectory = environment.rollout(params, episode, "dual2", "live", 0.10, sample=False)
    assert trajectory["actions"].shape == (16, design.HORIZON)
    assert trajectory["rewards"].shape == (16, design.HORIZON)


if __name__ == "__main__":
    test_conditions_and_stream_are_paired()
    test_task_meanings_are_balanced()
    test_noisy_rollout_shape()
    print("repetition_pressure_study tests passed")
