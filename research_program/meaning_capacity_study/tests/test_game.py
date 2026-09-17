from __future__ import annotations
import numpy as np
from research_program.meaning_capacity_study import design, environment, policy


def test_conditions_and_stream_are_paired():
    assert design.parse_condition("repeat4_quad2_worker_only_p10") == ("repeat4", "quad2", "worker_only", 0.10, "p10")
    a = design.episode_stream(82101, 16, "quad2", "repeat4", update=3)
    b = design.episode_stream(82101, 16, "quad2", "repeat4", update=3)
    for key in ("site_type", "goal", "target_bits", "partner_id", "message_uniforms", "action_uniforms", "noise_uniforms"):
        assert np.array_equal(a[key], b[key])
    assert a["target_bits"].shape == (16, 8)


def test_meanings_are_balanced_and_distinct():
    goal = design.goal_table()
    unique = design.target_bits(goal, "unique8")
    repeat = design.target_bits(goal, "repeat4")
    shared = design.target_bits(goal, "shared8")
    assert np.unique(unique, axis=0).shape[0] == 8
    assert np.all(unique.mean(axis=0) == 0.5)
    assert np.all(repeat.mean(axis=0) == 0.5)
    assert np.array_equal(repeat[:, 0], repeat[:, 1]) and np.array_equal(repeat[:, 2], repeat[:, 3])
    assert np.all(shared[:, 0] == shared[:, 1]) and np.all(shared[:, 0] == shared[:, -1])


def test_capacity_and_noisy_rollout_shape():
    assert design.state_count("quad2") == 17 and design.state_count("atomic16") == 17
    params = policy.make_policy(17, "atomic16")
    episode = design.episode_stream(82101, 16, "atomic16", "shared8", update=1)
    trajectory = environment.rollout(params, episode, "atomic16", "live", 0.10, sample=False)
    assert trajectory["actions"].shape == (16, design.HORIZON)
    assert trajectory["rewards"].shape == (16, design.HORIZON)

if __name__ == "__main__":
    test_conditions_and_stream_are_paired(); test_meanings_are_balanced_and_distinct(); test_capacity_and_noisy_rollout_shape(); print("meaning_capacity_study tests passed")
