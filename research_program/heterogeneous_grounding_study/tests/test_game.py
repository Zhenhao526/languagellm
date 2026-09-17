from __future__ import annotations

import numpy as np

from research_program.heterogeneous_grounding_study import design, environment, policy


def main():
    assert len(design.PARENT_CONDITIONS) == 4
    assert len(design.CHILD_CONDITIONS) == 48
    assert design.alphabet_size("dual2") ** design.message_length("dual2") == 4
    assert design.alphabet_size("mono4") ** design.message_length("mono4") == 4
    assert design.state_count("dual2", "slot_local") == 3
    assert design.surface_permutation("identity").tolist() == [0, 1]
    assert design.surface_permutation("swap").tolist() == [1, 0]
    assert design.parse_parent_condition("heterogeneous_dual2_factorized") == ("heterogeneous", "dual2", "factorized")

    seed = design.SEEDS[0]
    homogeneous = design.episode_stream(seed, 64, task="factorized", population_mode="homogeneous", update=1)
    heterogeneous = design.episode_stream(seed, 64, task="factorized", population_mode="heterogeneous", update=1)
    assert np.array_equal(homogeneous["site_semantic"], heterogeneous["site_semantic"])
    assert np.array_equal(homogeneous["goal"], heterogeneous["goal"])
    assert np.array_equal(homogeneous["partner_id"], heterogeneous["partner_id"])
    odd = np.isin(homogeneous["partner_id"], [1, 3])
    assert np.any(homogeneous["site_surface"][odd] != heterogeneous["site_surface"][odd])
    assert np.array_equal(
        homogeneous["site_surface"][homogeneous["partner_id"] % 2 == 0],
        heterogeneous["site_surface"][heterogeneous["partner_id"] % 2 == 0],
    )

    child_identity = design.episode_stream(seed, 64, task="factorized", population_mode="heterogeneous", child_mapping="identity", update=1)
    child_swap = design.episode_stream(seed, 64, task="factorized", population_mode="heterogeneous", child_mapping="swap", update=1)
    assert np.array_equal(child_identity["site_semantic"], child_swap["site_semantic"])
    assert np.array_equal(child_identity["goal"], child_swap["goal"])
    assert np.array_equal(child_identity["partner_id"], child_swap["partner_id"])
    target = child_identity["partner_id"] == design.TARGET_WORKER
    assert np.any(child_identity["site_surface"][target] != child_swap["site_surface"][target])
    assert not np.any(design.goal_index(design.episode_stream(seed, 256, task="factorized", population_mode="homogeneous", support="leave_one_out", heldout=design.heldout_goal(seed))["goal"]) == design.heldout_goal(seed))

    eval_ep = design.balanced_eval_stream(123, 4 * 32, "heldout", "factorized", 0, "homogeneous", "identity")
    assert np.bincount(eval_ep["partner_id"], minlength=design.WORKERS).tolist() == [32, 32, 32, 32]
    mixed = design.balanced_eval_stream(123, 4 * 32, "all", "factorized", 0, "homogeneous", None)
    explicit = design.goal_index(mixed["goal"])
    assert np.any(environment._permute_sequences(explicit, mixed["partner_id"]) != explicit)

    params = policy.make_policy(4, "dual2", "slot_local")
    tr = environment.rollout(params, eval_ep, "dual2", "factorized", "slot_local", "live", sample=False)
    assert tr["actions"].shape == (128, design.HORIZON)
    assert tr["rewards"].shape == (128, design.HORIZON)
    assert environment.recombination_override(params, eval_ep, "dual2", "factorized").shape == (128, 2)
    assert environment.oracle_team_return() == 4 / 6
    print("heterogeneous_grounding_study tests passed")


if __name__ == "__main__":
    main()
