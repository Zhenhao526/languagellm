from __future__ import annotations

import numpy as np

from research_program.object_surface_remap_study import design, environment, policy


def main():
    assert design.alphabet_size("dual2") ** design.message_length("dual2") == 4
    assert design.alphabet_size("mono4") ** design.message_length("mono4") == 4
    assert design.state_count("dual2", "slot_local") == 3
    assert design.surface_permutation("identity").tolist() == [0, 1]
    assert design.surface_permutation("swap").tolist() == [1, 0]

    episode = design.episode_stream(
        design.SEEDS[0],
        32,
        task="factorized",
        mapping="swap",
        support="leave_one_out",
        heldout=design.heldout_goal(design.SEEDS[0]),
        update=1,
    )
    assert episode["site_semantic"].shape == (32, 2, 2)
    assert np.array_equal(episode["site_surface"], 1 - episode["site_semantic"])
    assert not np.any(design.goal_index(episode["goal"]) == episode["heldout_goal"])

    eval_ep = design.balanced_eval_stream(123, 4 * 32, "heldout", "factorized", 0, "identity")
    assert np.bincount(eval_ep["partner_id"], minlength=design.WORKERS).tolist() == [32, 32, 32, 32]
    mixed = design.balanced_eval_stream(123, 4 * 32, "all", "factorized", 0, "identity")
    explicit = design.goal_index(mixed["goal"])
    assert np.any(environment._permute_sequences(explicit, mixed["partner_id"]) != explicit)
    params = policy.make_policy(4, "dual2", "slot_local")
    tr = environment.rollout(params, eval_ep, "dual2", "factorized", "slot_local", "live", sample=False)
    assert tr["actions"].shape == (128, design.HORIZON)
    assert tr["rewards"].shape == (128, design.HORIZON)
    assert environment.recombination_override(params, eval_ep, "dual2", "factorized").shape == (128, 2)
    assert environment.oracle_team_return() == 4 / 6
    print("object_surface_remap_study tests passed")


if __name__ == "__main__":
    main()
