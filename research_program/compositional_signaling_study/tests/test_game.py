"""Interface and intervention tests."""
import numpy as np
from research_program.compositional_signaling_study import design, environment, policy


def main():
    for form in design.FORMS:
        ep = design.episode_stream(75101, "rotating", "hidden", "factorized", 64)
        p = policy.make_policy(75101, form)
        tr = environment.rollout(p, ep, form, "rotating", "hidden", "factorized", "live", sample=False)
        assert tr["messages"].shape == (64, design.HORIZON)
        assert tr["actions"].shape == (64, design.HORIZON)
        assert np.all(tr["actions"][:, :design.ACTION_START] == 0)
        assert np.isfinite(environment.oracle_team_return(ep)).all()
        closed = environment.rollout(p, ep, form, "rotating", "hidden", "factorized", "live", message_mode="closed", sample=False)
        perm = environment.rollout(p, ep, form, "rotating", "hidden", "factorized", "live", message_mode="permuted", sample=False)
        assert np.all(closed["messages"][:, :design.message_length(form)] == design.NULL_MESSAGE)
        assert perm["messages"].shape == tr["messages"].shape
    assert np.array_equal(design.target_bits(np.array([[0,1],[1,1]], dtype=np.int8), "factorized"), np.array([[0,1],[1,1]], dtype=np.int8))
    assert np.array_equal(design.target_bits(np.array([[0,1],[1,1]], dtype=np.int8), "entangled"), np.array([[0,1],[1,0]], dtype=np.int8))
    print("compositional_signaling_study tests passed")

if __name__ == "__main__": main()
