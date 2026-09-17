"""Small interface tests."""
import numpy as np
from research_program.population_signaling_study import design, policy, environment


def main():
    ep = design.episode_stream(74101, "rotating", "hidden", "abundant", 32)
    p0 = policy.make_policy(74101)
    p = {"sender_logits_hidden": p0["sender_logits"], "sender_logits_visible": np.zeros((2 * design.WORKERS, design.ALPHABET_SIZE)), "worker_logits": p0["worker_logits"]}
    tr = environment.rollout(p, ep, "rotating", "hidden", "abundant", "live", sample=False)
    assert tr["messages"].shape == (32, design.HORIZON)
    assert tr["actions"].shape == (32, design.HORIZON)
    assert np.all(tr["actions"][:, 0] == 0)
    assert np.isfinite(environment.oracle_team_return(ep)).all()
    assert len(np.unique(ep["partner_id"])) > 1
    print("population_signaling_study tests passed")


if __name__ == "__main__":
    main()
