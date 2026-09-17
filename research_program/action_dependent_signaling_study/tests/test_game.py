from __future__ import annotations
import numpy as np
from .. import design, environment, policy


def run():
    p = policy.make_policy(76101, "dual2")
    for protocol in design.PROTOCOLS:
        ep = design.episode_stream(76101, "rotating", "hidden", "factorized", 64)
        tr = environment.rollout(p, ep, "dual2", "rotating", "hidden", "factorized", "live", protocol, sample=False)
        assert tr["messages"].shape == (64, design.HORIZON)
        assert np.all(np.isin(tr["actions"], [0, 1, 2]))
        expected = design.message_arrival_times(protocol, "dual2")
        assert np.all(tr["messages"][:, expected[0]] >= 0)
        assert np.all(tr["messages"][:, expected[1]] >= 0)
        closed = environment.rollout(p, ep, "dual2", "rotating", "hidden", "factorized", "live", protocol, message_mode="closed", sample=False)
        perm = environment.rollout(p, ep, "dual2", "rotating", "hidden", "factorized", "live", protocol, message_mode="permuted", sample=False)
        assert np.all(closed["messages"] == design.NULL_MESSAGE)
        assert np.all(np.isfinite(tr["rewards"])) and np.all(np.isfinite(perm["rewards"]))
    assert design.parse_condition(design.CONDITIONS[0])[-1] == "simultaneous"
    assert design.parse_condition(design.CONDITIONS[4])[-1] == "staged"

if __name__ == "__main__":
    run(); print("action_dependent_signaling_study tests passed")
