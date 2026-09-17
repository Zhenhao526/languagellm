"""Interface and intervention checks for the turnover study."""
from __future__ import annotations
import numpy as np
from .. import design, environment, policy


def main():
    p = policy.make_policy(76101)
    ep = design.episode_stream(76101, "parent", 32, update=1)
    natural = environment.rollout(p, ep, "live", message_mode="natural", sample=False)
    silent = environment.rollout(p, ep, "silent", message_mode="silent", sample=False)
    scrambled = environment.rollout(p, ep, "scrambled", message_mode="scrambled", sample=False)
    assert natural["messages"][:, 0].min() >= 0
    assert np.all(silent["messages"][:, 0] == design.NULL_MESSAGE)
    assert not np.array_equal(natural["messages"][:, 0], scrambled["messages"][:, 0])
    filtered = environment.rollout(p, ep, "live", message_mode="natural", sample=False, partner_filter=0)
    assert np.all(filtered["active"] == (ep["partner_id"] == 0))
    assert np.all(filtered["rewards"][~filtered["active"]] == 0)
    assert design.parse_condition("worker_live") == ("worker", "live")
    assert design.parse_condition("sender_scrambled") == ("sender", "scrambled")
    print("generation_transmission_study tests passed")


if __name__ == "__main__": main()
