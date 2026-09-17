"""Independent, low-cost invariants for the temporal scarcity game."""
from __future__ import annotations

import numpy as np

from .. import design, environment, model, runner


def test_design_and_pairing():
    assert len(design.CONDITIONS) == 32
    a = design.episode_stream(68101, "scarce", "persistent", 12, update=3)
    b = design.episode_stream(68101, "abundant", "persistent", 12, update=3)
    for key in ("site_type", "goal", "message_uniforms", "action_uniforms"):
        np.testing.assert_array_equal(a[key], b[key])
    assert np.all(a["capacity"] == 2) and np.all(b["capacity"] == design.HORIZON)
    held = design.episode_stream(68101, "scarce", "persistent", 12, evaluation=True)
    assert not np.array_equal(a["goal"], held["goal"])


def test_observation_masks():
    pi = design.encode_observation(own_goal=1, local_type=0, local_inventory=1,
                                   last_result=2, time=2, partner_goal=1,
                                   remote_type=1, remote_inventory=3, information="PI",
                                   sender_role=True)
    fi = design.encode_observation(own_goal=1, local_type=0, local_inventory=1,
                                   last_result=2, time=2, partner_goal=1,
                                   remote_type=1, remote_inventory=3, information="FI",
                                   sender_role=True)
    assert pi.shape == (design.OBS_DIM,)
    assert np.all(pi[14:19] == 0) and pi[19] == 0 and pi[20] == 1 and pi[21] == 0
    assert fi[15] == 1 and fi[17] == 1 and fi[18] == 0.5 and fi[19] == 1


def test_memory_pairing_and_forward():
    stateless = model.make_networks(68101, "stateless")
    recurrent = model.make_networks(68101, "recurrent")
    for a in design.AGENTS:
        for key in ("W_obs", "W_recv", "b_h", "W_msg", "b_msg", "W_act", "b_act"):
            np.testing.assert_array_equal(stateless[a][key], recurrent[a][key])
        assert np.all(stateless[a]["W_h"] == 0)
    x = np.zeros((3, design.OBS_DIM)); recv = np.zeros((3, design.RECV_DIM))
    h0 = np.zeros((3, design.HIDDEN)); h1 = np.ones_like(h0)
    s0 = model.forward(stateless[0], x, recv, h0, "stateless")
    s1 = model.forward(stateless[0], x, recv, h1, "stateless")
    np.testing.assert_array_equal(s0[1], s1[1])
    r0 = model.forward(recurrent[0], x, recv, h0, "recurrent")
    r1 = model.forward(recurrent[0], x, recv, h1, "recurrent")
    assert not np.array_equal(r0[1], r1[1])


def test_protocol_and_oracle():
    ep = design.episode_stream(68101, "scarce", "persistent", 16, update=2)
    nets = model.make_networks(68101, "recurrent")
    live = environment.run_episode_batch(nets, ep, "recurrent", "PI", "live",
                                         sample=True, collect=True)
    closed = environment.run_episode_batch(nets, ep, "recurrent", "PI", "live",
                                           sample=False, controls={"message_mode": "closed"},
                                           collect=False)
    permuted = environment.run_episode_batch(nets, ep, "recurrent", "PI", "live",
                                             sample=False, controls={"message_mode": "permuted"},
                                             collect=False)
    assert live["messages"].shape == (16, design.HORIZON, 2)
    assert np.all(live["messages"][:, 2:, :] == design.NULL_MESSAGE)
    assert np.all(ep["sender"] == 0)
    assert np.all(live["actions"][:, :, 0] == 0)
    assert np.all(live["actions"][:, 0, 1] == 0)
    assert np.all(closed["messages"] == design.NULL_MESSAGE)
    assert np.all(live["final_inventory"] >= 0)
    oracle = environment.oracle_team_return(ep)
    assert np.all(oracle >= -1e-12) and np.all(oracle <= 1 + 1e-12)
    assert np.all(oracle + 1e-12 >= live["team_return"])
    assert permuted["actions"].shape == closed["actions"].shape


def test_gradient_is_finite_and_updates():
    ep = design.episode_stream(68101, "scarce", "persistent", 8, update=1)
    nets = model.make_networks(68101, "recurrent")
    tr = environment.run_episode_batch(nets, ep, "recurrent", "PI", "live", sample=True)
    grads, diag = runner._backward(nets, tr, "recurrent", "PI", "live", 1)
    assert np.isfinite(diag["return_mean"])
    assert all(np.isfinite(g).all() for group in grads for g in group.values())
    old = model.parameter_hash(nets)
    norm, _ = model.adam_step(nets, grads, model.adam_state(nets), 1)
    assert np.isfinite(norm) and model.parameter_hash(nets) != old


def main():
    for fn in (test_design_and_pairing, test_observation_masks,
               test_memory_pairing_and_forward, test_protocol_and_oracle,
               test_gradient_is_finite_and_updates):
        fn()
    print("role_signaling_study tests passed")


if __name__ == "__main__":
    main()
