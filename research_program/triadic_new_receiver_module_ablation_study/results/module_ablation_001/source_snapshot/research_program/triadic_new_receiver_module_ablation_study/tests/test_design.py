"""Static checks for the module ablation."""
from __future__ import annotations

import numpy as np

from .. import design, runner


def test_grid():
    full = runner.load_full_reference()
    static = design.prepared(__import__("json").loads((design.TRANSMISSION_ROOT / "prepared.json").read_text()))
    assert len(design.CONDITIONS) == 8
    assert len(static["runs"]) == 64
    assert static["monitor_spec"]["world_count"] == 18720
    assert static["final_spec"]["world_count"] == 56160
    assert len(full["runs"]) == 32


def test_allowed_gradients():
    _, checkpoint = runner.load_source_result(66701, "static")
    networks = runner.source_runner.load_networks(checkpoint)
    gradients = [{key: np.ones_like(value) for key, value in network.items()} for network in networks]
    action = runner.apply_gradients(networks, gradients, "action_only")
    sender = runner.apply_gradients(networks, gradients, "sender_only")
    assert all(np.all(v == 0) for net in action[:8] for v in net.values())
    assert all(np.all(v == 1) for v in action[8].values())
    assert all(np.all(v == 0) for net in sender[:6] for v in net.values())
    assert all(np.all(v == 1) for net in sender[6:8] for v in net.values())
    assert all(np.all(v == 0) for v in sender[8].values())
