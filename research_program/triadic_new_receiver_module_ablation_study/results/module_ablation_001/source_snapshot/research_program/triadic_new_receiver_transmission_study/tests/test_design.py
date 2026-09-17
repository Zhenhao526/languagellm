"""Static tests for the frozen new-receiver design."""
from __future__ import annotations

import numpy as np

from .. import design
from .. import runner


def test_grid_and_specs():
    source = runner.load_source_prepared()
    static = design.prepared(source)
    assert design.SEEDS == tuple(range(66701, 66709))
    assert len(design.CONDITIONS) == 4
    assert static["monitor_spec"]["world_count"] == 18720
    assert static["final_spec"]["world_count"] == 56160
    assert len(static["runs"]) == 32


def test_fresh_c_and_zero_gradients():
    runner.load_source_prepared()
    _, checkpoint = runner.load_source_result(66701, "static")
    networks = runner.source_runner.load_networks(checkpoint)
    old_hash = runner.parameter_hash(networks, range(6, 9))
    fresh = runner.fresh_c_networks(66701, 0)
    assert runner.parameter_hash(fresh) != old_hash
    combined = runner.clone_networks(networks)
    combined[6:9] = fresh
    gradients = [{key: np.ones_like(value) for key, value in net.items()} for net in combined]
    zeroed = runner.zero_frozen_gradients(gradients, combined)
    assert all(np.all(value == 0) for net in zeroed[:6] for value in net.values())
    assert all(np.all(value == 1) for net in zeroed[6:] for value in net.values())


def test_source_manifest_is_present():
    assert runner.source_artifacts()
    assert runner.source_code_paths()
