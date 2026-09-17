"""Small deterministic runner checks; no optimizer or model training."""
from __future__ import annotations

import numpy as np

from .. import design, runner


def test_pair_condition_names():
    for condition in design.CONDITIONS:
        schedule, live = design.parse_condition(condition)
        assert schedule in design.SCHEDULES
        assert live is condition.endswith("_live")


def test_zero_step_preserves_frozen_hash():
    _, checkpoint = runner.load_source_result(66701, "rematched")
    networks = runner.source_runner.load_networks(checkpoint)
    before = runner.parameter_hash(networks, range(6))
    optimizer = runner.core.base.make_adam(networks)
    grads = [{key: np.zeros_like(value) for key, value in net.items()} for net in networks]
    runner.core.base.adam_step(networks, grads, optimizer, 1)
    assert runner.parameter_hash(networks, range(6)) == before
