"""Static checks for the two-generation chain."""
from __future__ import annotations

import json

from .. import design, runner


def test_grid_and_parsing():
    assert len(design.CONDITIONS) == 8
    assert len(design.prepared(json.loads((design.TRANSMISSION_ROOT / "prepared.json").read_text()))["runs"]) == 64
    for condition in design.CONDITIONS:
        generation, arm, schedule, live = design.parse_condition(condition)
        assert generation in design.GENERATIONS
        assert arm in design.ARMS
        assert schedule in design.SCHEDULES
        assert live is condition.endswith("_live")


def test_replaced_indices_and_fresh_initialization():
    assert runner.replaced_indices(2) == (0, 1, 2)
    assert runner.replaced_indices(3) == (3, 4, 5)
    left = runner.fresh_replaced_networks(66701, 2, 0)
    right = runner.fresh_replaced_networks(66701, 2, 0)
    assert all((a[k] == b[k]).all() for a, b in zip(left, right) for k in a)
