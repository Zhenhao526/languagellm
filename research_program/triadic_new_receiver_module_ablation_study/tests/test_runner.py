"""No-training deterministic runner checks."""
from __future__ import annotations

from .. import design, runner


def test_names():
    for condition in design.CONDITIONS:
        schedule, arm, live = design.parse_condition(condition)
        assert schedule in design.SCHEDULES
        assert arm in design.ARMS
        assert live is condition.endswith("_live")


def test_reference_audit():
    full = runner.load_full_reference()
    assert len(full["runs"]) == 32
