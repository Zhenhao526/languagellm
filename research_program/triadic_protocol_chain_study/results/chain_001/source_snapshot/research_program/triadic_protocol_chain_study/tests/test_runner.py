"""No-training runner checks."""
from __future__ import annotations

from .. import design, runner


def test_generation1_inputs_are_audited():
    result, checkpoint = runner.load_generation1_result(66701, "static")
    assert result["live"] is True
    assert checkpoint.is_file()


def test_config_matches_grid():
    assert runner.CONFIG["conditions"] == list(design.CONDITIONS)
    assert runner.CONFIG["generations"] == list(design.GENERATIONS)
