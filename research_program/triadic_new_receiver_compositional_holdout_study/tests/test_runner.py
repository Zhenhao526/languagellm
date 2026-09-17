from __future__ import annotations

from .. import design, runner


def test_reference_control_grid():
    static = runner.build_prepared()
    assert len(static["runs"]) == 32
    assert static["train_spec"]["world_count"] == 1248 * 18 * 6
    assert static["heldout_spec"]["world_count"] == 312 * 6 * 6
    assert len(static["control_runs"]) == 32
