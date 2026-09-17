"""Static preflight tests; no neural forwards or formal output reads."""
import json
from itertools import product
from pathlib import Path
import numpy as np

from . import runner, metrics, factor_cases


def test_condition_grid_and_parsing():
    assert len(runner.CONDITIONS) == 8
    assert {runner.parse_condition(c) for c in runner.CONDITIONS} == {(g, r, l) for g in metrics.REGIMES for r in metrics.RULES for l in (True, False)}


def test_static_factorial_support():
    static = runner.prepared()
    assert static['split']['heldout_resources'] == [5, 6]
    assert static['split']['train_need_count'] == 2088
    assert static['split']['heldout_need_count'] == 3288
    for regime in metrics.REGIMES:
        for part in ('heldout_resource', 'heldout_both'):
            groups = static['factor_edge_groups'][regime][part]
            assert all(groups['heldout_changed_actor']) and all(groups['seen_changed_actor'])


def test_subset_metric_synthetic():
    static = runner.prepared(); target = static['need_response_cases']['factorial_holdout']['heldout_resource']; groups = static['factor_edge_groups']['factorial_holdout']['heldout_resource']
    values = np.asarray([-1] * target['world_count'], dtype=np.int8)
    out = factor_cases.subset_metrics(target, values, groups['heldout_changed_actor'])
    assert out['complete_nine_strata'] and out['Q'] == 0.0


if __name__ == '__main__':
    test_condition_grid_and_parsing(); test_static_factorial_support(); test_subset_metric_synthetic(); print(json.dumps({'status': 'passed', 'real_neural_forward_samples': 0, 'formal_output_files_read': 0}))
