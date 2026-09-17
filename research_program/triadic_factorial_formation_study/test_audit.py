"""Pure audit arithmetic preflight."""
import json
import numpy as np
from . import audit, runner


def test_independent_static_matches():
    source = audit.read(runner.ORIGINAL / 'prepared.json')
    specs, response, groups = audit.independent_specs(source)
    static = runner.prepared()
    assert specs == static['partitions']
    assert response == static['need_response_cases']
    assert groups == static['factor_edge_groups']


def test_independent_subset_zero():
    static = runner.prepared(); c = static['need_response_cases']['factorial_holdout']['heldout_resource']; g = static['factor_edge_groups']['factorial_holdout']['heldout_resource']
    values = np.full(c['world_count'], -1, dtype=np.int8)
    out = audit.factor_subset_independent(c, values, g['heldout_changed_actor'])
    assert out['Q'] == 0.0 and out['Q_shuffle'] >= 0.0


if __name__ == '__main__':
    test_independent_static_matches(); test_independent_subset_zero(); print(json.dumps({'status': 'passed', 'real_neural_forward_samples': 0, 'formal_output_files_read': 0}))

