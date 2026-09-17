"""Regression tests for the post hoc factor-response correction."""
import numpy as np

from research_program.triadic_action_dependency_study import environment
from . import correction, factor_cases, runner


def test_elementwise_pair_comparison():
    static = runner.prepared()
    case_spec = static['need_response_cases']['saturated']['heldout_resource']
    groups = static['factor_edge_groups']['saturated']['heldout_resource']
    pair_ids = {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    by_need = np.asarray([
        pair_ids[environment.full_success_plans(tuple(need))[0][:2]]
        for need in case_spec['needs']
    ], dtype=np.int8)
    values = np.repeat(by_need, case_spec['n_backgrounds'])
    fixed = correction.corrected_subset(case_spec, values, groups['heldout_changed_actor'])
    buggy = factor_cases.subset_metrics(case_spec, values, groups['heldout_changed_actor'])
    assert fixed['Q'] == 1.0
    assert buggy['Q'] == 0.0


if __name__ == '__main__':
    test_elementwise_pair_comparison()
    print({'status': 'passed', 'real_neural_forward_samples': 0})
