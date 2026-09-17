from copy import deepcopy
import csv
import io
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research_program.triadic_action_dependency_study import dataset as d
from research_program.triadic_action_dependency_study import environment as e
from research_program.triadic_action_dependency_study import summarize_results as s
from research_program.triadic_action_dependency_study import plot_results as p


def fake_grid():
    return dict(status='completed', completed_run_count=24, runs=[dict(seed=seed, condition=condition, updates=6000,
        final={part: {'natural': {}, 'closed': {}} for part in s.PARTITIONS},
        monitor=[dict(update=t, monitor={part: {'natural': {}, 'closed': {}} for part in s.PARTITIONS}) for t in s.STEPS])
        for seed, condition in product(s.SEEDS, s.CONDITIONS)])


def fake_summary():
    rows = []
    pl, ll = [.1, .2, .15, .25], [.2, .1, .15, .35]
    for seed, condition, part, mode in product(s.SEEDS, s.CONDITIONS, s.PARTITIONS, s.MODES):
        index = s.SEEDS.index(seed)
        value = pl[index] if condition == 'PL_live' else ll[index] if condition == 'LL_live' else 0.
        for kind, steps in (('full_endpoint', (6000,)), ('two_background_monitor', s.STEPS)):
            for t in steps:
                v = value if kind == 'full_endpoint' else value / 4 + s.STEPS.index(t) / 100
                if mode == 'closed': v = 0.
                row = dict(seed=seed, condition=condition, partition=part, mode=mode, kind=kind, checkpoint=t,
                    world=dict(reward_mean=.2, full_success_rate=.1),
                    semantic_scope=dict(coverage='full_partition' if kind == 'full_endpoint' else 'all_needs_fixed_background_subset', saved_background_count=108 if kind == 'full_endpoint' else 2),
                    content=dict(macro={'both_endpoints_apt': v}, by_axis={a: {'metrics': {'both_endpoints_apt': v}} for a in ('kind', 'length', 'destination')}))
                rows.append(row)
    primary = dict(partition='new_needs_and_layouts', contrast='PL_live_minus_LL_live',
                   seed_values=[dict(seed=seed, difference=a-b) for seed, a, b in zip(s.SEEDS, pl, ll)])
    return dict(status='completed_read_only_summary', compact_records=rows, primary_comparison={'primary': primary})


class SummaryPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = d.make_prepared()['partitions']['new_needs']
        cls.spec = dict(original, layouts=original['layouts'][:2], private_sites=[[1, 2, 3]])
        cls.spec['world_count'] = len(cls.spec['needs']) * 2
        cls.spec['monitor_indices'] = list(range(cls.spec['world_count']))
        cls.pairs = d.content_pairs(cls.spec)

    def synthetic_cell(self, correct=True):
        spec = self.spec; packed = d.pack_states(spec); n = len(packed)
        actions = np.zeros((n, 3), dtype=np.int16)
        if correct:
            for i, row in enumerate(packed):
                plan = e.full_success_plans(row[:3], row[3:7])[0]
                actions[i] = d.plan_action_indices(plan)
        probabilities = np.eye(17)[actions]
        reward = np.full(n, float(correct)); executed = actions != 0
        values = dict(states=packed, state_indices=np.arange(n), messages=np.zeros((n, 2, 3, 4), dtype=np.int8),
            action_indices=actions, action_probabilities=probabilities, greedy_reward=reward,
            executed=executed, satisfied=executed,
            conditional_exact_expected_reward=reward.copy(), conditional_exact_full_success_probability=reward.copy(),
            conditional_exact_execution_probability=reward.copy())
        record = dict(worlds=n, information='PL', live=correct, reward_mean=float(correct),
            full_success_rate=float(correct), role_success_rate=float(correct), physical_execution_rate=float(correct))
        return values, record

    def test_complete_grid_and_no_success_selection(self):
        grid = fake_grid(); self.assertEqual(len(s.validate_completed_grid(grid)), 24)
        grid['runs'].pop()
        with self.assertRaises(ValueError): s.validate_completed_grid(grid)
        for state in ('running', 'failed'):
            grid = fake_grid(); grid['status'] = state
            with self.assertRaises(ValueError): s.validate_completed_grid(grid)

    def test_silent_alias_exact_and_live_separate(self):
        nat = dict(path='natural.npz', reused_natural=False, data_sha256='abc', live=False)
        closed = dict(nat, reused_natural=True)
        self.assertTrue(s.verify_alias(nat, closed, 'PL_silent'))
        closed['data_sha256'] = 'different'
        with self.assertRaises(ValueError): s.verify_alias(nat, closed, 'PL_silent')
        self.assertFalse(s.verify_alias(dict(nat, live=True), dict(nat, path='closed.npz'), 'PL_live'))

    def test_all_correct_pair_metrics_actual_layout(self):
        values, record = self.synthetic_cell()
        report = s.analyze_cell(values, record, self.spec, 'full_endpoint', 'PL_live', 'natural', self.pairs)
        self.assertEqual(report['world_metrics']['full_success_rate'], 1.)
        self.assertEqual(report['semantic_metrics']['content']['macro']['both_endpoints_apt'], 1.)
        self.assertEqual(report['semantic_metrics']['role']['macro']['both_endpoints_apt'], 1.)
        self.assertEqual(set(report['semantic_metrics']['case_mean_values']), set(s.metrics.PAIR_METRICS))

    def test_all_wait_keeps_every_failure_and_structural_zero(self):
        values, record = self.synthetic_cell(False)
        report = s.analyze_cell(values, record, self.spec, 'full_endpoint', 'PL_silent', 'natural', self.pairs)
        self.assertEqual(report['world_metrics']['failure_worlds'], self.spec['world_count'])
        self.assertEqual(report['semantic_metrics']['content']['macro']['both_endpoints_apt'], 0.)
        self.assertEqual(report['semantic_metrics']['case_count'], len(self.pairs['rows']))
        values['greedy_reward'][0] = .5
        with self.assertRaises(ValueError): s.analyze_cell(values, record, self.spec, 'full_endpoint', 'PL_silent', 'natural', self.pairs)

    def test_monitor_and_complete_endpoint_not_interchangeable(self):
        values, record = self.synthetic_cell()
        with self.assertRaises(ValueError): s.analyze_cell(values, record, self.spec, 'two_background_monitor', 'PL_live', 'natural', self.pairs)
        altered = deepcopy(values); altered['state_indices'] = altered['state_indices'][::-1]
        with self.assertRaises(ValueError): s.analyze_cell(altered, record, self.spec, 'full_endpoint', 'PL_live', 'natural', self.pairs)

    def test_case_mean_artifact_preserves_all_labels_and_failures(self):
        values, record = self.synthetic_cell(False)
        report = s.analyze_cell(values, record, self.spec, 'full_endpoint', 'PL_silent', 'natural', self.pairs)
        labels = deepcopy(report['semantic_metrics']['case_rows'])
        metadata = dict(kind='full_endpoint', seed=51101, condition='PL_silent', checkpoint=6000, partition='synthetic', mode='natural')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out / 'case_metrics').mkdir(); (out / 'reports').mkdir()
            catalog = dict(labels=labels, path='all_cases.json', sha256='synthetic')
            stream = io.StringIO()
            entry, cleaned = s.save_cell(out, 'synthetic', metadata, report, catalog, csv.writer(stream))
            self.assertNotIn('case_mean_values', cleaned['semantic_metrics'])
            json.dumps(cleaned, allow_nan=False)
            with np.load(out / entry['case_metrics']['path'], allow_pickle=False) as arrays:
                self.assertEqual(len(arrays['case_index']), len(self.pairs['rows']))
                self.assertEqual(set(arrays.files), {'case_index', *s.metrics.PAIR_METRICS})
                self.assertTrue(np.all(arrays['both_team_full_success'] == 0))
            self.assertEqual(s.sha(out / entry['report_path']), entry['report_sha256'])
            self.assertIn('content', stream.getvalue())

    def test_path_and_sha_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            execution = Path(tmp); directory = execution / 'seed_51101_PL_live'; directory.mkdir()
            target = directory / 'final_train_natural.npz'; target.write_bytes(b'synthetic-only')
            record = dict(path=str(target), data_sha256=s.sha(target))
            self.assertEqual(s.resolve_record(record, execution, 51101, 'PL_live', 'full_endpoint', 6000, 'train', 'natural'), target.resolve())
            target.write_bytes(b'changed')
            with self.assertRaises(ValueError): s.resolve_record(record, execution, 51101, 'PL_live', 'full_endpoint', 6000, 'train', 'natural')

    def test_figures_keep_all_seeds_negatives_and_distinct_6000_sets(self):
        summary = fake_summary(); data = p.figure_data(summary)
        self.assertEqual(len(data['endpoint']['macro']['PL_live']), 4)
        self.assertLess(data['primary']['seed_values'][0]['difference'], 0.)
        self.assertEqual(data['primary']['seed_values'][2]['difference'], 0.)
        self.assertNotEqual(data['endpoint']['macro']['PL_live'][0], data['trajectory']['new_needs_and_layouts']['PL_live']['natural'][0][-1])
        self.assertEqual(len(data['trajectory']), 4)
        broken = deepcopy(summary); broken['compact_records'].pop()
        with self.assertRaises(AssertionError): p.figure_data(broken)


if __name__ == '__main__': unittest.main()
