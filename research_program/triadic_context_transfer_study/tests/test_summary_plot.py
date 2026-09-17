"""Synthetic-only checks; never open an experimental result or checkpoint."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
import numpy as np
from research_program.triadic_context_transfer_study import summarize_results as s
from research_program.triadic_context_transfer_study import plot_results as p


def synthetic_spec():
    pairs = [(sender, listener) for sender in range(3) for listener in range(3) if sender != listener]
    return dict(axis=np.repeat(np.arange(3), 6),
        sender=np.tile([x[0] for x in pairs], 3), listener=np.tile([x[1] for x in pairs], 3),
        base_within_axis_weight=np.full(18, 1 / 6), endpoint_indices=np.zeros((18, 2), dtype=int),
        donor_endpoint_indices=np.zeros((2, 18, 2), dtype=int),
        eligible=np.stack((np.ones(18, dtype=bool), np.arange(18) % 2 == 0)),
        eligible_donor_count=1 + (np.arange(18) % 2 == 0))


def synthetic_grid():
    records, refs = [], []
    for seed in s.SEEDS:
        for condition in s.CONDITIONS:
            for part in s.PARTS:
                refs.append(dict(seed=seed, condition=condition, partition=part))
                for mode in (*s.metrics.CONTROL_MODES, *s.metrics.REMOTE_MODES):
                    shift = -1 if mode in s.metrics.CONTROL_MODES else 0
                    for direction in (0, 1):
                        silent = condition.endswith('_silent')
                        records.append(dict(seed=seed, condition=condition, partition=part, mode=mode,
                            shift_index=shift, direction=direction, worlds=18, is_silent_alias=silent,
                            new_forward_worlds=0 if silent else 18, path=None if silent else 'synthetic.npz',
                            data_sha256=None if silent else 'synthetic'))
    return dict(status='completed', records=records, natural_references=refs)


def synthetic_summary():
    policies = []
    for seed in s.SEEDS:
        for condition in s.CONDITIONS:
            index = seed - s.SEEDS[0]
            difference = (-.03 + .02 * index) * (1 if condition.startswith('PL') else .5)
            if condition.endswith('silent'):
                difference = 0.
            parts = {}
            for part in s.PARTS:
                modes = {}
                for mode, apt in [('remote_same_both', .14), ('remote_opposite_both', .14 + difference)]:
                    modes[mode] = dict(macro=dict(target_apt=apt, conservative_target_apt=apt / 2,
                        copy_donor_same_target_apt=.1, copy_donor_opposite_target_apt=.04, conservative_gate=.6))
                modes['contrast'] = dict(macro=dict(target_apt_gain=difference, conservative_target_apt_gain=difference / 2),
                    by_axis={a: dict(weight_sum=1., values=dict(target_apt_gain=difference)) for a in s.metrics.AXES})
                parts[part] = {'eligible': deepcopy(modes), 'all_other': deepcopy(modes)}
            policies.append(dict(seed=seed, condition=condition, partitions=parts))
    return dict(status='completed_read_only_summary', policies=policies)


class SummaryTests(unittest.TestCase):
    def test_complete_logical_grid_and_aliases(self):
        grid = synthetic_grid()
        lookup, refs = s.validate_grid(grid, {part: 1 for part in s.PARTS})
        self.assertEqual(len(lookup), 512)
        self.assertEqual(len(refs), 64)
        grid['records'].append(grid['records'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            s.validate_grid(grid, {part: 1 for part in s.PARTS})

    def test_missing_and_uncompleted_batches_fail(self):
        grid = synthetic_grid(); grid['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'completed'):
            s.validate_grid(grid, {part: 1 for part in s.PARTS})
        grid['status'] = 'completed'; grid['records'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing'):
            s.validate_grid(grid, {part: 1 for part in s.PARTS})

    def test_streaming_fixed_denominators_and_strata(self):
        spec = synthetic_spec()
        accumulator = s.StratifiedAccumulator()
        raw = np.arange(18) / 18
        for shift in range(2):
            for direction in (0, 1):
                for start in range(0, 18, 7):
                    selection = slice(start, min(start + 7, 18))
                    values = {'rate': raw[selection], 'signed': raw[selection] - .5}
                    weights = s.metrics.support_weights(spec, shift, 'eligible', selection)
                    accumulator.update(values, spec, selection, weights, direction)
        report = accumulator.finish()
        self.assertAlmostEqual(report['macro']['rate'], raw.mean(), places=12)
        self.assertAlmostEqual(report['macro']['signed'], raw.mean() - .5, places=12)
        for d in ('0', '1'):
            self.assertAlmostEqual(report['by_direction'][d]['macro']['rate'], raw.mean(), places=12)
        for sender in range(3):
            for listener in range(3):
                if sender != listener:
                    mask = (spec['sender'] == sender) & (spec['listener'] == listener)
                    self.assertAlmostEqual(report['by_sender_listener'][f'{sender}_{listener}']['macro']['rate'], raw[mask].mean(), places=12)

    def test_missing_direction_is_not_silently_renormalized(self):
        spec = synthetic_spec(); accumulator = s.StratifiedAccumulator()
        weights = s.metrics.support_weights(spec, -1, 'control')
        accumulator.update({'rate': np.ones(18)}, spec, slice(None), weights, 0)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            accumulator.finish()

    def test_primary_preserves_all_seeds_and_signs(self):
        summary = synthetic_summary()
        report = s.metrics.primary_comparison(summary['policies'])
        values = [r['PL_minus_LL'] for r in report['primary']['seed_pairs']]
        np.testing.assert_allclose(values, [-.015, -.005, .005, .015], atol=1e-16)
        self.assertEqual(len(report['all_partitions']), 4)

    def test_plot_extract_keeps_complete_grid(self):
        summary = synthetic_summary()
        rows = p.extract(summary)
        self.assertEqual(len(rows), 4 * 2 * 4 * 2)
        summary['policies'].pop()
        with self.assertRaisesRegex(ValueError, '16'):
            p.extract(summary)

    def test_synthetic_figures_and_overwrite_guard(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'summary.json'
            source.write_text(json.dumps(synthetic_summary()))
            report = p.plot(source, root / 'figures')
            self.assertEqual(len(report['outputs']), 6)
            self.assertTrue(all(r['bytes'] > 1000 for r in report['outputs'].values()))
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                p.plot(source, root / 'figures')


if __name__ == '__main__':
    unittest.main()
