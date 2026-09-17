"""Synthetic fixtures only: do not read or measure any real receiver record."""
import copy
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a


def messages(code=0):
    result = np.zeros((30, 16, 2), dtype=np.int64)
    result[..., 0] = code//7; result[..., 1] = code % 7
    return result


def old_record(logits, emitted):
    """Construct just the old fields required by exact replay, independently."""
    x = np.asarray(logits, dtype=np.float32); actions = x.argmax(2)
    targets = np.asarray(a.MAPS); codes = emitted[..., 0]*7+emitted[..., 1]
    u = np.asarray([any(tuple(v) == tuple(t) for v in actions) for t in targets])
    n = np.asarray([[tuple(actions[int(c)]) == tuple(t) for c in row] for t, row in zip(targets, codes)])
    order = np.sort(x, -1)
    maps = [dict(map_id=i, locations=list(t), partition='train', U_full49=bool(u[i]),
                 U_channel=bool(u[i]), natural_both=n[i].tolist()) for i, t in enumerate(a.MAPS)]
    summary = dict(U_full49=dict(numerator=int(u.sum()), denominator=30, rate=int(u.sum())/30),
                   N_both=dict(numerator=int(n.sum()), denominator=480, rate=int(n.sum())/480))
    return dict(receiver=dict(logits=x.tolist(), actions=actions.tolist(),
        unique_max_count=(x == x.max(-1, keepdims=True)).sum(-1).tolist(),
        top_two_margin=(order[..., -1]-order[..., -2]).tolist()),
        emitted=emitted.tolist(), delivered=emitted.tolist(), analysis=dict(maps=maps, summaries=dict(train=summary, heldout=None)))


class ProbabilityTests(unittest.TestCase):
    def test_no_neural_imports(self):
        self.assertNotIn('torch', sys.modules); self.assertNotIn('camp', sys.modules)

    def test_uniform_retains_all_strict_product_maxima(self):
        result = a.measure(np.zeros((49, 2, 6)), messages(), list(range(30)), [])
        for row in result['maps']:
            self.assertAlmostEqual(row['C'], 1/36)
            self.assertEqual(row['max_code_ids'], list(range(49)))
            self.assertEqual(row['first_max_code_id'], 0)
            self.assertEqual(row['first_max_message'], [0, 0])
            self.assertTrue(all(abs(v-1/36) < 1e-15 for v in row['E_G']))
            self.assertTrue(all(v == 0 for v in row['C_minus_E_G']))

    def test_same_complete_code_and_natural_gap(self):
        probabilities = np.full((49, 2, 6), 1/6)
        probabilities[0, 0] = [.8, .04, .04, .04, .04, .04]
        probabilities[0, 1] = [.198, .01, .198, .198, .198, .198]
        probabilities[1, 0] = [.01, .198, .198, .198, .198, .198]
        probabilities[1, 1] = [.04, .8, .04, .04, .04, .04]
        probabilities[2, 0] = [.4, .12, .12, .12, .12, .12]
        probabilities[2, 1] = [.12, .4, .12, .12, .12, .12]
        result = a.measure(np.log(probabilities), messages(0), list(range(30)), [])
        row = result['maps'][0]  # food0/water1
        self.assertAlmostEqual(row['C'], .16)
        self.assertNotAlmostEqual(row['C'], .8*.8)
        self.assertEqual(row['max_code_ids'], [2])
        self.assertEqual(row['max_messages'], [[0, 2]])
        self.assertTrue(all(abs(v-.008) < 1e-15 for v in row['E_G']))
        self.assertTrue(row['U'])
        self.assertFalse(any(row['N']))
        self.assertTrue(all(v > 0 for v in row['C_minus_E_G']))

    def test_equal_best_codes_are_not_collapsed(self):
        x = np.zeros((49, 2, 6)); x[[4, 9], 0, 0] = 3; x[[4, 9], 1, 1] = 3
        row = a.measure(x, messages(9), list(range(30)), [])['maps'][0]
        self.assertEqual(row['max_code_ids'], [4, 9]); self.assertEqual(row['first_max_message'], [0, 4])
        self.assertTrue(all(v == row['C'] for v in row['E_G']))

    def test_translation_invariance_and_temperature_changes_probability(self):
        x = np.zeros((49, 2, 6)); x[:, 0, 0] = 1; x[:, 1, 1] = 1
        base = a.measure(x, messages(), list(range(30)), [])
        shifted = a.measure(x+100, messages(), list(range(30)), [])
        self.assertEqual(base, shifted)
        scaled = a.measure(x*3, messages(), list(range(30)), [])
        self.assertEqual([r['U'] for r in base['maps']], [r['U'] for r in scaled['maps']])
        self.assertGreater(scaled['maps'][0]['C'], base['maps'][0]['C'])

    def test_nonfinite_and_probability_underflow_rejected(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            x = np.zeros((49, 2, 6)); x[0, 0, 0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError): a.measure(x, messages(), list(range(30)), [])
        x = np.zeros((49, 2, 6)); x[..., 1:] = -1000
        with self.assertRaisesRegex(ValueError, 'probability underflow'): a.measure(x, messages(), list(range(30)), [])

    def test_product_underflow_rejected_without_false_maximum_ties(self):
        x = np.full((49, 2, 6), -400.); x[..., 0] = 0
        self.assertTrue((a.stable_log_softmax(x)[1] > 0).all())
        with self.assertRaisesRegex(ValueError, 'product underflow'): a.measure(x, messages(), list(range(30)), [])

    def test_invalid_message_or_partition_refused(self):
        for m, train, held in [(messages().astype(float), list(range(30)), []),
                                (messages(49), list(range(30)), []), (messages(), [0], [0])]:
            with self.subTest(dtype=str(m.dtype), train_n=len(train)), self.assertRaises(ValueError):
                a.measure(np.zeros((49, 2, 6)), m, train, held)

    def test_original_U_N_actions_and_margin_remain_exact(self):
        x = np.zeros((49, 2, 6), np.float32); x[:, 0, 2] = 2; x[:, 1, 3] = 2
        x[0] = 0; x[0, 0, 0] = 3; x[0, 1, 1] = 3
        record = old_record(x, messages())
        result = a.measure(x, messages(), list(range(30)), [])
        self.assertEqual(a.check_old_record(record, result)['photo_N_checked'], 480)
        broken = copy.deepcopy(record); broken['analysis']['maps'][0]['natural_both'][0] = False
        with self.assertRaisesRegex(ValueError, 'exact replay'): a.check_old_record(broken, result)
        broken = copy.deepcopy(record); broken['receiver']['top_two_margin'][0][0] += .01
        with self.assertRaisesRegex(ValueError, 'margins'): a.check_old_record(broken, result)

    def test_six_factorial_contrasts(self):
        rows = [dict(seed=s, kind=k, gain=g, value=.01*(s-28101)+.1*(k=='joint')+.02*(g==3)+.03*(k=='joint' and g==3))
                for s, k, g in product(a.SEEDS, a.KINDS, a.GAINS)]
        effects = a.contrasts(rows, (), ('value',))[0]['effects']
        expected = dict(lambda_at_gain1=.1, lambda_at_gain3=.13, gain_at_lambda0=.02,
                        gain_at_lambda1=.05, interaction=.03, diagonal_joint3_minus_additive1=.15)
        for name, value in expected.items():
            for actual in effects[name]['seed_values']: self.assertAlmostEqual(actual, value)
        with self.assertRaises(ValueError): a.contrasts(rows[:-1], (), ('value',))
        with self.assertRaises(ValueError): a.contrasts(rows+[rows[0]], (), ('value',))

    def test_full_grid_averaging_auc_and_fixed_time_differences(self):
        entries = []
        for seed, condition, update, scout in product(a.SEEDS, a.CONDITIONS, a.UPDATES, range(2)):
            train, held = a.partition(condition)
            summaries = {}
            for part, ids in [('train', train), ('heldout', held)]:
                if not ids: summaries[part] = None; continue
                C = .1+update/10000
                summaries[part] = dict(C=C, C_n=len(ids), E_G=C/2, E_G_n=len(ids)*16,
                    C_minus_E_G=C/2, U=.5, N=.25, U_numerator=3, U_denominator=6,
                    N_numerator=24, N_denominator=96)
            entries.append(dict(seed=seed, condition=condition, update=update, scout=scout, summaries=summaries))
        result = a.summarize(entries)
        self.assertEqual(len(result['seed_values']), 384)
        self.assertEqual(len(result['means']), 96)
        self.assertEqual(len(result['early_auc_seed_values']), 48)
        self.assertEqual(len(result['time_change_seed_values']), 192)
        for row in result['early_auc_seed_values']:
            self.assertAlmostEqual(row['C_auc_0_1200'], .16)
            self.assertAlmostEqual(row['E_G_auc_0_1200'], .08)
        for row in result['time_change_seed_values']:
            self.assertAlmostEqual(row['C'], (row['end_update']-row['start_update'])/10000)
        for row in result['effects']['early_auc']:
            self.assertEqual(row['effects']['interaction']['seed_values'], [0.]*4)
        with self.assertRaises(ValueError): a.summarize(entries[:-1])

    def test_input_arrays_not_mutated(self):
        x = np.zeros((49, 2, 6)); m = messages(1)
        oldx, oldm = x.copy(), m.copy()
        a.measure(x, m, list(range(30)), [])
        self.assertTrue(np.array_equal(x, oldx)); self.assertTrue(np.array_equal(m, oldm))

    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.json'; a.write_new(path, dict(x=1))
            with self.assertRaises(FileExistsError): a.write_new(path, dict(x=2))
            self.assertEqual(a.read(path), dict(x=1))

    def test_execute_keeps_failure_without_partial_summary_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output/'bad_source.json'
            identity = dict(seed=28101, condition=a.CONDITIONS[0], update=0, scout=0)
            a.write_new(source, dict(identity, collector=1, receiver=dict(logits=[]), emitted=[]))
            digest = a.sha(source)
            a.write_new(output/'plan.json', dict(fixture=True))
            plan = dict(prepare_runtime=a.runtime(), source_records=[dict(identity, file=str(source), sha256=digest)])
            with patch.object(a, 'verify', return_value=plan), self.assertRaisesRegex(ValueError, '49x2x6'):
                a.execute(output)
            self.assertEqual(a.sha(source), digest)
            failure = a.read(output/'execution/failure.json')
            self.assertEqual(failure['status'], 'failed')
            self.assertFalse(failure['automatic_retry'])
            self.assertFalse((output/'execution/results.json').exists())
            with patch.object(a, 'verify', return_value=plan), self.assertRaises(FileExistsError): a.execute(output)


if __name__ == '__main__': unittest.main()
