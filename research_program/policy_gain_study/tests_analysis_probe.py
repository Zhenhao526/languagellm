"""Pure NumPy fixtures; no model imports, loads, forwards or training."""
import copy
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as a
import probe as p


def fixture():
    n = 9600
    maps = np.asarray(a.MAPS)
    mid = np.tile(np.arange(30), n//30)
    scout = np.repeat(np.arange(2), n//2)
    goals = np.tile([[0, 1], [1, 0]], (n//2, 1))
    menus = np.asarray([[np.roll(np.arange(6), i % 6), np.roll(np.arange(6), (i+2) % 6)] for i in range(n)])
    places = np.take_along_axis(maps[mid], goals, 1)
    actions = (menus == places[..., None]).argmax(2)
    photos = np.asarray(list(product(range(100, 104), range(200, 204))))
    codes = np.column_stack((mid//7, mid % 7))
    arrays = dict(scout=scout, episode=np.tile(np.arange(n//2), 2), positions=maps[mid], photo_ids=photos[np.arange(n) % 16],
        goals=goals, menu=menus, inventory=np.zeros((n, 2), int), history=np.zeros((n, 18), np.float32),
        sent=codes.copy(), delivered=codes.copy(), action=actions, place=places,
        successes=np.ones((n, 2), np.float32), reward=np.ones(n, np.float32))
    table = np.vstack((maps, np.tile(maps[0], (19, 1))))
    records = [dict(scout=s, collector=1-s, update=2400, photo_pairs=photos.tolist(),
        emitted=np.repeat(np.column_stack((np.arange(30)//7, np.arange(30) % 7))[:, None, :], 16, 1).tolist(),
        receiver=dict(actions=table.tolist())) for s in range(2)]
    return arrays, records


class AnalysisTests(unittest.TestCase):
    def test_import_does_not_load_neural_runtime(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('camp', sys.modules)

    def test_factorial_truth_and_independent_seed_order(self):
        rows = [dict(family='split', seed=seed, kind=kind, gain=gain,
            J=.001*(seed-28101)+.1*(kind == 'joint')+.02*(gain == 3)+.03*(kind == 'joint' and gain == 3))
            for seed, kind, gain in product(a.SEEDS, a.KINDS, a.GAINS)]
        result = a.paired_factorial(list(reversed(rows)), ('family',), ('J',))[0]
        self.assertEqual(result['seeds'], list(a.SEEDS))
        for name, expected in [('lambda_at_gain1', .1), ('lambda_at_gain3', .13), ('gain_at_lambda0', .02),
                               ('gain_at_lambda1', .05), ('interaction', .03),
                               ('diagonal_joint3_minus_additive1', .15)]:
            for value in result['effects'][name]['seed_values']: self.assertAlmostEqual(value, expected)

    def test_factorial_missing_and_duplicate_cells_refused(self):
        rows = [dict(seed=s, kind=k, gain=g, J=0.) for s, k, g in product(a.SEEDS, a.KINDS, a.GAINS)]
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.subTest(n=len(bad)), self.assertRaises(ValueError): a.paired_factorial(bad, (), ('J',))

    def test_auc_fixed_interval_and_trapezoid(self):
        points = {u: dict(N=u/1200) for u in (0, 100, 300, 600, 1200)}
        self.assertAlmostEqual(p.normalized_auc(points, 'N'), .5)
        points[2400] = dict(N=1)
        with self.assertRaises(ValueError): p.normalized_auc(points, 'N')

    def test_unique_max_and_stop_on_exact_tie(self):
        logits = np.zeros((49, 2, 6)); logits[..., 4] = 1
        self.assertTrue(np.all(np.asarray(a.measurement.lookup_from_logits(logits)['actions']) == 4))
        logits[2, 0, 1] = 1
        with self.assertRaises(a.measurement.TieError): a.measurement.lookup_from_logits(logits)

    def test_receiver_menu_is_only_final_gather(self):
        check = a.measurement.menu_gather_source_check(a.WORK / 'redesign_v0.8/camp.py')
        self.assertIn('720', check['proof'])

    def test_same_code_U_is_not_separate_query_maxima(self):
        actions = np.tile([1, 2], (49, 1))
        actions[0] = [0, 2]; actions[1] = [1, 1]
        messages = np.zeros((30, 16, 2), int)
        result = a.measurement.analyze_arrays(actions, messages, messages, list(range(30)), [], False)
        row = result['maps'][0]  # target [0, 1]
        self.assertTrue(row['food_marginal'] and row['water_marginal'])
        self.assertFalse(row['U_full49'])
        self.assertFalse(any(row['natural_both']))

    def test_normal_world_and_same_photo_anchor(self):
        arrays, records = fixture()
        before = {k: v.copy() for k, v in arrays.items()}
        anchors = p.anchor_endpoint(records, arrays, 1.)
        for row in anchors.values():
            self.assertEqual(row['actual_worlds'], 4800)
            self.assertEqual(row['physical_actions_checked'], 9600)
            self.assertEqual(row['same_photo_message_rows_checked'], 4800)
        self.assertTrue(all(np.array_equal(v, before[k]) for k, v in arrays.items()))

    def test_wrong_action_or_sender_photo_is_rejected(self):
        arrays, records = fixture()
        changed = copy.deepcopy(records)
        changed[0]['receiver']['actions'][0] = [2, 3]
        with self.assertRaises(AssertionError): p.anchor_endpoint(changed, arrays, 1.)
        changed = copy.deepcopy(records)
        changed[0]['emitted'][0][0] = [6, 6]
        with self.assertRaises(ValueError): p.anchor_endpoint(changed, arrays, 1.)

    def test_native_reward_is_not_scaled_by_policy_gain(self):
        arrays, _ = fixture()
        arrays['reward'][:] = 3
        with self.assertRaises(ValueError): a.settle_arrays(arrays, 'normal', 1.)

    def test_incomplete_batch_rejected_before_hash_or_weight_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'status.json').write_text(json.dumps(dict(status='running', completed_runs=63, expected_runs=64)))
            (base / 'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'All 64'): a.complete_inventory(base)

    def test_training_loss_and_finiteness(self):
        rows = []
        for i in range(2400):
            weight = .02 if i < 2100 else 0.
            comp = dict(policy_loss=.1, weighted_policy_loss=.3, value_loss=.2, entropy=.5)
            rows.append(dict(update=i+1, batch_identity=i, policy_gain=3, entropy_weight=weight,
                outcome_counts={'00': 0, '01': 0, '10': 0, '11': 512}, single_accuracy=1., both_accuracy=1.,
                reward=1., world_sha256=f'{i:064x}', agents=[dict(loss=.5-weight*.5, gradient_norm=1.,
                    components=[comp.copy(), comp.copy()]) for _ in range(2)]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training.jsonl'
            def write(): path.write_text('\n'.join(json.dumps(r) for r in rows))
            write()
            self.assertEqual(a.training_summary(path, 3, 'joint')['gradient_observations'], 4800)
            rows[30]['agents'][0]['loss'] += .1; write()
            with self.assertRaisesRegex(ValueError, 'Total role'): a.training_summary(path, 3, 'joint')
            rows[30]['agents'][0]['loss'] = float('nan'); write()
            with self.assertRaisesRegex(ValueError, 'Nonfinite'): a.training_summary(path, 3, 'joint')

    def test_summary_full_1024_cells_and_fixed_early_auc(self):
        records = []
        for seed, condition, update, scout in product(a.SEEDS, a.CONDITIONS, a.UPDATES, range(2)):
            train, held = a.measurement.map_partition(a.CONDITIONS[condition]['plan']['split'])
            table = np.vstack((np.asarray(a.MAPS), np.tile([0, 1], (19, 1))))
            codes = np.repeat(np.column_stack((np.arange(30)//7, np.arange(30) % 7))[:, None, :], 16, 1)
            stats = a.measurement.analyze_arrays(table, codes, codes, train, held)['summaries']
            records.append(dict(seed=seed, condition=condition, update=update, scout=scout, summaries=stats))
        summary, effects = p.summarize(records)
        self.assertEqual(len(records), 1024)
        self.assertEqual(len(summary['early_auc_seed_values']), 48)  # 3 family/partition × 16 factorial cells
        for row in summary['early_auc_seed_values']:
            self.assertEqual(row['U_auc_0_1200'], 1.)
            self.assertEqual(row['N_auc_0_1200'], 1.)
        self.assertEqual(len(effects['early_auc']), 6)
        for row in effects['early_auc']:
            self.assertEqual(row['effects']['interaction']['seed_values'], [0.]*4)


if __name__ == '__main__': unittest.main()
