"""Synthetic-only checks: no Torch import, saved weight loading, or policy inference."""
import copy
import hashlib
import importlib.util
from itertools import permutations, product
from pathlib import Path
import sys
import unittest

import numpy as np

PATH = Path(__file__).with_name('v08_formation_trajectory.py')
spec = importlib.util.spec_from_file_location('trajectory_under_test', PATH)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def fixture(blocked=False):
    actions = np.asarray([(min(f, 5), min(w, 5)) for f, w in product(range(7), repeat=2)], dtype=np.int64)
    messages = np.tile(np.asarray(subject.MAPS)[:, None, :], (1, 16, 1))
    delivered = np.zeros_like(messages) if blocked else messages.copy()
    logits = np.full((49, 2, 6), -2., dtype=np.float32)
    for code in range(49):
        logits[code, np.arange(2), actions[code]] = 3.
    train, held = subject.map_partition(1)
    record = dict(seed=27101, condition='split1_additive', update=2400, scout=0, collector=1,
                  receiver=dict(logits=logits.tolist(), **subject.lookup_from_logits(logits)),
                  emitted=messages.tolist(), delivered=delivered.tolist(),
                  photo_pairs=list(map(list, product(range(40, 44), range(50, 54)))),
                  analysis=subject.analyze_arrays(actions, messages, delivered, train, held, blocked))
    return record


def anchor(record):
    """Independent construction of the old protocol schema."""
    actions = np.asarray(record['receiver']['actions'], dtype=np.int64)
    table = [dict(message=[i//7, i%7], actions_by_goal=a.tolist(), menu_action_counts_by_goal=[
        [720*int(site == j) for j in range(6)] for site in a]) for i, a in enumerate(actions)]
    rows = []
    for m, world in enumerate(subject.MAPS):
        decoded = actions[np.asarray(record['delivered'][m]) @ np.asarray([7, 1])]
        correct = decoded == world
        def counts(messages):
            unique, count = np.unique(messages, axis=0, return_counts=True)
            return [dict(message=u.tolist(), count=int(n)) for u, n in zip(unique, count)]
        for goal in range(2):
            rows.append(dict(map_id=m, food_location=world[0], water_location=world[1], sender_goal=goal,
                             emitted_messages=counts(record['emitted'][m]), delivered_messages=counts(record['delivered'][m]),
                             native=subject.proportion(int(correct[:, goal].sum()), 16),
                             switched=subject.proportion(int(correct[:, 1-goal].sum()), 16),
                             both=subject.proportion(int(correct.all(1).sum()), 16)))
    physical = np.broadcast_to(actions[:, :, None], (49, 2, 720)).copy()
    return dict(scout=0, collector=1, receiver_decoder_table=table,
                menu_audit=dict(all_menu_permutations_physically_equivalent=True, menus_used_after_check=1,
                                cases=49*2*720, physical_action_sha256=hashlib.sha256(physical.tobytes()).hexdigest()),
                phases=dict(validation=dict(photo_pairs=record['photo_pairs'], codebook=rows)))


class FormationTests(unittest.TestCase):
    def test_import_and_pure_checks_do_not_load_torch(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('camp', sys.modules)
        self.assertEqual(len(subject.CONDITIONS), 15)
        self.assertEqual(len(set(product(subject.SEEDS, subject.CONDITIONS, subject.UPDATES, range(2)))), 960)

    def test_unique_max_all_720_menus(self):
        rng = np.random.default_rng(78)
        logits = rng.normal(size=(49, 2, 6))
        lookup = subject.lookup_from_logits(logits)
        for menu in permutations(range(6)):
            menu = np.asarray(menu)
            np.testing.assert_array_equal(menu[logits[..., menu].argmax(-1)], lookup['actions'])

    def test_any_max_tie_and_nonfinite_rejected(self):
        logits = np.zeros((49, 2, 6))
        logits[..., 0] = 1
        logits[23, 1, 4] = 1
        with self.assertRaises(subject.TieError) as caught:
            subject.lookup_from_logits(logits)
        self.assertEqual(caught.exception.details['tied_code_goal'], [[23, 1]])
        for invalid in (np.nan, np.inf, -np.inf):
            bad = logits.copy()
            bad[0, 0, 0] = invalid
            with self.assertRaises(ValueError):
                subject.lookup_from_logits(bad)

    def test_full_coverage_and_same_photo_counts(self):
        r = fixture()
        for part, maps in [('all', 30), ('train', 24), ('heldout', 6)]:
            m = r['analysis']['summaries'][part]
            self.assertEqual(m['U_full49'], subject.proportion(maps, maps))
            self.assertEqual(m['N_both'], subject.proportion(maps*16, maps*16))
            self.assertEqual(m['N_single']['denominator'], maps*16*2)
            self.assertIsNone(m['fraction_failures_U0']['rate'])
        for m, row in enumerate(r['analysis']['maps']):
            self.assertEqual(row['natural_actions'], [list(subject.MAPS[m])]*16)

    def test_blocked_keeps_emitted_separate_and_admissible_U(self):
        r = fixture(True)
        s = r['analysis']['summaries']['all']
        self.assertEqual(s['U_full49']['rate'], 1.)
        self.assertEqual(s['U_channel']['rate'], 0.)  # code00 -> (0,0), never a valid distinct pair
        self.assertEqual(s['N_both']['rate'], 0.)
        self.assertEqual(s['fraction_failures_U0']['rate'], 1.)
        self.assertEqual(s['optimal_uniform_goal_channel'], subject.proportion(10, 60))
        self.assertNotEqual(r['emitted'], r['delivered'])

    def test_marginals_and_available_but_unused_distinct(self):
        r = fixture()
        actions = np.tile([0, 0], (49, 1))
        actions[1] = [1, 0]
        actions[2] = [0, 2]
        messages = np.zeros((30, 16, 2), dtype=np.int64)
        a = subject.analyze_arrays(actions, messages, messages, *subject.map_partition(1))
        selected = a['maps'][subject.MAPS.index((1, 2))]
        self.assertEqual(selected['category'], 'both_marginals_no_joint_code')
        self.assertTrue(all(c == 'N_failure_U0' for c in selected['natural_class']))
        actions[3] = [1, 2]
        a = subject.analyze_arrays(actions, messages, messages, *subject.map_partition(1))
        selected = a['maps'][subject.MAPS.index((1, 2))]
        self.assertTrue(all(c == 'N_failure_U1' for c in selected['natural_class']))

    def test_endpoint_anchors_detect_decoder_photo_and_goal_duplicate_change(self):
        r = fixture()
        source = anchor(r)
        self.assertEqual(subject.anchor_direction(r, source)['codebook_rows_matched'], 60)
        for field in ('decoder', 'photos', 'frequency', 'duplicate'):
            bad = copy.deepcopy(source)
            if field == 'decoder':
                bad['receiver_decoder_table'][8]['actions_by_goal'][0] = 5
            elif field == 'photos':
                bad['phases']['validation']['photo_pairs'][0][0] = 999
            elif field == 'frequency':
                bad['phases']['validation']['codebook'][0]['emitted_messages'][0]['count'] = 15
            else:
                bad['phases']['validation']['codebook'][1]['both']['numerator'] = 0
            with self.assertRaises(ValueError):
                subject.anchor_direction(r, bad)

    def test_update_zero_all_conditions_except_delivery(self):
        r = fixture()
        r['update'] = 0
        blocked = copy.deepcopy(r)
        blocked['condition'] = 'blocked_joint'
        blocked['delivered'] = np.zeros((30,16,2), dtype=int).tolist()
        self.assertTrue(subject.update0_cache_verify(blocked, r))
        for key in ('emitted', 'receiver', 'photo_pairs'):
            bad = copy.deepcopy(blocked)
            bad[key] = []
            with self.assertRaises(ValueError):
                subject.update0_cache_verify(bad, r)

    def test_four_seed_summary_no_split0_replication(self):
        records = []
        for seed, condition, update, scout in product(subject.SEEDS, subject.CONDITIONS, subject.UPDATES, range(2)):
            train, held = subject.map_partition(subject.expected_plan(condition)['split'])
            r = fixture()
            metrics = subject.analyze_arrays(r['receiver']['actions'], r['emitted'], r['emitted'], train, held)['summaries']
            records.append(dict(seed=seed, condition=condition, update=update, scout=scout, summaries=metrics))
        summaries = subject.summarize(records)
        self.assertEqual(len(summaries), 3*8*(3+2+2))
        self.assertTrue(all(r['seeds'] == list(subject.SEEDS) for r in summaries))
        self.assertTrue(all(r['metrics']['N_both']['seed_values'] == [1.]*4 for r in summaries))
        with self.assertRaises(ValueError):
            subject.summarize(records[:-1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
