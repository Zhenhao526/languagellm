"""Pure checks for the post-hoc v0.8 checkpoint probe; never neural inference."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
from itertools import permutations, product
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


MAPS = list(permutations(range(6), 2))


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def unique_lookup_reference(logits):
    import numpy as np
    a = np.asarray(logits)
    require(a.shape == (49, 2, 6), '49 codes, 2 needs, 6 physical locations')
    require(np.isfinite(a).all(), 'finite logits')
    output = []
    margins = []
    for code in a:
        selected, delta = [], []
        for row in code:
            values = [float(x) for x in row]
            best = max(values)
            require(values.count(best) == 1, 'unique maximum required before menu collapse')
            selected.append(values.index(best))
            delta.append(sorted(values)[-1] - sorted(values)[-2])
        output.append(selected); margins.append(delta)
    return np.array(output, int), np.array(margins)


def natural_counts_reference(actions, emitted, delivered, blocked):
    import numpy as np
    table = np.asarray(actions)
    sent, received = np.asarray(emitted), np.asarray(delivered)
    require(table.shape == (49, 2) and sent.shape == received.shape == (30, 16, 2), 'fixed map/photo/code axes')
    require(np.issubdtype(sent.dtype, np.integer) and np.issubdtype(received.dtype, np.integer), 'integer channel')
    require(((received >= 0) & (received < 7)).all(), '49-code channel')
    require(np.array_equal(received, np.zeros_like(sent) if blocked else sent), 'actual channel delivery')
    rows = []
    for map_id, target in enumerate(MAPS):
        codebook = Counter(tuple(map(int, x)) for x in received[map_id])
        success = sum(n for code, n in codebook.items() if tuple(table[code[0] * 7 + code[1]]) == target)
        all_outputs = {tuple(x) for x in table}
        allowed_outputs = {tuple(table[0])} if blocked else all_outputs
        u, legal_u = int(target in all_outputs), int(target in allowed_outputs)
        require(u or not success, 'natural success requires a correct joint code')
        require(legal_u or not success, 'natural success requires an allowed correct joint code')
        rows.append({'map_id': map_id, 'U': u, 'U_channel_allowed': legal_u,
                     'N': success, 'denominator': 16, 'message_histogram': codebook})
    return rows


class PureTrajectoryReferenceTests(unittest.TestCase):
    def test_unique_maximum_preserves_all_720_physical_menu_actions(self):
        import numpy as np
        logits = np.arange(49 * 2 * 6, dtype=np.float64).reshape(49, 2, 6)
        actions, _ = unique_lookup_reference(logits)
        for menu in permutations(range(6)):
            perm = np.array(menu)
            physical = perm[np.argmax(logits[:, :, perm], axis=2)]
            np.testing.assert_array_equal(physical, actions)

    def test_exact_tie_and_nonfinite_rejected(self):
        import numpy as np
        logits = np.arange(49 * 2 * 6, dtype=np.float64).reshape(49, 2, 6)
        for mode in ('tie', 'nan', 'inf'):
            with self.subTest(mode=mode):
                bad = logits.copy()
                if mode == 'tie':
                    bad[11, 1, 4] = bad[11, 1, 5]
                else:
                    bad[11, 1, 4] = float(mode)
                with self.assertRaises(AssertionError):
                    unique_lookup_reference(bad)

    def test_fixed_map_and_photo_axes_keep_all_480_messages(self):
        import numpy as np
        table = np.array(MAPS + [(0, 0)] * 19)
        sent = np.array([[[i // 7, i % 7]] * 16 for i in range(30)])
        rows = natural_counts_reference(table, sent, sent, False)
        self.assertEqual(sum(x['N'] for x in rows), 480)
        self.assertEqual(sum(x['denominator'] for x in rows), 480)
        corrupted = sent.copy(); corrupted[0, 15] = sent[1, 0]
        rows = natural_counts_reference(table, corrupted, corrupted, False)
        self.assertEqual(rows[0]['N'], 15)
        self.assertEqual(rows[1]['N'], 16)

    def test_blocked_raw_emission_is_not_delivered_message(self):
        import numpy as np
        table = np.array(MAPS + [(0, 0)] * 19)
        sent = np.array([[[i // 7, i % 7]] * 16 for i in range(30)])
        received = np.zeros_like(sent)
        rows = natural_counts_reference(table, sent, received, True)
        self.assertEqual(sum(x['U'] for x in rows), 30)
        self.assertEqual(sum(x['U_channel_allowed'] for x in rows), 1)
        self.assertEqual(sum(x['N'] for x in rows), 16)


class AuditedTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('v08_formation_trajectory.py')
        spec = importlib.util.spec_from_file_location('formation_under_review', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_import_and_pure_checks_do_not_import_torch(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('camp', sys.modules)

    def test_lookup_matches_independent_max_and_rejects_ties(self):
        import numpy as np
        logits = np.arange(49 * 2 * 6).reshape(49, 2, 6)
        expected, margins = unique_lookup_reference(logits)
        actual = self.module.lookup_from_logits(logits)
        np.testing.assert_array_equal(actual['actions'], expected)
        np.testing.assert_array_equal(actual['top_two_margin'], margins)
        self.assertTrue(np.equal(actual['unique_max_count'], 1).all())
        for bad_value in [float('nan'), float('inf')]:
            with self.subTest(bad_value=bad_value):
                bad = logits.astype(float); bad[3, 1, 2] = bad_value
                with self.assertRaises(ValueError):
                    self.module.lookup_from_logits(bad)
        tied = logits.copy(); tied[3, 1, 4] = tied[3, 1, 5]
        with self.assertRaises(self.module.TieError) as error:
            self.module.lookup_from_logits(tied)
        self.assertIn([3, 1], error.exception.details['tied_code_goal'])
        np.testing.assert_array_equal(error.exception.details['logits'], tied)

    def test_480_natural_counts_match_independent_photo_index_reference(self):
        import numpy as np
        actions = np.array(MAPS + [(0, 0)] * 19)
        emitted = np.array([[[i // 7, i % 7]] * 16 for i in range(30)])
        emitted[24, 15] = [0, 0]
        for blocked in (False, True):
            with self.subTest(blocked=blocked):
                delivered = np.zeros_like(emitted) if blocked else emitted
                actual = self.module.analyze_arrays(actions, emitted, delivered, list(range(24)), list(range(24, 30)), blocked)
                independent = natural_counts_reference(actions, emitted, delivered, blocked)
                for record, ref in zip(actual['maps'], independent):
                    self.assertEqual((record['map_id'], record['U_full49'], record['U_channel'], sum(record['natural_both'])),
                                     (ref['map_id'], ref['U'], ref['U_channel_allowed'], ref['N']))
                self.assertEqual(actual['summaries']['all']['N_both']['denominator'], 480)
                self.assertEqual(actual['summaries']['heldout']['N_both']['denominator'], 96)
                self.assertEqual(actual['summaries']['heldout']['U_full49']['denominator'], 6)
                self.assertEqual(actual['summaries']['all']['N_both']['numerator'], sum(x['N'] for x in independent))

    def test_update_zero_is_shared_before_channel_intervention(self):
        record = {'seed': 27101, 'scout': 0, 'collector': 1, 'update': 0, 'condition': 'split1_additive',
                  'receiver': {'logits': [1, 2], 'actions': [1]}, 'emitted': [[[1, 2]]],
                  'delivered': [[[1, 2]]], 'photo_pairs': [[11, 15]]}
        blocked = deepcopy(record); blocked['condition'] = 'blocked_joint'; blocked['delivered'] = [[[0, 0]]]
        self.assertTrue(self.module.update0_cache_verify(blocked, record))
        for key, different in [('receiver', {'logits': [1, 3], 'actions': [1]}), ('emitted', [[[2, 1]]]),
                               ('photo_pairs', [[12, 15]])]:
            with self.subTest(key=key):
                bad = deepcopy(blocked); bad[key] = different
                with self.assertRaises(ValueError):
                    self.module.update0_cache_verify(bad, record)

    def test_endpoint_table_and_frequency_anchor(self):
        import numpy as np
        actions = np.array(MAPS + [(0, 0)] * 19)
        emitted = np.array([[[i // 7, i % 7]] * 16 for i in range(30)])
        emitted[2, 14] = [0, 0]
        photos = [[a, b] for a in range(4) for b in range(4, 8)]
        for blocked in (False, True):
            with self.subTest(blocked=blocked):
                delivered = np.zeros_like(emitted) if blocked else emitted.copy()
                record = {'scout': 0, 'collector': 1, 'receiver': {'actions': actions.tolist()},
                          'emitted': emitted.tolist(), 'delivered': delivered.tolist(), 'photo_pairs': photos,
                          'analysis': self.module.analyze_arrays(actions, emitted, delivered, list(range(30)), [], blocked)}
                rows = []
                for m, target in enumerate(MAPS):
                    truth = [[int(x == t) for x, t in zip(actions[int(code[0]) * 7 + int(code[1])], target)] for code in delivered[m]]
                    em = [dict(message=list(code), count=n) for code, n in sorted(Counter(map(tuple, emitted[m])).items())]
                    dm = [dict(message=list(code), count=n) for code, n in sorted(Counter(map(tuple, delivered[m])).items())]
                    prop = lambda n: dict(numerator=n, denominator=16, rate=n / 16)
                    for g in (0, 1):
                        rows.append(dict(map_id=m, food_location=target[0], water_location=target[1], sender_goal=g,
                            emitted_messages=em, delivered_messages=dm, native=prop(sum(x[g] for x in truth)),
                            switched=prop(sum(x[1-g] for x in truth)), both=prop(sum(x[0] * x[1] for x in truth))))
                source = {'scout': 0, 'collector': 1, 'receiver_decoder_table': [
                    {'message': [i // 7, i % 7], 'actions_by_goal': action.tolist(),
                     'menu_action_counts_by_goal': [[720 if j == site else 0 for j in range(6)] for site in action]}
                    for i, action in enumerate(actions)], 'phases': {'validation': {'photo_pairs': photos, 'codebook': rows}},
                    'menu_audit': {'all_menu_permutations_physically_equivalent': True, 'menus_used_after_check': 1,
                                  'cases': 49 * 2 * 720, 'physical_action_sha256': hashlib.sha256(
                                      np.repeat(actions[:, :, None], 720, axis=2).tobytes()).hexdigest()}}
                proof = self.module.anchor_direction(record, source)
                self.assertEqual(proof['codebook_rows_matched'], 60)
                self.assertFalse(proof['source_per_photo_match_available'])
                bad = deepcopy(source); bad['phases']['validation']['codebook'][1]['emitted_messages'][0]['message'] = [6, 6]
                with self.assertRaises(ValueError):
                    self.module.anchor_direction(record, bad)

    def test_existing_execution_stops_before_torch_or_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); (output / 'execution').mkdir()
            marker = output / 'execution' / 'preserved.txt'; marker.write_text('existing evidence')
            with patch.object(self.module, 'verify', return_value={}):
                with self.assertRaises(FileExistsError):
                    self.module.execute(output)
            self.assertEqual(marker.read_text(), 'existing evidence')
            self.assertNotIn('torch', sys.modules)

    def test_frozen_plan_source_and_snapshot_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / 'out'; output.mkdir()
            source = root / 'test_source.py'; source.write_text('value=1\n')
            snapshot = output / 'code_snapshot' / 'test_source.py'; snapshot.parent.mkdir(); snapshot.write_bytes(source.read_bytes())
            plan = {'source_files_sha256': {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
            (output / 'plan.json').write_text(json.dumps(plan)); (output / 'freeze.json').write_text(json.dumps(
                {'plan_sha256': hashlib.sha256((output / 'plan.json').read_bytes()).hexdigest()}))
            with patch.object(self.module, 'WORK', root), patch.object(self.module, 'CODE_FILES', ('test_source.py',)):
                self.assertEqual(self.module.verify(output), plan)
                snapshot.write_text('changed snapshot')
                with self.assertRaises(ValueError):
                    self.module.verify(output)
                snapshot.write_bytes(source.read_bytes()); source.write_text('changed source')
                with self.assertRaises(ValueError):
                    self.module.verify(output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
