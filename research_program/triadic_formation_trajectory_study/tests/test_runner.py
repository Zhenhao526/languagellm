"""Integration with fake module functions; never load a checkpoint or train."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np
from research_program.triadic_formation_trajectory_study import runner as r


def networks(time):
    return tuple((time, head) for head in range(9))


def fake_forward(network, x):
    time, head = network
    value = np.rint(x @ (np.arange(x.shape[1]) % 7 + 1)).astype(int) + time // 100 + head
    if head % 3 == 2:
        z = np.full((len(x), 17), -5.)
        z[np.arange(len(x)), value % 17] = 5.
    else:
        z = np.full((len(x), 4, 8), -5.)
        for pos in range(4):
            z[np.arange(len(x)), pos, (value + pos) % 8] = 5.
        z = z.reshape(len(x), 32)
    return z, None


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only static old rows/metadata, no saved policy message/action output.
        data = r.dataset.POSITION / 'dataset'
        validation = r.load_npz(data / 'validation.npz')
        part = r.read(data / 'validation.json')['source_part_spec']
        rows = np.array([np.flatnonzero((validation['axis'] == 0) & (validation['sender'] == s))[0]
                         for s in range(3)])
        cls.validation = {k: v[rows].copy() for k, v in validation.items()}
        pool = r.dataset.validation_pool(cls.validation, part)
        cls.pool_index = pool['arrays']; cls.spec = r.local_spec(cls.validation, cls.pool_index)
        cls.states = cls.pool_index['validation_pool_states']; cls.full_ids = cls.pool_index['validation_pool_endpoint_indices']
        cls.x = r.ch.observations(cls.states, 'LL')

    def natural(self, time, live=True):
        # caller owns a fake-forward patch; no parameter arrays exist in this test.
        pool, record = r.build_natural(networks(time), self.x, self.states, self.full_ids, live)
        return pool, record

    def test_local_full_roundtrip_and_truth_not_replaced(self):
        full = self.full_ids
        self.assertGreater(int(full.max()), len(full))
        for a, b in (('endpoint_indices', 'recipient_pool_rows'), ('donor_endpoint_indices', 'donor_pool_rows')):
            np.testing.assert_array_equal(self.spec[a], self.pool_index[b])
            np.testing.assert_array_equal(full[self.spec[a]], self.validation[a])
        np.testing.assert_array_equal(self.spec['correct_actions'], self.validation['correct_actions'])
        before = self.validation['endpoint_indices'].copy()
        broken = dict(self.pool_index); broken['validation_pool_endpoint_indices'] = full + 1
        with self.assertRaisesRegex(AssertionError, 'mapping'):
            r.local_spec(self.validation, broken)
        np.testing.assert_array_equal(self.validation['endpoint_indices'], before)

    def test_complete_grid_and_six_by_six_budget_without_forward(self):
        c = r.CONFIG; cells = list(r.pr.cell_specs()); self.assertEqual(len(cells), 36)
        self.assertEqual(tuple(r.STEPS), (0, 100, 500, 1500, 3000, 6000))
        got = dict(new_train_message_worlds=0, new_train_message_module_samples=0,
                   new_validation_natural_worlds=0, new_validation_natural_module_samples=0,
                   new_position_worlds=0, new_position_module_samples=0,
                   new_cross_time_worlds=0, new_cross_time_module_samples=0,
                   train_records=0, natural_records=0, position_records=0, cross_time_records=0,
                   new_position_npz=0, new_cross_time_npz=0, reused_position_records=0,
                   reused_cross_time_records=0, parameter_loads=0)
        for seed in r.SEEDS:
            for condition in r.CONDITIONS:
                live = condition.endswith('_live')
                got['parameter_loads'] += 5 + int(live)
                for time in r.STEPS:
                    got['train_records'] += 1; got['natural_records'] += 1
                    if time != 6000:
                        got['new_train_message_worlds'] += 419904
                        got['new_train_message_module_samples'] += 419904 * 6
                        got['new_validation_natural_worlds'] += 554
                        got['new_validation_natural_module_samples'] += 554 * 9
                    for unit, arm, direction in cells:
                        got['position_records'] += 1; got['reused_position_records'] += time == 6000
                        if live and time != 6000:
                            got['new_position_npz'] += 1; got['new_position_worlds'] += 144
                            got['new_position_module_samples'] += 144 * (6 if unit < 4 else 3)
                for receiver in r.STEPS:
                    for donor in r.STEPS:
                        reused = receiver == donor == 6000
                        got['cross_time_records'] += 4; got['reused_cross_time_records'] += 4 * reused
                        if live and not reused:
                            got['new_cross_time_npz'] += 4; got['new_cross_time_worlds'] += 4 * 144
                            got['new_cross_time_module_samples'] += 4 * 144 * 6
        got['total_new_module_samples'] = sum(v for k, v in got.items() if k.endswith('_module_samples'))
        for name, value in got.items():
            self.assertEqual(value, c[name], name)

    @patch.object(r.pr.core.base, 'actor_forward', side_effect=fake_forward)
    def test_training_tail_batch_six_heads_and_natural_nine_heads(self, forward):
        x = self.x[:5]
        self.assertEqual(len(x), 5)
        with patch.object(r, 'BATCH', 2):
            messages, record = r.generate_train(networks(100), x, True)
        self.assertEqual([call.args[0][1] for call in forward.call_args_list], [0, 3, 6, 1, 4, 7] * 3)
        self.assertEqual([len(call.args[1]) for call in forward.call_args_list], [2]*12 + [1]*6)
        self.assertEqual(record, dict(new_forward_worlds=5, new_network_samples=30,
                                     neural_forward_calls=18, actions_generated=False))
        forward.reset_mock()
        natural, nr = r.build_natural(networks(100), x, self.states[:5], self.full_ids[:5], True)
        np.testing.assert_array_equal(messages, natural['messages'])
        self.assertEqual(forward.call_count, 9); self.assertEqual(nr['new_network_samples'], 45)
        np.testing.assert_array_equal(natural['state_indices'], self.full_ids[:5])

    @patch.object(r.pr.core.base, 'actor_forward', side_effect=fake_forward)
    def test_new_position_uses_local_indices_and_old_helper_count_fields(self, forward):
        pool, _ = self.natural(100)
        for unit in (0, 4):
            for direction in (0, 1):
                forward.reset_mock(); ids = self.spec['endpoint_indices'][:, direction]
                data, record = r.position_data({}, 100, networks(100), self.x[ids], pool, self.spec,
                                               self.full_ids, unit, 'opposite', direction, True)
                np.testing.assert_array_equal(data['recipient_indices'], ids)
                np.testing.assert_array_equal(data['donor_indices'], self.spec['donor_endpoint_indices'][:, 1-direction])
                self.assertEqual(record['receiver_update'], 100); self.assertEqual(record['donor_update'], 100)
                self.assertEqual(record['index_scope'], 'local554_pool'); self.assertFalse(record['is_reused'])
                self.assertEqual(record['new_network_samples'], len(ids) * (6 if unit < 4 else 3))
                self.assertEqual(record['neural_forward_calls'], forward.call_count)

    @patch.object(r.pr.core.base, 'actor_forward', side_effect=fake_forward)
    def test_6000_live_positions_keep_old_full_ids_and_zero_new_forward(self, forward):
        pool, _ = self.natural(6000)
        with TemporaryDirectory() as temp:
            for unit, arm, direction in ((0, 'opposite', 0), (4, 'same', 1), (0, 'sham', 1)):
                ids = self.spec['endpoint_indices'][:, direction]
                data, old = r.pr.execute_cell(networks(6000), self.x[ids], pool, self.spec, unit, arm, direction, True)
                for key in ('recipient_indices', 'donor_indices', 'counterfactual_recipient_indices'):
                    data[key] = self.full_ids[data[key]]
                path = Path(temp) / f'{unit}_{arm}_{direction}.npz'; np.savez(path, **data)
                old.update(path=str(path), data_sha256=r.sha(path)); policy = {'prior_position_records': [old]}
                with patch.object(r.pr.core.base, 'actor_forward', side_effect=AssertionError('reused means no forward')):
                    got, record = r.position_data(policy, 6000, None, self.x[ids], pool, self.spec,
                                                  self.full_ids, unit, arm, direction, True)
                for key in data:
                    np.testing.assert_array_equal(got[key], data[key])
                self.assertTrue(record['is_reused']); self.assertEqual(record['index_scope'], 'prior_full_domain')
                self.assertEqual(record['new_network_samples'], 0); self.assertEqual(record['new_forward_worlds'], 0)
                self.assertEqual(record['neural_forward_calls'], 0)
                self.assertEqual(record['old_actual_forward_worlds'], len(ids))
                changed = dict(data); changed['donor_indices'] = data['donor_indices'] + 1
                with patch.object(r, 'load_npz', return_value=changed), self.assertRaisesRegex(AssertionError, 'donor'):
                    r.position_data(policy, 6000, None, self.x[ids], pool, self.spec, self.full_ids, unit, arm, direction, True)

    @patch.object(r.pr.core.base, 'actor_forward', side_effect=fake_forward)
    def test_6000_silent_positions_recreate_routes_not_fake_npz_or_forward(self, forward):
        pool, _ = self.natural(6000, False); ids = self.spec['endpoint_indices'][:, 1]
        data, old = r.pr.execute_cell(None, self.x[ids], pool, self.spec, 4, 'opposite', 1, False)
        old.update(path=None, data_sha256=None)
        with patch.object(r.pr.core.base, 'actor_forward', side_effect=AssertionError('silent must not forward')):
            got, record = r.position_data({'prior_position_records': [old]}, 6000, None, self.x[ids], pool,
                                          self.spec, self.full_ids, 4, 'opposite', 1, False)
        np.testing.assert_array_equal(got['messages'], pool['messages'][ids])
        np.testing.assert_array_equal(got['action_indices'], pool['action_indices'][ids])
        self.assertTrue(record['is_silent_alias'] and record['is_reused'])
        self.assertEqual(record['new_network_samples'], 0); self.assertEqual(record['new_forward_worlds'], 0)
        self.assertEqual(record['routing_sha256'], old['routing_sha256'])
        old['routing_sha256'] = {}
        with self.assertRaisesRegex(AssertionError, 'routes'):
            r.position_data({'prior_position_records': [old]}, 6000, None, self.x[ids], pool,
                             self.spec, self.full_ids, 4, 'opposite', 1, False)

    @patch.object(r.pr.core.base, 'actor_forward', side_effect=fake_forward)
    def test_cross_time_uses_receiver_network_and_natural_with_donor_packets(self, forward):
        receiver, _ = self.natural(100); donor, _ = self.natural(500)
        self.assertFalse(np.array_equal(receiver['messages'], donor['messages']))
        before = deepcopy(receiver)
        for arm in ('same', 'opposite'):
            for direction in (0, 1):
                ids = self.spec['endpoint_indices'][:, direction]
                di = self.spec['donor_endpoint_indices'][:, direction if arm == 'same' else 1-direction]
                with patch.object(r.ch, 'cross_time_whole', wraps=r.ch.cross_time_whole) as call:
                    data, record = r.whole_data({}, 100, 500, networks(100), self.x[ids], receiver, donor,
                                                self.spec, arm, direction, True)
                self.assertEqual(call.call_args.args[0], networks(100))
                np.testing.assert_array_equal(call.call_args.args[1], self.x[ids])
                np.testing.assert_array_equal(call.call_args.args[2], receiver['messages'][ids])
                np.testing.assert_array_equal(call.call_args.args[4], donor['messages'][di, :, self.spec['sender']])
                self.assertEqual((record['receiver_update'], record['donor_update']), (100, 500))
                self.assertFalse(record['is_reused']); self.assertEqual(record['index_scope'], 'local554_pool')
                self.assertEqual(record['new_network_samples'], 6 * len(ids))
                self.assertEqual(record['neural_forward_calls'], 6)
                np.testing.assert_array_equal(data['donor_packets'], donor['messages'][di, :, self.spec['sender']])
        for key in before:
            np.testing.assert_array_equal(receiver[key], before[key])

    @patch.object(r.pr.core.base, 'actor_forward', side_effect=fake_forward)
    def test_silent_cross_time_action_is_receiver_natural_not_donor(self, forward):
        receiver, _ = self.natural(100, False); donor, _ = self.natural(500, False)
        ids = self.spec['endpoint_indices'][:, 0]
        with patch.object(r.pr.core.base, 'actor_forward', side_effect=AssertionError('no silent intervention forwards')):
            data, record = r.whole_data({}, 100, 500, None, self.x[ids], receiver, donor,
                                        self.spec, 'opposite', 0, False)
        np.testing.assert_array_equal(data['messages'], receiver['messages'][ids])
        np.testing.assert_array_equal(data['action_indices'], receiver['action_indices'][ids])
        np.testing.assert_array_equal(data['action_probabilities'], receiver['action_probabilities'][ids])
        self.assertTrue(record['is_silent_alias']); self.assertFalse(record['is_reused'])
        self.assertEqual(record['new_forward_worlds'], 0); self.assertEqual(record['new_network_samples'], 0)

    @patch.object(r.pr.core.base, 'actor_forward', side_effect=fake_forward)
    def test_6000_whole_live_and_silent_anchors_both_arms_and_directions(self, forward):
        with TemporaryDirectory() as temp:
            for live in (False, True):
                pool, _ = self.natural(6000, live)
                for arm in ('same', 'opposite'):
                    for direction in (0, 1):
                        ids = self.spec['endpoint_indices'][:, direction]
                        # Generate a synthetic earlier reference through the same physical API,
                        # then demand that the endpoint branch reuse bytes without a new call.
                        data, _ = r.whole_data({}, 3000, 6000, networks(6000) if live else None,
                                              self.x[ids], pool, pool, self.spec, arm, direction, live)
                        path = Path(temp) / f'{live}_{arm}_{direction}.npz'; np.savez(path, **data)
                        old = dict(arm=arm, direction=direction, path=str(path), data_sha256=r.sha(path))
                        with patch.object(r.pr.core.base, 'actor_forward', side_effect=AssertionError('endpoint reuse')):
                            got, record = r.whole_data({'prior_whole_references': [old]}, 6000, 6000, None,
                                                      self.x[ids], pool, pool, self.spec, arm, direction, live)
                        for key in data:
                            np.testing.assert_array_equal(got[key], data[key])
                        self.assertTrue(record['is_reused']); self.assertEqual(record['new_network_samples'], 0)
                        self.assertEqual(record['is_silent_alias'], not live)
                        self.assertEqual((record['receiver_update'], record['donor_update']), (6000, 6000))
                        bad = dict(data); bad['patched_outward_packets'] = (data['patched_outward_packets'] + 1) % 8
                        with patch.object(r, 'load_npz', return_value=bad), self.assertRaisesRegex(AssertionError, 'packet'):
                            r.whole_data({'prior_whole_references': [old]}, 6000, 6000, None,
                                         self.x[ids], pool, pool, self.spec, arm, direction, live)

    def test_refuse_overwrite_and_direction_combine(self):
        with TemporaryDirectory() as temp:
            with patch.object(r.pr, 'verify', side_effect=AssertionError('must not inspect old policy')):
                with self.assertRaisesRegex(AssertionError, 'overwrite'):
                    r.prepare(temp)
            out = Path(temp); (out / 'execution').mkdir()
            with patch.object(r, 'verify', return_value={}):
                with self.assertRaises(FileExistsError):
                    r.execute(out)
            path = out / 'existing.npz'; path.write_bytes(b'preserve')
            with self.assertRaisesRegex(AssertionError, 'overwrite'):
                r.save_array(path, {'x': np.zeros(1)})
            self.assertEqual(path.read_bytes(), b'preserve')
        got = r.combine({1: {'x': np.array([8, 9])}, 0: {'x': np.array([3, 4])}})
        np.testing.assert_array_equal(got['x'], [[3, 4], [8, 9]])


if __name__ == '__main__':
    unittest.main()
