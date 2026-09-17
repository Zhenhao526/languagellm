"""Static historical-source and index-pool tests: no model/NN preparation."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np
from research_program.triadic_formation_trajectory_study import dataset as d


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_load = np.load
        cls.array_reads = []
        def restricted(path, *args, **kwargs):
            path = Path(path)
            if path.name not in ('discovery.npz', 'validation.npz'):
                raise AssertionError('Only previous static arrays may be deserialized: ' + str(path))
            cls.array_reads.append(str(path))
            return original_load(path, *args, **kwargs)
        with patch.object(d.np, 'load', side_effect=restricted):
            cls.bundle = d.load_sources()
            cls.prepared = d.make_prepared(cls.bundle)

    def test_only_two_static_arrays_loaded_not_parameters_or_outputs(self):
        self.assertEqual(len(self.array_reads), 2)
        self.assertEqual({Path(p).name for p in self.array_reads}, {'discovery.npz', 'validation.npz'})
        self.assertTrue(all('/dataset/' in p for p in self.array_reads))
        self.assertEqual(len(self.bundle['checkpoints']), 96)
        self.assertTrue(all(not row['array_parameters_loaded'] and row['neural_forward_calls'] == 0
                            for row in self.bundle['checkpoints']))

    def test_all_checkpoints_exist_and_match_original_monitor_and_audit(self):
        rows = self.bundle['checkpoints']
        self.assertEqual([(r['seed'], r['condition'], r['update']) for r in rows],
                         [(s, c, u) for s in d.SEEDS for c in d.CONDITIONS for u in d.UPDATES])
        self.assertEqual(len({r['path'] for r in rows}), 96)
        for row in rows:
            self.assertEqual(Path(row['path']).name, f"checkpoint_{row['update']:04d}.npz")
            self.assertEqual(self.bundle['source_sha256'][row['path']], row['sha256'])
            self.assertEqual(Path(row['path']).stat().st_size, row['bytes'])
            self.assertGreater(row['bytes'], 0)
            if row['update'] == 6000:
                self.assertEqual(set(row['saved_natural_6000']), {'train', d.HELDOUT})
                self.assertEqual(row['prior_policy_tag'], f"seed_{row['seed']}_{row['condition']}")
                self.assertTrue(all(not r['current_bytes_read'] for r in row['saved_natural_6000'].values()))
                self.assertEqual(row['source_position_result_sha256'], self.bundle['source_sha256'][row['source_position_result']])

    def test_pool_union_and_original_row_direction_order(self):
        old = self.bundle['validation']['arrays']; arrays = self.prepared['pool']['arrays']
        recipient, donor = old['endpoint_indices'], old['donor_endpoint_indices']
        expected = sorted(set(map(int, recipient.ravel())) | set(map(int, donor.ravel())))
        self.assertEqual(len(set(recipient.ravel())), 284)
        self.assertEqual(len(set(donor.ravel())), 280)
        self.assertEqual(len(set(recipient.ravel()) & set(donor.ravel())), 10)
        self.assertEqual(len(expected), 554)
        np.testing.assert_array_equal(arrays['validation_pool_endpoint_indices'], expected)
        np.testing.assert_array_equal(arrays['validation_pool_endpoint_indices'][arrays['recipient_pool_rows']], recipient)
        np.testing.assert_array_equal(arrays['validation_pool_endpoint_indices'][arrays['donor_pool_rows']], donor)
        self.assertEqual(arrays['recipient_pool_rows'].shape, (144, 2))
        self.assertEqual(arrays['donor_pool_rows'].shape, (144, 2))

    def test_independent_world_packing_for_every_unique_validation_world(self):
        arrays = self.prepared['pool']['arrays']
        spec = self.bundle['context_prepared']['partitions'][d.HELDOUT]
        nlayout, nowner = len(spec['layouts']), len(spec['private_sites'])
        expected = []
        for index in arrays['validation_pool_endpoint_indices']:
            need, remainder = divmod(int(index), nlayout * nowner)
            layout, owner = divmod(remainder, nowner)
            expected.append(spec['needs'][need] + spec['layouts'][layout] + spec['private_sites'][owner])
        np.testing.assert_array_equal(arrays['validation_pool_states'], expected)
        self.assertEqual(len({tuple(row) for row in expected}), 554)

    def test_discovery_complete_train_domain_and_no_truth_replacement(self):
        arrays = self.prepared['pool']['arrays']
        np.testing.assert_array_equal(arrays['train_world_indices'], np.arange(419904))
        self.assertEqual(len(np.unique(self.prepared['discovery']['arrays']['endpoint_indices'])), 326592)
        for name in ('discovery', 'validation'):
            self.assertEqual(set(self.prepared[name]['arrays']), set(self.bundle[name]['arrays']))
            for key in self.bundle[name]['arrays']:
                np.testing.assert_array_equal(self.prepared[name]['arrays'][key], self.bundle[name]['arrays'][key])
        self.assertEqual(self.prepared['separation_proof'], self.bundle['separation_proof'])

    def test_budget_six_senders_only_vs_validation_nine_and_final_reuse(self):
        budget = self.prepared['pool']['metadata']['budget']
        full, reused = budget['full_six_checkpoint_matrix'], budget['first_five_if_6000_saved_outputs_reused']
        self.assertEqual(full['policy_checkpoint_cells'], 96)
        self.assertEqual(full['training_message_world_samples'], 40310784)
        self.assertEqual(full['training_message_module_samples'], 241864704)
        self.assertEqual(full['validation_natural_world_samples'], 53184)
        self.assertEqual(full['validation_natural_module_samples'], 478656)
        self.assertEqual(reused['policy_checkpoint_cells'], 80)
        self.assertEqual(reused['training_message_world_samples'], 33592320)
        self.assertEqual(reused['training_message_module_samples'], 201553920)
        self.assertEqual(reused['validation_natural_world_samples'], 44320)
        self.assertEqual(reused['validation_natural_module_samples'], 398880)

    def test_pool_rejects_out_of_range_or_malformed_indices(self):
        a = self.bundle['validation']['arrays']; spec = self.bundle['context_prepared']['partitions'][d.HELDOUT]
        for error in ('negative', 'too_large', 'float', 'shape'):
            v = {k: a[k].copy() for k in ('endpoint_indices', 'donor_endpoint_indices')}
            if error == 'negative':
                v['endpoint_indices'] = v['endpoint_indices'].astype(np.int64); v['endpoint_indices'][0, 0] = -1
            elif error == 'too_large':
                v['endpoint_indices'][0, 0] = spec['world_count']
            elif error == 'float':
                v['endpoint_indices'] = v['endpoint_indices'].astype(float)
            else:
                v['donor_endpoint_indices'] = v['donor_endpoint_indices'][:, 0]
            with self.subTest(error=error), self.assertRaises(ValueError):
                d.validation_pool(v, spec)

    def test_determinism_input_immutability_and_existing_output_rejection(self):
        before = {name: {k: sha256(v.tobytes()).hexdigest() for k, v in self.bundle[name]['arrays'].items()}
                  for name in ('discovery', 'validation')}
        repeated = d.make_prepared(self.bundle)
        for k in repeated['pool']['arrays']:
            np.testing.assert_array_equal(repeated['pool']['arrays'][k], self.prepared['pool']['arrays'][k])
        for name in before:
            for k, digest in before[name].items():
                self.assertEqual(sha256(self.bundle[name]['arrays'][k].tobytes()).hexdigest(), digest)
        with TemporaryDirectory() as temp, patch.object(d, 'load_sources', side_effect=AssertionError('not allowed')):
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                d.prepare(temp)


if __name__ == '__main__':
    unittest.main()
