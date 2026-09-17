import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research_program.triadic_semantic_probe_study import runner as r


class RunnerTests(unittest.TestCase):
    def test_prepare_and_verify_do_not_load_models_or_build_observations(self):
        source = Path(r.__file__)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(r, 'collect_inputs', return_value=([{'fake_policy': i} for i in range(16)], {})), \
             patch.object(r, 'source_files', return_value={str(source): r.sha(source)}), \
             patch.object(r.ch.core, 'load_networks', side_effect=AssertionError('Model load forbidden')), \
             patch.object(r.ch, 'observations', side_effect=AssertionError('Observation build forbidden')):
            out = Path(temporary) / 'run'
            self.assertEqual(r.prepare(out)['status'], 'prepared_without_forward')
            self.assertEqual(len(r.verify(out)['policies']), 16)
            with self.assertRaises(AssertionError): r.prepare(out)
            snapshot = out / 'source_snapshot' / source.relative_to(r.ROOT)
            snapshot.write_text('changed')
            with self.assertRaises(AssertionError): r.verify(out)

    def test_existing_execution_refuses_before_models_or_datasets(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(r, 'verify', return_value={}), \
             patch.object(r.ch.core, 'load_networks', side_effect=AssertionError('Unexpected model load')):
            out = Path(temporary); (out / 'execution').mkdir()
            with self.assertRaises(FileExistsError): r.execute(out)

    def test_all_donor_modes_and_both_directions(self):
        spec = dict(endpoint_indices=np.array([[3, 4], [7, 8], [11, 12]]),
                    remote_donor_partition=np.array([1, 0, 1], dtype=np.int8),
                    remote_donor_endpoint_indices=np.array([[31, 41], [71, 81], [111, 121]]))
        rows = np.array([0, 2])
        for direction in (0, 1):
            for mode in r.MODES:
                parts, ids = r.donor_selection(spec, rows, 'train', direction, mode)
                endpoint = direction if mode == 'sham' or mode.startswith('remote_same') else 1-direction
                source = 'remote_donor_endpoint_indices' if mode.startswith('remote_') else 'endpoint_indices'
                self.assertTrue(np.array_equal(ids, spec[source][rows, endpoint]))
                self.assertTrue(np.array_equal(parts, spec['remote_donor_partition'][rows] if mode.startswith('remote_') else [0, 0]))

    def test_budget_and_prespecified_mode_matrix(self):
        self.assertEqual(len(r.MODES), 10)
        self.assertEqual(5 * 16 * 46656, r.CONFIG['new_natural_worlds'])
        self.assertEqual(8 * 17280 * 2 * 10, r.CONFIG['new_intervention_worlds'])
        self.assertEqual(r.CONFIG['new_natural_worlds'] * 9 + r.CONFIG['new_intervention_worlds'] * 6,
                         r.CONFIG['new_network_samples'])


if __name__ == '__main__': unittest.main()
