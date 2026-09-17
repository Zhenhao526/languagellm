import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qwen_language_diagnostics.map_presentation import prepare, verify_map_prepared, execute


class MapPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name) / 'MAP'
        cls.manifest = prepare(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_saved_inputs_and_chained_provenance(self):
        self.assertEqual(verify_map_prepared(self.out), self.manifest)
        data = json.loads((self.out / 'prepared_cases.json').read_text())
        self.assertEqual(len(data['cases']), 8)
        self.assertEqual(self.manifest['model_calls'], 16)
        self.assertEqual(self.manifest['max_backend_calls'], 16)
        self.assertEqual(len(self.manifest['source_sha256']), 14)
        self.assertTrue(self.manifest['prior_result_files_sha256'])
        self.assertFalse(self.manifest['unlocks_symbolic_experiment'])

    def test_preparation_refuses_overwrite(self):
        before = (self.out / 'prepared_cases.json').read_bytes()
        with self.assertRaises(FileExistsError):
            prepare(self.out)
        self.assertEqual((self.out / 'prepared_cases.json').read_bytes(), before)

    def test_changed_input_refused_before_loading_weights(self):
        path = self.out / 'prepared_cases.json'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b' ')
            with patch('qwen_language_diagnostics.map_presentation.Backend') as backend:
                with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
                    execute(self.out)
                backend.assert_not_called()
        finally:
            path.write_bytes(original)

    def test_changed_snapshot_refused_before_loading_weights(self):
        path = self.out / 'code_snapshot/qwen_language_diagnostics/map_cases.py'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with patch('qwen_language_diagnostics.map_presentation.Backend') as backend:
                with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                    execute(self.out)
                backend.assert_not_called()
        finally:
            path.write_bytes(original)

    def test_started_run_or_existing_inference_cannot_be_retried(self):
        path = self.out / 'status.json'
        original = path.read_bytes()
        inference = self.out / 'inference.jsonl'
        try:
            for status in ('running', 'failed', 'completed', 'prepared'):
                with self.subTest(status=status):
                    path.write_text(json.dumps({'status': status}))
                    if status == 'prepared':
                        inference.write_text('{}\n')
                    with patch('qwen_language_diagnostics.map_presentation.Backend') as backend:
                        with self.assertRaisesRegex(RuntimeError, 'Refuse to rerun'):
                            execute(self.out)
                        backend.assert_not_called()
        finally:
            path.write_bytes(original)
            inference.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
