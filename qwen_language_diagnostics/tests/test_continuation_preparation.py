import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qwen_language_diagnostics.continuation import prepare, verify_continuation_prepared, execute


class ContinuationPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name) / 'CONT'
        cls.manifest = prepare(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_frozen_provenance_and_limits(self):
        self.assertEqual(verify_continuation_prepared(self.out), self.manifest)
        self.assertEqual(len(self.manifest['source_sha256']), 16)
        self.assertEqual(self.manifest['max_backend_calls'], 180)
        self.assertEqual(self.manifest['max_new_steps_per_arm'], 3)
        self.assertEqual(self.manifest['original_deadline'], 12)
        self.assertFalse(self.manifest['modifies_model_deadline'])
        self.assertFalse(self.manifest['unlocks_symbolic_experiment'])
        self.assertTrue(self.manifest['run_second_arm_regardless_of_first_outcome'])

    def test_refuses_preparation_overwrite(self):
        before = (self.out / 'prepared_cases.json').read_bytes()
        with self.assertRaises(FileExistsError):
            prepare(self.out)
        self.assertEqual((self.out / 'prepared_cases.json').read_bytes(), before)

    def test_modified_input_refused_before_model_load(self):
        path = self.out / 'prepared_cases.json'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b' ')
            with patch('qwen_language_diagnostics.continuation.Backend') as backend:
                with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
                    execute(self.out)
                backend.assert_not_called()
        finally:
            path.write_bytes(original)

    def test_modified_snapshot_refused_before_model_load(self):
        path = self.out / 'code_snapshot/qwen_language_diagnostics/continuation.py'
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with patch('qwen_language_diagnostics.continuation.Backend') as backend:
                with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                    execute(self.out)
                backend.assert_not_called()
        finally:
            path.write_bytes(original)

    def test_started_run_and_partial_log_cannot_retry(self):
        status_path = self.out / 'status.json'
        original = status_path.read_bytes()
        inference = self.out / 'inference.jsonl'
        try:
            for status in ('running', 'failed', 'completed', 'prepared'):
                with self.subTest(status=status):
                    status_path.write_text(json.dumps({'status': status}))
                    if status == 'prepared':
                        inference.write_text('{}\n')
                    with patch('qwen_language_diagnostics.continuation.Backend') as backend:
                        with self.assertRaisesRegex(RuntimeError, 'Refuse to rerun'):
                            execute(self.out)
                        backend.assert_not_called()
        finally:
            status_path.write_bytes(original)
            inference.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
