import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qwen_language_diagnostics.action_semantics import prepare, verify_semantic_prepared, execute


class SemanticPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.out=Path(cls.temp.name)/'M'
        cls.manifest=prepare(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_saved_cases_and_source_provenance(self):
        self.assertEqual(verify_semantic_prepared(self.out),self.manifest)
        data=json.loads((self.out/'prepared_cases.json').read_text())
        self.assertEqual(len(data['cases']),8)
        self.assertEqual(self.manifest['model_calls'],16)
        self.assertEqual(self.manifest['max_backend_calls'],16)
        self.assertFalse(self.manifest['unlocks_symbolic_experiment'])

    def test_preparation_refuses_overwrite(self):
        before=(self.out/'prepared_cases.json').read_bytes()
        with self.assertRaises(FileExistsError):prepare(self.out)
        self.assertEqual((self.out/'prepared_cases.json').read_bytes(),before)

    def test_changed_input_refused_before_loading_weights(self):
        path=self.out/'prepared_cases.json'; original=path.read_bytes()
        try:
            path.write_bytes(original+b' ')
            with patch('qwen_language_diagnostics.action_semantics.Backend') as backend:
                with self.assertRaisesRegex(RuntimeError,'inputs changed'):execute(self.out)
                backend.assert_not_called()
        finally:path.write_bytes(original)

    def test_changed_source_snapshot_refused_before_loading_weights(self):
        path=self.out/'code_snapshot/qwen_language_diagnostics/semantic_cases.py'; original=path.read_bytes()
        try:
            path.write_bytes(original+b'\n')
            with patch('qwen_language_diagnostics.action_semantics.Backend') as backend:
                with self.assertRaisesRegex(RuntimeError,'Source changed'):execute(self.out)
                backend.assert_not_called()
        finally:path.write_bytes(original)

    def test_no_retry_of_started_run(self):
        path=self.out/'status.json'; original=path.read_bytes()
        try:
            for status in ('running','failed','completed'):
                with self.subTest(status=status):
                    path.write_text(json.dumps({'status':status}))
                    with patch('qwen_language_diagnostics.action_semantics.Backend') as backend:
                        with self.assertRaisesRegex(RuntimeError,'Refuse to rerun'):execute(self.out)
                        backend.assert_not_called()
        finally:path.write_bytes(original)


if __name__=='__main__':unittest.main()
