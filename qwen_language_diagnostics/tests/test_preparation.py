import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from qwen_language_diagnostics.experiment import ROOT, prepare, verify_prepared, execute


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.latest_path = ROOT/'LATEST'
        cls.latest_before = cls.latest_path.read_bytes() if cls.latest_path.exists() else None
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name)/'prepared'
        cls.manifest = prepare(cls.out)

    @classmethod
    def tearDownClass(cls):
        if cls.latest_before is None:
            cls.latest_path.unlink(missing_ok=True)
        else:
            cls.latest_path.write_bytes(cls.latest_before)
        cls.temp.cleanup()

    def test_prepared_provenance_and_researcher_targets(self):
        self.assertEqual(verify_prepared(self.out),self.manifest)
        data=json.loads((self.out/'prepared_cases.json').read_text())
        self.assertEqual([len(c['history']) for c in data['history']],[1,8])
        for case in data['history']:
            self.assertIn(case['diagnostic_target_action'],[m['action'] for m in case['original_record']['menus'][case['agent']]])
            before=json.loads(case['prompt'][1]['content'])
            after=json.loads(case['empty_history_prompt'][1]['content'])
            before['你自己的历史']=[]
            self.assertEqual(before,after)
            self.assertNotIn('diagnostic_target_action',after)

    def test_refuse_preparation_overwrite(self):
        before=(self.out/'prepared_cases.json').read_bytes()
        with self.assertRaises(FileExistsError):prepare(self.out)
        self.assertEqual((self.out/'prepared_cases.json').read_bytes(),before)

    def test_tampered_input_refused_before_model_loading(self):
        path=self.out/'prepared_cases.json'; original=path.read_bytes()
        try:
            path.write_bytes(original+b' ')
            with patch('qwen_language_diagnostics.experiment.Backend') as backend:
                with self.assertRaisesRegex(RuntimeError,'inputs changed'):execute(self.out)
                backend.assert_not_called()
        finally:path.write_bytes(original)

    def test_tampered_source_snapshot_refused_before_model_loading(self):
        path=self.out/'code_snapshot/qwen_language_diagnostics/experiment.py'; original=path.read_bytes()
        try:
            path.write_bytes(original+b'\n')
            with patch('qwen_language_diagnostics.experiment.Backend') as backend:
                with self.assertRaisesRegex(RuntimeError,'Source changed'):execute(self.out)
                backend.assert_not_called()
        finally:path.write_bytes(original)

    def test_started_run_refused_without_retry(self):
        path=self.out/'status.json'; original=path.read_bytes()
        try:
            path.write_text(json.dumps({'status':'completed'}))
            with patch('qwen_language_diagnostics.experiment.Backend') as backend:
                with self.assertRaisesRegex(RuntimeError,'Refuse to rerun'):execute(self.out)
                backend.assert_not_called()
        finally:path.write_bytes(original)


if __name__=='__main__':unittest.main()
