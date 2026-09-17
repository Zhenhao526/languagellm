"""Model-free tests for the fixed CAP replay/shared-plan diagnostic.

The frozen Backend.decide executes, but infer is a source-recording fixture.
No model initializer or tokenizer/weight loader is called here.
"""
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

from qwen_language_v3.backend import Backend
from research_program.triadic_task import environment as env
from research_program.triadic_task import qwen_execution_diagnostic as diag


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def explicit_lexical_witness(state):
    """Independently enumerate at most 24 semantic plans, using actual physics."""
    for i, j in combinations(range(3), 2):
        for site, destination in product(env.SITES, env.DESTINATIONS):
            actions = {agent: {'kind': 'wait'} for agent in env.AGENTS}
            actions[env.AGENTS[i]] = {'kind': 'transport', 'site': site,
                'destination': destination, 'partner': env.AGENTS[j]}
            actions[env.AGENTS[j]] = {'kind': 'transport', 'site': site,
                'destination': destination, 'partner': env.AGENTS[i]}
            if env.settle(state, actions, require_match=True)['reward'] == 1:
                return actions
    raise AssertionError('No full-success semantic plan')


class RecordingBackend(Backend):
    def __init__(self, prepared):
        self.calls = 0
        self.entries = []
        self.cases = {case['case_id']: case for case in prepared['cases']}

    def infer(self, messages, *, mode, label, seed=0, limit=32, choices=None, temperature=0):
        case = self.cases[label['case_id']]
        source = case['original_analysis' if mode == 'private_analysis' else 'original_formal']
        if label['arm'] == 'original_replay':
            output = source['output']
        elif mode == 'private_analysis':
            output = 'AUDIT_PLAN_PRIVATE_' + case['case_id']
        else:
            output = str(next(m['id'] for m in case['menu'] if m['action'] == case['expected_plan_action']))
        self.calls += 1
        entry = deepcopy(source)
        entry.update(call=self.calls, messages=deepcopy(messages), mode=mode,
                     label=deepcopy(label), seed=seed, temperature=temperature, output=output)
        # Original rendered hashes are lookup values only when messages match.
        # This tests comparator plumbing; it is not fresh tokenizer attestation.
        entry['prompt_sha256'] = (source['prompt_sha256'] if messages == source['messages'] else
                                  hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest())
        self.entries.append(entry)
        return output


class ExecutionDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = [Path(diag.__file__), Path(env.__file__),
                       Path(diag.__file__).resolve().parents[2] / 'qwen_language_v3/backend.py',
                       Path(diag.__file__).resolve().parents[2] / 'qwen_language_v3/run.py']
        cls.hashes_before = {str(path): digest(path) for path in cls.sources}
        cls.prepared = diag.build_cases()
        cls.prepared_before = deepcopy(cls.prepared)
        cls.backend = RecordingBackend(cls.prepared)
        cls.result = diag.run_cases(cls.backend, cls.prepared)

    @classmethod
    def tearDownClass(cls):
        for path, original in cls.hashes_before.items():
            if digest(path) != original:
                raise AssertionError(f'A source changed during the audit: {path}')

    def test_all_twelve_source_snapshots_and_36_action_decisions_retained(self):
        cases = self.prepared['cases']
        self.assertEqual(len(cases), 36)
        self.assertEqual(len({case['case_id'] for case in cases}), 36)
        self.assertEqual(len({case['source_trial_id'] for case in cases}), 12)
        self.assertEqual(set(Counter(case['source_trial_id'] for case in cases).values()), {3})
        source_calls = []
        for case in cases:
            analysis, formal = case['original_analysis'], case['original_formal']
            self.assertEqual(analysis['mode'], 'private_analysis')
            self.assertEqual(formal['mode'], 'action')
            self.assertEqual(analysis['label']['window'], None)
            self.assertEqual(formal['call'], analysis['call'] + 1)
            self.assertEqual(analysis['seed'], formal['seed'])
            self.assertEqual(case['seed'], analysis['seed'])
            self.assertEqual(case['original_base_messages'], analysis['messages'][:-1])
            self.assertEqual(case['original_base_messages'], formal['messages'][:-2])
            self.assertEqual(formal['messages'][-2], {'role': 'assistant', 'content': analysis['output']})
            self.assertEqual(case['choices'], list(range(17)))
            source_calls.extend((analysis['call'], formal['call']))
        self.assertEqual(len(set(source_calls)), 72)
        expected_calls = [18 * i + offset for i in range(12) for offset in range(13, 19)]
        self.assertEqual(sorted(source_calls), expected_calls)

    def test_plan_is_only_new_base_message_and_public_transcript_is_unmodified(self):
        for case in self.prepared['cases']:
            original, planned = case['original_base_messages'], case['plan_base_messages']
            self.assertEqual(len(original), 2)
            self.assertEqual(len(planned), 3)
            self.assertEqual(planned[:2], original)
            self.assertEqual(planned[-1], case['shared_plan_message'])
            self.assertEqual(planned[-1]['role'], 'user')
            self.assertTrue(any(word in planned[-1]['content'] for word in ('实验者', '研究者')))
            original_data = json.loads(original[1]['content'])
            self.assertEqual(len(original_data['本轮已公开广播']), 6)
            self.assertEqual(original_data, json.loads(planned[1]['content']))
            self.assertEqual(len(original_data['你自己的历史']), case['episode'] - 1)
            self.assertEqual(original_data['本轮完整信息观察']['你是'], case['agent'])
            self.assertEqual(len(original_data['本次可选动作']), 17)
            self.assertFalse(re.search(r'(?:编号|ID)\s*[:：=]?\s*\d+', planned[-1]['content']))

    def test_same_plan_is_semantically_valid_lexical_and_shared_by_all_three(self):
        by_trial = {}
        for case in self.prepared['cases']:
            by_trial.setdefault(case['source_trial_id'], []).append(case)
            state = env.State(**case['state'])
            expected = explicit_lexical_witness(state)
            self.assertEqual(case['shared_plan'], expected)
            self.assertEqual(case['expected_plan_action'], expected[case['agent']])
            self.assertEqual(env.settle(state, case['shared_plan'], require_match=True)['reward'], 1)
            for action in case['shared_plan'].values():
                self.assertNotIn('id', action)
                self.assertNotIn('编号', action)
        for cases in by_trial.values():
            self.assertEqual({c['agent'] for c in cases}, set(env.AGENTS))
            self.assertEqual(cases[0]['shared_plan_message'], cases[1]['shared_plan_message'])
            self.assertEqual(cases[1]['shared_plan_message'], cases[2]['shared_plan_message'])

    def test_complete_144_calls_two_stage_original_wrapper(self):
        self.assertIs(RecordingBackend.decide, Backend.decide)
        self.assertEqual(self.backend.calls, 144)
        self.assertEqual(len(self.backend.entries), 144)
        self.assertEqual(Counter(e['label']['arm'] for e in self.backend.entries),
                         {'original_replay': 72, 'shared_plan_added': 72})
        self.assertEqual(Counter(e['mode'] for e in self.backend.entries),
                         {'private_analysis': 72, 'action': 72})
        self.assertEqual([e['label']['arm'] for e in self.backend.entries[:72]], ['original_replay'] * 72)
        self.assertEqual([e['label']['arm'] for e in self.backend.entries[72:]], ['shared_plan_added'] * 72)
        by_case = {c['case_id']: c for c in self.prepared['cases']}
        for analysis, formal in zip(self.backend.entries[::2], self.backend.entries[1::2]):
            self.assertEqual(analysis['label']['case_id'], formal['label']['case_id'])
            case = by_case[analysis['label']['case_id']]
            self.assertEqual(analysis['seed'], case['seed'])
            self.assertEqual(formal['seed'], case['seed'])
            self.assertEqual(analysis['temperature'], 0)
            self.assertEqual(formal['temperature'], 0)
            base = case['original_base_messages'] if analysis['label']['arm'] == 'original_replay' else case['plan_base_messages']
            self.assertEqual(analysis['messages'][:-1], base)
            self.assertEqual(formal['messages'][:-2], base)
            self.assertEqual(formal['messages'][-2], {'role': 'assistant', 'content': analysis['output']})
            self.assertEqual(formal['output'] in tuple(map(str, range(17))), True)

    def test_all_original_stage_messages_and_outputs_reproduce_source_fixture(self):
        by_case = {c['case_id']: c for c in self.prepared['cases']}
        for entry in self.backend.entries[:72]:
            case = by_case[entry['label']['case_id']]
            source = case['original_analysis' if entry['mode'] == 'private_analysis' else 'original_formal']
            for key in ('messages', 'output', 'seed', 'mode', 'temperature', 'prompt_sha256'):
                self.assertEqual(entry[key], source[key])

    def test_probe_histories_are_not_updated_and_partner_probe_analysis_is_hidden(self):
        self.assertEqual(self.prepared, self.prepared_before)
        for entry in self.backend.entries:
            messages = entry['messages']
            data = json.loads(messages[1]['content'])
            self.assertNotIn('AUDIT_PLAN_PRIVATE_', json.dumps(data, ensure_ascii=False))
            seen = [c['case_id'] for c in self.prepared['cases']
                    if 'AUDIT_PLAN_PRIVATE_' + c['case_id'] in json.dumps(messages, ensure_ascii=False)]
            expected = ([entry['label']['case_id']] if entry['label']['arm'] == 'shared_plan_added'
                        and entry['mode'] == 'action' else [])
            self.assertEqual(seen, expected)
            for key in ('seed', 'source_call_ids', 'source_trial_id', 'expected_plan_action', 'case_id'):
                self.assertNotIn(key, data)

    def test_reproduction_gate_requires_every_recorded_field(self):
        report = diag.reproduction_report(self.backend.entries, self.prepared)
        self.assertTrue(report['complete_exact_reproduction'])
        self.assertTrue(report['effect_interpretable'])
        self.assertEqual(len(report['checks']), 72)
        for field in diag.REPRO_FIELDS:
            with self.subTest(field=field):
                changed = deepcopy(self.backend.entries)
                value = changed[0][field]
                changed[0][field] = (not value if type(value) is bool else value + 1
                                     if type(value) in (int, float) else value + '_DIFFERENT'
                                     if type(value) is str else value + [{'role': 'user', 'content': 'DIFFERENT'}])
                rejected = diag.reproduction_report(changed, self.prepared)
                self.assertFalse(rejected['complete_exact_reproduction'])
                self.assertFalse(rejected['effect_interpretable'])
                self.assertEqual(rejected['effect_status'], 'effect_uninterpretable')
                self.assertFalse(rejected['checks'][0]['equal'][field])
        with self.assertRaises(ValueError):
            diag.reproduction_report(self.backend.entries[:-1], self.prepared)

    def test_original_output_difference_does_not_stop_the_plan_arm(self):
        class ChangedFirstAnalysis(RecordingBackend):
            def infer(self, *args, **kwargs):
                output = super().infer(*args, **kwargs)
                if self.calls == 1:
                    output += '_DELIBERATE_REPLAY_DIFFERENCE'
                    self.entries[-1]['output'] = output
                return output
        fake = ChangedFirstAnalysis(self.prepared)
        result = diag.run_cases(fake, self.prepared)
        self.assertEqual(fake.calls, 144)
        self.assertEqual(len(result['decisions']), 72)
        self.assertEqual(len(result['trials']), 24)
        report = diag.reproduction_report(fake.entries, self.prepared)
        self.assertFalse(report['effect_interpretable'])
        self.assertEqual(result['arms'][1]['full_success_trials'], 12)

    def test_joint_settlement_and_reference_plan_match_are_distinct_fields(self):
        self.assertEqual(self.result['model_calls'], 144)
        self.assertEqual(len(self.result['trials']), 24)
        case_by_trial = {case['source_trial_id']: case for case in self.prepared['cases']}
        for row in self.result['trials']:
            case = case_by_trial[row['source_trial_id']]
            outcome = env.settle(env.State(**case['state']), row['actions'], require_match=True)
            self.assertEqual(row['outcome'], outcome)
            same = row['actions'] == case['shared_plan']
            self.assertEqual(row['all_actions_match_reference_plan'], same)
            self.assertEqual(row['full_success_via_different_plan'], outcome['reward'] == 1 and not same)
        self.assertEqual(self.result['arms'][0]['full_success_trials'], 4)
        self.assertEqual(self.result['arms'][1]['full_success_trials'], 12)
        self.assertEqual(self.result['arms'][1]['individual_actions_matching_reference_plan'], 36)

    def test_module_import_alone_does_not_import_backend_or_model_runtime(self):
        script = ('import sys; import research_program.triadic_task.qwen_execution_diagnostic; '
                  'assert all(m not in sys.modules for m in '
                  '("qwen_language_v3.backend","mlx","mlx.core","mlx_lm","torch"))')
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_model_runtimes_not_imported_and_core_sources_unchanged(self):
        for module in ('mlx', 'mlx.core', 'mlx_lm', 'torch'):
            self.assertNotIn(module, sys.modules)
        for path, original in self.hashes_before.items():
            self.assertEqual(digest(path), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
