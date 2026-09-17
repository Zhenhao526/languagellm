"""Independent capability-runner audit: actual two-stage wrapper, fake infer.

No model initializer, weight loading, MLX, or training is used. Oracle actions
in the positive-gate fixture are researcher test data and never enter prompts.
Run: python3 -m unittest discover -s research_program/triadic_task/tests -v
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from random import Random
import sys
import unittest
from unittest.mock import patch

from research_program.triadic_task import environment as env
from research_program.triadic_task import qwen_capability as cap
from qwen_language_v3.backend import Backend


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token(label, private):
    prefix = 'AUDIT_PRIVATE_' if private else 'AUDIT_PUBLIC_'
    return prefix + f"g{label['group']}_e{label['episode']}_w{label['window']}_{label['agent']}"


class RecordingBackend(Backend):
    """Use the frozen Backend.decide verbatim, replace only model infer."""
    def __init__(self, prepared=None, fail_first=False):
        self.calls = 0
        self.entries = []
        self.oracle = {}
        if prepared is not None:
            for case in prepared['trials']:
                actions = env.sufficient_information_witness(env.State(**case['state']))
                if fail_first and case is prepared['trials'][0]:
                    actions = {a: {'kind': 'wait'} for a in env.AGENTS}
                for agent in env.AGENTS:
                    self.oracle[(case['group'], case['episode'], agent)] = str(next(
                        m['id'] for m in case['menus'][agent] if m['action'] == actions[agent]))

    def infer(self, messages, *, mode, label, seed=0, limit=32, choices=None, temperature=0):
        data = json.loads(messages[1]['content'])
        if mode == 'private_analysis':
            answer = token(label, True)
        elif mode == 'natural_message':
            answer = token(label, False)
        elif mode == 'action':
            answer = self.oracle.get((label['group'], label['episode'], label['agent']))
            if answer is None:
                answer = str(next(m['编号'] for m in data['本次可选动作'] if m['动作'] == '等待'))
        else:
            raise AssertionError(f'Unexpected inference mode {mode}')
        self.calls += 1
        self.entries.append({'call': self.calls, 'messages': deepcopy(messages), 'mode': mode,
                             'label': deepcopy(label), 'seed': seed, 'limit': limit,
                             'choices': deepcopy(choices), 'temperature': temperature, 'output': answer})
        return answer


class QwenCapabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_paths = [Path(cap.__file__), Path(env.__file__), cap.BACKEND,
                            cap.WORK / 'qwen_language_v3/run.py']
        cls.source_before = {str(p): digest(p) for p in cls.source_paths}
        cls.prepared = cap.build_cases()
        cls.prepared_before = deepcopy(cls.prepared)
        cls.backend = RecordingBackend()
        cls.callbacks = []
        cls.result = cap.run_cases(cls.backend, cls.prepared, on_trial=cls.callbacks.append)

    @classmethod
    def tearDownClass(cls):
        for path, original in cls.source_before.items():
            if digest(path) != original:
                raise AssertionError(f'Source modified during audit: {path}')

    def test_actual_two_stage_wrapper_and_full_failure_budget(self):
        self.assertIs(RecordingBackend.decide, Backend.decide)
        self.assertEqual(len(self.backend.entries), 216)
        self.assertEqual(self.backend.calls, cap.MAX_CALLS)
        self.assertEqual(len(self.result['records']), 12)
        self.assertEqual(len(self.callbacks), 12)
        self.assertEqual([r['outcome']['reward'] for r in self.result['records']], [0.] * 12)
        self.assertFalse(self.result['gate']['passed'])
        self.assertEqual([r['call_range'] for r in self.result['records']],
                         [[18 * i + 1, 18 * (i + 1)] for i in range(12)])
        self.assertEqual(Counter(e['mode'] for e in self.backend.entries),
                         {'private_analysis': 108, 'natural_message': 72, 'action': 36})
        self.assertEqual(self.prepared, self.prepared_before)

    def test_two_synchronous_barriers_in_complete_inference_messages(self):
        for trial_index, row in enumerate(self.result['records']):
            for window_index, window in enumerate((1, 2, None)):
                expected_transcript = [] if window == 1 else row['messages'][:3] if window == 2 else row['messages']
                for agent_index, agent in enumerate(env.AGENTS):
                    index = trial_index * 18 + (window_index * 3 + agent_index) * 2
                    analysis, formal = self.backend.entries[index:index + 2]
                    self.assertEqual(analysis['label']['agent'], agent)
                    self.assertEqual(analysis['label']['window'], window)
                    self.assertEqual(analysis['messages'][:2], formal['messages'][:2])
                    self.assertEqual(len(analysis['messages']), 3)
                    self.assertEqual(len(formal['messages']), 4)
                    self.assertEqual(analysis['messages'][-1]['role'], 'user')
                    self.assertEqual(formal['messages'][-2],
                                     {'role': 'assistant', 'content': analysis['output']})
                    data = json.loads(analysis['messages'][1]['content'])
                    self.assertEqual(data['本轮已公开广播'], expected_transcript)
                    self.assertEqual(data['你自己的历史'], row['private_histories_before'][agent])
                    self.assertEqual(data['本轮完整信息观察']['你是'], agent)
                    if window is not None:
                        self.assertNotIn('本次可选动作', data)
                    else:
                        self.assertEqual(len(data['本次可选动作']), 17)
                        self.assertEqual(set(data), {'你自己的历史', '本轮完整信息观察', '本轮已公开广播',
                                                     '本次可选动作', '当前决策'})

    def test_private_analysis_only_enters_its_own_immediate_formal_call(self):
        for i, entry in enumerate(self.backend.entries):
            messages = entry['messages']
            text = json.dumps(messages, ensure_ascii=False)
            expected = self.backend.entries[i - 1]['output'] if entry['label']['stage'] == 'formal' else None
            markers = [e['output'] for e in self.backend.entries if e['mode'] == 'private_analysis']
            actual = [marker for marker in markers if marker in text]
            self.assertEqual(actual, [expected] if expected else [])
            data = json.loads(messages[1]['content'])
            self.assertNotIn('AUDIT_PRIVATE_', json.dumps(data, ensure_ascii=False))
        self.assertNotIn('AUDIT_PRIVATE_', json.dumps(self.result['histories_after'], ensure_ascii=False))

    def test_group_and_agent_history_ownership(self):
        by_group = {g: [r for r in self.result['records'] if r['group'] == g] for g in cap.GROUPS}
        for group, records in by_group.items():
            for index, row in enumerate(records):
                for agent in env.AGENTS:
                    history = row['private_histories_before'][agent]
                    self.assertEqual(len(history), index)
                    expected = []
                    for earlier in records[:index]:
                        expected.append({'本轮观察': earlier['observations'][agent],
                                         '公开广播': earlier['messages'],
                                         '自己提交的动作': earlier['actions'][agent],
                                         '自己可见结果': earlier['outcome']['individual_feedback'][agent]})
                    self.assertEqual(history, expected)
                    for item in history:
                        self.assertEqual(set(item), {'本轮观察', '公开广播', '自己提交的动作', '自己可见结果'})
                        self.assertEqual(item['本轮观察']['self'], agent)
                        self.assertEqual(set(item['自己可见结果']), {'executed', 'own_need_satisfied', 'team_reward'})
                    text = json.dumps(history, ensure_ascii=False)
                    for other in cap.GROUPS:
                        if other != group:
                            self.assertNotIn(f'AUDIT_PUBLIC_g{other}_', text)
            for agent in env.AGENTS:
                self.assertEqual(len(self.result['histories_after'][group][agent]), 4)
        # Current full-information inputs are the same across context groups;
        # group ids/seeds appear in researcher labels, not in first prompts.
        first = self.prepared['first_prompts']
        self.assertEqual(first['301'], first['302'])
        self.assertEqual(first['302'], first['303'])

    def test_current_observation_is_exact_full_information_control(self):
        for case in self.prepared['trials']:
            state = env.State(**case['state'])
            for agent in env.AGENTS:
                observation = case['observations'][agent]
                self.assertEqual(observation, env.observe(state, agent, shared_needs=True, full_information=True))
                self.assertEqual(len(observation['visible_materials']), 4)
                self.assertEqual(set(observation['shared_needs']), set(env.AGENTS))
                self.assertEqual(observation['information_control'], 'full_information')
        # Complete semantic information is authorized here; raw world ids,
        # source seeds, witnesses and result objects are not model input fields.
        forbidden_keys = {'state', 'researcher', 'witness', 'seed', 'scene_seed', 'menu_seeds',
                          'case_id', 'selections', 'outcome', 'private_histories_before'}
        def keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(keys(x) for x in value.values()), set())
            if isinstance(value, list):
                return set().union(*(keys(x) for x in value), set())
            return set()
        for entry in self.backend.entries:
            data = json.loads(entry['messages'][1]['content'])
            self.assertFalse(keys(data) & forbidden_keys)
            text = json.dumps(entry['messages'], ensure_ascii=False)
            for seed in cap.SCENE_SEEDS:
                self.assertNotIn(str(seed), text)
            for seed in (e['seed'] for e in self.backend.entries):
                self.assertNotIn(str(seed), text)

    def test_researcher_case_fields_and_callback_outputs_do_not_enter_prompts(self):
        prepared = deepcopy(self.prepared)
        for case in prepared['trials']:
            case['researcher'] = {'secret': 'DO_NOT_SHOW_RESEARCHER'}
            case['witness'] = 'DO_NOT_SHOW_WITNESS'
            case['previous_other_group_output'] = 'DO_NOT_SHOW_OTHER_GROUP'
        fake = RecordingBackend()
        def mutate_callback(row):
            row['messages'].append({'text': 'DO_NOT_SHOW_CALLBACK_MUTATION'})
            row['observations']['A']['own_need']['value'] = 'DO_NOT_SHOW_CALLBACK_MUTATION'
        result = cap.run_cases(fake, prepared, on_trial=mutate_callback)
        for entry in fake.entries:
            self.assertNotIn('DO_NOT_SHOW_', json.dumps(entry['messages'], ensure_ascii=False))
        self.assertNotIn('DO_NOT_SHOW_', json.dumps(result['histories_after'], ensure_ascii=False))

    def test_private_menus_are_complete_and_do_not_depend_on_target_values(self):
        for case in self.prepared['trials']:
            for agent in env.AGENTS:
                menu = case['menus'][agent]
                self.assertEqual([m['id'] for m in menu], list(range(17)))
                expected = env.all_actions(agent)
                self.assertEqual({json.dumps(m['action'], sort_keys=True) for m in menu},
                                 {json.dumps(action, sort_keys=True) for action in expected})
                order = list(range(17))
                Random(case['menu_seeds'][agent]).shuffle(order)
                self.assertEqual(menu, [{'id': i, 'action': expected[k]} for i, k in enumerate(order)])
        changed_world = env.State((0, 6, 9), (3, 2, 1, 0), (3, 1, 2))
        with patch.object(cap.env, 'draw_state', return_value=changed_world):
            changed = cap.build_cases()
        self.assertNotEqual(changed['scenarios'], self.prepared['scenarios'])
        for old, new in zip(self.prepared['trials'], changed['trials']):
            self.assertEqual(old['menus'], new['menus'])
            self.assertEqual(old['menu_seeds'], new['menu_seeds'])

    def test_seed_scope_is_108_unique_decisions_with_shared_two_stage_seed(self):
        decision_seeds = []
        for analysis, formal in zip(self.backend.entries[::2], self.backend.entries[1::2]):
            self.assertEqual(analysis['seed'], formal['seed'])
            label = analysis['label']
            formal_mode = 'action' if label['window'] is None else 'natural_message'
            expected = cap.stable_seed('generation', label['group'], label['episode'],
                                       label['window'] or 0, label['agent'], formal_mode)
            self.assertEqual(analysis['seed'], expected)
            self.assertEqual(analysis['temperature'], 0)
            self.assertEqual(formal['temperature'], 0 if formal_mode == 'action' else .7)
            decision_seeds.append(analysis['seed'])
        self.assertEqual(len(decision_seeds), 108)
        self.assertEqual(len(set(decision_seeds)), 108)
        self.assertEqual(Counter(e['seed'] for e in self.backend.entries),
                         {seed: 2 for seed in decision_seeds})
        menu_seeds = [seed for case in self.prepared['trials'] for seed in case['menu_seeds'].values()]
        self.assertEqual(len(menu_seeds), 36)
        self.assertEqual(len(set(menu_seeds)), 36)
        self.assertFalse(set(menu_seeds) & set(decision_seeds))
        self.assertEqual(len(set(cap.SCENE_SEEDS)), 4)
        self.assertEqual(Counter(c['scene_seed'] for c in self.prepared['trials']),
                         {seed: 3 for seed in cap.SCENE_SEEDS})

    def test_full_success_gate_uses_actual_settlement_and_preserves_old_gate(self):
        oracle = RecordingBackend(self.prepared)
        success = cap.run_cases(oracle, self.prepared)
        self.assertTrue(success['gate']['passed'])
        self.assertEqual(success['gate']['full_success_trials'], 12)
        partial = cap.run_cases(RecordingBackend(self.prepared, fail_first=True), self.prepared)
        self.assertFalse(partial['gate']['passed'])
        self.assertEqual(partial['gate']['full_success_trials'], 11)
        self.assertEqual(partial['model_calls'], 216)
        self.assertTrue(success['gate']['new_task_only'])
        self.assertFalse(success['gate']['modifies_original_v3_gate'])
        self.assertFalse(success['gate']['starts_symbolic_experiment'])
        with self.assertRaises(ValueError):
            cap.capability_gate(self.result['records'][:-1], self.prepared)
        forged = deepcopy(self.result['records'])
        forged[0]['outcome']['reward'] = 1.
        with self.assertRaises(ValueError):
            cap.capability_gate(forged, self.prepared)

    def test_call_cap_checks_before_inference_and_rejects_bad_counter(self):
        backend = RecordingBackend()
        backend.calls = 215
        with self.assertRaises(ValueError):
            cap._decide(backend, [], mode='natural_message',
                        label={'group': 301, 'episode': 1, 'window': 1, 'agent': 'A'})
        self.assertEqual(backend.calls, 215)
        self.assertEqual(backend.entries, [])
        class BrokenCounter:
            calls = 0
            def decide(self, *args, **kwargs):
                self.calls += 1
                return ''
        with self.assertRaises(ValueError):
            cap._decide(BrokenCounter(), [], mode='natural_message',
                        label={'group': 301, 'episode': 1, 'window': 1, 'agent': 'A'})

    def test_sources_and_module_imports_remain_model_free(self):
        self.assertEqual(digest(cap.BACKEND), cap.BACKEND_SHA)
        self.assertNotIn('mlx', sys.modules)
        self.assertNotIn('mlx.core', sys.modules)
        self.assertNotIn('mlx_lm', sys.modules)
        self.assertNotIn('torch', sys.modules)
        for path, original in self.source_before.items():
            self.assertEqual(digest(path), original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
