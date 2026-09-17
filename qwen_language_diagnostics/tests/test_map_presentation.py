"""Map-wording contrasts: deterministic world checks and a fake backend."""

from collections import deque
from copy import deepcopy
import json
import unittest

from qwen_language_diagnostics import map_presentation
from qwen_language_diagnostics.map_cases import build_map_cases
from qwen_language_diagnostics.semantic_cases import build_semantic_cases
from qwen_language_v3.environment import RULES, World


class FakeBackend:
    def __init__(self, responses, *, mutate_received_prompt=False):
        self.responses = deque(responses)
        self.calls = 0
        self.decisions = []
        self.analyses = []
        self.mutate_received_prompt = mutate_received_prompt

    def decide(self, prompt, **kwargs):
        if not self.responses:
            raise AssertionError('Unexpected extra decision or retry')
        response = self.responses.popleft()
        formal, analysis = response if isinstance(response, tuple) else (response, 'Unscored private text.')
        self.decisions.append({'prompt': deepcopy(prompt), **deepcopy(kwargs)})
        self.analyses.append(analysis)
        self.calls += 2
        if self.mutate_received_prompt:
            prompt.append({'role': 'assistant', 'content': 'Do not retain this backend-private mutation.'})
        return str(formal)


class MapCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = build_map_cases()

    def test_fixed_case_order_seed_truth_balance_and_answer_order(self):
        cases = self.prepared['cases']
        expected = [(question, representation) for question in ('G1', 'G2', 'J1', 'J2')
                    for representation in ('original', 'explicit_edges')]
        self.assertEqual([(case['question_id'], case['representation']) for case in cases], expected)
        self.assertEqual(len({case['case_id'] for case in cases}), 8)
        self.assertEqual({case['seed'] for case in cases}, {20260916})
        truths = {'G1': True, 'G2': False, 'J1': True, 'J2': False}
        for case in cases:
            self.assertIs(case['truth'], truths[case['question_id']])
            self.assertEqual(case['choices'], [0, 1])
            self.assertEqual(case['order'], 'affirm_first')
            self.assertEqual(case['state_id'], 'S0')
            self.assertEqual(case['correct_choice'], 0 if case['truth'] else 1)
        self.assertEqual(sum(case['truth'] for case in cases), 4)

    def test_paired_prompts_change_exactly_one_map_sentence_and_no_user_bytes(self):
        change = self.prepared['presentation_change']
        old, new = change['original'], change['explicit_edges']
        self.assertNotEqual(old, new)
        self.assertEqual(change['replacement_count'], 1)
        self.assertEqual(RULES.count(old), 1)
        for question in ('G1', 'G2', 'J1', 'J2'):
            pair = {case['representation']: case for case in self.prepared['cases']
                    if case['question_id'] == question}
            original, explicit = pair['original'], pair['explicit_edges']
            self.assertEqual(original['prompt'][1], explicit['prompt'][1])
            self.assertEqual(original['statement'], explicit['statement'])
            self.assertEqual(original['answers'], explicit['answers'])
            self.assertEqual(original['seed'], explicit['seed'])
            before = original['prompt'][0]['content']
            after = explicit['prompt'][0]['content']
            self.assertEqual(before.count(old), 1)
            self.assertEqual(after.count(new), 1)
            self.assertEqual(after, before.replace(old, new, 1))
            self.assertEqual(after.replace(new, old, 1), before)
            self.assertIn(RULES, before)
            self.assertIn(RULES.replace(old, new, 1), after)

    def test_independent_inputs_contain_no_researcher_answers_or_previous_responses(self):
        for case in self.prepared['cases']:
            with self.subTest(case=case['case_id']):
                self.assertEqual([message['role'] for message in case['prompt']], ['system', 'user'])
                data = json.loads(case['prompt'][1]['content'])
                self.assertEqual(data['你自己的历史'], [])
                self.assertEqual(data['当前已可见广播'], [])
                self.assertNotIn('本次可选动作', data)
                serialized = json.dumps(case['prompt'], ensure_ascii=False)
                for key in ('truth', 'correct_choice', 'checks', 'evaluated_facts',
                            'anchor', 'original_analysis', 'original_formal', 'state_after'):
                    self.assertNotIn('"'+key+'"', serialized)

    def test_j1_original_is_exact_m1_affirm_first_base_prompt_and_seed(self):
        j1 = next(case for case in self.prepared['cases'] if case['case_id'] == 'J1_original')
        m1 = next(case for case in build_semantic_cases()['cases']
                  if case['question_id'] == 'M1' and case['order'] == 'affirm_first')
        self.assertEqual(j1['prompt'], m1['prompt'])
        self.assertEqual(j1['seed'], m1['seed'])
        self.assertEqual(j1['choices'], m1['choices'])
        self.assertEqual(j1['answers'], m1['answers'])
        self.assertEqual(j1['statement'], m1['statement'])
        anchor = self.prepared['anchor']
        self.assertEqual(anchor['prompt'], m1['prompt'])
        self.assertEqual(anchor['seed'], m1['seed'])
        self.assertEqual(anchor['source_case_id'], 'M1_affirm_first')
        self.assertEqual(anchor['case_id'], 'J1_original')

    def _wood_target(self):
        state = self.prepared['states']['S0']
        goal = next(goal for goal in state['goals'] if goal['kind'] == '木材')
        candidates = [oid for oid, item in state['items'].items()
                      if all(item[key] == goal[key] for key in ('kind', 'length', 'condition'))
                      and not item['delivered']]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(goal['destination'], '营地')
        self.assertEqual(goal['length'], '长')
        return candidates[0]

    def test_g1_direct_single_step_is_legal_and_refutes_g2_required_detour(self):
        states = self.prepared['states']
        before = deepcopy(states)
        initial = states['S0']
        self.assertEqual(initial['positions']['A'], '林地')
        actions = {'A': {'kind': 'move', 'destination': '营地'},
                   'B': {'kind': 'wait'}, 'C': {'kind': 'wait'}}
        for question in ('G1', 'G2'):
            check = self.prepared['checks'][question]
            self.assertEqual(check['actions'], actions)
            world = World.from_state_dict(initial)
            self.assertIn(actions['A'], [item['action'] for item in world.action_menu('A')])
            feedback = world.step(deepcopy(actions))
            self.assertTrue(feedback['A']['action_succeeded'])
            self.assertEqual(world.positions['A'], '营地')
            self.assertEqual(world.t, initial['t']+1)
            self.assertEqual(world.state_dict(), states['G1_after'])
            self.assertEqual(feedback, check['feedback'])
            self.assertIs(check['truth'], question == 'G1')
        self.assertEqual(states, before)

    def test_j1_matching_partners_and_destination_carry_ground_wood_directly(self):
        states = self.prepared['states']
        initial, check = states['S0'], self.prepared['checks']['J1']
        oid = self._wood_target()
        self.assertEqual(initial['items'][oid]['location'], '林地')
        self.assertEqual(initial['items'][oid]['carriers'], [])
        actions = {
            'A': {'kind': 'carry_together', 'partner': 'C', 'destination': '营地',
                  'item': initial['handles']['A'][oid]},
            'B': {'kind': 'wait'},
            'C': {'kind': 'carry_together', 'partner': 'A', 'destination': '营地',
                  'item': initial['handles']['C'][oid]},
        }
        self.assertEqual(check['actions'], actions)
        world = World.from_state_dict(initial)
        feedback = world.step(deepcopy(actions))
        self.assertEqual(world.state_dict(), states['J1_after'])
        self.assertEqual(feedback, check['feedback'])
        for agent in ('A', 'C'):
            self.assertTrue(feedback[agent]['action_succeeded'])
            self.assertEqual(world.positions[agent], '营地')
            self.assertEqual(world.inventory[agent]['cargo'], oid)
        # Physical location is given by the carriers; carried items have no
        # ground-location value in this environment's state representation.
        self.assertIsNone(world.items[oid]['location'])
        self.assertEqual(set(world.items[oid]['carriers']), {'A', 'C'})
        self.assertIs(check['truth'], True)

    def test_j2_legal_but_different_destinations_fail_without_physical_change(self):
        states = self.prepared['states']
        initial, check = states['S0'], self.prepared['checks']['J2']
        oid = self._wood_target()
        actions = {
            'A': {'kind': 'carry_together', 'partner': 'C', 'destination': '营地',
                  'item': initial['handles']['A'][oid]},
            'B': {'kind': 'wait'},
            'C': {'kind': 'carry_together', 'partner': 'A', 'destination': '河滩',
                  'item': initial['handles']['C'][oid]},
        }
        self.assertEqual(check['actions'], actions)
        world = World.from_state_dict(initial)
        for agent in ('A', 'C'):
            self.assertIn(actions[agent], [item['action'] for item in world.action_menu(agent)])
        feedback = world.step(deepcopy(actions))
        self.assertEqual(world.state_dict(), states['J2_after'])
        self.assertEqual(feedback, check['feedback'])
        for agent in ('A', 'C'):
            self.assertFalse(feedback[agent]['action_succeeded'])
        self.assertEqual(world.positions, initial['positions'])
        self.assertEqual(world.items, initial['items'])
        self.assertEqual(world.inventory, initial['inventory'])
        self.assertEqual(world.t, initial['t']+1)
        self.assertIs(check['truth'], False)


class MapRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = build_map_cases()

    def test_formal_correct_answers_fixed_16_calls_with_private_input_copies(self):
        prepared = deepcopy(self.prepared)
        before = deepcopy(prepared)
        backend = FakeBackend([
            (case['correct_choice'], f"私有文本认为应选 {1-case['correct_choice']}。")
            for case in prepared['cases']
        ], mutate_received_prompt=True)
        emitted = []
        result = map_presentation.run_cases(backend, prepared, lambda row: emitted.append(deepcopy(row)))
        self.assertEqual(len(result), 8)
        self.assertTrue(all(row['correct'] for row in result))
        self.assertEqual(backend.calls, 16)
        self.assertFalse(backend.responses)
        self.assertEqual(len(emitted), 8)
        self.assertEqual(prepared, before)
        self.assertEqual([row['case_id'] for row in result], [case['case_id'] for case in prepared['cases']])
        self.assertEqual([row['prompt'] for row in backend.decisions], [case['prompt'] for case in prepared['cases']])
        self.assertEqual([row['call_ids'] for row in result], [[i, i+1] for i in range(1, 17, 2)])
        self.assertEqual({row['seed'] for row in backend.decisions}, {20260916})

    def test_formal_errors_do_not_trigger_extra_questions_or_retries(self):
        prepared = deepcopy(self.prepared)
        backend = FakeBackend([
            (1-case['correct_choice'], f"私有文本知道正确编号为 {case['correct_choice']}。")
            for case in prepared['cases']
        ])
        emitted = []
        result = map_presentation.run_cases(backend, prepared, lambda row: emitted.append(deepcopy(row)))
        self.assertEqual(len(result), 8)
        self.assertTrue(all(not row['correct'] for row in result))
        self.assertEqual(backend.calls, 16)
        self.assertFalse(backend.responses)
        self.assertEqual(len(emitted), 8)
        self.assertEqual([row['prompt'] for row in backend.decisions], [case['prompt'] for case in prepared['cases']])

    def test_always_affirming_or_denying_each_scores_four_of_eight(self):
        for fixed_choice in (0, 1):
            with self.subTest(fixed_choice=fixed_choice):
                backend = FakeBackend([fixed_choice] * 8)
                result = map_presentation.run_cases(backend, deepcopy(self.prepared))
                self.assertEqual(sum(row['correct'] for row in result), 4)
                self.assertEqual(backend.calls, 16)
                self.assertFalse(backend.responses)


if __name__ == '__main__':
    unittest.main()
