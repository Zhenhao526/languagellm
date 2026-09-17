"""Balanced action-semantics checks using real world replay and a fake model."""

from collections import deque
from copy import deepcopy
import json
import unittest

from qwen_language_diagnostics import action_semantics
from qwen_language_diagnostics.semantic_cases import build_semantic_cases
from qwen_language_v3.environment import World


class FakeBackend:
    def __init__(self, responses, *, mutate_received_prompt=False):
        self.responses = deque(responses)
        self.calls = 0
        self.decisions = []
        self.inferences = []
        self.mutate_received_prompt = mutate_received_prompt

    def decide(self, prompt, **kwargs):
        if not self.responses:
            raise AssertionError('Unexpected extra decision or retry')
        response = self.responses.popleft()
        formal, analysis = response if isinstance(response, tuple) else (response, 'Unscored private text.')
        self.decisions.append({'prompt': deepcopy(prompt), **deepcopy(kwargs)})
        self.calls += 2
        self.inferences.extend([
            {'call': self.calls - 1, 'mode': 'private_analysis', 'output': analysis},
            {'call': self.calls, 'mode': 'action', 'output': str(formal)},
        ])
        if self.mutate_received_prompt:
            prompt.append({'role': 'assistant', 'content': 'Backend-private mutation; not a later example.'})
        return str(formal)

    def stats(self):
        return {'calls': self.calls, 'fake_backend': True}


class SemanticCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = build_semantic_cases()

    def test_truth_and_numeric_option_positions_are_both_balanced(self):
        cases = self.prepared['cases']
        self.assertEqual(len(cases), 8)
        self.assertEqual({case['seed'] for case in cases}, {20260916})
        self.assertEqual(len({case['case_id'] for case in cases}), 8)
        expected_truth = {'M1': True, 'M2': False, 'M3': False, 'M4': True}
        for question_id, truth in expected_truth.items():
            pair = [case for case in cases if case['question_id'] == question_id]
            self.assertEqual(len(pair), 2)
            self.assertEqual({case['order'] for case in pair}, {'affirm_first', 'deny_first'})
            for case in pair:
                self.assertIs(case['truth'], truth)
                self.assertEqual(case['choices'], [0, 1])
                affirmative_index = 0 if case['order'] == 'affirm_first' else 1
                self.assertEqual(case['correct_choice'], affirmative_index if truth else 1-affirmative_index)
        self.assertEqual(sum(case['truth'] for case in cases), 4)
        self.assertEqual(sum(case['correct_choice'] == 0 for case in cases), 4)

    def test_always_affirming_and_always_denying_each_score_only_four_of_eight(self):
        cases = self.prepared['cases']
        affirm = [0 if case['order'] == 'affirm_first' else 1 for case in cases]
        deny = [1-choice for choice in affirm]
        self.assertEqual(sum(choice == case['correct_choice'] for choice, case in zip(affirm, cases)), 4)
        self.assertEqual(sum(choice == case['correct_choice'] for choice, case in zip(deny, cases)), 4)

    def test_order_pair_changes_only_answer_options_not_facts_or_statement(self):
        for question_id in ('M1', 'M2', 'M3', 'M4'):
            pair = {case['order']: case for case in self.prepared['cases']
                    if case['question_id'] == question_id}
            affirm, deny = pair['affirm_first'], pair['deny_first']
            self.assertEqual(affirm['statement'], deny['statement'])
            self.assertEqual(affirm['state_id'], deny['state_id'])
            self.assertEqual(affirm['seed'], deny['seed'])
            self.assertEqual(affirm['answers'], list(reversed(deny['answers'])))
            self.assertEqual(affirm['prompt'][0], deny['prompt'][0])
            a = json.loads(affirm['prompt'][1]['content'])
            b = json.loads(deny['prompt'][1]['content'])
            options_a = a.pop('答案选项')
            options_b = b.pop('答案选项')
            self.assertEqual(a, b)
            self.assertEqual([row['含义'] for row in options_a],
                             [row['含义'] for row in reversed(options_b)])

    def test_prompts_are_independent_and_do_not_reveal_researcher_truth(self):
        for case in self.prepared['cases']:
            with self.subTest(case=case['case_id']):
                self.assertEqual([message['role'] for message in case['prompt']], ['system', 'user'])
                data = json.loads(case['prompt'][1]['content'])
                serialized = json.dumps(case['prompt'], ensure_ascii=False)
                for key in ('correct_choice', 'truth', 'evaluated_facts', 'checks', 'state_after'):
                    self.assertNotIn('"'+key+'"', serialized)
                self.assertEqual(data['你自己的历史'], [])
                self.assertEqual(data['当前已可见广播'], [])
                self.assertNotIn('本次可选动作', data)

    def _target(self):
        state = self.prepared['states']['S0']
        goal = next(goal for goal in state['goals'] if goal['kind'] == '木材')
        matches = [oid for oid, item in state['items'].items()
                   if all(item[field] == goal[field] for field in ('kind', 'length', 'condition'))
                   and not item['delivered']]
        self.assertEqual(len(matches), 1)
        self.assertEqual(goal['length'], '长')
        return matches[0], goal['destination']

    def test_m1_matching_actions_move_ground_wood_and_both_carriers_in_one_step(self):
        states, check = self.prepared['states'], self.prepared['checks']['M1']
        before = deepcopy(states)
        oid, destination = self._target()
        initial = states['S0']
        self.assertEqual(initial['items'][oid]['carriers'], [])
        self.assertEqual(initial['inventory']['A']['cargo'], None)
        self.assertEqual(initial['inventory']['C']['cargo'], None)
        actions = {
            'A': {'kind': 'carry_together', 'partner': 'C', 'destination': destination,
                  'item': initial['handles']['A'][oid]},
            'B': {'kind': 'wait'},
            'C': {'kind': 'carry_together', 'partner': 'A', 'destination': destination,
                  'item': initial['handles']['C'][oid]},
        }
        self.assertEqual(check['actions'], actions)
        world = World.from_state_dict(initial)
        feedback = world.step(actions)
        self.assertEqual(world.state_dict(), states['S1'])
        self.assertEqual(feedback, check['feedback'])
        for agent in ('A', 'C'):
            self.assertTrue(feedback[agent]['action_succeeded'])
            self.assertEqual(world.positions[agent], destination)
            self.assertEqual(world.inventory[agent]['cargo'], oid)
        self.assertEqual(set(world.items[oid]['carriers']), {'A', 'C'})
        # A carried object's location is represented by its carriers' positions,
        # not the ground-location field, which is cleared on joint transport.
        self.assertIsNone(world.items[oid]['location'])
        self.assertEqual(world.t, initial['t']+1)
        self.assertIs(check['truth'], True)
        self.assertEqual(states, before)

    def test_m2_single_actor_pickup_of_long_wood_fails_without_moving_or_loading(self):
        states, check = self.prepared['states'], self.prepared['checks']['M2']
        initial = states['S0']
        oid, _ = self._target()
        actions = {'A': {'kind': 'wait'}, 'B': {'kind': 'wait'},
                   'C': {'kind': 'pickup', 'item': initial['handles']['C'][oid]}}
        self.assertEqual(check['actions'], actions)
        world = World.from_state_dict(initial)
        feedback = world.step(actions)
        self.assertEqual(world.state_dict(), states['M2_after'])
        self.assertEqual(feedback, check['feedback'])
        self.assertFalse(feedback['C']['action_succeeded'])
        self.assertEqual(world.positions, initial['positions'])
        self.assertEqual(world.inventory, initial['inventory'])
        self.assertEqual(world.items[oid], initial['items'][oid])
        self.assertEqual(world.score, 0)
        self.assertEqual(world.t, initial['t']+1)
        self.assertIs(check['truth'], False)

    def test_m3_arrival_is_not_delivery_or_automatic_score(self):
        state = self.prepared['states']['S1']
        check = self.prepared['checks']['M3']
        oid, destination = self._target()
        world = World.from_state_dict(state)
        self.assertIsNone(world.items[oid]['location'])
        self.assertTrue(all(world.positions[agent] == destination for agent in world.items[oid]['carriers']))
        self.assertFalse(world.items[oid]['delivered'])
        self.assertEqual(set(world.items[oid]['carriers']), {'A', 'C'})
        self.assertEqual(world.score, 0)
        self.assertTrue(all(goal['delivered'] == 0 for goal in world.goals))
        self.assertIsNone(check['actions'])
        self.assertIsNone(check['feedback'])
        self.assertIs(check['truth'], False)

    def test_m4_joint_delivery_counts_one_item_and_releases_both_carriers(self):
        states, check = self.prepared['states'], self.prepared['checks']['M4']
        initial = states['S1']
        oid, _ = self._target()
        actions = {
            'A': {'kind': 'deliver_together', 'partner': 'C', 'item': initial['handles']['A'][oid]},
            'B': {'kind': 'wait'},
            'C': {'kind': 'deliver_together', 'partner': 'A', 'item': initial['handles']['C'][oid]},
        }
        self.assertEqual(check['actions'], actions)
        world = World.from_state_dict(initial)
        before_delivered = sum(goal['delivered'] for goal in world.goals)
        feedback = world.step(actions)
        self.assertEqual(world.state_dict(), states['S2'])
        self.assertEqual(feedback, check['feedback'])
        for agent in ('A', 'C'):
            self.assertTrue(feedback[agent]['action_succeeded'])
            self.assertIsNone(world.inventory[agent]['cargo'])
        self.assertTrue(world.items[oid]['delivered'])
        self.assertEqual(world.items[oid]['carriers'], [])
        self.assertEqual(sum(goal['delivered'] for goal in world.goals) - before_delivered, 1)
        self.assertEqual(world.score, 0.5)
        self.assertIs(check['truth'], True)


class SemanticRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = build_semantic_cases()

    def test_eight_formal_correct_answers_run_once_even_when_analysis_disagrees(self):
        prepared = deepcopy(self.prepared)
        before = deepcopy(prepared)
        responses = [(case['correct_choice'], f"私有分析认为应选 {1-case['correct_choice']}。")
                     for case in prepared['cases']]
        backend = FakeBackend(responses, mutate_received_prompt=True)
        emitted = []
        result = action_semantics.run_cases(backend, prepared, lambda row: emitted.append(deepcopy(row)))
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

    def test_eight_formal_errors_do_not_add_retries_or_change_the_budget(self):
        prepared = deepcopy(self.prepared)
        responses = [(1-case['correct_choice'], f"私有分析知道正确编号是 {case['correct_choice']}。")
                     for case in prepared['cases']]
        backend = FakeBackend(responses)
        result = action_semantics.run_cases(backend, prepared)
        self.assertEqual(len(result), 8)
        self.assertTrue(all(not row['correct'] for row in result))
        self.assertEqual(backend.calls, 16)
        self.assertFalse(backend.responses)
        self.assertEqual([row['prompt'] for row in backend.decisions], [case['prompt'] for case in prepared['cases']])

    def test_always_affirming_and_denying_are_graded_as_half_correct(self):
        for stance in ('affirm', 'deny'):
            with self.subTest(stance=stance):
                prepared = deepcopy(self.prepared)
                choices = [0 if case['order'] == 'affirm_first' else 1 for case in prepared['cases']]
                if stance == 'deny':
                    choices = [1-choice for choice in choices]
                backend = FakeBackend(choices)
                result = action_semantics.run_cases(backend, prepared)
                self.assertEqual(sum(row['correct'] for row in result), 4)
                self.assertEqual(backend.calls, 16)
                self.assertFalse(backend.responses)


if __name__ == '__main__':
    unittest.main()
