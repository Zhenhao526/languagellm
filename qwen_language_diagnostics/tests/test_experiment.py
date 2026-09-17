"""Decision-path and counterfactual-integrity tests; no model is loaded."""

from collections import deque
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from qwen_language_diagnostics import experiment
from qwen_language_diagnostics.history_cases import load_history_cases
from qwen_language_v3.environment import AGENTS, World


class FakeBackend:
    """Keep contradictory analysis as evidence, but return only formal output."""

    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = 0
        self.decisions = []
        self.inferences = []

    def decide(self, prompt, **kwargs):
        if not self.responses:
            raise AssertionError('Unexpected extra decision or retry')
        response = self.responses.popleft()
        if isinstance(response, tuple):
            formal, analysis = response
        else:
            formal, analysis = response, 'Private analysis is not the formal answer.'
        self.decisions.append({'prompt': deepcopy(prompt), **deepcopy(kwargs)})
        self.calls += 2
        self.inferences.extend([
            {'call': self.calls - 1, 'mode': 'private_analysis', 'output': analysis},
            {'call': self.calls, 'mode': 'action', 'output': str(formal)},
        ])
        return str(formal)

    def stats(self):
        return {'calls': self.calls, 'fake_backend': True}


class RuleCasesTests(unittest.TestCase):
    def test_all_rule_cases_share_seed_and_balance_numeric_answer_position(self):
        full = experiment.make_rule_cases('full')
        card = experiment.make_rule_cases('card')
        self.assertEqual(len(full), 4)
        self.assertEqual(len(card), 4)
        self.assertEqual({case['seed'] for case in full + card}, {20260915})
        self.assertEqual({(case['length'], case['order']) for case in full}, {
            ('长', 'can_first'), ('长', 'must_first'),
            ('短', 'can_first'), ('短', 'must_first'),
        })
        self.assertEqual(len({case['case_id'] for case in full + card}), 8)
        for case in full + card:
            self.assertEqual(case['choices'], [0, 1])
            expected = 0 if case['order'] == 'can_first' else 1
            self.assertEqual(case['correct_choice'], expected)
            self.assertEqual(case['answers'][expected], experiment.CAN)
            data = json.loads(case['prompt'][1]['content'])
            self.assertEqual(data['答案选项'][expected]['含义'], experiment.CAN)

    def test_length_order_and_rule_source_are_the_only_prompt_contrasts(self):
        full = {(case['length'], case['order']): case
                for case in experiment.make_rule_cases('full')}
        card = {(case['length'], case['order']): case
                for case in experiment.make_rule_cases('card')}
        for key, original in full.items():
            shortened = card[key]
            self.assertEqual(original['prompt'][1], shortened['prompt'][1])
            self.assertEqual(original['prompt'][0]['content'].replace(
                experiment.RULES, experiment.RULE_CARD), shortened['prompt'][0]['content'])
            self.assertEqual(original['seed'], shortened['seed'])
        for order in ('can_first', 'must_first'):
            long = full['长', order]['prompt']
            short = full['短', order]['prompt']
            self.assertEqual(long[0], short[0])
            long_data, short_data = json.loads(long[1]['content']), json.loads(short[1]['content'])
            question = long_data.pop('问题')
            self.assertEqual(question.replace('干长纤维', '干短纤维'), short_data.pop('问题'))
            self.assertEqual(long_data, short_data)
        for length in ('长', '短'):
            first = json.loads(full[length, 'can_first']['prompt'][1]['content'])
            second = json.loads(full[length, 'must_first']['prompt'][1]['content'])
            options = first.pop('答案选项')
            reversed_options = second.pop('答案选项')
            self.assertEqual([x['含义'] for x in options], [x['含义'] for x in reversed(reversed_options)])
            self.assertEqual(first, second)

    def test_all_semantic_truth_values_are_yes_not_negative_rule_controls(self):
        # Numeric option reversal is not a yes/no truth-value counterbalance.
        cases = experiment.make_rule_cases('full') + experiment.make_rule_cases('card')
        self.assertEqual({case['correct_meaning'] for case in cases}, {experiment.CAN})
        self.assertTrue(all(case['answers'][case['correct_choice']] != experiment.MUST for case in cases))
        for case in cases:
            data = json.loads(case['prompt'][1]['content'])
            self.assertTrue(all(item['种类'] == '纤维' for item in data['当前状态']['地面物品']))
            self.assertEqual(data['当前状态']['普通携带位'], '空闲')

    def test_rule_grade_uses_formal_output_not_private_analysis(self):
        case = experiment.make_rule_cases()[0]
        right = case['correct_choice']
        wrong = 1 - right
        formal_correct = experiment.evaluate_rule(
            FakeBackend([(right, f'我认为应选错误编号 {wrong}。')]), case)
        formal_wrong = experiment.evaluate_rule(
            FakeBackend([(wrong, f'我知道正确编号是 {right}。')]), case)
        self.assertTrue(formal_correct['correct'])
        self.assertFalse(formal_wrong['correct'])
        self.assertEqual(formal_correct['call_ids'], [1, 2])
        self.assertEqual(formal_wrong['meaning'], experiment.MUST)

    def test_bad_rule_source_and_out_of_menu_formal_answer_are_rejected(self):
        with self.assertRaises(ValueError):
            experiment.make_rule_cases('unplanned_hint')
        case = experiment.make_rule_cases()[0]
        backend = FakeBackend([7])
        with self.assertRaises(ValueError):
            experiment.evaluate_rule(backend, case)
        self.assertEqual(backend.calls, 2)


class HistoryAndScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This validates/reconstructs frozen JSON inputs without inference.
        cls.history_cases = load_history_cases(experiment.SOURCE_RUN)

    def prepared(self):
        return {'rule_full': experiment.make_rule_cases('full'),
                'rule_card': experiment.make_rule_cases('card'),
                'history': deepcopy(self.history_cases)}

    @staticmethod
    def recorded_choices(cases):
        return [case['original_record']['selections'][case['agent']]['id']
                for case in cases for _ in range(2)]

    def test_correct_formal_rules_skip_card_despite_wrong_analysis_and_use_16_calls(self):
        prepared = self.prepared()
        before = deepcopy(prepared)
        responses = [(case['correct_choice'], f"我认为应选 {1-case['correct_choice']}。")
                     for case in prepared['rule_full']]
        responses += self.recorded_choices(prepared['history'])
        backend = FakeBackend(responses)
        emitted = []
        result = experiment.run_cases(backend, prepared, lambda row: emitted.append(deepcopy(row)))
        self.assertFalse(result['rule_card_triggered'])
        self.assertEqual(len(result['rule_results']), 4)
        self.assertEqual(len(result['history_results']), 4)
        self.assertEqual(backend.calls, 16)
        self.assertEqual(len(emitted), 8)
        self.assertFalse(backend.responses)
        self.assertEqual(prepared, before)
        expected_prompts = [case['prompt'] for case in prepared['rule_full']]
        expected_prompts += [case[field] for case in prepared['history']
                             for field in ('prompt', 'empty_history_prompt')]
        self.assertEqual([row['prompt'] for row in backend.decisions], expected_prompts)
        self.assertEqual([row['call_ids'] for row in emitted], [[n, n+1] for n in range(1, 17, 2)])

    def test_formal_error_triggers_exactly_one_card_batch_then_all_s_arms_24_calls(self):
        prepared = self.prepared()
        full = [(case['correct_choice'], 'Analysis need not agree.') for case in prepared['rule_full']]
        first = prepared['rule_full'][0]
        full[0] = (1-first['correct_choice'], f"正确编号明确是 {first['correct_choice']}。")
        # The card is deliberately wrong too: no further card or repair retry.
        card = [(1-case['correct_choice'], f"正确编号是 {case['correct_choice']}。")
                for case in prepared['rule_card']]
        backend = FakeBackend(full + card + self.recorded_choices(prepared['history']))
        emitted = []
        result = experiment.run_cases(backend, prepared, lambda row: emitted.append(deepcopy(row)))
        self.assertTrue(result['rule_card_triggered'])
        self.assertEqual(len(result['rule_results']), 8)
        self.assertTrue(all(not row['correct'] for row in result['rule_results'][4:]))
        self.assertEqual(len(result['history_results']), 4)
        self.assertEqual(backend.calls, 24)
        self.assertFalse(backend.responses)
        self.assertEqual(len(emitted), 12)
        expected_prompts = [case['prompt'] for section in ('rule_full', 'rule_card') for case in prepared[section]]
        expected_prompts += [case[field] for case in prepared['history'] for field in ('prompt', 'empty_history_prompt')]
        self.assertEqual([row['prompt'] for row in backend.decisions], expected_prompts)
        self.assertEqual([(row['case_id'], row['arm']) for row in result['history_results']], [
            (case['case_id'], arm) for case in prepared['history'] for arm in ('original', 'empty_history')])

    def test_s_action_changes_do_not_trigger_card_or_extra_retries(self):
        prepared = self.prepared()
        responses = [case['correct_choice'] for case in prepared['rule_full']]
        for case in prepared['history']:
            original = case['original_record']['selections'][case['agent']]['id']
            other = next(item['id'] for item in case['original_record']['menus'][case['agent']]
                         if item['id'] != original)
            responses.extend([other, other])
        backend = FakeBackend(responses)
        result = experiment.run_cases(backend, prepared)
        self.assertFalse(result['rule_card_triggered'])
        self.assertTrue(all(not row['action_matches_recorded'] for row in result['history_results']))
        self.assertEqual(backend.calls, 16)
        self.assertFalse(backend.responses)

    def test_history_arm_changes_only_supplied_history_and_preserves_inputs(self):
        for saved in self.history_cases:
            with self.subTest(case=saved['case_id']):
                case = deepcopy(saved)
                before = deepcopy(case)
                choice = case['original_record']['selections'][case['agent']]['id']
                original, empty = FakeBackend([choice]), FakeBackend([choice])
                experiment.evaluate_history(original, case, 'original')
                experiment.evaluate_history(empty, case, 'empty_history')
                self.assertEqual(case, before)
                self.assertEqual(original.decisions[0]['prompt'], case['prompt'])
                self.assertEqual(empty.decisions[0]['prompt'], case['empty_history_prompt'])
                self.assertEqual(original.decisions[0]['seed'], case['seed'])
                self.assertEqual(empty.decisions[0]['seed'], case['seed'])
                self.assertEqual(original.decisions[0]['choices'], empty.decisions[0]['choices'])
                self.assertEqual(original.decisions[0]['prompt'][0], empty.decisions[0]['prompt'][0])
                expected = json.loads(original.decisions[0]['prompt'][1]['content'])
                expected['你自己的历史'] = []
                actual = json.loads(empty.decisions[0]['prompt'][1]['content'])
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual['当前已可见广播']), 12)
                self.assertNotIn('original_record', actual)
                self.assertNotIn('state_before', actual)
                self.assertNotIn('original_analysis', actual)
                self.assertNotIn('original_formal', actual)

    def test_only_selected_actor_changes_and_settlement_happens_after_decision(self):
        for saved in self.history_cases:
            with self.subTest(case=saved['case_id']):
                case = deepcopy(saved)
                record, agent = case['original_record'], case['agent']
                original = record['selections'][agent]['id']
                selected = next(item for item in record['menus'][agent] if item['id'] != original)
                expected_actions = deepcopy(record['actions'])
                expected_actions[agent] = deepcopy(selected['action'])
                expected_world = World.from_state_dict(record['state_before'])
                expected_feedback = expected_world.step(deepcopy(expected_actions))
                backend = FakeBackend([selected['id']])
                actual_step = World.step
                commit_count = []

                def checked_step(world, actions):
                    self.assertEqual(len(backend.decisions), 1)
                    self.assertEqual(world.state_dict(), record['state_before'])
                    self.assertEqual(actions, expected_actions)
                    commit_count.append(1)
                    return actual_step(world, actions)

                with patch.object(World, 'step', checked_step):
                    result = experiment.evaluate_history(backend, case, 'empty_history')
                self.assertEqual(len(commit_count), 1)
                self.assertEqual(result['selection'], selected)
                self.assertEqual(result['actions'], expected_actions)
                self.assertEqual(result['feedback'], expected_feedback)
                self.assertEqual(result['state_after'], expected_world.state_dict())
                self.assertEqual(result['score_after_one_step'], expected_world.score)
                self.assertFalse(result['action_matches_recorded'])
                for other in AGENTS:
                    if other != agent:
                        self.assertEqual(result['actions'][other], record['actions'][other])

    def test_other_actors_researcher_actions_never_enter_model_prompt(self):
        # C step9: A's recorded joint action can be replaced by wait in the
        # researcher-only counterfactual without changing any of C's input.
        original = deepcopy(next(case for case in self.history_cases if case['agent'] == 'C'))
        altered = deepcopy(original)
        other = next(agent for agent in AGENTS if agent != original['agent']
                     and altered['original_record']['actions'][agent]['kind'] != 'wait')
        altered['original_record']['actions'][other] = {'kind': 'wait'}
        selected = original['original_record']['selections']['C']['id']
        first, second = FakeBackend([selected]), FakeBackend([selected])
        result_a = experiment.evaluate_history(first, original, 'original')
        result_b = experiment.evaluate_history(second, altered, 'original')
        self.assertEqual(first.decisions, second.decisions)
        self.assertNotEqual(result_a['actions'][other], result_b['actions'][other])

    def test_result_mutation_does_not_write_into_frozen_case(self):
        case = deepcopy(self.history_cases[0])
        before = deepcopy(case)
        agent = case['agent']
        choice = case['original_record']['selections'][agent]['id']
        result = experiment.evaluate_history(FakeBackend([choice]), case, 'original')
        result['selection']['action']['kind'] = 'corrupted'
        result['recorded_selection']['action']['kind'] = 'corrupted'
        result['actions'][agent]['kind'] = 'corrupted'
        result['feedback'][agent]['result'] = 'corrupted'
        result['state_after']['positions'][agent] = 'corrupted'
        self.assertEqual(case, before)

    def test_researcher_target_changes_grade_but_not_prompt_or_selected_action(self):
        plain = deepcopy(self.history_cases[0])
        agent = plain['agent']
        selected = plain['original_record']['selections'][agent]
        matching = deepcopy(plain)
        matching['diagnostic_target_action'] = deepcopy(selected['action'])
        different = deepcopy(plain)
        different['diagnostic_target_action'] = deepcopy(next(
            item['action'] for item in plain['original_record']['menus'][agent]
            if item['action'] != selected['action']))
        backends = [FakeBackend([selected['id']]) for _ in range(3)]
        results = [experiment.evaluate_history(backend, case, 'original')
                   for backend, case in zip(backends, (plain, matching, different))]
        self.assertIsNone(results[0]['matches_diagnostic_target'])
        self.assertTrue(results[1]['matches_diagnostic_target'])
        self.assertFalse(results[2]['matches_diagnostic_target'])
        self.assertEqual(backends[0].decisions, backends[1].decisions)
        self.assertEqual(backends[1].decisions, backends[2].decisions)
        self.assertEqual(results[0]['actions'], results[1]['actions'])
        self.assertEqual(results[1]['actions'], results[2]['actions'])
        data = json.loads(backends[1].decisions[0]['prompt'][1]['content'])
        self.assertNotIn('diagnostic_target_action', data)
        self.assertNotIn('diagnostic_target_scope', data)

    def test_returned_diagnostic_target_is_not_an_alias_of_frozen_case(self):
        case = deepcopy(self.history_cases[0])
        selected = case['original_record']['selections'][case['agent']]
        case['diagnostic_target_action'] = deepcopy(selected['action'])
        before = deepcopy(case)
        result = experiment.evaluate_history(FakeBackend([selected['id']]), case, 'original')
        result['diagnostic_target_action']['kind'] = 'corrupted'
        self.assertEqual(case, before)

    def test_unknown_history_arm_is_rejected_without_backend_call(self):
        backend = FakeBackend([])
        with self.assertRaises(ValueError):
            experiment.evaluate_history(backend, deepcopy(self.history_cases[0]), 'rewrite_history')
        self.assertEqual(backend.calls, 0)


if __name__ == '__main__':
    unittest.main()
