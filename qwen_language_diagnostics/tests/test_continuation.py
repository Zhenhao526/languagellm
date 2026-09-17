"""No-model continuation tests using frozen outputs and real World settlement."""
from copy import deepcopy
import hashlib
import json
import unittest

from qwen_language_diagnostics import continuation as runner
from qwen_language_diagnostics.continuation_cases import build_continuation_cases
from qwen_language_v3.agents import current_status
from qwen_language_v3.backend import Backend
from qwen_language_v3.environment import AGENTS, World
from qwen_language_v3.run import inference_seed


class FakeBackend:
    """Exercise the unchanged two-pass adapter without a model or tokenizer."""

    decide = Backend.decide

    def __init__(self, *, source_calls=None, choice_plan=None):
        self.calls = 0
        self.rows = []
        self.choice_plan = choice_plan or {}
        self.source_calls = {
            (r['label']['step'], r['label']['window'], r['label']['agent'],
             r['label']['stage']): r for r in (source_calls or [])
        }

    def infer(self, messages, *, mode, label, seed=0, limit=32, choices=None,
              temperature=0):
        key = (label['step'], label['window'], label['agent'], label['stage'])
        if self.source_calls:
            source = self.source_calls[key]
            for field, actual in (('messages', messages), ('mode', mode),
                                  ('seed', seed), ('temperature', temperature)):
                if actual != source[field]:
                    raise AssertionError(f'Source replay mismatch at {key}: {field}')
            row = deepcopy(source)
            output = source['output']
        else:
            arm, step, agent = label['arm_id'], label['step'], label['agent']
            if mode == 'private_analysis':
                output = f'PRIVATE_ONLY/{arm}/{step}/{label["window"]}/{agent}'
            elif mode == 'natural_message':
                output = f'{arm}/{step}/{label["window"]}/{agent}'
            elif mode == 'action':
                data = json.loads(messages[1]['content'])
                wait = next(m['编号'] for m in data['本次可选动作'] if m['动作'] == '等待')
                output = str(self.choice_plan.get((arm, step, agent), wait))
                if output not in list(map(str, choices)):
                    raise AssertionError('Fake action is outside the real current menu')
            else:
                raise AssertionError(mode)
            row = {
                'mode': mode, 'seed': seed, 'temperature': temperature,
                'messages': deepcopy(messages), 'output': output,
                # Fake-only marker, never reported as a real template digest.
                'prompt_sha256': hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
                'fresh_cache': True, 'native_thinking': False, 'valid': True,
                'finish_reason': 'stop',
            }
        self.calls += 1
        row['call'] = self.calls
        row['label'] = deepcopy(label)
        self.rows.append(row)
        return output


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def delivery_plan(prepared, arm_id):
    """Researcher-only legal two-step fixture, never a real model instruction."""
    world = World.from_state_dict(prepared['state'])
    wood = next(oid for oid, item in world.items.items()
                if item['kind'] == '木材' and item['length'] == '长'
                and item['condition'] == '干' and not item['delivered'])
    fiber = world.inventory['B']['cargo']
    plan = {}
    for step in (9, 10):
        actions = {'B': ({'kind': 'move', 'destination': '营地'} if step == 9 else
                          {'kind': 'deliver', 'item': world.handles['B'][fiber]})}
        for actor, partner in (('A', 'C'), ('C', 'A')):
            action = {'kind': 'carry_together' if step == 9 else 'deliver_together',
                      'item': world.handles[actor][wood], 'partner': partner}
            if step == 9:
                action['destination'] = '营地'
            actions[actor] = action
        for actor in AGENTS:
            plan[(arm_id, step, actor)] = next(m['id'] for m in world.action_menu(actor)
                                              if m['action'] == actions[actor])
        feedback = world.step(actions)
        if not all(feedback[a]['action_succeeded'] for a in AGENTS):
            raise AssertionError('Two-step fixture does not execute successfully')
    if world.status != 'success' or world.t != 10:
        raise AssertionError('Two-step fixture does not complete the true goals')
    return plan


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = build_continuation_cases()

    def assert_inputs_follow_records(self, backend, runs):
        """Rebuild owner histories and visible windows independently of runner."""
        by_key = {}
        for run in runs:
            arm = run['summary']['arm_id']
            memory = deepcopy(next(x['histories'] for x in self.prepared['arms']
                                   if x['arm_id'] == arm))
            for row in run['steps']:
                self.assertEqual(row['private_histories_before_step'], memory)
                for owner in AGENTS:
                    by_key[(arm, row['step'], owner)] = (row, deepcopy(memory[owner]))
                    entry = {'观察': deepcopy(row['observations'][owner]),
                             '公开广播': deepcopy(row['messages']),
                             '自己执行的动作': deepcopy(row['actions'][owner]),
                             '自己可见结果': deepcopy(row['feedback'][owner])}
                    if World.from_state_dict(row['state_after']).done:
                        entry['任务段结束'] = {'目标完成比例': row['score'], '已用动作步': row['step']}
                    memory[owner].append(entry)
                self.assertEqual(row['history_lengths_after'], {a: len(memory[a]) for a in AGENTS})
            self.assertEqual(run['histories_after'], memory)

        for index in range(0, len(backend.rows), 2):
            analysis, formal = backend.rows[index:index + 2]
            label = analysis['label']
            row, history = by_key[(label['arm_id'], label['step'], label['agent'])]
            data = json.loads(analysis['messages'][1]['content'])
            owner, window = label['agent'], label['window']
            self.assertEqual(data['你自己的历史'], history)
            self.assertEqual(data['当前私有观察'], row['observations'][owner])
            self.assertEqual(data['当前观察的直读'], current_status(row['observations'][owner]))
            expected_broadcasts = (row['messages'] if window is None else
                                   [m for m in row['messages'] if m['window'] < window])
            self.assertEqual(data['当前已可见广播'], expected_broadcasts)
            if window is None:
                self.assertEqual(data['本次可选动作'],
                                 [{'编号': m['id'], '动作': m['description']} for m in row['menus'][owner]])
            else:
                self.assertNotIn('本次可选动作', data)
                self.assertEqual(data['当前决策'], {'广播窗口': window, '本步剩余符号额度': None})
            stage = 'action' if window is None else 'message'
            expected_seed = inference_seed(17, 1, row['step'], window or 0, owner, stage)
            self.assertEqual(analysis['seed'], expected_seed)
            self.assertEqual(formal['seed'], expected_seed)
            self.assertEqual(analysis['temperature'], 0)
            self.assertEqual(formal['temperature'], 0 if window is None else .7)
            self.assertEqual(formal['messages'][:2], analysis['messages'][:2])
            self.assertEqual(formal['messages'][2], {'role': 'assistant', 'content': analysis['output']})
            self.assertEqual(len(analysis['messages']), 3)
            self.assertEqual(len(formal['messages']), 4)
            self.assertEqual(analysis['messages'][0],
                             self.prepared['first_prompts'][label['arm_id']][owner][0])

    def test_source_seed_sequence_and_fresh_start_boundaries(self):
        prepared = self.prepared
        self.assertEqual([r['call'] for r in prepared['source_calls']], list(range(241, 331)))
        for arm, initial_length in zip(prepared['arms'], (8, 0)):
            for owner in AGENTS:
                data = json.loads(prepared['first_prompts'][arm['arm_id']][owner][1]['content'])
                self.assertEqual(len(data['你自己的历史']), initial_length)
                self.assertEqual(data['当前已可见广播'], [])
                self.assertEqual(data['当前私有观察']['step'], 8)
                self.assertEqual(data['当前私有观察']['steps_remaining'], 4)
        for call in prepared['source_calls']:
            label = call['label']
            self.assertEqual(call['seed'], inference_seed(
                17, 1, label['step'], label['window'] or 0, label['agent'],
                'action' if label['window'] is None else 'message'))

    def test_retained_history_exactly_replays_all_90_source_calls_and_three_steps(self):
        backend = FakeBackend(source_calls=self.prepared['source_calls'])
        run = runner.run_arm(backend, self.prepared, 'original_history')
        self.assertEqual(backend.calls, 90)
        self.assertEqual([r['step'] for r in run['steps']], [9, 10, 11])
        for new, old in zip(run['steps'], self.prepared['source_records']):
            for field in runner.SOURCE_FIELDS:
                self.assertEqual(new[field], old[field], f'step {new["step"]}: {field}')
        reproduction = runner.reproduction([run], self.prepared, backend.rows)
        self.assertTrue(reproduction['complete_90_call_reproduction'])
        self.assertEqual(reproduction['calls_compared'], 90)
        self.assert_inputs_follow_records(backend, [run])

    def test_two_arms_use_180_calls_even_when_first_arm_does_not_complete(self):
        backend = FakeBackend()
        callbacks = []
        runs = runner.run_cases(backend, self.prepared, on_arm=lambda r: callbacks.append(r['summary']['arm_id']))
        self.assertEqual(callbacks, ['original_history', 'reset_history'])
        self.assertEqual(backend.calls, 180)
        self.assertEqual([r['summary']['model_calls'] for r in runs], [90, 90])
        self.assertEqual([r['summary']['success'] for r in runs], [False, False])
        self.assertEqual([r['summary']['end_t'] for r in runs], [11, 11])
        self.assertEqual([r['seed'] for r in backend.rows[:90]], [r['seed'] for r in backend.rows[90:]])
        self.assert_inputs_follow_records(backend, runs)

    def test_observation_cutoff_keeps_original_deadline_and_ongoing_engine(self):
        backend = FakeBackend()
        run = runner.run_arm(backend, self.prepared, 'reset_history')
        summary = run['summary']
        self.assertEqual((summary['end_reason'], summary['engine_status']), ('observation_limit', 'running'))
        self.assertEqual((summary['start_t'], summary['end_t'], summary['new_steps']), (8, 11, 3))
        final = World.from_state_dict(run['steps'][-1]['state_after'])
        self.assertFalse(final.done)
        self.assertEqual(final.max_steps, 12)
        for row in run['steps']:
            for owner in AGENTS:
                self.assertFalse(row['feedback'][owner]['done'])
                self.assertEqual(row['observations'][owner]['steps_remaining'], 13 - row['step'])
        for memory in run['histories_after'].values():
            self.assertEqual(len(memory), 3)
            self.assertTrue(all('任务段结束' not in r for r in memory))

    def test_reset_is_only_initial_then_keeps_new_own_history_and_public_windows(self):
        backend = FakeBackend()
        run = runner.run_arm(backend, self.prepared, 'reset_history')
        self.assertEqual([r['history_lengths_after'] for r in run['steps']],
                         [{a: n for a in AGENTS} for n in (1, 2, 3)])
        self.assert_inputs_follow_records(backend, [run])
        for call in backend.rows:
            if call['mode'] != 'private_analysis':
                continue
            data = json.loads(call['messages'][1]['content'])
            self.assertNotIn('PRIVATE_ONLY', json.dumps(data, ensure_ascii=False))
            for old in data['你自己的历史']:
                self.assertTrue(all(m['text'].startswith('reset_history/') for m in old['公开广播']))
        first_window = [r for r in backend.rows if r['mode'] == 'private_analysis'
                        and r['label']['step'] == 9 and r['label']['window'] == 1]
        self.assertEqual(len(first_window), 3)
        self.assertTrue(all(json.loads(r['messages'][1]['content'])['当前已可见广播'] == [] for r in first_window))

    def test_current_actions_do_not_move_world_before_other_actors_choose(self):
        world = World.from_state_dict(self.prepared['state'])
        move = next(m['id'] for m in world.action_menu('A')
                    if m['action'] == {'kind': 'move', 'destination': '营地'})
        backend = FakeBackend(choice_plan={('reset_history', 9, 'A'): move})
        run = runner.run_arm(backend, self.prepared, 'reset_history')
        self.assert_inputs_follow_records(backend, [run])
        first = run['steps'][0]
        self.assertEqual(first['state_before']['positions']['A'], '林地')
        self.assertEqual(first['state_after']['positions']['A'], '营地')
        c_action_analysis = next(r for r in backend.rows if r['mode'] == 'private_analysis'
                                 and r['label']['step'] == 9 and r['label']['agent'] == 'C'
                                 and r['label']['window'] is None)
        obs = json.loads(c_action_analysis['messages'][1]['content'])['当前私有观察']
        self.assertIn('A', [x['agent'] for x in obs['nearby_agents']])

    def test_two_step_completion_stops_early_and_second_arm_starts_independently(self):
        before = fingerprint(self.prepared)
        backend = FakeBackend(choice_plan=delivery_plan(self.prepared, 'original_history'))
        runs = runner.run_cases(backend, self.prepared)
        first, second = runs
        self.assertEqual(backend.calls, 150)
        self.assertEqual((first['summary']['new_steps'], first['summary']['end_t'],
                          first['summary']['end_reason'], first['summary']['engine_status']),
                         (2, 10, 'environment_terminal', 'success'))
        self.assertEqual(first['summary']['score'], 1)
        self.assertEqual(first['summary']['model_calls'], 60)
        self.assertEqual(second['summary']['model_calls'], 90)
        self.assertEqual(second['steps'][0]['state_before'], self.prepared['state'])
        self.assertEqual(second['steps'][0]['private_histories_before_step'], {a: [] for a in AGENTS})
        self.assertEqual(second['summary']['score'], 0)
        for memory in first['histories_after'].values():
            self.assertEqual(len(memory), 10)
            self.assertNotIn('任务段结束', memory[-2])
            self.assertEqual(memory[-1]['任务段结束'], {'目标完成比例': 1.0, '已用动作步': 10})
        self.assertEqual(fingerprint(self.prepared), before)
        self.assert_inputs_follow_records(backend, runs)

    def test_joint_events_count_one_object_and_delivery_requires_its_own_step(self):
        backend = FakeBackend(choice_plan=delivery_plan(self.prepared, 'reset_history'))
        run = runner.run_arm(backend, self.prepared, 'reset_history')
        carry, deliver = run['steps']
        self.assertEqual(carry['outcomes']['successful_joint_object_actions'],
                         {'carry_together': 1, 'deliver_together': 0})
        self.assertEqual(carry['outcomes']['delivered_this_step'], 0)
        self.assertEqual(carry['score'], 0)
        self.assertEqual(deliver['outcomes']['successful_joint_object_actions'],
                         {'carry_together': 0, 'deliver_together': 1})
        self.assertEqual(deliver['outcomes']['delivered_this_step'], 2)
        self.assertEqual(deliver['score'], 1)

    def test_step_and_arm_callbacks_cannot_mutate_live_histories_or_later_arms(self):
        backend = FakeBackend()
        def corrupt_step(row):
            row['messages'].clear()
            row['state_after']['positions']['A'] = 'callback_only'
            row['private_histories_before_step']['A'].clear()
        def corrupt_arm(run):
            run['histories_after']['A'].append({'callback_only': True})
        runs = runner.run_cases(backend, self.prepared, on_step=corrupt_step, on_arm=corrupt_arm)
        self.assert_inputs_follow_records(backend, runs)
        self.assertTrue(all(len(row['messages']) == 12 for run in runs for row in run['steps']))
        self.assertNotIn('callback_only', json.dumps(runs, ensure_ascii=False))

    def test_a_started_backend_is_rejected_before_a_new_batch(self):
        backend = FakeBackend()
        backend.calls = 2
        with self.assertRaisesRegex(RuntimeError, 'fresh backend'):
            runner.run_cases(backend, self.prepared)
        self.assertEqual(backend.rows, [])

    def test_budget_cap_blocks_the_next_decision_before_any_inference(self):
        backend = FakeBackend()
        backend.calls = 180
        with self.assertRaisesRegex(RuntimeError, 'cap exceeded'):
            runner.run_arm(backend, self.prepared, 'reset_history')
        self.assertEqual(backend.rows, [])

    def test_incomplete_two_pass_adapter_is_rejected(self):
        class OneCallBackend:
            calls = 0
            def decide(self, *args, **kwargs):
                self.calls += 1
                return ''
        with self.assertRaisesRegex(RuntimeError, 'exactly two calls'):
            runner.run_arm(OneCallBackend(), self.prepared, 'reset_history')


if __name__ == '__main__':
    unittest.main()
