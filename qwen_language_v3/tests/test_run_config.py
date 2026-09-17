"""Configuration and history-policy checks; no model or world rollout."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from qwen_language_v3 import run
from qwen_language_v3.environment import AGENTS


class RunConfigTests(unittest.TestCase):
    def test_explicit_seeds_and_variants_preserve_planned_order(self):
        args, conditions, groups, scenarios = run.parse_config([
            '--phase', 'pilot', '--conditions', 'immediate', '--groups', '17',
            '--episodes', '3', '--variants', '0,22,0',
            '--seeds', '93101,93123,93201', '--history-mode', 'continuous',
        ])
        self.assertEqual(args.history_mode, 'continuous')
        self.assertEqual(conditions, ['immediate'])
        self.assertEqual(groups, [17])
        self.assertEqual(scenarios, [
            {'episode': 1, 'seed': 93101, 'variant': 0},
            {'episode': 2, 'seed': 93123, 'variant': 22},
            {'episode': 3, 'seed': 93201, 'variant': 0},
        ])

    def test_omitted_seeds_keep_documented_deterministic_default(self):
        _, _, _, scenarios = run.parse_config([
            '--episodes', '3', '--variants', '0,22,0',
        ])
        self.assertEqual([row['seed'] for row in scenarios], [6101, 7135, 6101])

    def test_history_defaults_follow_phase_and_explicit_mode_is_respected(self):
        cases = [
            ([], 'independent_episodes'),
            (['--phase', 'calibration'], 'independent_episodes'),
            (['--phase', 'pilot'], 'continuous'),
            (['--phase', 'pilot', '--history-mode', 'independent_episodes'], 'independent_episodes'),
            (['--phase', 'calibration', '--history-mode', 'continuous'], 'continuous'),
        ]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                args, _, _, _ = run.parse_config(arguments)
                self.assertEqual(args.history_mode, expected)

    def test_independent_episode_replaces_histories_without_mutating_old_snapshot(self):
        other = {a: [{'owner': a, 'group': 18}] for a in AGENTS}
        old = {a: [{'owner': a, 'episode': 1}] for a in AGENTS}
        memories = {('natural', 17): old, ('natural', 18): other}
        before = deepcopy(old)
        fresh = run.episode_history(memories, 'natural', 17, 'independent_episodes')
        self.assertIs(fresh, memories['natural', 17])
        self.assertEqual(fresh, {a: [] for a in AGENTS})
        self.assertEqual(old, before)
        self.assertIs(memories['natural', 18], other)
        fresh['A'].append({'owner': 'A', 'episode': 2})
        self.assertEqual(fresh['B'], [])
        self.assertEqual(fresh['C'], [])
        second = run.episode_history(memories, 'natural', 17, 'independent_episodes')
        self.assertEqual(second, {a: [] for a in AGENTS})
        self.assertIsNot(second, fresh)

    def test_continuous_mode_carries_only_requested_group_condition_and_owner(self):
        memories = {
            (condition, group): {a: [] for a in AGENTS}
            for condition in ('immediate', 'delayed') for group in (17, 18)
        }
        first = run.episode_history(memories, 'immediate', 17, 'continuous')
        first['A'].append({'only_owner_memory': 'A / immediate / 17'})
        second = run.episode_history(memories, 'immediate', 17, 'continuous')
        self.assertIs(second, first)
        self.assertEqual(second['A'], [{'only_owner_memory': 'A / immediate / 17'}])
        self.assertEqual(second['B'], [])
        self.assertEqual(second['C'], [])
        self.assertEqual(memories['delayed', 17], {a: [] for a in AGENTS})
        self.assertEqual(memories['immediate', 18], {a: [] for a in AGENTS})

    def test_invalid_history_mode_does_not_modify_memories(self):
        memories = {('immediate', 17): {a: [] for a in AGENTS}}
        before = deepcopy(memories)
        with self.assertRaises(ValueError):
            run.episode_history(memories, 'immediate', 17, 'summarized_notes')
        self.assertEqual(memories, before)

    def test_invalid_parameters_fail_before_directory_or_backend_creation(self):
        invalid = [
            ['--episodes', '0'], ['--episodes', '-1'],
            ['--conditions', 'unknown'], ['--conditions', ''],
            ['--conditions', 'immediate,immediate'],
            ['--groups', '17,17'], ['--groups', ''], ['--groups', 'abc'],
            ['--episodes', '2', '--variants', '0'],
            ['--variants', '-1'], ['--variants', 'abc'], ['--variants', '0,'], ['--variants', ''],
            ['--episodes', '2', '--seeds', '93101'],
            ['--seeds', 'abc'], ['--seeds', '93101,'], ['--seeds', ''],
            ['--history-mode', 'private_notes'],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / 'not-created' / 'run'
            for arguments in invalid:
                with self.subTest(arguments=arguments), \
                     patch.object(sys, 'argv', ['run', '--run-dir', str(run_dir), *arguments]), \
                     patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir reached')) as mkdir, \
                     patch.object(run, 'Backend', side_effect=AssertionError('backend reached')) as backend, \
                     redirect_stderr(io.StringIO()):
                    with self.assertRaises((SystemExit, ValueError)):
                        run.main()
                    mkdir.assert_not_called()
                    backend.assert_not_called()
                    self.assertFalse(run_dir.exists())

    def test_explicit_empty_lists_are_not_silently_replaced_by_defaults(self):
        for option in ('--variants', '--seeds'):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises((SystemExit, ValueError)):
                    run.parse_config([option, ''])

    def _run_without_model(self, *, conditions, groups='17', history_mode, episodes, variants, seeds):
        calls = []

        def fake_episode(backend, run_dir, **kwargs):
            histories = kwargs['histories']
            calls.append({
                **{key: value for key, value in kwargs.items() if key != 'histories'},
                'before': deepcopy(histories),
            })
            for agent in AGENTS:
                histories[agent].append({
                    'owner': agent, 'condition': kwargs['condition'],
                    'group': kwargs['group'], 'episode': kwargs['episode'],
                })

        class NoModelBackend:
            def stats(self):
                return {'calls': 0, 'test_backend': True}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'test_source'
            root.mkdir()
            run_dir = Path(temporary) / 'run'
            argv = [
                'run', '--phase', 'pilot' if conditions == 'immediate' else 'calibration',
                '--conditions', conditions, '--groups', groups,
                '--episodes', str(episodes), '--variants', variants, '--seeds', seeds,
                '--history-mode', history_mode, '--run-dir', str(run_dir),
            ]
            with patch.object(sys, 'argv', argv), \
                 patch.object(run, 'ROOT', root), \
                 patch.object(run, 'Backend', return_value=NoModelBackend()) as backend, \
                 patch.object(run, 'run_episode', side_effect=fake_episode), \
                 patch('qwen_language_v3.analyze.write_report'), \
                 patch('qwen_language_v3.inspect_run.inspect'), \
                 redirect_stdout(io.StringIO()):
                run.main()
            backend.assert_called_once()
            manifest = json.loads((run_dir / 'manifest.json').read_text())
            status = json.loads((run_dir / 'status.json').read_text())
            self.assertEqual(status['status'], 'completed')
            self.assertEqual(manifest['history_mode'], history_mode)
            self.assertEqual((root / 'LATEST_RUN').read_text().strip(), str(run_dir.resolve()))
        return calls, manifest

    def test_main_resets_every_control_episode(self):
        calls, manifest = self._run_without_model(
            conditions='full_information,natural', history_mode='independent_episodes',
            episodes=2, variants='0,22', seeds='93101,93123',
        )
        self.assertEqual(len(calls), 4)
        for call in calls:
            self.assertEqual(call['before'], {agent: [] for agent in AGENTS})
        self.assertEqual({(call['condition'], call['episode']) for call in calls}, {
            ('full_information', 1), ('full_information', 2),
            ('natural', 1), ('natural', 2),
        })
        self.assertEqual([row['seed'] for row in manifest['scenarios']], [93101, 93123])

    def test_main_continues_three_symbolic_episodes_without_cross_owner_memory(self):
        calls, manifest = self._run_without_model(
            conditions='immediate', history_mode='continuous',
            episodes=3, variants='0,22,0', seeds='93101,93123,93201',
        )
        self.assertEqual([call['episode'] for call in calls], [1, 2, 3])
        self.assertEqual([call['seed'] for call in calls], [93101, 93123, 93201])
        self.assertEqual([call['variant'] for call in calls], [0, 22, 0])
        for call in calls:
            for agent in AGENTS:
                self.assertEqual([row['episode'] for row in call['before'][agent]], list(range(1, call['episode'])))
                self.assertTrue(all(row['owner'] == agent for row in call['before'][agent]))
        self.assertEqual(manifest['alphabet'], '@#%&*+=~')
        self.assertEqual(manifest['max_message_symbols'], 32)
        self.assertEqual(manifest['per_agent_step_symbols'], 64)
        self.assertEqual(manifest['windows'], 4)

    def test_main_continuous_groups_and_conditions_do_not_share_records(self):
        calls, _ = self._run_without_model(
            conditions='immediate,delayed', groups='17,18', history_mode='continuous',
            episodes=2, variants='0,22', seeds='93101,93123',
        )
        self.assertEqual(len(calls), 8)
        for call in calls:
            for agent in AGENTS:
                self.assertEqual(len(call['before'][agent]), call['episode'] - 1)
                for row in call['before'][agent]:
                    self.assertEqual((row['owner'], row['condition'], row['group']),
                                     (agent, call['condition'], call['group']))

    def test_step_two_probe_is_saved_before_action_in_a_three_step_episode(self):
        class ThreeStepWorld:
            def __init__(self):
                self.t = 0
                self.witness = []

            @property
            def done(self):
                return self.t == 3

            @property
            def score(self):
                return 1 if self.done else 0

            def observe(self, agent):
                return {'agent': agent, 'step': self.t, 'location': '营地',
                        'ground_items': [], 'carried_items': []}

            def action_menu(self, agent):
                return [{'id': 0, 'description': '等待', 'action': {'kind': 'wait'}}]

            def state_dict(self):
                return {'t': self.t, 'test_world': True}

            def step(self, actions):
                self.t += 1
                return {agent: {'success': True, 'result': '等待。'} for agent in AGENTS}

        class NoModelBackend:
            def decide(self, prompt, **kwargs):
                return '0' if kwargs['mode'] == 'action' else '@'

            def stats(self):
                return {'calls': 0, 'test_backend': True}

        for episode in (1, 3):
            with self.subTest(episode=episode), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary)
                histories = {agent: [{'old_owner_memory': agent}] for agent in AGENTS}
                with patch.object(run, 'World', return_value=ThreeStepWorld()), redirect_stdout(io.StringIO()):
                    result = run.run_episode(
                        NoModelBackend(), run_dir, condition='immediate', group=17,
                        episode=episode, seed=93101, variant=0, histories=histories, phase='pilot',
                    )
                self.assertEqual(result['steps'], 3)
                probe_path = run_dir / f'probe_pilot_immediate_17_{episode}_2.json'
                self.assertTrue(probe_path.exists())
                case = json.loads(probe_path.read_text())
                self.assertEqual(case['episode'], episode)
                self.assertEqual(case['step'], 2)
                self.assertEqual(case['state_before']['t'], 1)
                self.assertEqual(case['state_after']['t'], 2)
                self.assertTrue(all(message['step'] == 2 for message in case['messages']))
                for agent in AGENTS:
                    self.assertEqual(case['observations'][agent]['step'], 1)
                    history = case['private_histories_before_step'][agent]
                    self.assertEqual(history[0], {'old_owner_memory': agent})
                    self.assertEqual(len(history), 2)
                    self.assertEqual(history[1]['观察']['step'], 0)
                    self.assertEqual(history[1]['观察']['agent'], agent)
                    self.assertEqual(len(histories[agent]), 4)
                self.assertEqual(list(run_dir.glob(f'probe_pilot_*_{episode}_2.json')), [probe_path])


if __name__ == '__main__':
    unittest.main()
