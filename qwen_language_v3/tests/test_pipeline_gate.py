"""Capability admission and pipeline sequencing with temporary, no-model records."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from qwen_language_v3 import execute_locked_pipeline as pipeline
from qwen_language_v3.environment import World


PLANNED = [{'variant': 0, 'seed': 92015}, {'variant': 22, 'seed': 92038}]
SYMBOLIC = [
    {'variant': 0, 'seed': 93101}, {'variant': 22, 'seed': 93123},
    {'variant': 0, 'seed': 93204},
]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def fixture(path, condition='full_information', failed_episode=None):
    """Replay the engine witness, or wait to timeout, without an agent/model."""
    path.mkdir()
    manifest = {
        'phase': 'calibration', 'conditions': [condition], 'groups': [17],
        'episodes_per_group': 2, 'history_mode': 'independent_episodes',
        'scenarios': [{'episode': i + 1, **sc} for i, sc in enumerate(PLANNED)],
    }
    write_json(path / 'manifest.json', manifest)
    episodes, steps = [], []
    for episode, sc in enumerate(PLANNED, 1):
        world = World(sc['seed'], variant=sc['variant'])
        initial = world.state_dict()
        witness = deepcopy(world.witness)
        label = {'phase': 'calibration', 'condition': condition, 'group': 17,
                 'episode': episode, 'seed': sc['seed']}
        write_json(path / f'world_calibration_{condition}_17_{episode}.json',
                   {'initial_state': initial, 'witness': witness})
        write_json(path / f'checkpoint_calibration_{condition}_17_{episode}_before.json',
                   {a: [] for a in 'ABC'})
        while not world.done:
            step = world.t + 1
            before = world.state_dict()
            actions = ({a: {'kind': 'wait'} for a in 'ABC'} if episode == failed_episode
                       else witness[step - 1])
            feedback = world.step(actions)
            steps.append({**label, 'step': step, 'state_before': before,
                          'state_after': world.state_dict(), 'actions': actions,
                          'feedback': feedback, 'score': world.score})
        episodes.append({**label, 'variant': sc['variant'], 'steps': world.t,
                         'score': world.score, 'success': world.status == 'success'})
    write_rows(path / 'episodes.jsonl', episodes)
    write_rows(path / 'steps.jsonl', steps)
    write_json(path / 'status.json', {'status': 'completed'})
    return path


class CapabilityGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = fixture(Path(self.temporary.name) / 'run')

    def check(self):
        return pipeline.check_capability(self.path, 'full_information', PLANNED)

    def test_two_fresh_successful_scenarios_pass(self):
        result = self.check()
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['cases']), 2)
        self.assertTrue(all(row['fresh_histories_verified'] for row in result['cases']))

    def test_any_failed_scenario_blocks_gate_after_both_cases(self):
        for failed_episode in (1, 2):
            with self.subTest(failed_episode=failed_episode):
                path = fixture(Path(self.temporary.name) / f'failed-{failed_episode}',
                               failed_episode=failed_episode)
                result = pipeline.check_capability(path, 'full_information', PLANNED)
                self.assertFalse(result['passed'])
                self.assertEqual(len(result['cases']), 2)
                self.assertFalse(result['cases'][failed_episode - 1]['all_goals_delivered'])

    def test_success_claim_with_an_undelivered_physical_goal_is_rejected(self):
        rows = read_rows(self.path / 'steps.jsonl')
        terminal = next(row for row in reversed(rows) if row['episode'] == 2)
        terminal['state_after']['goals'][0]['delivered'] = 0
        write_rows(self.path / 'steps.jsonl', rows)
        with self.assertRaises(RuntimeError):
            self.check()

    def test_nonempty_initial_private_history_is_rejected(self):
        write_json(self.path / 'checkpoint_calibration_full_information_17_2_before.json',
                   {'A': [{'previous_goal': 'old'}], 'B': [], 'C': []})
        with self.assertRaises(RuntimeError):
            self.check()

    def test_missing_episode_is_rejected(self):
        write_rows(self.path / 'episodes.jsonl', read_rows(self.path / 'episodes.jsonl')[:1])
        with self.assertRaises(RuntimeError):
            self.check()

    def test_missing_physical_step_is_rejected(self):
        write_rows(self.path / 'steps.jsonl', read_rows(self.path / 'steps.jsonl')[1:])
        with self.assertRaises(RuntimeError):
            self.check()

    def test_unfinished_or_failed_run_status_is_rejected(self):
        for status in ('running', 'failed'):
            with self.subTest(status=status):
                write_json(self.path / 'status.json', {'status': status})
                with self.assertRaises(RuntimeError):
                    self.check()

    def test_summary_score_or_success_disagreement_is_rejected(self):
        original = read_rows(self.path / 'episodes.jsonl')
        for field, value in [('score', .5), ('success', False)]:
            with self.subTest(field=field):
                rows = deepcopy(original)
                rows[0][field] = value
                write_rows(self.path / 'episodes.jsonl', rows)
                with self.assertRaises(RuntimeError):
                    self.check()

    def test_nonfinite_summary_score_is_rejected(self):
        rows = read_rows(self.path / 'episodes.jsonl')
        rows[0]['score'] = float('nan')
        write_rows(self.path / 'episodes.jsonl', rows)
        with self.assertRaises((RuntimeError, ValueError)):
            self.check()

    def test_manifest_scenario_mismatch_is_rejected(self):
        manifest = json.loads((self.path / 'manifest.json').read_text())
        manifest['scenarios'][0]['seed'] += 1
        write_json(self.path / 'manifest.json', manifest)
        with self.assertRaises(RuntimeError):
            self.check()

    def test_trajectory_seed_mismatch_is_rejected(self):
        rows = read_rows(self.path / 'steps.jsonl')
        rows[0]['seed'] += 1
        write_rows(self.path / 'steps.jsonl', rows)
        with self.assertRaises(RuntimeError):
            self.check()

    def test_terminal_world_identity_mismatch_is_rejected(self):
        rows = read_rows(self.path / 'steps.jsonl')
        rows[-1]['state_after']['variant'] = 999
        write_rows(self.path / 'steps.jsonl', rows)
        with self.assertRaises(RuntimeError):
            self.check()

    def test_nonterminal_feedback_cannot_support_a_completed_success(self):
        rows = read_rows(self.path / 'steps.jsonl')
        for feedback in rows[-1]['feedback'].values():
            feedback['done'] = False
            feedback['status'] = 'running'
        write_rows(self.path / 'steps.jsonl', rows)
        with self.assertRaises(RuntimeError):
            self.check()

    def test_missing_physical_goal_cannot_shrink_the_success_denominator(self):
        rows = read_rows(self.path / 'steps.jsonl')
        rows[-1]['state_after']['goals'].pop(0)
        write_rows(self.path / 'steps.jsonl', rows)
        with self.assertRaises(RuntimeError):
            self.check()

    def test_goal_counters_without_delivered_physical_items_are_rejected(self):
        rows = read_rows(self.path / 'steps.jsonl')
        for item in rows[-1]['state_after']['items'].values():
            item['delivered'] = False
        write_rows(self.path / 'steps.jsonl', rows)
        with self.assertRaises(RuntimeError):
            self.check()

    def test_nonfinite_terminal_step_score_is_rejected(self):
        rows = read_rows(self.path / 'steps.jsonl')
        rows[-1]['score'] = float('nan')
        write_rows(self.path / 'steps.jsonl', rows)
        with self.assertRaises((RuntimeError, ValueError)):
            self.check()

    def test_timeout_feedback_disagrees_with_full_success(self):
        rows = read_rows(self.path / 'steps.jsonl')
        for feedback in rows[-1]['feedback'].values():
            feedback['status'] = 'timeout'
        write_rows(self.path / 'steps.jsonl', rows)
        with self.assertRaises(RuntimeError):
            self.check()


class PipelineSequencingTests(unittest.TestCase):
    def exercise(self, failure_condition=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'source'
            root.mkdir()
            batch = Path(temporary) / 'batch'
            write_json(root / 'experiment_locked.json', {
                'capability_scenarios': PLANNED, 'symbolic_scenarios': SYMBOLIC,
                'frozen_probes': [{'episode': 1, 'step': 2}, {'episode': 3, 'step': 2}],
            })
            write_json(root / 'core_source_frozen.json', {})
            commands = []

            def no_process(command, **kwargs):
                commands.append(command)
                module = command[2]
                if module == 'qwen_language_v3.run':
                    condition = command[command.index('--conditions') + 1]
                    path = Path(command[command.index('--run-dir') + 1])
                    if condition == 'immediate':
                        path.mkdir()
                        write_json(path / 'manifest.json', {'phase': 'pilot'})
                    else:
                        fixture(path, condition, 1 if condition == failure_condition else None)
                elif module not in ('qwen_language_v3.report_suite',
                                    'qwen_language_v3.replay_snapshot', 'qwen_language_v3.probe'):
                    self.fail(f'Unexpected command: {command}')

            with patch.object(pipeline, 'ROOT', root), \
                 patch.object(pipeline, 'verify_frozen'), \
                 patch.object(sys, 'argv', ['pipeline', '--batch-dir', str(batch)]), \
                 patch.object(pipeline.subprocess, 'run', side_effect=no_process), \
                 redirect_stdout(io.StringIO()):
                pipeline.main()
            status = json.loads((batch / 'pipeline_status.json').read_text())
            gates = json.loads((batch / 'capability_gates.json').read_text())
            return commands, status, gates

    def test_full_information_failure_finishes_two_cases_and_does_not_unlock_natural(self):
        commands, status, gates = self.exercise('full_information')
        runs = [c for c in commands if c[2] == 'qwen_language_v3.run']
        self.assertEqual([c[c.index('--conditions') + 1] for c in runs], ['full_information'])
        self.assertEqual(len(gates[0]['cases']), 2)
        self.assertFalse(status['symbolic_started'])
        self.assertEqual(status['outcome'], 'full_information_capability_not_passed')

    def test_natural_failure_does_not_unlock_symbols_or_probes(self):
        commands, status, gates = self.exercise('natural')
        runs = [c for c in commands if c[2] == 'qwen_language_v3.run']
        self.assertEqual([c[c.index('--conditions') + 1] for c in runs], ['full_information', 'natural'])
        self.assertEqual([len(gate['cases']) for gate in gates], [2, 2])
        self.assertFalse(any(c[2] == 'qwen_language_v3.probe' for c in commands))
        self.assertFalse(status['symbolic_started'])

    def test_all_four_successes_unlock_only_the_locked_symbolic_plan(self):
        commands, status, gates = self.exercise()
        runs = [c for c in commands if c[2] == 'qwen_language_v3.run']
        self.assertEqual([c[c.index('--conditions') + 1] for c in runs],
                         ['full_information', 'natural', 'immediate'])
        for command in runs[:2]:
            self.assertEqual(command[command.index('--history-mode') + 1], 'independent_episodes')
            self.assertEqual(command[command.index('--seeds') + 1], '92015,92038')
        pilot = runs[2]
        self.assertEqual(pilot[pilot.index('--episodes') + 1], '3')
        self.assertEqual(pilot[pilot.index('--variants') + 1], '0,22,0')
        self.assertEqual(pilot[pilot.index('--seeds') + 1], '93101,93123,93204')
        self.assertEqual(pilot[pilot.index('--history-mode') + 1], 'continuous')
        probes = [c for c in commands if c[2] == 'qwen_language_v3.probe']
        self.assertEqual([(c[c.index('--episode') + 1], c[c.index('--step') + 1]) for c in probes],
                         [('1', '2'), ('3', '2')])
        self.assertTrue(status['symbolic_started'])
        self.assertTrue(all(gate['passed'] for gate in gates))

    def test_existing_batch_is_not_overwritten_and_starts_no_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary) / 'existing'
            batch.mkdir()
            sentinel = batch / 'original.txt'
            sentinel.write_text('keep original result')
            with patch.object(pipeline, 'verify_frozen'), \
                 patch.object(pipeline, 'ROOT', Path(temporary)), \
                 patch.object(sys, 'argv', ['pipeline', '--batch-dir', str(batch)]), \
                 patch.object(pipeline.subprocess, 'run') as subprocess_run:
                write_json(Path(temporary) / 'experiment_locked.json', {})
                with self.assertRaises(FileExistsError):
                    pipeline.main()
            subprocess_run.assert_not_called()
            self.assertEqual(sentinel.read_text(), 'keep original result')


if __name__ == '__main__':
    unittest.main()
