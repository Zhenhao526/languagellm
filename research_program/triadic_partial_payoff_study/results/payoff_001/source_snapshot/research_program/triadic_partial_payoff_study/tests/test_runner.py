"""Pure runner tests: static tasks, stub rollout, and a two-step mock loop.

No real network creation, forward/backward call, parameter load, or training.
"""
import builtins
from contextlib import ExitStack
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_program.triadic_partial_payoff_study import runner as r
from research_program.triadic_action_dependency_study import environment as env


def tiny_spec():
    return dict(needs=[(0, 1, 6), (0, 4, 12)], layouts=[(0, 1, 2, 3)],
                private_sites=[(1, 2, 3), (2, 3, 1)], world_count=4,
                monitor_indices=[0, 1])


class RunnerTests(unittest.TestCase):
    def test_public_layout_mask_equals_official_observations(self):
        spec = tiny_spec()
        spec['layouts'] += [(3, 1, 0, 2)]
        spec['world_count'] *= 2
        arrays = r.make_arrays(spec)
        official = r.dataset.make_arrays(spec)
        self.assertNotIn('states', arrays)
        for key in ('packed_states', 'rewards', 'joint_actions', 'x_FI', 'x_PL'):
            np.testing.assert_array_equal(arrays[key], official[key])
        self.assertFalse(np.shares_memory(arrays['x_FI'], arrays['x_PL']))
        for actor in range(3):
            np.testing.assert_array_equal(arrays['x_PL'][:, actor, actor*7:(actor+1)*7],
                                          arrays['x_FI'][:, actor, actor*7:(actor+1)*7])
            for other in range(3):
                if other != actor:
                    np.testing.assert_array_equal(arrays['x_PL'][:, actor, other*7:(other+1)*7], 0)
        np.testing.assert_array_equal(arrays['x_PL'][:, :, 21:53], arrays['x_FI'][:, :, 21:53])
        np.testing.assert_array_equal(arrays['x_PL'][:, :, 53], 0)

    def test_evaluate_separates_native_utility_roles_and_full_success(self):
        arrays = r.make_arrays(tiny_spec())
        greedy = np.asarray([[2, 0, 1], [1, 1, 0], [0, 0, 0], [6, 0, 5]])
        logits = np.full((4, 3, 17), -0.7)
        logits[np.arange(4)[:, None], np.arange(3), greedy] = 2.3
        messages = (np.arange(4*2*3*4).reshape(4, 2, 3, 4) % 8).astype(np.int8)
        def stub(networks, x, live):
            self.assertIsNone(networks)
            np.testing.assert_array_equal(x, arrays['x_PL'])
            self.assertTrue(live)
            return dict(action_logits=logits.copy(), messages=messages.copy())
        shifted = logits - logits.max(axis=2, keepdims=True)
        probs = np.exp(shifted); probs /= probs.sum(axis=2, keepdims=True)
        # Enumerate all4913 outcomes, including zero-payoff nonstructural actions.
        joint_actions = list(product(range(17), repeat=3))
        native = np.zeros(4); full = np.zeros(4); physical = np.zeros(4)
        partial = np.zeros(4)
        menus = [env.all_actions(a) for a in env.AGENTS]
        for world, row in enumerate(arrays['packed_states']):
            state = env.State(tuple(row[:3]), tuple(row[3:7]), tuple(row[7:10]))
            for choices in joint_actions:
                mass = np.prod([probs[world, a, choices[a]] for a in range(3)])
                truth = env.settle(state, {a: menus[i][choices[i]] for i, a in enumerate(env.AGENTS)})
                native[world] += mass * truth['reward']
                full[world] += mass * truth['full_success']
                partial[world] += mass * (truth['reward'] == .5)
                physical[world] += mass * any(v['executed'] for v in truth['individual_feedback'].values())
        with tempfile.TemporaryDirectory() as td, patch.object(r.core, 'rollout', side_effect=stub) as forward:
            records, data = {}, {}
            for alpha in (.5, .1):
                path = Path(td)/f'a{alpha}.npz'
                records[alpha] = r.evaluate(None, arrays, 'PL', True, alpha, np.arange(4), path)
                with np.load(path) as z:
                    data[alpha] = {k:z[k].copy() for k in z.files}
                got = data[alpha]
                np.testing.assert_array_equal(got['action_indices'], greedy)
                np.testing.assert_array_equal(got['messages'], messages)
                np.testing.assert_array_equal(got['states'], arrays['packed_states'])
                np.testing.assert_array_equal(got['state_indices'], np.arange(4))
                np.testing.assert_array_equal(got['greedy_reward'], [1, .5, 0, .5])
                np.testing.assert_array_equal(got['greedy_utility'], [1, alpha, 0, alpha])
                np.testing.assert_allclose(got['action_probabilities'], probs, atol=1e-14, rtol=0)
                for field, expected in (
                    ('conditional_exact_expected_reward', native),
                    ('conditional_exact_full_success_probability', full),
                    ('conditional_exact_execution_probability', physical),
                    ('conditional_exact_expected_utility', full+alpha*partial),
                    ('conditional_full_posterior_mass', full/(full+alpha*partial))):
                    np.testing.assert_allclose(got[field], expected, atol=2e-14, rtol=0)
                rec = records[alpha]
                self.assertEqual(rec['reward_mean'], .5)
                self.assertAlmostEqual(rec['utility_mean'], (1+2*alpha)/4)
                self.assertEqual(rec['full_success_rate'], .25)
                self.assertEqual(rec['role_success_rate'], .5)
                self.assertEqual(rec['physical_execution_rate'], .75)
                self.assertEqual(sum(x['worlds'] for x in rec['raw_joint_role_counts']), 4)
                self.assertEqual(sorted(x['worlds'] for x in rec['raw_joint_role_counts']), [1, 1, 2])
                self.assertEqual({tuple(x['partner_indices']):x['worlds'] for x in rec['raw_joint_role_counts']},
                                 {(2,-1,0):2, (1,0,-1):1, (-1,-1,-1):1})
                self.assertEqual(sum(x['worlds'] for x in rec['raw_joint_action_counts']), 4)
                self.assertEqual({tuple(x['action_indices']):x['worlds'] for x in rec['raw_joint_action_counts']},
                                 {tuple(row):1 for row in greedy})
                self.assertEqual(rec['data_sha256'], r.sha(path))
                self.assertEqual(got['action_indices'].dtype, np.int16)
                self.assertEqual(got['messages'].dtype, np.int8)
            self.assertEqual(forward.call_count, 2)
            for key in data[.5]:
                if key not in ('greedy_utility', 'conditional_exact_expected_utility', 'conditional_full_posterior_mass'):
                    np.testing.assert_array_equal(data[.5][key], data[.1][key])
            with self.assertRaises(AssertionError):
                r.evaluate(None, arrays, 'PL', True, .1, np.arange(4), Path(td)/'a0.1.npz')
            self.assertEqual(forward.call_count, 2)

    def test_evaluate_batch_tail_and_rejects_invalid_indices(self):
        source = r.make_arrays(tiny_spec())
        arrays = {key:np.repeat(source[key][:1], 1025, axis=0) for key in ('packed_states','rewards','x_FI','x_PL')}
        sizes = []
        def stub(networks, x, live):
            sizes.append(len(x))
            return dict(action_logits=np.zeros((len(x),3,17)), messages=np.zeros((len(x),2,3,4),np.int8))
        with tempfile.TemporaryDirectory() as td, patch.object(r.core, 'rollout', side_effect=stub):
            r.evaluate(None, arrays, 'PL', False, .1, np.arange(1025), Path(td)/'tail.npz')
            self.assertEqual(sizes, [1024, 1])
            for case, ix in enumerate(([], [0, 0], [-1], [1025], [[0, 1]])):
                with self.subTest(indices=ix), self.assertRaises((AssertionError, ValueError, IndexError)):
                    r.evaluate(None, arrays, 'PL', False, .1, ix, Path(td)/f'bad_{case}.npz')
            self.assertEqual(sizes, [1024, 1])

    def test_silent_closed_alias_and_live_closed_forward(self):
        record = dict(path='natural', worlds=4, reused_natural=False, nested={'x':1})
        with tempfile.TemporaryDirectory() as td, patch.object(r, 'evaluate', return_value=record) as evaluate:
            silent = r.evaluate_modes(None, None, 'PL_silent', .1, [0], Path(td)/'silent')
            self.assertEqual(evaluate.call_count, 1)
            self.assertTrue(silent['closed']['reused_natural'])
            self.assertEqual(silent['closed']['path'], silent['natural']['path'])
            silent['closed']['nested']['x'] = 2
            self.assertEqual(silent['natural']['nested']['x'], 1)
            evaluate.reset_mock()
            live = r.evaluate_modes(None, None, 'PL_live', .1, [0], Path(td)/'live')
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual([call.args[3] for call in evaluate.call_args_list], [True, False])
            self.assertFalse(live['closed']['reused_natural'])
            self.assertTrue(str(evaluate.call_args_list[1].args[-1]).endswith('_closed.npz'))

    def test_primary_uses_all_four_paired_complete_world_cells(self):
        deltas = [.1, -.2, .3, -.1]
        runs = []
        for i, seed in enumerate(r.SEEDS):
            for payoff, _, condition in r.CELLS:
                role = {'a50_PL_silent':.2, 'a50_PL_live':.4, 'a10_PL_silent':.4,
                        'a10_PL_live':.6+deltas[i]}.get(payoff+'_'+condition, .999)
                stats = dict(role_success_rate=role, full_success_rate=role/2,
                             reward_mean=.5+role/3, utility_mean=.4)
                runs.append(dict(seed=seed,payoff=payoff,condition=condition,
                    final={'new_needs_and_layouts':dict(natural=stats, closed={'role_success_rate':-99}),
                           'train':dict(natural={'role_success_rate':-99})}))
        original = deepcopy(runs)
        got = r.primary(runs)
        self.assertEqual(runs, original)
        self.assertEqual(got['metric'], 'role_success_rate')
        self.assertEqual(got['partition'], 'new_needs_and_layouts')
        self.assertAlmostEqual(got['mean_difference'], np.mean(deltas))
        for row, delta in zip(got['paired_seeds'], deltas):
            self.assertAlmostEqual(row['contrasts']['role_success_rate'], delta)
            self.assertAlmostEqual(row['contrasts']['full_success_rate'], delta/2)
            self.assertEqual(len(row['cells']), 6)
        self.assertEqual(r.primary(list(reversed(runs))), got)
        with self.assertRaises(AssertionError):r.primary(runs[:-1])
        with self.assertRaises(AssertionError):r.primary(runs[:-1]+[runs[0]])

    def test_prepared_budget_actual_vs_alias(self):
        with patch.object(r.core, 'make_networks', side_effect=AssertionError('no model')), \
             patch.object(r.core, 'rollout', side_effect=AssertionError('no forward')):
            prepared = r.prepared()
        budget = prepared['budget']
        actual_per_step = sum(2 if c.endswith('_live') else 1 for _ in r.SEEDS for _, _, c in r.CELLS)
        alias_per_step = sum(not c.endswith('_live') for _ in r.SEEDS for _, _, c in r.CELLS)
        self.assertEqual(actual_per_step, 32)
        self.assertEqual(budget['actual_monitor_files'], actual_per_step*6*4)
        self.assertEqual(budget['actual_monitor_files'], 768)
        self.assertEqual(budget['actual_final_files'], actual_per_step*4)
        self.assertEqual(budget['actual_final_files'], 128)
        self.assertEqual(budget['silent_closed_aliases'], alias_per_step*7*4)
        self.assertEqual(budget['silent_closed_aliases'], 448)
        self.assertEqual(budget['checkpoints'], 24*6)
        self.assertEqual(budget['training_world_samples'], 24*6000*256)
        self.assertEqual(budget['message_trajectories'], 24*6000*256*2)
        self.assertEqual(budget['categorical_symbol_samples'], 24*6000*256*2*24)
        self.assertEqual(budget['evaluation_network_samples'], 9*(budget['actual_monitor_worlds']+budget['actual_final_worlds']))

    def test_prepare_refuses_overwrite_and_binds_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root/'fixture.txt'; source.write_text('static fixture')
            static = dict(partitions={},budget={'runs':24})
            with patch.object(r,'ROOT',root), \
                 patch.object(r,'sources',side_effect=lambda:{str(source):r.sha(source)}), \
                 patch.object(r,'prepared',return_value=static), \
                 patch.object(r.core,'make_networks',side_effect=AssertionError('no model')):
                out = root/'run'
                self.assertEqual(r.prepare(out)['status'],'prepared_without_training')
                r.verify(out)
                with self.assertRaises(AssertionError):r.prepare(out)
                (out/'source_snapshot/fixture.txt').write_text('changed')
                with self.assertRaises(AssertionError):r.verify(out)

    def test_existing_execution_refuses_before_spawn(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'execution').mkdir()
            with patch.object(r,'verify',return_value=({},{})), \
                 patch.object(r.multiprocessing,'get_context',side_effect=AssertionError('no spawn')):
                with self.assertRaises(FileExistsError):r.execute(out)

    def test_two_step_mock_preserves_initial_and_exogenous_stream_pairing(self):
        arrays0 = r.make_arrays(tiny_spec())
        arrays = {part:deepcopy(arrays0) for part in r.PARTS}
        static = {'partitions':{part:tiny_spec() for part in r.PARTS}}
        captured=[]
        def mock_train(nets,x,native,live,uniforms,update,alpha):
            captured.append(dict(x=x.copy(),native=native.copy(),uniforms=uniforms.copy(),
                                 live=live,update=update,alpha=alpha))
            return [],dict(entropy_coefficient=r.core.base.entropy_coefficient(update))
        def mock_checkpoint(path,nets,opt,update,wr,mr):
            path.write_text(json.dumps({'step':update,'seed':nets['seed']}))
            return r.sha(path)
        def short_range(*args):
            return (1,6000) if args==(1,6001) else builtins.range(*args)
        with tempfile.TemporaryDirectory() as td, ExitStack() as stack:
            stack.enter_context(patch.object(r,'range',side_effect=short_range,create=True))
            stack.enter_context(patch.object(r.core,'make_networks',side_effect=lambda seed:{'seed':seed}))
            stack.enter_context(patch.object(r.core,'parameter_hash',side_effect=lambda nets:str(nets['seed'])))
            stack.enter_context(patch.object(r.core.base,'make_adam',return_value={}))
            stack.enter_context(patch.object(r.core.base,'adam_step',return_value=(0.,1.)))
            stack.enter_context(patch.object(r.core,'save_checkpoint',side_effect=mock_checkpoint))
            stack.enter_context(patch.object(r,'evaluate_modes',return_value={'natural':{'stub':True},'closed':{'stub':True}}))
            stack.enter_context(patch.object(r.utility,'training_gradients',side_effect=mock_train))
            stack.enter_context(patch.object(r.core,'rollout',side_effect=AssertionError('no forward')))
            execution=Path(td); results=[]; logs=[]
            for payoff,alpha,condition in r.CELLS:
                result=r.train_run(r.SEEDS[0],payoff,alpha,condition,static,arrays,execution)
                results.append(result)
                log=(execution/r.name(r.SEEDS[0],payoff,condition)/'training.jsonl').read_text().splitlines()
                logs.append([json.loads(line) for line in log])
            self.assertEqual(len(captured),12)
            self.assertEqual(len({x['initial_parameter_sha256'] for x in results}),1)
            for step in range(2):
                for key in ('world_uniforms_sha256','batch_indices_sha256','batch_states_sha256',
                            'sample_uniforms_sha256','entropy_coefficient'):
                    self.assertEqual(len({lines[step][key] for lines in logs}),1,key)
                for cell in range(1,6):
                    np.testing.assert_array_equal(captured[step]['native'],captured[2*cell+step]['native'])
                    np.testing.assert_array_equal(captured[step]['uniforms'],captured[2*cell+step]['uniforms'])
            for condition in range(3):
                for step in range(2):
                    np.testing.assert_array_equal(captured[2*condition+step]['x'],captured[6+2*condition+step]['x'])
            for part in r.PARTS:
                for key in arrays0:np.testing.assert_array_equal(arrays[part][key],arrays0[key])
            with self.assertRaises(FileExistsError):
                r.train_run(r.SEEDS[0],*r.CELLS[0],static,arrays,execution)


if __name__=='__main__':unittest.main()
