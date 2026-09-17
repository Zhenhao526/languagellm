"""Engineering tests for checkpoint readaptation; no formal training runs."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agents import ResourceAgent
import run_contact as training


def equal_states(left, right):
    return len(left) == len(right) and all(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
                                          for a, b in zip(left, right))


class ContactTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_contact_schedule_seed_and_historical_exposure(self):
        config = training.CONFIG
        self.assertEqual(config['additional_updates'], 600)
        self.assertEqual(config['training_seed_offset'], 2000)
        self.assertEqual(config['task'], 'full')
        self.assertEqual(config['action_entropy_coefficient'], 0)
        self.assertEqual(config['signalling_coefficient'], 0)
        generator = np.random.default_rng((101 + 2000) * 10000 + 301)
        expected = np.concatenate([generator.permutation(3) for _ in range(200)])
        np.testing.assert_array_equal(training.contact_schedule(101, 'fixed_to_fixed'), np.zeros(600))
        for condition in ('fixed_to_rotating', 'rotating_to_rotating'):
            actual = training.contact_schedule(101, condition)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.bincount(actual), [200, 200, 200])
        fixed = np.asarray([[0, 1200, 0, 0], [1200, 0, 0, 0], [0, 0, 0, 1200], [0, 0, 1200, 0]])
        np.testing.assert_array_equal(training.initial_edges('fixed_to_fixed'), fixed)
        np.testing.assert_array_equal(training.initial_edges('fixed_to_rotating'), fixed)
        np.testing.assert_array_equal(training.initial_edges('rotating_to_rotating'), (1 - np.eye(4)) * 400)
        self.assertEqual(training.seen_pairs(fixed), ((0, 1), (2, 3)))
        after_contact = fixed.copy()
        after_contact[0, 2] = after_contact[2, 0] = 1
        after_contact[1, 3] = after_contact[3, 1] = 1
        self.assertEqual(training.seen_pairs(after_contact), ((0, 1), (0, 2), (1, 3), (2, 3)))
        with self.assertRaises(ValueError):
            training.origin_condition('invalid')

    def test_mixed_diagnostics_exclude_duplicate_resource_actions(self):
        n = 4
        assignment = np.asarray([[2, 0], [3, 1]])
        masks = [np.asarray([True, False, False, True]), np.zeros(n, dtype=bool),
                 np.ones(n, dtype=bool), np.asarray([False, True, False, False])]
        kinds = np.zeros((n, 2, 2, 2), dtype=np.int64)
        agents, action_entropies, sender_entropies = [], [], []
        class Probe:
            def __init__(self, logits):
                self.logits = logits
            def act(self, *args):
                return self.logits
        for who, mask in enumerate(masks):
            logits = torch.zeros(n, 2)
            logits[torch.from_numpy(mask)] = torch.tensor([12., -12.])
            probability = logits.softmax(-1)
            action_entropies.append(-(probability * logits.log_softmax(-1)).sum(-1))
            sender_entropies.append(torch.linspace(.1, .4, n) + who * .1)
            agents.append(Probe(logits))
        for slot, pair in enumerate(assignment):
            for local, who in enumerate(pair):
                kinds[masks[who], slot, local, 1] = 1
        result = {'representations': [(None, None)] * 4, 'received': [torch.zeros(n, dtype=torch.int64)] * 4,
                  'act_outputs': [(None, None, h) for h in action_entropies],
                  'send_outputs': [(None, None, h) for h in sender_entropies]}
        records = training.mixed_diagnostics(agents, result, kinds, assignment)
        for who, mask in enumerate(masks):
            self.assertEqual(records[who]['n'], int(mask.sum()))
            if not mask.any():
                self.assertIsNone(records[who]['action_entropy'])
                self.assertIsNone(records[who]['maximum_action_probability'])
            else:
                self.assertLess(records[who]['action_entropy'], 1e-6)
                self.assertGreater(records[who]['maximum_action_probability'], .999999)
                self.assertAlmostEqual(records[who]['sender_entropy'], float(sender_entropies[who][mask].mean()))
        self.assertGreater(float(action_entropies[0].mean()), .3)

    def test_saved_optimizer_and_rng_are_weights_only_loadable_and_resume_exactly(self):
        torch.manual_seed(301)
        agents = [ResourceAgent() for _ in range(4)]
        optimizers = [torch.optim.Adam(a.parameters(), lr=.0003) for a in agents]
        bank = training.ImageBank()
        n = 16
        public, _ = training.private_public(n, 16)
        world = np.random.default_rng(7001)
        layout = np.random.default_rng(7002)
        policies = [np.random.default_rng(7003 + i) for i in range(4)]
        rotation = np.asarray([1, 2, 0])
        edges = np.zeros((4, 4), dtype=np.int64)
        def step(models, opts, world_rng, layout_rng, policy_rngs, index):
            assignment = training.assign_slots(int(rotation[index]), layout_rng.integers(2, size=3))
            kinds = training.sample_scenes(world_rng, 2 * n).reshape(n, 2, 2, 2)
            features, _ = bank.sample(kinds, 'train', world_rng)
            result = training.rollout(models, features, kinds, assignment, public, policy_rngs)
            training.optimize(models, opts, result, entropy_weight=0.)
            return assignment
        pair = step(agents, optimizers, world, layout, policies, 0)
        for a, b in pair:
            edges[a, b] += 1
            edges[b, a] += 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'state.pt'
            training.save_resume_state(path, agents, optimizers, 1, world, layout, policies, rotation, edges)
            saved = torch.load(path, weights_only=True)
            self.assertEqual(saved['additional_updates_completed'], 1)
            self.assertEqual(saved['torch_rng_state'].dtype, torch.uint8)
            np.testing.assert_array_equal(saved['matching_schedule'].numpy(), rotation)
            recovered = [ResourceAgent() for _ in range(4)]
            recovered_opts = [torch.optim.Adam(a.parameters(), lr=.0003) for a in recovered]
            for agent, state, optimizer, optimizer_state in zip(recovered, saved['agents'], recovered_opts, saved['optimizers']):
                agent.load_state_dict(state, strict=True)
                optimizer.load_state_dict(optimizer_state)
            def restore(state):
                rng = np.random.default_rng()
                rng.bit_generator.state = copy.deepcopy(state)
                return rng
            recovered_world, recovered_layout = restore(saved['world_rng']), restore(saved['layout_rng'])
            recovered_policies = [restore(state) for state in saved['policy_rngs']]
            step(agents, optimizers, world, layout, policies, 1)
            step(recovered, recovered_opts, recovered_world, recovered_layout, recovered_policies, 1)
            self.assertTrue(equal_states([a.state_dict() for a in agents], [a.state_dict() for a in recovered]))
            self.assertEqual(world.bit_generator.state, recovered_world.bit_generator.state)
            self.assertEqual(layout.bit_generator.state, recovered_layout.bit_generator.state)
            for a, b in zip(policies, recovered_policies):
                self.assertEqual(a.bit_generator.state, b.bit_generator.state)

    def test_real_origin_endpoint_evaluation_is_identical_without_updates(self):
        self.assertTrue(training.validate_source()['passed'])
        bank = training.ImageBank()
        for origin in ('fixed', 'rotating'):
            folder = training.SOURCE / f'{origin}_s101'
            agents = training.load_population(folder / 'checkpoint_1200.pt')
            expected = json.loads((folder / 'learning_curve.json').read_text())[-1]['evaluation']
            actual = training.checkpoint_population(agents, bank, 101 + 700000, n=512, horizon=16)
            self.assertEqual(actual, expected)

    def test_actual_training_same_origin_condition_isolation_and_saved_state(self):
        config = copy.deepcopy(training.CONFIG)
        config.update(additional_updates=3, batch_size_per_agent=32, checkpoints=[0, 1, 3],
                      protocol_probe_n=32, protocol_support_n=64, evaluation_n=32,
                      intervention_n=16, alignment_n=16)
        bank = training.ImageBank()
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(training, 'evaluate_population', return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            names = [('ff', 'fixed_to_fixed'), ('forced_ff', 'fixed_to_rotating'),
                     ('fr', 'fixed_to_rotating'), ('rr', 'rotating_to_rotating')]
            for name, condition in names:
                origin = training.origin_condition(condition)
                expected = json.loads((training.SOURCE / f'{origin}_s101/learning_curve.json').read_text())[-1]['evaluation']
                with patch.object(training, 'checkpoint_population', return_value=expected):
                    if name == 'forced_ff':
                        with patch.object(training, 'contact_schedule', return_value=np.zeros(3, dtype=np.int64)):
                            training.train(101, condition, bank, root / name, config)
                    else:
                        training.train(101, condition, bank, root / name, config)
            final = {name: torch.load(root / name / 'final_training_state.pt', weights_only=True) for name, _ in names}
            self.assertTrue(equal_states(final['ff']['agents'], final['forced_ff']['agents']))
            records = {name: [json.loads(line) for line in (root / name / 'training_metrics.jsonl').read_text().splitlines()]
                       for name, _ in names}
            for name, condition in names:
                folder = root / name
                origin = training.SOURCE / f'{training.origin_condition(condition)}_s101/checkpoint_1200.pt'
                self.assertEqual((folder / 'initial.pt').read_bytes(), origin.read_bytes())
                self.assertEqual((folder / 'checkpoint_0000.pt').read_bytes(), origin.read_bytes())
                self.assertTrue(json.loads((folder / 'zero_update_validation.json').read_text())['passed'])
                metadata = json.loads((folder / 'origin.json').read_text())
                self.assertEqual(metadata['training_seed'], 2101)
                self.assertEqual(metadata['initial_optimizer_state_entries'], [0, 0, 0, 0])
                self.assertEqual([r['world_slots_sha256'] for r in records[name]], [r['world_slots_sha256'] for r in records['ff']])
                self.assertEqual([r['layout_bits'] for r in records[name]], [r['layout_bits'] for r in records['ff']])
                for row in records[name]:
                    self.assertEqual(row['task'], 'full')
                    self.assertEqual(row['aux_weight'], 0)
                    self.assertEqual(row['action_entropy_weight'], 0)
                    self.assertEqual(row['lifetime_update'], 1200 + row['update'])
                    self.assertTrue(all('mixed_resource_diagnostics' in a for a in row['agents']))
                saved = final[name]
                checkpoint = torch.load(folder / 'checkpoint_0003.pt', weights_only=True)
                self.assertTrue(equal_states(saved['agents'], checkpoint))
                self.assertEqual(saved['additional_updates_completed'], 3)
                self.assertTrue(all(int(state['step']) == 3 for optimizer in saved['optimizers'] for state in optimizer['state'].values()))
                torch.load(folder / 'protocol_reference.pt', weights_only=True)
                curve = json.loads((folder / 'learning_curve.json').read_text())
                for record in curve:
                    cumulative = np.asarray(record['cumulative_edge_training_updates'])
                    expected_pairs = [list(pair) for pair in training.seen_pairs(cumulative)]
                    self.assertEqual(record['protocol_drift']['current_trained_pairs'], expected_pairs)
                self.assertEqual(curve[0]['protocol_drift']['current_trained_pairs'],
                                 [list(pair) for pair in (training.ALL_PAIRS if condition == 'rotating_to_rotating' else training.ORIGINAL_PAIRS)])


if __name__ == '__main__':
    unittest.main()
