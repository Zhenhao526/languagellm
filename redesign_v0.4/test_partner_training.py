"""Engineering probes of partner routing, independent learning and task bounds.

The deterministic protocols below are not supplied to any learning experiment.
"""
import contextlib
import copy
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agents import ResourceAgent
from resource_env import FEASIBLE_SCENES
import run_partners as training


class TaggedProbe(ResourceAgent):
    """Send an agent-specific token and pick the first option, for wiring only."""
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def observe(self, own_features, public):
        self.last_private_features = own_features.detach().clone()
        return super().observe(own_features, public)

    def send(self, local):
        logits = torch.full((len(local), 5), -100.)
        logits[:, self.tag] = 0.
        return logits

    def act(self, options, local, received):
        self.last_received = received.detach().clone()
        return torch.tensor([0., -100.]).expand(len(local), 2)


class CountingAdam(torch.optim.Adam):
    def __init__(self, parameters):
        super().__init__(parameters, lr=.0003)
        self.step_count = 0

    def step(self, *args, **kwargs):
        self.step_count += 1
        return super().step(*args, **kwargs)


class PartnerTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_routing_private_inputs_pair_rewards_and_independent_rng_consumption(self):
        n = 16
        agents = [TaggedProbe(i) for i in range(4)]
        assignment = np.asarray([[3, 1], [0, 2]])
        features = torch.arange(4.).reshape(1, 2, 2, 1, 1).expand(n, 2, 2, 2, 1024).clone()
        # Pair slot 0 necessarily succeeds; slot 1 can succeed but both probes
        # choose food. Scoring the camps together would conceal this failure.
        kinds = np.tile(np.asarray([[[0, 0], [1, 1]], [[0, 1], [0, 1]]]), (n, 1, 1, 1))
        public, _ = training.private_public(n, 16)
        rngs = [np.random.default_rng(71 + i) for i in range(4)]
        expected = [np.random.default_rng(71 + i) for i in range(4)]
        for rng in expected:
            rng.random((n, 1))
            rng.random((n, 1))
        result = training.rollout(agents, features, kinds, assignment, public, rngs)
        np.testing.assert_array_equal(result['partners'], [2, 3, 0, 1])
        for slot, pair in enumerate(assignment):
            for local_slot, who in enumerate(pair):
                torch.testing.assert_close(agents[who].last_private_features, features[:, slot, local_slot])
                torch.testing.assert_close(agents[who].last_received, torch.full((n,), int(result['partners'][who])))
                self.assertEqual(result['sent'][who].dtype, torch.int64)
                self.assertFalse(result['sent'][who].requires_grad)
                self.assertEqual(rngs[who].bit_generator.state, expected[who].bit_generator.state)
        for who in (1, 3):
            torch.testing.assert_close(result['targets'][who], torch.zeros(n))
        for who in (0, 2):
            torch.testing.assert_close(result['targets'][who], -torch.ones(n))
        changed = kinds.copy()
        changed[:, 1, 1] = [1, 0]
        alternative = training.rollout(agents, features, changed, assignment, public, rngs)
        for who in (1, 3):
            torch.testing.assert_close(result['targets'][who], alternative['targets'][who])
        for who in (0, 2):
            torch.testing.assert_close(alternative['targets'][who], torch.zeros(n))

    def test_private_gradient_and_exactly_one_optimizer_step_per_agent(self):
        torch.manual_seed(193)
        agents = [ResourceAgent() for _ in range(4)]
        training.assert_independent(agents)
        with self.assertRaises(ValueError):
            training.assert_independent([agents[0], agents[0], agents[2], agents[3]])
        n = 16
        features = torch.randn(n, 2, 2, 2, 1024)
        kinds = training.sample_scenes(np.random.default_rng(72), 2 * n).reshape(n, 2, 2, 2)
        public, _ = training.private_public(n, 16)
        assignment = training.assign_slots(2, [1, 0, 1])
        rngs = [np.random.default_rng(103 + i) for i in range(4)]
        result = training.rollout(agents, features, kinds, assignment, public, rngs)
        loss = -(result['send_outputs'][0][1] + result['act_outputs'][0][1]).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in agents[0].parameters()))
        self.assertTrue(all(p.grad is None for agent in agents[1:] for p in agent.parameters()))
        self.assertIsNone(features.grad)
        optimizers = [CountingAdam(agent.parameters()) for agent in agents]
        before = [[p.detach().clone() for p in agent.parameters()] for agent in agents]
        for step in (1, 2):
            result = training.rollout(agents, features, kinds, assignment, public, rngs)
            metrics = training.optimize(agents, optimizers, result, .05)
            self.assertEqual([entry['agent'] for entry in metrics], list(range(4)))
            self.assertEqual([optimizer.step_count for optimizer in optimizers], [step] * 4)
            for optimizer in optimizers:
                self.assertTrue(all(int(state['step']) == step for state in optimizer.state.values()))
        for initial, agent in zip(before, agents):
            self.assertTrue(any(not torch.equal(old, new) for old, new in zip(initial, agent.parameters())))

    def test_balanced_exposure_layout_and_auxiliary_withdrawal(self):
        rotation = training.matching_schedule(101, 1200)
        np.testing.assert_array_equal(np.bincount(rotation, minlength=3), [400, 400, 400])
        for block in rotation.reshape(-1, 3):
            np.testing.assert_array_equal(np.sort(block), [0, 1, 2])
        for index, bits in itertools.product(range(3), itertools.product((0, 1), repeat=3)):
            assignment = training.assign_slots(index, bits)
            self.assertEqual({frozenset(pair) for pair in assignment}, {frozenset(pair) for pair in training.MATCHINGS[index]})
            partners = training.partners_for(assignment)
            np.testing.assert_array_equal(partners[partners], np.arange(4))
            self.assertTrue(np.all(partners != np.arange(4)))
        edges = np.zeros((4, 4), dtype=int)
        for index in rotation:
            for a, b in training.MATCHINGS[index]:
                edges[a, b] += 1
                edges[b, a] += 1
        np.testing.assert_array_equal(edges, (1 - np.eye(4, dtype=int)) * 400)
        np.testing.assert_array_equal(edges.sum(axis=1), [1200] * 4)
        for update in range(1, 1201):
            plan = training.schedule(update)
            self.assertEqual(plan['aux_weight'], 0)
            self.assertEqual(plan['action_entropy_weight'], .05 if update <= 900 else 0)

    def test_four_symbol_private_protocol_solves_all_six_pairs(self):
        # Each policy's own rank is an identity carried in its fixed parameters,
        # not a supplied partner ID. Forced resources reuse the extreme tokens.
        def send(own, rank):
            if own[0] == own[1]:
                return 0 if own[0] == 0 else 3
            return rank

        def act(own, rank, received):
            if own[0] == own[1]:
                return 0
            if received == 0:
                resource = 1
            elif received == 3:
                resource = 0
            else:
                resource = int(rank > received)
            return int(np.flatnonzero(own == resource)[0])

        successes = 0
        for left, right in itertools.combinations(range(4), 2):
            for scene in FEASIBLE_SCENES:
                message_left, message_right = send(scene[0], left), send(scene[1], right)
                self.assertTrue(0 <= message_left < 4 and 0 <= message_right < 4)
                selected_left = scene[0, act(scene[0], left, message_right)]
                selected_right = scene[1, act(scene[1], right, message_left)]
                self.assertNotEqual(selected_left, selected_right)
                successes += 1
        self.assertEqual(successes, 6 * 14)

    def test_group_no_current_message_upper_bounds_by_exhaustive_enumeration(self):
        # Grant perfect visual categorization. On FF/WW resource choice is
        # forced; each policy has just two binary choices for FW and WF.
        policies = np.asarray([[0, a, b, 1] for a, b in itertools.product((0, 1), repeat=2)])
        edges = list(itertools.combinations(range(4), 2))
        same = FEASIBLE_SCENES[..., 0] == FEASIBLE_SCENES[..., 1]
        for scenes, expected_all, expected_subset in (
                (FEASIBLE_SCENES, 13 / 21, 5 / 7),
                (FEASIBLE_SCENES[same.sum(axis=1) == 1], .5, .5)):
            indices = scenes[..., 0] * 2 + scenes[..., 1]
            maxima = np.zeros(3)
            for ids in itertools.product(range(4), repeat=4):
                counts = [int((policies[ids[a], indices[:, 0]] != policies[ids[b], indices[:, 1]]).sum()) for a, b in edges]
                scores = np.asarray([sum(counts) / 6, (counts[0] + counts[5]) / 2, sum(counts[1:5]) / 4]) / len(scenes)
                maxima = np.maximum(maxima, scores)
            np.testing.assert_allclose(maxima, [expected_all, expected_subset, expected_subset], rtol=0, atol=1e-15)

    def test_actual_training_fixed_matching_equivalence_and_paired_world_slots(self):
        config = copy.deepcopy(training.CONFIG)
        config.update(course_updates=3, exploratory_full_updates=3, pure_reward_updates=3,
                      batch_size_per_agent=32, checkpoints=[0, 3, 6, 9],
                      checkpoint_evaluation_n=32, evaluation_n=32, intervention_n=16, alignment_n=16)
        bank = training.ImageBank()
        def fake_checkpoint(*args, **kwargs):
            return {'aggregates': {group: {'tasks': {'full': {
                'normal': {'mean_reward_per_step': .5}, 'shuffle': {'mean_reward_per_step': .5}}}}
                for group in ('all', 'original', 'cross')}}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(training, 'checkpoint_population', side_effect=fake_checkpoint), \
                patch.object(training, 'evaluate_population', return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            prepared = root / 'prepared.pt'
            original = torch.load(training.ROOT / 'results/pilot_001/prepared_s101.pt', weights_only=True)
            extra = training.make_agents(1101)
            torch.save(original + [agent.state_dict() for agent in extra], prepared)
            training.train(101, 'fixed', bank, prepared, root / 'fixed', config)
            with patch.object(training, 'matching_schedule', return_value=np.zeros(9, dtype=int)):
                training.train(101, 'rotating', bank, prepared, root / 'forced_fixed', config)
            training.train(101, 'rotating', bank, prepared, root / 'rotating', config)
            snapshots = {name: torch.load(root / name / 'checkpoint_0009.pt', weights_only=True)
                         for name in ('fixed', 'forced_fixed', 'rotating')}
            for left, right in zip(snapshots['fixed'], snapshots['forced_fixed']):
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            records = {name: [json.loads(line) for line in (root / name / 'training_metrics.jsonl').read_text().splitlines()]
                       for name in snapshots}
            for name in snapshots:
                self.assertEqual((root / name / 'initial.pt').read_bytes(), prepared.read_bytes())
                self.assertEqual([row['world_slots_sha256'] for row in records[name]], [row['world_slots_sha256'] for row in records['fixed']])
                self.assertEqual([row['layout_bits'] for row in records[name]], [row['layout_bits'] for row in records['fixed']])
                for row in records[name]:
                    self.assertEqual([a['agent'] for a in row['agents']], [0, 1, 2, 3])
                    for agent in row['agents']:
                        self.assertEqual(agent['received_sha256'], row['agents'][agent['partner']]['sent_sha256'])
            self.assertTrue(any(a['private_input_sha256'] != b['private_input_sha256']
                                for left, right in zip(records['fixed'], records['rotating'])
                                for a, b in zip(left['agents'], right['agents'])))
            result = json.loads((root / 'rotating/result.json').read_text())
            np.testing.assert_array_equal(result['edge_training_updates'], (1 - np.eye(4, dtype=int)) * 3)
            self.assertEqual(result['individual_training_choices_per_agent'], 9 * 32)
            self.assertEqual(result['joint_dyad_training_steps'], 9 * 32 * 2)


if __name__ == '__main__':
    unittest.main()
