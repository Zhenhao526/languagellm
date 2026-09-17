"""Independent finite-state and information-routing checks; no training runs."""
from __future__ import annotations

import copy
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bounds_audit
import camp
import run_stages


class IndependentCampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def agents_and_bank(self, vocab=5, length=2):
        torch.manual_seed(77071)
        project = nn.Sequential(nn.Linear(1024, 64), nn.Tanh())
        agents = [camp.CampAgent(project, vocab, length) for _ in range(2)]
        bank = SimpleNamespace(pools={(split, kind): np.array([2 * kind, 2 * kind + 1])
                                     for split in ('train', 'test') for kind in (0, 1)})
        banks = [torch.randn(4, 64) for _ in range(2)]
        return agents, banks, bank

    def test_all_environment_transitions_and_inventory_conservation(self):
        cases = list(product(permutations(range(4), 2), product(range(3), repeat=2),
                             range(2), range(4), range(3)))
        positions = np.array([row[0] for row in cases])
        inventory = np.array([row[1] for row in cases])
        goals = np.array([row[2] for row in cases])
        actions = np.array([row[3] for row in cases])
        refills = np.array([(row[4] + .5) / 3 for row in cases])
        before_positions, before_inventory = positions.copy(), inventory.copy()
        observed = camp.collect(positions, inventory, goals, actions, refills, replenish=True)
        expected_positions, expected_inventory, expected_reward = [], [], []
        expected_gathered, expected_overflow = [], []
        for world, stored, goal, action, refill_index in cases:
            food, water = world
            outcome = 0 if action == food else 1 if action == water else None
            got = [int(outcome == 0), int(outcome == 1)]
            extra = [max(stored[kind] + got[kind] - 2, 0) for kind in range(2)]
            held = [stored[kind] + got[kind] - extra[kind] for kind in range(2)]
            reward = int(held[goal] > 0)
            held[goal] -= reward
            new_world = list(world)
            if outcome is not None:
                choices = [place for place in range(4) if place != world[1 - outcome]]
                new_world[outcome] = choices[refill_index]
            expected_positions.append(new_world)
            expected_inventory.append(held)
            expected_reward.append(reward)
            expected_gathered.append(got)
            expected_overflow.append(extra)
        for actual, expected in zip(observed, [expected_positions, expected_inventory,
                                             expected_reward, expected_gathered, expected_overflow]):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(positions, before_positions)
        np.testing.assert_array_equal(inventory, before_inventory)
        self.assertEqual(len(cases), 2592)

    def test_stock_full_still_collects_and_replenishes_before_consumption(self):
        next_positions, after, reward, gathered, overflow = camp.collect(
            np.array([[0, 1]]), np.array([[2, 0]]), np.array([0]),
            np.array([0]), np.array([.9]), replenish=True)
        np.testing.assert_array_equal(next_positions, [[3, 1]])
        np.testing.assert_array_equal(after, [[1, 0]])
        np.testing.assert_array_equal(gathered, [[1, 0]])
        np.testing.assert_array_equal(overflow, [[1, 0]])
        np.testing.assert_array_equal(reward, [1])

    def test_empty_choice_preserves_world_but_can_consume_stored_resource(self):
        next_positions, after, reward, gathered, overflow = camp.collect(
            np.array([[0, 1]]), np.array([[0, 1]]), np.array([1]),
            np.array([3]), np.array([.9]), replenish=True)
        np.testing.assert_array_equal(next_positions, [[0, 1]])
        np.testing.assert_array_equal(after, [[0, 0]])
        np.testing.assert_array_equal(gathered, [[0, 0]])
        np.testing.assert_array_equal(overflow, [[0, 0]])
        np.testing.assert_array_equal(reward, [1])

    def test_exact_information_bounds_and_independent_history_tree(self):
        bounds = bounds_audit.static_channel_bounds()
        self.assertEqual(bounds['goal_hidden']['1']['fraction'], '1/4')
        self.assertEqual(bounds['goal_known']['4']['fraction'], '1')
        self.assertEqual(bounds['goal_hidden']['5']['fraction'], '17/24')
        self.assertEqual(bounds['goal_hidden']['12']['fraction'], '1')
        value = bounds_audit.no_message_value(3, (1,) * 12, (0, 0))
        self.assertEqual(value, Fraction(265, 216))
        self.assertEqual(value, bounds_audit.independent_history_tree(3))

    def test_holdout_is_balanced_and_disjoint(self):
        all_maps = set(map(tuple, camp.MAPS))
        train = set(map(tuple, camp.MAPS[camp.TRAIN_MAPS]))
        heldout = set(map(tuple, camp.MAPS[camp.HOLDOUT]))
        self.assertFalse(train & heldout)
        self.assertEqual(train | heldout, all_maps)
        self.assertEqual((len(train), len(heldout)), (8, 4))
        for pool, count in [(train, 2), (heldout, 1)]:
            for kind in (0, 1):
                for location in range(4):
                    self.assertEqual(sum(world[kind] == location for world in pool), count)

    def test_private_world_changes_cannot_change_blocked_first_action(self):
        agents, banks, bank = self.agents_and_bank()
        plan = dict(known=False, vocab=5, length=2, blocked=True)
        _, _, original = run_stages.rollout(agents, banks, bank, plan, 47001, 32,
                                            greedy=False, trace=True)
        initial = run_stages.initial_world
        def changed_world(*args, **kwargs):
            positions, inventory = initial(*args, **kwargs)
            return (positions + 1) % 4, inventory
        with patch.object(run_stages, 'initial_world', side_effect=changed_world):
            _, _, changed = run_stages.rollout(agents, banks, bank, plan, 47001, 32,
                                               greedy=False, trace=True)
        for first, second in zip(original, changed):
            self.assertFalse(np.array_equal(first['positions'], second['positions']))
            for key in ('goals', 'menu', 'delivered', 'action', 'place', 'history'):
                np.testing.assert_array_equal(first[key], second[key])

    def test_collector_history_has_only_its_actual_previous_events(self):
        agents, banks, bank = self.agents_and_bank()
        plan = dict(known=False, vocab=5, length=2, horizon=3)
        _, _, records = run_stages.rollout(agents, banks, bank, plan, 47002, 32,
                                           greedy=False, trace=True)
        for scout in (0, 1):
            direction = [record for record in records if record['scout'][0] == scout]
            self.assertEqual(len(direction), 3)
            expected = np.zeros((16, 14), np.float32)
            for step, record in enumerate(direction):
                np.testing.assert_array_equal(record['history'], expected)
                np.testing.assert_array_equal(record['step'], step)
                actual_event = np.column_stack((np.eye(4)[record['place']], record['gathered'],
                                               record['gathered'].sum(1) == 0)).astype(np.float32)
                expected = np.column_stack((expected[:, 7:], actual_event)).astype(np.float32)
            for earlier, later in zip(direction, direction[1:]):
                np.testing.assert_array_equal(earlier['next_positions'], later['positions'])
                np.testing.assert_array_equal(earlier['next_inventory'], later['inventory'])

    def test_menu_permutation_only_relabels_receiver_actions(self):
        agents, _, _ = self.agents_and_bank()
        agent = agents[1]
        menus = torch.tensor(list(permutations(range(4))))
        n = len(menus)
        messages = torch.tensor([[1, 3]]).repeat(n, 1)
        goals = torch.tensor([[1., 0.]]).repeat(n, 1)
        inventory, history = torch.zeros(n, 2), torch.zeros(n, 14)
        logits, _ = agent.receive(messages, goals, inventory, history, menus)
        physical_logits = torch.empty_like(logits).scatter(1, menus, logits)
        torch.testing.assert_close(physical_logits, physical_logits[:1].expand_as(physical_logits))

    def test_sequence_capacity_and_only_integer_delivery(self):
        agents, _, _ = self.agents_and_bank()
        sequences = torch.tensor(list(product(range(5), repeat=2)), dtype=torch.int64)
        self.assertEqual(len(sequences), 25)
        self.assertEqual(len(set((5 * sequences[:, 0] + sequences[:, 1]).tolist())), 25)
        logits, _ = agents[1].receive(sequences, torch.ones(25, 2), torch.zeros(25, 2),
                                      torch.zeros(25, 14), torch.arange(4).repeat(25, 1))
        self.assertEqual(tuple(logits.shape), (25, 4))
        with self.assertRaises(AssertionError):
            agents[1].receive(sequences.float(), torch.ones(25, 2), torch.zeros(25, 2),
                               torch.zeros(25, 14), torch.arange(4).repeat(25, 1))

    def test_receiver_loss_cannot_backpropagate_into_sender(self):
        agents, _, _ = self.agents_and_bank()
        h = agents[0].observe(torch.randn(8, 260))
        tokens, _, _, _ = agents[0].send(h, torch.zeros(8, 2), torch.zeros(8, 2),
                                       np.random.default_rng(47003), False)
        self.assertFalse(tokens.requires_grad)
        logits, value = agents[1].receive(tokens, torch.ones(8, 2), torch.zeros(8, 2),
                                         torch.zeros(8, 14), torch.arange(4).repeat(8, 1))
        (logits.square().mean() + value.square().mean()).backward()
        self.assertTrue(all(parameter.grad is None for parameter in agents[0].parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in agents[1].parameters()))
        self.assertTrue(all(not parameter.requires_grad for agent in agents for parameter in agent.project.parameters()))
        all_pointers = [p.data_ptr() for agent in agents for p in agent.parameters()]
        self.assertEqual(len(all_pointers), len(set(all_pointers)))

    def test_replay_equals_immediate_in_value_and_gradient(self):
        agents, _, _ = self.agents_and_bank()
        immediate, replay = agents[0], copy.deepcopy(agents[0])
        visual = torch.randn(8, 260)
        h0 = immediate.observe(visual, 0, 'retain')
        h1 = replay.observe(visual, 3, 'replay')
        torch.testing.assert_close(h0, h1, rtol=0, atol=0)
        h0.square().sum().backward()
        h1.square().sum().backward()
        for (name, parameter), (other_name, other_parameter) in zip(immediate.named_parameters(), replay.named_parameters()):
            self.assertEqual(name, other_name)
            if parameter.grad is None:
                self.assertIsNone(other_parameter.grad)
            else:
                torch.testing.assert_close(parameter.grad, other_parameter.grad, rtol=0, atol=0)

    def test_reset_removes_scene_and_shuffle_preserves_goal_inventory_groups(self):
        agents, _, _ = self.agents_and_bank()
        first = agents[0].observe(torch.randn(8, 260), 3, 'reset')
        second = agents[0].observe(torch.randn(8, 260), 3, 'reset')
        torch.testing.assert_close(first, torch.zeros_like(first), rtol=0, atol=0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        sent = np.arange(32).reshape(16, 2)
        goals = np.array([0, 1] * 8)
        inventory = np.array([[0, 0]] * 8 + [[1, 0]] * 8)
        shuffled = run_stages.shuffle_conditional(sent, goals, inventory, np.random.default_rng(47004))
        for goal, inv0 in product(range(2), range(2)):
            rows = (goals == goal) & (inventory[:, 0] == inv0)
            self.assertEqual(set(map(tuple, sent[rows])), set(map(tuple, shuffled[rows])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
