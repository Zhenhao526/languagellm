"""Meaningful freeze, same-stream, resume, origin, and exact-bound checks."""
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
import run_mechanism as mechanism
import mechanism_bounds as bounds


def equal_states(left, right):
    return all(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a) for a, b in zip(left, right))


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.bank = mechanism.ImageBank()

    def test_fixed_design_and_trainable_counts(self):
        self.assertEqual(mechanism.CONFIRMATORY, [404, 505, 606, 707, 808, 909, 1001])
        self.assertEqual(mechanism.CONFIG['additional_updates'], 600)
        self.assertEqual(mechanism.CONFIG['training_seed_offset'], 2000)
        self.assertEqual(mechanism.CONFIG['checkpoints'], [0, 1, 5, 10, 20, 50, 100, 200, 400, 600])
        counts = {'both_learn': 17927, 'sender_only': 9094, 'listener_only': 13250,
                  'behavior_frozen': 4417, 'all_plastic': 83527}
        agents = [ResourceAgent() for _ in range(4)]
        for condition in mechanism.FLAGS:
            opts = mechanism.configure(agents, condition)
            self.assertTrue(all(not o.state for o in opts))
            self.assertEqual(sum(p.numel() for p in agents[0].parameters() if p.requires_grad), counts[condition])
            for module, enabled in mechanism.FLAGS[condition].items():
                self.assertTrue(all(p.requires_grad == enabled for a in agents for p in getattr(a, module).parameters()))

    def test_whole_functions_and_baseline_isolation(self):
        origin = mechanism.origin_folder(101) / 'checkpoint_1200.pt'
        for condition in mechanism.FACTORIAL:
            agents = mechanism.partners.load_population(origin)
            optimizers = mechanism.configure(agents, condition)
            initial = mechanism.module_digests(agents)
            reference = mechanism.protocol_reference(agents, self.bank, 92101, n=32, support_n=64)
            public, _ = mechanism.contact.private_public(32, 16)
            world = np.random.default_rng(103)
            policies = [np.random.default_rng(20 + i) for i in range(4)]
            for step in range(3):
                kinds = mechanism.contact.sample_scenes(world, 64).reshape(32, 2, 2, 2)
                features, _ = self.bank.sample(kinds, 'train', world)
                assignment = mechanism.partners.assign_slots(step, [0, 1, 0])
                result = mechanism.partners.rollout(agents, features, kinds, assignment, public, policies)
                mechanism.partners.optimize(agents, optimizers, result, entropy_weight=0.)
            drift = mechanism.protocol_drift(agents, reference)
            self.assertTrue(mechanism.assert_frozen(agents, condition, initial, drift)['passed'])
            final = mechanism.module_digests(agents)
            self.assertTrue(all(final[i]['value'] != initial[i]['value'] for i in range(4)))
            for name, active in mechanism.FLAGS[condition].items():
                if active:
                    self.assertTrue(any(final[i][name] != initial[i][name] for i in range(4)))

    def test_head_freeze_without_projection_freeze_does_change_function(self):
        agents = mechanism.partners.load_population(mechanism.origin_folder(101) / 'checkpoint_1200.pt')
        reference = mechanism.protocol_reference(agents, self.bank, 901, n=32, support_n=64)
        for agent in agents:
            agent.sender.requires_grad_(False)
            agent.actor.requires_grad_(False)
            with torch.no_grad():
                agent.project[0].weight.add_(.02)
        drift = mechanism.protocol_drift(agents, reference)
        self.assertTrue(any(s['overall']['mean_probability_total_variation'] > 0 for s in drift['senders']))
        self.assertTrue(any(symbol['mean_action_probability_total_variation'] > 0
                            for receiver in drift['listeners'] for context in receiver['by_own_context'].values()
                            for symbol in context['all_five_symbols']))

    def test_restore_freeze_flags_adam_and_random_stream_exactly(self):
        agents = mechanism.partners.load_population(mechanism.origin_folder(101) / 'checkpoint_1200.pt')
        optimizers = mechanism.configure(agents, 'listener_only')
        public, _ = mechanism.contact.private_public(32, 16)
        world, layout = np.random.default_rng(71), np.random.default_rng(72)
        policies = [np.random.default_rng(73 + i) for i in range(4)]
        schedule = np.asarray([1, 2, 0])
        def step(models, opts, w, l, p, index):
            assignment = mechanism.partners.assign_slots(int(schedule[index]), l.integers(2, size=3))
            kinds = mechanism.contact.sample_scenes(w, 64).reshape(32, 2, 2, 2)
            features, _ = self.bank.sample(kinds, 'train', w)
            result = mechanism.partners.rollout(models, features, kinds, assignment, public, p)
            mechanism.partners.optimize(models, opts, result, entropy_weight=0.)
        step(agents, optimizers, world, layout, policies, 0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'state.pt'
            mechanism.save_state(path, agents, optimizers, 'listener_only', 1, world, layout, policies, schedule, np.zeros((4, 4), dtype=np.int64))
            models, opts, w, l, p, saved = mechanism.restore_state(path)
            self.assertEqual(saved['trainability'], mechanism.FLAGS['listener_only'])
            self.assertTrue(all(not x.requires_grad for a in models for x in a.project.parameters()))
            step(agents, optimizers, world, layout, policies, 1)
            step(models, opts, w, l, p, 1)
            self.assertTrue(equal_states([a.state_dict() for a in agents], [a.state_dict() for a in models]))
            self.assertEqual(world.bit_generator.state, w.bit_generator.state)
            for left, right in zip(policies, p):
                self.assertEqual(left.bit_generator.state, right.bit_generator.state)

    def test_all_plastic_exactly_replays_original_contact_and_factorial_inputs_match(self):
        config = copy.deepcopy(mechanism.CONFIG)
        config.update(additional_updates=3, batch_size_per_agent=32, checkpoints=[0, 1, 3],
                      protocol_probe_n=32, protocol_support_n=64, evaluation_n=32,
                      intervention_n=16, alignment_n=16)
        origin = mechanism.origin_folder(101)
        expected = json.loads((origin / 'learning_curve.json').read_text())[-1]['evaluation']
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(mechanism.contact, 'checkpoint_population', return_value=expected), \
                patch.object(mechanism.contact, 'evaluate_population', return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            mechanism.contact.train(101, 'fixed_to_rotating', self.bank, root / 'old', config)
            for condition in mechanism.FACTORIAL + ['all_plastic']:
                mechanism.train(101, condition, self.bank, root / condition, config)
            old_state = torch.load(root / 'old/final_training_state.pt', weights_only=True)
            new_state = torch.load(root / 'all_plastic/final_training_state.pt', weights_only=True)
            self.assertTrue(equal_states(old_state['agents'], new_state['agents']))
            self.assertEqual(old_state['world_rng'], new_state['world_rng'])
            self.assertEqual(old_state['policy_rngs'], new_state['policy_rngs'])
            old_rows = [json.loads(s) for s in (root / 'old/training_metrics.jsonl').read_text().splitlines()]
            for condition in mechanism.FACTORIAL + ['all_plastic']:
                rows = [json.loads(s) for s in (root / condition / 'training_metrics.jsonl').read_text().splitlines()]
                self.assertEqual([r['world_slots_sha256'] for r in rows], [r['world_slots_sha256'] for r in old_rows])
                self.assertEqual([r['slot_assignment'] for r in rows], [r['slot_assignment'] for r in old_rows])
                self.assertEqual([[a['private_input_sha256'] for a in r['agents']] for r in rows],
                                 [[a['private_input_sha256'] for a in r['agents']] for r in old_rows])
                state = torch.load(root / condition / 'final_training_state.pt', weights_only=True)
                self.assertEqual(state['policy_rngs'], old_state['policy_rngs'])
                self.assertEqual(state['world_rng'], old_state['world_rng'])
                self.assertEqual(state['layout_rng'], old_state['layout_rng'])
                self.assertEqual((root / condition / 'initial.pt').read_bytes(), (origin / 'checkpoint_1200.pt').read_bytes())
            self.assertEqual([r['agents'] for r in old_rows], [r['agents'] for r in [json.loads(s) for s in (root / 'all_plastic/training_metrics.jsonl').read_text().splitlines()]])

    def test_new_origin_initialization_has_no_reuse_or_seed_overlap(self):
        used = [seed + offset for seed in mechanism.EXPLORATORY + mechanism.CONFIRMATORY for offset in (0, 1000)]
        self.assertEqual(len(used), len(set(used)))
        def practice(agents, bank, seed, updates, batch):
            self.assertEqual((updates, batch), (200, 64))
            return [{'heldout_need_sensitive_choice': 1.} for _ in agents]
        with tempfile.TemporaryDirectory() as temp, patch.object(mechanism, 'individual_practice', side_effect=practice):
            path, report = mechanism.prepare_new_population(404, self.bank, Path(temp), mechanism.partners.CONFIG)
            self.assertEqual(report['source_seeds'], [404, 1404])
            self.assertEqual(report['reused_agents'], [])
            self.assertTrue(report['passed'])
            states = torch.load(path, weights_only=True)
            expected = mechanism.make_agents(404) + mechanism.make_agents(1404)
            self.assertTrue(equal_states(states, [a.state_dict() for a in expected]))


class BoundTests(unittest.TestCase):
    def test_sender_indistinguishability_and_perfect_information_limits(self):
        p = np.zeros((4, 16, 2, 5))
        p[..., 0] = 1
        result = bounds.sender_information_bound(p, bounds.GROUPS['all'])
        self.assertAlmostEqual(result['one_restricted_success_upper_bound'], .5)
        self.assertAlmostEqual(result['full_success_upper_bound'], 5 / 7)
        p[:, :, 1, 0] = 0
        p[:, :, 1, 1] = 1
        self.assertEqual(bounds.sender_information_bound(p, bounds.GROUPS['all'])['full_success_upper_bound'], 1.)

    def test_sender_conflicting_codes_respect_partner_graph(self):
        p = np.zeros((4, 16, 2, 5))
        for who in range(4):
            for q in range(2):
                p[who, :, q, q ^ (who % 2)] = 1
        self.assertEqual(bounds.sender_information_bound(p, bounds.GROUPS['original'])['full_success_upper_bound'], 1.)
        self.assertAlmostEqual(bounds.sender_information_bound(p, bounds.GROUPS['cross'])['full_success_upper_bound'], 5 / 7)
        self.assertAlmostEqual(bounds.sender_information_bound(p, bounds.GROUPS['all'])['full_success_upper_bound'], 17 / 21)

    def test_listener_conflicts_respect_partner_graph_and_oracle_direction(self):
        p = np.zeros((4, 16, 5, 2))
        for who in range(4):
            for symbol in range(5):
                p[who, :, symbol, (symbol % 2) ^ (who % 2)] = 1
        self.assertEqual(bounds.listener_compatibility_bound(p, bounds.GROUPS['original'])['full_success_upper_bound'], 1.)
        self.assertAlmostEqual(bounds.listener_compatibility_bound(p, bounds.GROUPS['cross'])['full_success_upper_bound'], 5 / 7)
        self.assertAlmostEqual(bounds.listener_compatibility_bound(p, bounds.GROUPS['all'])['full_success_upper_bound'], 17 / 21)
        result = bounds.listener_compatibility_bound(p, bounds.GROUPS['original'])
        self.assertEqual(result['best_symbol_by_sender_time_resource'][0][0], [0, 1])

    def test_exact_enumeration_covers_repeated_pairs_both_orders_all_times_and_probabilities(self):
        class Bank:
            pools = {('test', 0): np.asarray([0, 1]), ('test', 1): np.asarray([2, 3])}
            features = torch.eye(4)
        class Agent:
            def observe(self, own, public):
                return own, public
            def send(self, local):
                logits = torch.zeros(len(local), 5)
                logits[:, 0] = 2
                return logits
            def act(self, own, local, received):
                return torch.zeros(len(local), 2)
        result = bounds.frozen_function_bounds([Agent() for _ in range(4)], Bank())
        self.assertEqual(result['restricted_ordered_photo_pairs_by_resource'], [4, 4])
        self.assertEqual(result['mixed_ordered_photo_pairs'], 8)
        for mode in ('normal', 'stochastic'):
            self.assertAlmostEqual(result['fixed_sender'][mode]['all']['full_success_upper_bound'], 5 / 7)
            self.assertAlmostEqual(result['fixed_listener'][mode]['all']['full_success_upper_bound'], 5 / 7)
        self.assertEqual(np.asarray(result['sender_distributions_agent_time_resource_symbol']['normal']).shape, (4, 16, 2, 5))


if __name__ == '__main__':
    unittest.main()
