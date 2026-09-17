import unittest
from unittest.mock import patch
from itertools import product
import numpy as np
from research_program.triadic_semantic_probe_study import channel_intervention as ch


def fake_forward(module, inputs):
    width = 17 if module % 3 == 2 else 32
    weighted = inputs @ (1 + np.arange(inputs.shape[-1]) % 11)
    centers = weighted.astype(np.int64) % width
    return -np.square(np.arange(width)[None, :] - centers[:, None]).astype(float), {}


class InterventionTests(unittest.TestCase):
    def fixture(self):
        x = np.zeros((3, 3, 54)); senders = np.arange(3, dtype=np.int16)
        x[:, :, 1] = np.array([1, 3, 6])
        donor_x = x.copy(); donor_x[np.arange(3), senders, 0] = [2, 4, 7]
        nets = list(range(9))
        natural = ch.core.rollout(nets, x, True)
        donor = ch.core.rollout(nets, donor_x, True)
        packets = donor['messages'][np.arange(3), :, senders, :]
        return x, donor_x, senders, nets, natural, donor, packets

    def test_only_outward_payloads_change_and_visibility_is_preserved(self):
        messages = np.arange(36, dtype=np.int8).reshape(3, 3, 4) % 8
        senders = np.arange(3, dtype=np.int16); replacement = np.full((3, 4), 7, dtype=np.int8)
        actual = ch.replace_outward(messages, senders, replacement, True)
        original = ch.core.routed_window(messages, True)
        self.assertTrue(np.array_equal(actual[:, :, 96:], original[:, :, 96:]))
        for row, sender, viewer in product(range(3), range(3), range(3)):
            expected = replacement[row] if sender == senders[row] and viewer != sender else messages[row, sender]
            block = actual[row, viewer, sender*32:(sender+1)*32].reshape(4, 8)
            self.assertTrue(np.array_equal(block, np.eye(8)[expected]))

    @patch.object(ch.core.base, 'actor_forward', side_effect=fake_forward)
    def test_local_both_reproduces_other_agents_inputs_by_construction(self, _):
        x, dx, senders, nets, natural, donor, packets = self.fixture()
        actual = ch.intervene(nets, x, natural['messages'], senders, packets, (0, 1), True)
        for row in range(3):
            for agent in range(3):
                if agent != senders[row]:
                    self.assertTrue(np.array_equal(actual['action_inputs'][row, agent], donor['action_inputs'][row, agent]))
        self.assertTrue(np.array_equal(actual['messages'][np.arange(3), 1, senders], natural['messages'][np.arange(3), 1, senders]))

    @patch.object(ch.core.base, 'actor_forward', side_effect=fake_forward)
    def test_silent_all_interventions_leave_outputs_unchanged(self, _):
        x, dx, senders, nets, natural, donor, packets = self.fixture()
        silent = ch.core.rollout(nets, x, False)
        for windows in ((0,), (1,), (0, 1)):
            actual = ch.intervene(nets, x, silent['messages'], senders, packets, windows, False)
            self.assertTrue(np.array_equal(actual['action_inputs'], silent['action_inputs']))
            self.assertTrue(np.array_equal(actual['messages'], silent['messages']))

    @patch.object(ch.core.base, 'actor_forward', side_effect=fake_forward)
    def test_second_window_intervention_preserves_generated_second_messages(self, _):
        x, dx, senders, nets, natural, donor, packets = self.fixture()
        actual = ch.intervene(nets, x, natural['messages'], senders, packets, (1,), True)
        self.assertTrue(np.array_equal(actual['messages'], natural['messages']))
        self.assertTrue(np.array_equal(actual['first_routes'], ch.core.routed_window(natural['messages'][:, 0], True)))

    @patch.object(ch.core.base, 'actor_forward', side_effect=fake_forward)
    def test_same_fact_same_background_sham_is_exact(self, _):
        x, dx, senders, nets, natural, donor, packets = self.fixture()
        own = natural['messages'][np.arange(3), :, senders, :]
        actual = ch.intervene(nets, x, natural['messages'], senders, own, (0, 1), True)
        self.assertTrue(np.array_equal(actual['action_inputs'], natural['action_inputs']))
        self.assertTrue(np.array_equal(actual['messages'], natural['messages']))

    def test_native_settlement_all_4913_actions(self):
        env = ch.core.base.env
        joint = np.array(list(product(range(17), repeat=3)), dtype=np.int16)
        for needs, layout in (((0, 5, 7), (0, 1, 2, 3)), ((2, 8, 11), (3, 1, 0, 2))):
            state = env.State(needs, layout, (1, 2, 3))
            packed = np.tile(np.array([*needs, *layout, 1, 2, 3], dtype=np.int16), (len(joint), 1))
            actual = ch.settle(packed, joint)
            expected = []
            for row in joint:
                outcome = env.settle(state, {a: env.all_actions(a)[row[i]] for i, a in enumerate(env.AGENTS)}, require_match=True)
                expected.append(outcome['reward'])
            self.assertTrue(np.array_equal(actual['greedy_reward'], expected))
            self.assertTrue(np.array_equal(actual['satisfied'].sum(-1)/2, expected))

    def test_probability_rounding_tie_matches_original_greedy_convention(self):
        def nearly_tied(module, inputs):
            if module % 3 == 1:
                logits = np.zeros((len(inputs), 32)); logits[:, 1::8] = 1e-18
                return logits, {}
            return fake_forward(module, inputs)
        with patch.object(ch.core.base, 'actor_forward', side_effect=nearly_tied):
            x, dx, senders, nets, natural, donor, packets = self.fixture()
            self.assertTrue(np.all(natural['messages'][:, 1] == 0))
            actual = ch.intervene(nets, x, natural['messages'], senders, packets, (0,), True)
            self.assertTrue(np.all(actual['messages'][:, 1] == 0))


if __name__ == '__main__':
    unittest.main()
