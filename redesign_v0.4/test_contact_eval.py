"""Engineering checks for policy drift; synthetic rules are not learned evidence."""
import io
import unittest

import numpy as np
import torch

from contact_eval import ALL_PAIRS, CONTEXTS, ORIGINAL_PAIRS, protocol_drift, protocol_reference


class ToyBank:
    def sample(self, kinds, split, rng):
        assert split == "test"
        return torch.from_numpy(kinds[..., None].astype(np.float32)), rng.integers(1000, size=kinds.shape)


class ToyAgent:
    def __init__(self, symbol):
        self.symbol = symbol
        self.reverse_symbols = set()

    def observe(self, features, public):
        assert features.shape[1:] == (2, 1)
        assert public.shape == (len(features), 3)
        return features[..., 0], features[..., 0]

    def send(self, local):
        logits = torch.zeros(len(local), 5)
        logits[:, self.symbol] = 20
        return logits

    def act(self, options, local, received):
        assert received.dtype == torch.int64 and not received.requires_grad
        logits = 10 * options
        reverse = torch.tensor([int(symbol) in self.reverse_symbols for symbol in received], dtype=torch.bool)
        return torch.where(reverse[:, None], -logits, logits)


class ContactDriftTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.agents = [ToyAgent(i) for i in range(4)]
        self.reference = protocol_reference(self.agents, ToyBank(), 183, n=96, support_n=512)

    def test_identity_has_zero_drift_and_reference_loads_weights_only(self):
        buffer = io.BytesIO()
        torch.save(self.reference, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        report = protocol_drift(self.agents, restored)
        for sender in report["senders"]:
            self.assertEqual(sender["overall"]["mean_probability_total_variation"], 0)
            self.assertEqual(sender["overall"]["greedy_raw_symbol_change_rate"], 0)
        for listener in report["listeners"]:
            for context in CONTEXTS:
                for symbol in listener["by_own_context"][context]["all_five_symbols"]:
                    self.assertEqual(symbol["mean_action_probability_total_variation"], 0)
                    self.assertEqual(symbol["greedy_resource_change_rate"], 0)

    def test_sender_change_does_not_create_listener_drift(self):
        self.agents[0].symbol = 4
        report = protocol_drift(self.agents, self.reference)
        self.assertEqual(report["senders"][0]["overall"]["greedy_raw_symbol_change_rate"], 1)
        for listener in report["listeners"]:
            for context in CONTEXTS:
                effects = listener["by_own_context"][context]["all_five_symbols"]
                self.assertTrue(all(x["mean_action_probability_total_variation"] == 0 for x in effects))

    def test_listener_resource_drift_is_separated_from_duplicate_options(self):
        self.agents[0].reverse_symbols = set(range(5))
        report = protocol_drift(self.agents, self.reference)
        contexts = report["listeners"][0]["by_own_context"]
        for symbol in contexts["mixed"]["all_five_symbols"]:
            self.assertEqual(symbol["greedy_resource_change_rate"], 1)
        for context in ("food_food", "water_water"):
            self.assertTrue(all(symbol["greedy_resource_change_rate"] == 0 for symbol in contexts[context]["all_five_symbols"]))

    def test_unused_symbol_drift_does_not_enter_initial_support_summary(self):
        self.agents[0].reverse_symbols = {4}
        report = protocol_drift(self.agents, self.reference)
        mixed = report["listeners"][0]["by_own_context"]["mixed"]
        self.assertEqual(mixed["all_five_symbols"][4]["greedy_resource_change_rate"], 1)
        self.assertEqual(mixed["initial_received_support_summary"]["symbols"], [1])
        self.assertEqual(mixed["initial_received_support_summary"]["equal_symbol_mean"]["greedy_resource_change_rate"], 0)

    def test_new_partner_symbols_do_not_rewrite_initial_support_mask(self):
        report = protocol_drift(self.agents, self.reference, current_trained_pairs=ALL_PAIRS)
        receiver = report["listeners"][0]
        self.assertEqual(receiver["initial_declared_partners"], [1])
        self.assertEqual(receiver["current_declared_partners"], [1, 2, 3])
        mixed = receiver["by_own_context"]["mixed"]
        self.assertEqual(mixed["initial_received_support_summary"]["symbols"], [1])
        self.assertEqual(mixed["received_support_change"]["current_symbols"], [1, 2, 3])

    def test_reference_observation_changes_are_rejected(self):
        self.reference["probe"]["public"][0, 2] = -1
        with self.assertRaises(AssertionError):
            protocol_drift(self.agents, self.reference)


if __name__ == "__main__":
    unittest.main()
