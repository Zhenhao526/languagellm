"""Engineering tests use an explicit artificial protocol, never learned evidence."""
import unittest

import numpy as np
import torch

from curriculum_eval import (CURRICULUM_SCENES, _shuffle, coordination_bounds,
                             evaluate, message_intervention, sample_curriculum)


class ToyBank:
    """Tests only: exposes resource bits instead of pretrained visual features."""
    def sample(self, kinds, split, rng):
        ids = rng.integers(10000, size=kinds.shape)
        return torch.from_numpy(kinds[..., None].astype(np.float32)), ids


class ProtocolAgent:
    """An oracle communication rule validates scoring and interventions."""
    def __init__(self, identity):
        self.identity = identity

    def observe(self, features, public):
        assert features.shape[1:] == (2, 1)
        assert torch.all(public[:, :2] == 0)
        return features[..., 0].long(), features[..., 0].long()

    def send(self, local):
        # 1 = two food options; 2 = two water options; 3 = mixed options.
        symbol = torch.where(local[:, 0] == local[:, 1], local[:, 0] + 1, 3)
        return torch.nn.functional.one_hot(symbol, 5).float() * 40

    def act(self, options, local, received):
        assert received.dtype == torch.int64 and not received.requires_grad
        assert received.ndim == 1
        target = torch.full_like(received, self.identity)
        target = torch.where(received == 1, 1, target)
        target = torch.where(received == 2, 0, target)
        return (options == target[:, None]).float() * 40


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.bank = ToyBank()
        self.agents = [ProtocolAgent(0), ProtocolAgent(1)]

    def test_scene_set_and_exact_no_message_bounds(self):
        self.assertEqual(CURRICULUM_SCENES.shape, (8, 2, 2))
        self.assertEqual(len(np.unique(CURRICULUM_SCENES.reshape(8, 4), axis=0)), 8)
        same = CURRICULUM_SCENES[..., 0] == CURRICULUM_SCENES[..., 1]
        np.testing.assert_array_equal(same.sum(axis=1), np.ones(8))
        np.testing.assert_array_equal(same.sum(axis=0), [4, 4])
        self.assertEqual(coordination_bounds("curriculum")["no_message_expected_success_upper_bound"], .5)
        self.assertEqual(coordination_bounds("full")["no_message_expected_success_upper_bound"], 5 / 7)
        self.assertEqual(sample_curriculum(np.random.default_rng(1), 0).shape, (0, 2, 2))
        with self.assertRaises(ValueError):
            sample_curriculum(np.random.default_rng(1), -1)

    def test_perfect_protocol_and_paired_message_disruptions(self):
        outputs = {}
        traces = {}
        for mode in ("normal", "shuffle", "blank"):
            outputs[mode], traces[mode] = evaluate(self.agents, self.bank, 891, n=8192,
                                                  task="curriculum", mode=mode, trace=True)
        self.assertEqual(outputs["normal"]["balanced_gathering"], 1)
        for mode in ("shuffle", "blank"):
            self.assertAlmostEqual(outputs[mode]["balanced_gathering"], .5, delta=.025)
            self.assertEqual(outputs[mode]["external_cases_sha256"], outputs["normal"]["external_cases_sha256"])
            for base, changed in zip(traces["normal"], traces[mode]):
                for field in ("kinds", "image_ids", "remaining", "sent"):
                    self.assertEqual(base[field], changed[field])
        for direction in outputs["normal"]["by_direction"]:
            self.assertEqual(direction["success"], 1)
        full, _ = evaluate(self.agents, self.bank, 122, n=4096, task="full")
        self.assertEqual(full["balanced_gathering"], 1)

    def test_stochastic_evaluation_keeps_external_cases_paired(self):
        normal, tr_normal = evaluate(self.agents, self.bank, 271, n=512, task="curriculum",
                                     greedy=False, trace=True)
        shuffled, tr_shuffle = evaluate(self.agents, self.bank, 271, n=512, task="curriculum",
                                        mode="shuffle", greedy=False, trace=True)
        self.assertEqual(normal["balanced_gathering"], 1)
        self.assertEqual(normal["external_cases_sha256"], shuffled["external_cases_sha256"])
        self.assertEqual([x["sent"] for x in tr_normal], [x["sent"] for x in tr_shuffle])

    def test_shuffle_preserves_each_senders_clock_conditional_counts(self):
        remaining = np.tile(np.arange(1, 17), 20)
        sent = np.random.default_rng(17).integers(5, size=(320, 2))
        delivered = _shuffle(sent, remaining, np.random.default_rng(27))
        self.assertTrue(np.any(sent != delivered))
        for time in range(1, 17):
            for sender in range(2):
                np.testing.assert_array_equal(np.bincount(sent[remaining == time, sender], minlength=5),
                                              np.bincount(delivered[remaining == time, sender], minlength=5))

    def test_fixed_observation_intervention_changes_resources_and_success(self):
        result = message_intervention(self.agents, self.bank, 93, n=2048, task="curriculum")
        for direction in result["directions"]:
            sensitivity = direction["observed_symbol_sensitivity"]
            self.assertEqual(sensitivity["fraction_cases_resource_changes_for_some_used_symbol"], 1)
            self.assertAlmostEqual(sensitivity["mean_food_probability_range"], 1)
            for effect in direction["interventions"]:
                if effect["replacement_symbol"] in (1, 2):
                    self.assertEqual(effect["resource_flip_rate"], 1)
                    self.assertEqual(effect["baseline_success"], 1)
                    self.assertEqual(effect["intervened_success"], 0)


if __name__ == "__main__":
    unittest.main()
