"""Environment invariants for the visual communication pilot.

Run from the workspace root with:
python -m unittest discover -s pilot_round1/tests -p 'test_env.py' -v
"""

from __future__ import annotations

import unittest

import numpy as np

from pilot_round1.env import VisualWorld


class VisualWorldTests(unittest.TestCase):
    def assert_batches_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)

    def test_shapes_dtypes_ranges_and_correct_indices(self):
        batch = VisualWorld(12).sample(37)
        self.assertEqual(batch["target_images"].shape, (37, 1, 32, 32))
        self.assertEqual(batch["candidate_images"].shape, (37, 4, 1, 32, 32))
        for name in ("target_images", "candidate_images"):
            self.assertEqual(batch[name].dtype, np.float32)
            self.assertTrue(np.isfinite(batch[name]).all())
            self.assertGreaterEqual(float(batch[name].min()), 0.0)
            self.assertLessEqual(float(batch[name].max()), 1.0)
        for name in ("target_ids", "candidate_ids", "correct_indices"):
            self.assertEqual(batch[name].dtype, np.int64)
        np.testing.assert_array_equal(
            np.sort(batch["candidate_ids"], axis=1),
            np.tile(np.arange(4), (37, 1)),
        )
        np.testing.assert_array_equal(
            batch["candidate_ids"][np.arange(37), batch["correct_indices"]],
            batch["target_ids"],
        )

    def test_seed_reproduces_entire_sampling_sequence(self):
        left, right = VisualWorld(901), VisualWorld(901)
        for batch_size in (3, 19, 1):
            self.assert_batches_equal(left.sample(batch_size), right.sample(batch_size))

    def test_rng_state_restores_sequence_without_aliasing(self):
        world = VisualWorld(40)
        world.sample(5)
        state = world.get_rng_state()
        expected = world.sample(9)
        world.sample(2)
        world.set_rng_state(state)
        self.assert_batches_equal(expected, world.sample(9))
        # Obtaining a snapshot must not hand callers mutable internal state.
        original = world.get_rng_state()
        unrelated_copy = world.get_rng_state()
        unrelated_copy["state"]["state"] = 0
        self.assertEqual(world.get_rng_state(), original)

    def test_external_probe_rng_does_not_advance_training_stream(self):
        world = VisualWorld(5)
        state = world.get_rng_state()
        probe = world.render_class(2, 7, np.random.default_rng(19))
        repeat = world.render_class(2, 7, np.random.default_rng(19))
        np.testing.assert_array_equal(probe, repeat)
        self.assertEqual(state, world.get_rng_state())

    def test_matched_views_are_not_copied_pixels(self):
        batch = VisualWorld(95).sample(128)
        matching_views = batch["candidate_images"][
            np.arange(128), batch["correct_indices"]
        ]
        identical = np.all(batch["target_images"] == matching_views, axis=(1, 2, 3))
        self.assertFalse(identical.any())
        # The random nuisance variation also applies within a single class.
        for class_id in range(4):
            images = VisualWorld(78).render_class(class_id, 4)
            self.assertFalse(np.array_equal(images[0], images[1]))

    def test_targets_and_positions_have_no_fixed_schedule(self):
        world = VisualWorld(409)
        targets, positions, candidates = [], [], []
        # Modest chunks avoid allocating a large image dataset for a metadata test.
        for _ in range(16):
            batch = world.sample(256)
            targets.append(batch["target_ids"])
            positions.append(batch["correct_indices"])
            candidates.append(batch["candidate_ids"])
        targets = np.concatenate(targets)
        positions = np.concatenate(positions)
        candidates = np.concatenate(candidates)
        for values in (targets, positions):
            fractions = np.bincount(values, minlength=4) / len(values)
            self.assertTrue(np.all(np.abs(fractions - 0.25) < 0.04), fractions)
        # A fixed first-position guess should obtain chance performance.
        self.assertLess(abs(np.mean(positions == 0) - 0.25), 0.04)
        # Sampling is with replacement: repeats are allowed, without alternation.
        repeat_rate = np.mean(targets[1:] == targets[:-1])
        self.assertLess(abs(repeat_rate - 0.25), 0.04)
        self.assertGreater(len(np.unique(candidates, axis=0)), 20)
        # Candidate order should not encode which category is the target.
        for class_id in range(4):
            first = candidates[targets == class_id, 0]
            fractions = np.bincount(first, minlength=4) / len(first)
            self.assertTrue(np.all(np.abs(fractions - 0.25) < 0.07), fractions)

    def test_empty_batches_and_invalid_arguments(self):
        world = VisualWorld(3)
        empty = world.sample(0)
        self.assertEqual(empty["target_images"].shape, (0, 1, 32, 32))
        self.assertEqual(empty["candidate_images"].shape, (0, 4, 1, 32, 32))
        self.assertEqual(world.render_class(0, 0).shape, (0, 1, 32, 32))
        for size in (-1,):
            with self.assertRaises(ValueError):
                world.sample(size)
        for size in (1.5, True):
            with self.assertRaises(TypeError):
                world.sample(size)
        with self.assertRaises(ValueError):
            world.render_class(4, 1)
        with self.assertRaises(ValueError):
            world.render_class(-1, 1)


if __name__ == "__main__":
    unittest.main()
