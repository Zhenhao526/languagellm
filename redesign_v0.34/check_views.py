import unittest

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "redesign_v0.28"))
import world
import views


class Tests(unittest.TestCase):
    def setUp(self):
        self.projected = torch.arange(12 * 64, dtype=torch.float32).reshape(12, 64) / 100.
        self.worlds = {"map_id": np.asarray([0, 1, 2, 3], np.int64), "photo_ids": np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], np.int64), "positions": np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], np.int64), "shown": np.asarray([0, 1, 0, 1], np.int64)}

    def test_full_delegates(self):
        expected = world.frames(self.projected, self.worlds)
        actual = views.frames(self.projected, self.worlds, "full")
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def test_masks(self):
        for view, visible in (("food_only", 0), ("water_only", 1)):
            frames, bits = views.frames(self.projected, self.worlds, view)
            self.assertEqual(tuple(frames.shape), (4, 2, 390))
            self.assertTrue(torch.equal(bits[:, 1], torch.zeros(4)))
            for row in range(4):
                hidden = 1 - visible
                hidden_site = self.worlds["positions"][row, hidden]
                self.assertTrue(torch.equal(frames[row, 0, hidden_site * 64:(hidden_site + 1) * 64], torch.zeros(64)))
                self.assertEqual(float(frames[row, 0, 384 + hidden_site]), 0.0)
                if self.worlds["shown"][row] == hidden:
                    self.assertTrue(torch.equal(frames[row, 1, hidden_site * 64:(hidden_site + 1) * 64], torch.zeros(64)))
                    self.assertEqual(float(frames[row, 1, 384 + hidden_site]), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
