import unittest

import numpy as np
import torch
from torch import nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "redesign_v0.8"))
from camp import CampAgent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "redesign_v0.21"))
import social_model as communication
import dual


class Tests(unittest.TestCase):
    def test_loss_roles_and_shapes(self):
        torch.set_num_threads(1)
        sender_f = CampAgent(nn.Sequential(nn.Linear(8, 64)), 7, 2)
        sender_w = CampAgent(nn.Sequential(nn.Linear(8, 64)), 7, 2)
        receiver = CampAgent(nn.Sequential(nn.Linear(8, 64)), 7, 2)
        for i, agent in enumerate((sender_f, sender_w, receiver)):
            communication.reset_communication(agent, 91000 + i)
        h_f = torch.randn(12, 96)
        h_w = torch.randn(12, 96)
        uniforms = np.random.default_rng(91001).random((12, 4), dtype=np.float32)
        positions = np.asarray([(i % 6, (i + 1) % 6) for i in range(12)], np.int64)
        lf, lw, lr, trace = dual.dual_loss(sender_f, sender_w, receiver, h_f, h_w, uniforms, positions, .02)
        self.assertEqual(tuple(trace["messages"].shape), (12, 2))
        self.assertEqual(tuple(trace["action_logits"].shape), (12, 2, 6))
        self.assertTrue(torch.isfinite(lf + lw + lr))
        grads = torch.autograd.grad(lf, communication.trainable_groups(sender_f)["sender"], retain_graph=True, allow_unused=True)
        used = [g for g in grads if g is not None]
        self.assertTrue(used and all(torch.isfinite(g).all() for g in used))


if __name__ == "__main__":
    unittest.main(verbosity=2)
