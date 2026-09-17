import unittest
import hashlib
import numpy as np
from research_program.triadic_semantic_probe_study import audit_execution as a


def fake_probability(module, inputs):
    width = 17 if module % 3 == 2 else 32
    code = (inputs @ (np.arange(inputs.shape[1]) % 7 + 1)).astype(int)
    logits = -np.abs(np.arange(width)[None, :] - (code % width)[:, None]).astype(float)
    if width == 32:
        logits = logits.reshape(len(inputs), 4, 8)
    exp = np.exp(logits-logits.max(-1, keepdims=True))
    return exp/exp.sum(-1, keepdims=True)


class IndependentAuditTests(unittest.TestCase):
    def test_outgoing_override_keeps_own_packets_and_visibility(self):
        messages = np.arange(36, dtype=np.int8).reshape(3, 3, 4) % 8
        replacements = np.full((3, 4), 6, dtype=np.int8)
        senders = np.arange(3, dtype=np.int8)
        routed = a.recipient_views(messages, True, senders, replacements)
        for row in range(3):
            for viewer in range(3):
                self.assertTrue(np.array_equal(routed[row, viewer, -3:], np.ones(3)))
                for speaker in range(3):
                    block = routed[row, viewer, speaker*32:(speaker+1)*32].reshape(4, 8)
                    expected = replacements[row] if speaker == row and viewer != row else messages[row, speaker]
                    self.assertTrue(np.array_equal(block.argmax(-1), expected))
                    self.assertTrue(np.array_equal(block.sum(-1), np.ones(4)))
        silent = a.recipient_views(messages, False, senders, replacements)
        self.assertTrue(np.array_equal(silent, a.recipient_views(messages, False)))

    def test_second_window_override_does_not_rewrite_generated_history(self):
        x = np.zeros((3, 3, 54)); m = np.zeros((3, 2, 3, 4), dtype=np.int8)
        senders = np.arange(3, dtype=np.int8); donor = np.full((3, 2, 4), 7, dtype=np.int8)
        natural_second = a.intervention_forward(list(range(9)), x, m, senders, donor, (1,), False, fake_probability)
        changed = a.intervention_forward(list(range(9)), x, m, senders, donor, (1,), True, fake_probability)
        # The sender sees its actual freshly generated packet, never the donor.
        for row in range(3):
            own = changed['second_routes'][row, row, 32*row:32*(row+1)].reshape(4, 8).argmax(-1)
            self.assertTrue(np.array_equal(own, changed['messages'][row, 1, row]))
        self.assertTrue(np.array_equal(changed['messages'][:, 0], m[:, 0]))
        self.assertEqual(natural_second['action_probabilities'].shape, (3, 3, 17))

    def test_mask_retains_all_full_success_actions(self):
        # AB require wood/L; C requires fiber/R. Only AB can both be satisfied.
        states = np.tile([0, 0, 4, 0, 1, 2, 3, 1, 2, 3], (3, 1))
        masks = a.success_masks(states, np.arange(3))
        self.assertTrue(np.array_equal(masks, [34, 34, 1]))
        # Relocating the two wood objects changes sites, not semantic acceptance.
        states[:, 3:7] = [2, 3, 0, 1]
        self.assertTrue(np.array_equal(a.success_masks(states, np.arange(3)), [(1 << 9)+(1 << 13), (1 << 9)+(1 << 13), 1]))

    def test_exact_budget(self):
        b = a.expected_budgets()
        self.assertEqual(b['new_forward_worlds'], 6497280)
        self.assertEqual(b['new_network_samples'], 50181120)
        self.assertEqual(b['new_natural_npz']+b['new_intervention_npz'], 480)

    def test_hash_descriptor_and_unequal_axis_weights(self):
        values = np.array([1, 2], dtype=np.int8)
        expected = hashlib.sha256(b'{"dtype":"|i1","shape":[2]}\n'+values.tobytes()).hexdigest()
        self.assertEqual(a.array_sha(values), expected)
        spec = dict(classification=np.ones(6, dtype=np.int8), axis=np.repeat(np.arange(3), 2),
                    content_within_axis_weight=np.array([.25, .75, .5, .5, .75, .25]))
        result = a.metric_summary({'two_directions': np.array([[0, 1], [1, 0], [0, 0], [1, 1], [1, 1], [0, 0]])}, spec, 1)
        self.assertTrue(np.array_equal(result['two_directions']['by_axis'], [[.75, .25], [.5, .5], [.75, .75]]))
        self.assertTrue(np.allclose(result['two_directions']['macro'], [2/3, .5]))


if __name__ == '__main__':
    unittest.main()
