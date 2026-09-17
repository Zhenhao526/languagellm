"""Exact counting/selection and packet coding, using only synthetic symbols."""
from fractions import Fraction
from unittest import TestCase

import numpy as np

from research_program.triadic_position_reuse_study import discovery as d


def count_fixture():
    total = np.ones((3, 3, 3), dtype=np.int64)
    changes = np.zeros((3, 3, 3, 8), dtype=np.int64)
    for actor in range(3):
        total[actor, :, actor] = 0
    return changes, total


def messages_from_counts(changes, totals):
    messages, endpoints, axes, senders, listeners = [], [], [], [], []
    for actor in range(3):
        for axis in range(3):
            for listener in range(3):
                if actor == listener:
                    continue
                for index in range(int(totals[actor, axis, listener])):
                    before = np.zeros((2, 3, 4), dtype=np.int8)
                    after = before.copy()
                    for position in range(8):
                        if index < changes[actor, axis, listener, position]:
                            after[position//4, actor, position % 4] = 1
                    first = len(messages)
                    messages.extend((before, after))
                    endpoints.append((first, first+1))
                    axes.append(axis); senders.append(actor); listeners.append(listener)
    return (dict(endpoint_indices=np.asarray(endpoints), axis=np.asarray(axes),
                 sender=np.asarray(senders), listener=np.asarray(listeners)),
            np.asarray(messages))


def as_fraction(payload):
    return Fraction(payload['numerator'], payload['denominator'])


class DiscoveryTests(TestCase):
    def test_listener_strata_are_equal_not_pooled_and_integer_counts_survive(self):
        change, total = count_fixture()
        total[0, 0, 1] = 1; total[0, 0, 2] = 3
        change[0, 0, 1, 0] = 1
        change[0, 0, 2, 1] = 2
        spec, messages = messages_from_counts(change, total)
        before = messages.copy()
        result = d.select_positions(spec, messages)
        self.assertEqual(as_fraction(result['exact_response_rates'][0][0][0]), Fraction(1, 2))
        self.assertEqual(as_fraction(result['exact_response_rates'][0][0][1]), Fraction(1, 3))
        self.assertEqual(result['selected_positions'][0][0], 0)
        # Pooling listener rows would reverse these two candidates:1/4 vs2/4.
        self.assertGreater(Fraction(2, 4), Fraction(1, 4))
        np.testing.assert_array_equal(result['stratum_change_counts'], change)
        np.testing.assert_array_equal(result['stratum_pair_counts'], total)
        np.testing.assert_array_equal(messages, before)

    def test_mathematical_tie_uses_smallest_position_and_saves_all_maxima(self):
        change, total = count_fixture()
        total[0, 0, 1] = 3; total[0, 0, 2] = 6
        change[0, 0, 1, 0] = 1
        change[0, 0, 2, 1] = 2
        result = d.selection_from_counts(change, total)
        self.assertEqual(as_fraction(result['exact_specificity_scores'][0][0][0]), Fraction(1, 6))
        self.assertEqual(as_fraction(result['exact_specificity_scores'][0][0][1]), Fraction(1, 6))
        self.assertEqual(result['maximizing_positions'][0][0], [0, 1])
        self.assertEqual(result['selected_positions'][0][0], 0)

    def test_smaller_than_previous_tolerance_is_still_strictly_greater(self):
        change, total = count_fixture()
        for listener in (1, 2):
            total[0, 0, listener] = 10**13
            change[0, 0, listener, 0] = 1
            change[0, 0, listener, 1] = 2
        result = d.selection_from_counts(change, total)
        scores = result['exact_specificity_scores'][0][0]
        self.assertEqual(as_fraction(scores[1])-as_fraction(scores[0]), Fraction(1, 10**13))
        self.assertEqual(result['maximizing_positions'][0][0], [1])
        self.assertEqual(result['selected_positions'][0][0], 1)
        self.assertNotIn('tie_tolerance', result)

    def test_negative_maximum_and_shared_position_are_allowed(self):
        change, total = count_fixture()
        for actor in range(3):
            for listener in range(3):
                if listener != actor:
                    change[actor, 1:, listener, :] = 1
        result = d.selection_from_counts(change, total)
        self.assertEqual(result['selected_positions'], [[0, 0, 0]]*3)
        self.assertEqual(result['maximizing_positions'][0][0], list(range(8)))
        self.assertEqual([as_fraction(v) for v in result['exact_specificity_scores'][0][0]], [Fraction(-1)]*8)

    def test_invalid_missing_or_impossible_counts_and_bad_axis_are_rejected(self):
        change, total = count_fixture()
        fixtures = []
        impossible = change.copy(); impossible[0, 0, 1, 0] = 2
        fixtures.append((impossible, total))
        missing = total.copy(); missing[0, 0, 1] = 0
        fixtures.append((change, missing))
        self_listener = total.copy(); self_listener[0, 0, 0] = 1
        fixtures.append((change, self_listener))
        fixtures.append((change.astype(float), total))
        for changes, totals in fixtures:
            with self.subTest(kind=str(totals.dtype), maximum=int(changes.max())):
                with self.assertRaises(ValueError):
                    d.selection_from_counts(changes, totals)
        spec, messages = messages_from_counts(change, total)
        spec['axis'][0] = 3
        with self.assertRaises(ValueError):
            d.select_positions(spec, messages)

    def test_packet_codes_unique_base8_order_and_legal_validation(self):
        numbers = np.arange(4096, dtype=np.uint32)
        digits = ((numbers[:, None] // (8**np.arange(4))) % 8).astype(np.int8)
        packet = np.zeros((4096, 2, 4), dtype=np.int8)
        packet[:, 0] = digits
        np.testing.assert_array_equal(d.packet_codes(packet), numbers)
        packet[:, 1] = digits; packet[:, 0] = 0
        np.testing.assert_array_equal(d.packet_codes(packet), numbers*4096)
        samples = np.random.default_rng(613).integers(0, 8, (128, 2, 4), dtype=np.int8)
        codes = d.packet_codes(samples)
        decoded = ((codes[:, None] // (8**np.arange(8))) % 8).reshape(-1, 2, 4)
        np.testing.assert_array_equal(decoded, samples)
        self.assertEqual(int(d.packet_codes(np.full((2, 4), 7, dtype=np.int8))), 8**8-1)
        for invalid in (np.zeros((2, 4), dtype=float), np.zeros((2, 4), dtype=bool),
                        np.full((2, 4), -1), np.full((2, 4), 8), np.zeros((8,), dtype=int)):
            with self.subTest(shape=invalid.shape, dtype=str(invalid.dtype)):
                with self.assertRaises(ValueError):
                    d.packet_codes(invalid)

    def test_training_sets_include_all419904_worlds_not_discovery_references(self):
        messages = np.zeros((419904, 2, 3, 4), dtype=np.int8)
        messages[0, 0, 0, 0] = 1
        # This packet lies outside a hypothetical referenced discovery subset.
        messages[-1, 1, 2, 3] = 7
        codes = d.training_code_sets(messages)
        np.testing.assert_array_equal(codes[0], [0, 1])
        np.testing.assert_array_equal(codes[1], [0])
        np.testing.assert_array_equal(codes[2], [0, 7*8**7])
        np.testing.assert_array_equal(d.packet_codes(messages[:18, :, 2]), 0)
        with self.assertRaises(ValueError):
            d.training_code_sets(messages[:18])
