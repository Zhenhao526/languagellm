"""Synthetic tests only: no experimental files, checkpoints or model calls."""
import itertools
import argparse
import hashlib
import json
import sys
from pathlib import Path
import unittest

import numpy as np

import structural_assay as assay


def toy(reverse=False):
    maps = np.asarray(list(itertools.permutations(range(6), 2)), np.int64)
    rows = list(itertools.product((0, 1), ((10, 20), (11, 21)), range(30)))
    mids = np.asarray([r[2] for r in rows], np.int64)
    positions = maps[mids]
    tokens = positions[:, ::-1].copy() if reverse else positions.copy()
    # All 49 messages have a legal greedy action pair, including unused digit 6.
    actions = np.asarray([(min(a, 5), min(b, 5))
                          for a, b in itertools.product(range(7), repeat=2)], np.int64)
    if reverse:
        actions = actions[:, ::-1]
    logits = np.full((49, 2, 6), -8.0, np.float32)
    for code, goal in itertools.product(range(49), range(2)):
        logits[code, goal, actions[code, goal]] = 8
    return {"map_id": mids, "positions": positions,
            "photo_ids": np.asarray([r[1] for r in rows], np.int64),
            "shown": np.asarray([r[0] for r in rows], np.int64),
            "tokens": tokens, "receiver_logits": logits}


def slow_reference(raw, p, permutations):
    """Target-first Python enumeration; no production donor/compose helpers."""
    old = set(map(int, assay.world.partition(p)["old"]))
    code = raw["tokens"][:, 0] * 7 + raw["tokens"][:, 1]
    action = raw["receiver_logits"].argmax(-1)
    values = {}
    for assignment in assay.ASSIGNMENTS:
        expected = []
        for permutation in (None, *permutations):
            inverse_lookup = None if permutation is None else list(permutation)
            row_rates = []
            for t in range(len(code)):
                # Match both photo IDs and the mask before enumerating donors.
                context = [j for j in range(len(code))
                    if raw["map_id"][j] in old
                    and raw["shown"][j] == raw["shown"][t]
                    and tuple(raw["photo_ids"][j]) == tuple(raw["photo_ids"][t])]
                fs = [j for j in context if raw["positions"][j, 0] == raw["positions"][t, 0]
                      and raw["positions"][j, 1] != raw["positions"][t, 1]]
                ws = [j for j in context if raw["positions"][j, 1] == raw["positions"][t, 1]
                      and raw["positions"][j, 0] != raw["positions"][t, 0]]
                good = []
                for f, w in itertools.product(fs, ws):
                    x, y = (f, w) if assignment == "food_water" else (w, f)
                    first, second = int(code[x]), int(code[y])
                    if permutation is not None:
                        first, second = int(permutation[first]), int(permutation[second])
                    joined = 7 * (first // 7) + second % 7
                    if permutation is not None:
                        joined = inverse_lookup.index(joined)
                    good.append(tuple(action[joined]) == tuple(raw["positions"][t]))
                row_rates.append(sum(good) / len(good))
            expected.append(row_rates)
        values[assignment] = np.asarray(expected)
    return values


class StructuralAssayTests(unittest.TestCase):
    def test_factorized_positive_and_reversed_assignment(self):
        references = assay.make_references()
        self.assertEqual(references.shape, (200, 49))
        for reversed_code in (False, True):
            score = assay.evaluate_protocol(toy(reversed_code), 1, references)
            winning = "water_food" if reversed_code else "food_water"
            losing = "food_water" if reversed_code else "water_food"
            for group in assay.GROUPS:
                good = score[winning][group]["pooled"]
                self.assertEqual(good["natural_J"], 1.0)
                self.assertEqual(good["recombined_J"], 1.0)
                self.assertLess(good["recoding"]["mean"], 0.2)
                self.assertEqual(score[losing][group]["pooled"]["recombined_J"], 0.0)

    def test_donors_are_old_and_match_both_context_fields(self):
        raw = toy()
        for p in (1, 2, 3):
            t, f, w, counts = assay.donor_trials(raw, p)
            old = set(map(int, assay.partitions(p)["old18"]))
            self.assertTrue(set(map(int, raw["map_id"][f])).issubset(old))
            self.assertTrue(set(map(int, raw["map_id"][w])).issubset(old))
            for donors in (f, w):
                np.testing.assert_array_equal(raw["photo_ids"][donors], raw["photo_ids"][t])
                np.testing.assert_array_equal(raw["shown"][donors], raw["shown"][t])
            self.assertTrue(np.all(raw["positions"][f, 0] == raw["positions"][t, 0]))
            self.assertTrue(np.all(raw["positions"][f, 1] != raw["positions"][t, 1]))
            self.assertTrue(np.all(raw["positions"][w, 1] == raw["positions"][t, 1]))
            self.assertTrue(np.all(raw["positions"][w, 0] != raw["positions"][t, 0]))
            np.testing.assert_array_equal(counts, np.where(np.isin(raw["map_id"], list(old)), 4, 9))

    def test_constant_message_is_a_recoding_invariant_negative_control(self):
        raw = toy()
        raw["tokens"][:] = (0, 1)
        result = assay.evaluate_protocol(raw, 2)
        for assignment, group, mask in itertools.product(assay.ASSIGNMENTS, assay.GROUPS, assay.MASKS):
            c = result[assignment][group][mask]
            self.assertAlmostEqual(c["recombined_J"], c["natural_J"])
            np.testing.assert_allclose(c["recoding"]["rates"], c["recombined_J"], atol=0, rtol=0)

    def test_random_protocol_matches_separate_slow_enumeration(self):
        raw = toy()
        rng = np.random.default_rng(84024)
        raw["tokens"] = rng.integers(0, 7, raw["tokens"].shape)
        raw["receiver_logits"] = rng.normal(size=(49, 2, 6))
        permutations = assay.make_references()[:3]
        fast, details = assay.evaluate_protocol(raw, 3, permutations, return_details=True)
        slow = slow_reference(raw, 3, permutations)
        for assignment in assay.ASSIGNMENTS:
            expected = slow[assignment]
            actual = np.vstack([details[assignment]["target_rates"],
                                details[assignment]["reference_target_rates"]])
            np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
            for group, ids in assay.partitions(3).items():
                select = np.isin(raw["map_id"], ids)
                c = fast[assignment][group]["pooled"]
                self.assertAlmostEqual(c["recombined_J"], np.mean(np.asarray(expected)[0, select]))
                np.testing.assert_allclose(c["recoding"]["rates"],
                    np.asarray(expected)[1:, select].mean(-1), atol=1e-15, rtol=0)

    def test_missing_duplicate_and_invalid_permutation_are_rejected(self):
        raw = toy()
        for indices in (np.arange(len(raw["tokens"]) - 1), np.r_[np.arange(len(raw["tokens"])), 0]):
            broken = {k: v if k == "receiver_logits" else v[indices] for k, v in raw.items()}
            with self.assertRaises(ValueError):
                assay.evaluate_protocol(broken, 1)
        invalid = np.zeros((1, 49), np.int64)
        with self.assertRaises(ValueError):
            assay.evaluate_protocol(raw, 1, invalid)

    def test_reference_quantiles_are_computed_after_cell_means(self):
        raw = toy()
        a = assay.evaluate_protocol(raw, 1, assay.make_references()[:3])
        b = assay.evaluate_protocol(toy(True), 1, assay.make_references()[:3])
        # The noncommuting example protects against averaging interval endpoints.
        ac, bc = a["food_water"]["new12"]["pooled"], b["food_water"]["new12"]["pooled"]
        ac["recoding"]["rates"] = [0.0, 0.0, 1.0]
        bc["recoding"]["rates"] = [1.0, 0.0, 0.0]
        merged = assay.average_scores([a, b])["food_water"]["new12"]["pooled"]
        np.testing.assert_array_equal(merged["recoding"]["rates"], [0.5, 0.0, 0.5])
        self.assertAlmostEqual(merged["recoding"]["quantiles"]["0.975"], 0.5)


def audit_development():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--partition", required=True, type=int)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()
    if not re_development_path(args.protocol):
        raise ValueError("This bounded replay accepts only a development 0040 protocol")
    raw = assay.read_npz(args.protocol)
    permutations = assay.make_references()[:3]
    _, actual = assay.evaluate_protocol(raw, args.partition, permutations, return_details=True)
    expected = slow_reference(raw, args.partition, permutations)
    nvalues, max_error = 0, 0.0
    for assignment in assay.ASSIGNMENTS:
        values = np.vstack([actual[assignment]["target_rates"],
                            actual[assignment]["reference_target_rates"]])
        np.testing.assert_allclose(values, expected[assignment], atol=0, rtol=0)
        nvalues += values.size
        max_error = max(max_error, float(np.abs(values - expected[assignment]).max()))
    payload = {"status": "passed", "development_only": True,
        "protocol": str(args.protocol.resolve()), "protocol_sha256": assay.sha(args.protocol),
        "targets": len(raw["tokens"]), "assignments": list(assay.ASSIGNMENTS),
        "reference_indices": [0, 1, 2], "identity_included": True,
        "target_rate_comparisons": nvalues, "max_absolute_error": max_error,
        "assay_sha256": assay.sha(Path(assay.__file__)),
        "audit_source_sha256": assay.sha(__file__),
        "model_calls": 0, "training_updates": 0}
    args.audit_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False))


def re_development_path(path):
    return path.name in ("protocol_0040_d0.npz", "protocol_0040_d1.npz")


if __name__ == "__main__":
    if "--protocol" in sys.argv:
        audit_development()
    else:
        unittest.main(verbosity=2)
