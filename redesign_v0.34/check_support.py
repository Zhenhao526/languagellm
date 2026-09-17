import unittest

import numpy as np

import support


class Tests(unittest.TestCase):
    def test_design_and_schedule(self):
        self.assertEqual(support.CONDITIONS, ("single_full", "dual_same_full", "dual_complementary"))
        self.assertEqual(len(support.team_slots("single_full")), 4)
        self.assertEqual(len(support.team_slots("dual_same_full")), 4)
        self.assertEqual({x[0] for x in support.team_slots("dual_same_full")}, {0, 1, 2, 3})
        self.assertEqual(support.team_slots("dual_complementary", step=0), support.team_slots("dual_complementary", schedule="A"))
        self.assertNotEqual(support.team_slots("dual_complementary", step=1), support.team_slots("dual_complementary", schedule="A"))
        self.assertEqual(set(x[0] for x in support.team_slots("dual_complementary", schedule="B")), {0, 1, 2, 3})
        self.assertEqual(set(x[1] for x in support.team_slots("dual_complementary", schedule="B")), {0, 1, 2, 3})
        self.assertEqual(set(x[2] for x in support.team_slots("dual_complementary", schedule="B")), {0, 1, 2, 3})

    def test_graphs(self):
        for p in (1, 2, 3):
            for c in support.CONDITIONS:
                g = support.groups(p, c)
                self.assertEqual(len(g["train12"]), 12)
                self.assertEqual(len(g["target12"]), 12)
                self.assertEqual(len(np.intersect1d(g["train12"], g["target12"])), 0)

    def test_fixture_pairing(self):
        table = {"map_id": np.zeros(720, np.int64), "photo_ids": np.asarray([(f, w) for _ in range(30) for f in range(6) for w in range(2)] * 2, np.int64)}
        table["positions"] = support.MAPS[table["map_id"]]
        table["shown"] = np.repeat(np.arange(2), 360)
        # Replace the synthetic map IDs with a legal complete table from the
        # inherited layout while testing only fixture shape and pairing.
        table["map_id"] = np.tile(np.arange(30), 24)
        table["positions"] = support.MAPS[table["map_id"]]
        a = support.fixture(99528, 1, 0, 7, "single_full", table)
        for condition in support.CONDITIONS:
            b = support.fixture(99528, 1, 0, 7, condition, table)
            np.testing.assert_array_equal(a["indices"], b["indices"])
            np.testing.assert_array_equal(a["uniforms"], b["uniforms"])
            self.assertEqual(a["indices"].shape, (240,))
            self.assertEqual(a["uniforms"].shape, (240, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
