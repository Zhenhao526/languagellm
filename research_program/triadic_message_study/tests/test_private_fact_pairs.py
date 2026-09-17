import unittest

from research_program.triadic_message_study import private_fact_pairs as p


class PrivateFactPairsTests(unittest.TestCase):
    def test_24_profiles_have_valid_mutual_actions(self):
        self.assertEqual(len(p.PROFILES), 24)
        for row in p.PROFILES:
            active = [a for a in range(3) if row[a]]
            self.assertEqual(len(active), 2)
            i, j = active
            ai, aj = p.MENUS[i][row[i]], p.MENUS[j][row[j]]
            self.assertEqual((ai['partner'], aj['partner']), (p.env.AGENTS[j], p.env.AGENTS[i]))
            self.assertEqual((ai['site'], ai['destination']), (aj['site'], aj['destination']))

    def test_semantic_predicates_all_96_combinations(self):
        for need in range(12):
            for material in range(4):
                for destination in range(2):
                    self.assertEqual(p.independent_accepts(need, material, destination),
                                     p.env.accepts(need, material, destination))

    def test_known_pair_and_both_actor_permutations(self):
        low = p.env.State((0, 6, 9), (2, 3, 0, 1))
        high = p.swapped_private_state(low, 0)
        self.assertEqual(high.layout, (2, 3, 1, 0))
        self.assertEqual(p.env.observe(low, 'A', shared_needs=False),
                         p.env.observe(high, 'A', shared_needs=False))
        self.assertFalse(set(p.fullsuccess_projections(low.needs, low.layout)[0]) &
                         set(p.fullsuccess_projections(high.needs, high.layout)[0]))
        self.assertEqual(p.swapped_private_state(high, 0), low)
        self.assertNotEqual(p.env.observe(low, 'A', shared_needs=True, full_information=True),
                            p.env.observe(high, 'A', shared_needs=True, full_information=True))

    def test_same_partition_filter_and_dedup(self):
        layouts = [(2, 3, 0, 1), (2, 3, 1, 0)]
        prepared = {'partitions': {'example': {'needs': [(0, 6, 9)], 'layouts': layouts,
            'private_sites': [(1, 2, 3)], 'world_count': 2}}}
        result = p.build_pairs(prepared)
        self.assertEqual(result['total_pairs'], 1)
        self.assertEqual(result['rows'][0]['listener'], 'A')
        self.assertLess(result['rows'][0]['state_low'], result['rows'][0]['state_high'])
        prepared['partitions']['example'].update(layouts=layouts[:1], world_count=1)
        self.assertEqual(p.build_pairs(prepared)['total_pairs'], 0)

    def test_wait_in_fullsuccess_projection_prevents_false_necessity(self):
        # All three need wood; another two can solve while the listener waits.
        state = p.env.State((0, 0, 0), (2, 3, 0, 1))
        other = p.swapped_private_state(state, 0)
        a = p.fullsuccess_projections(state.needs, state.layout)[0]
        b = p.fullsuccess_projections(other.needs, other.layout)[0]
        self.assertIn(0, a)
        self.assertIn(0, b)
        self.assertTrue(set(a) & set(b))


if __name__ == '__main__':
    unittest.main()
