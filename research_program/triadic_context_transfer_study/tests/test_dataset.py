"""No policy reads or forwards: set/index/weight/native-rule checks only."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from research_program.triadic_context_transfer_study import dataset as d


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared, cls.sources = d.load_source()
        cls.part = cls.prepared['partitions']['new_needs_and_layouts']
        cls.built = d.make_spec(cls.part)
        cls.a, cls.m = cls.built['arrays'], cls.built['metadata']

    def test_complete_case_order_and_all_four_counts(self):
        expected = {'train': 209952, 'new_needs': 77760, 'new_layouts': 69984, 'new_needs_and_layouts': 25920}
        for part, count in expected.items():
            with self.subTest(part=part):
                spec = self.prepared['partitions'][part]
                result = d.make_spec(spec)
                a, m = result['arrays'], result['metadata']
                original = d.source_dataset.content_pairs(spec)['rows']
                selected = [row for row in original if row['classification'] == 'content']
                self.assertEqual(m['content_cases'], selected)
                self.assertEqual(len(a['case_index']), count)
                self.assertEqual(a['eligible'].shape, (len(spec['layouts']) - 1, count))
                self.assertTrue(np.all(a['eligible_donor_count'] > 0))
                self.assertEqual(len(np.unique(a['case_index'])), len(selected))
                self.assertTrue(np.all(np.bincount(a['case_index']) == len(spec['layouts']) * 6))

    def test_all_other_mappings_are_bijections_without_self_and_need_independent(self):
        for split in ('train', 'new_layouts'):
            layouts = self.prepared['partitions'][split]['layouts']
            mappings, _ = d.layout_mapping(layouts)
            n = len(layouts)
            for mapping in mappings:
                self.assertEqual(sorted(mapping.tolist()), list(range(n)))
                self.assertFalse(np.any(mapping == np.arange(n)))
            for recipient in range(n):
                self.assertEqual(set(mappings[:, recipient]), set(range(n)) - {recipient})
        other = d.make_spec(self.prepared['partitions']['new_layouts'])
        np.testing.assert_array_equal(other['arrays']['donor_layout_index'], self.a['donor_layout_index'])

    def test_index_decoding_preserves_needs_owners_and_partition(self):
        a, spec = self.a, self.part
        nl = len(spec['layouts'])
        original = np.asarray([r['endpoint_need_indices'] for r in self.m['content_cases']])[a['case_index']]
        for ids in (a['endpoint_indices'], *a['donor_endpoint_indices']):
            np.testing.assert_array_equal(ids // (nl * 6), original)
            np.testing.assert_array_equal(ids % 6, np.broadcast_to(a['owner_index'][:, None], ids.shape))
            self.assertTrue(np.all(ids < spec['world_count']))
        for k, ids in enumerate(a['donor_endpoint_indices']):
            expected = a['donor_layout_index'][k, a['recipient_layout_index']]
            np.testing.assert_array_equal((ids // 6) % nl, np.broadcast_to(expected[:, None], ids.shape))

    def test_correct_actions_native_replay_and_truth_copy_exclusion(self):
        a, spec = self.a, self.part
        # Fixed arithmetic sample, no behavioral filter; covers all axes and actors.
        for row in range(0, len(a['axis']), 113):
            for donor in [-1, *range(a['donor_endpoint_indices'].shape[0])]:
                indices = a['endpoint_indices'][row] if donor < 0 else a['donor_endpoint_indices'][donor, row]
                actions = a['correct_actions'][row] if donor < 0 else a['donor_correct_actions'][donor, row]
                states = d.source_dataset.pack_states(spec, indices)
                for endpoint, packed in enumerate(states):
                    state = d.env.State(packed[:3], packed[3:7], packed[7:])
                    chosen = {agent: d.env.all_actions(agent)[int(actions[endpoint, actor])]
                              for actor, agent in enumerate(d.env.AGENTS)}
                    self.assertEqual(d.env.settle(state, chosen)['reward'], 1)
                    self.assertEqual(len(d.env.full_success_plans(state.needs, state.layout)), 1)
                if donor >= 0 and a['eligible'][donor, row]:
                    listener = a['listener'][row]
                    self.assertTrue(np.all(actions[:, listener] != a['correct_actions'][row, :, listener]))

    def test_eligibility_is_exact_both_material_site_change(self):
        a = self.a
        layouts = self.part['layouts']
        for k in range(a['eligible'].shape[0]):
            donor_layout = a['donor_layout_index'][k, a['recipient_layout_index']]
            for row in range(0, len(a['axis']), 29):
                rlayout = layouts[a['recipient_layout_index'][row]]
                dlayout = layouts[donor_layout[row]]
                expected = [rlayout.index(int(m)) != dlayout.index(int(m)) for m in a['target_materials'][row]]
                self.assertEqual(a['target_site_changed'][k, row].tolist(), expected)
                self.assertEqual(bool(a['eligible'][k, row]), all(expected))
        np.testing.assert_array_equal(a['eligible'].sum(0), a['eligible_donor_count'])

    def test_primary_and_all_other_weights_keep_every_recipient_uniform(self):
        a = self.a
        primary_direction = a['base_within_axis_weight'][None] * a['eligible'] / a['eligible_donor_count'][None] / 2
        np.testing.assert_allclose(primary_direction.sum(0) * 2, a['base_within_axis_weight'], rtol=0, atol=1e-18)
        for axis in range(3):
            mask = a['axis'] == axis
            self.assertAlmostEqual(float(primary_direction[:, mask].sum() * 2), 1, places=12)
            self.assertAlmostEqual(float(a['base_within_axis_weight'][mask].sum()), 1, places=12)
            for sender in range(3):
                for listener in range(3):
                    if sender == listener:
                        continue
                    stratum = mask & (a['sender'] == sender) & (a['listener'] == listener)
                    self.assertAlmostEqual(float(a['base_within_axis_weight'][stratum].sum()), 1 / 6, places=12)

    def test_listener_view_equal_matches_official_observation_not_full_input(self):
        a = self.a
        for row in range(0, len(a['axis']), 139):
            listener = int(a['listener'][row])
            ids = a['endpoint_indices'][row]
            packed = d.source_dataset.pack_states(self.part, ids)
            states = [d.env.State(p[:3], p[3:7], p[7:]) for p in packed]
            for mode in ('PL', 'LL'):
                self.assertEqual(d.env.observe(states[0], d.env.AGENTS[listener], information=mode),
                                 d.env.observe(states[1], d.env.AGENTS[listener], information=mode))
            for k in range(a['eligible'].shape[0]):
                donor = d.source_dataset.pack_states(self.part, a['donor_endpoint_indices'][k, row])
                ds = d.env.State(donor[0, :3], donor[0, 3:7], donor[0, 7:])
                self.assertNotEqual(d.env.observe(states[0], d.env.AGENTS[listener], information='PL'),
                                    d.env.observe(ds, d.env.AGENTS[listener], information='PL'))
                self.assertEqual(bool(a['view_equal_LL'][k, row]),
                    d.env.observe(states[0], d.env.AGENTS[listener]) == d.env.observe(ds, d.env.AGENTS[listener]))

    def test_input_immutability_and_refusal_to_overwrite(self):
        before = deepcopy(self.part)
        d.make_spec(self.part)
        self.assertEqual(before, self.part)
        with TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                d.prepare(Path(temp))


if __name__ == '__main__':
    unittest.main()
