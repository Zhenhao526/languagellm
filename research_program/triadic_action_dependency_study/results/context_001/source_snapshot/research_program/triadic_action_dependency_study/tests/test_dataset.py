import hashlib
from itertools import permutations, product
import json
import unittest
import numpy as np

from research_program.triadic_action_dependency_study import dataset as d
from research_program.triadic_action_dependency_study import environment as e


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.prepared = d.make_prepared()

    def test_orbit_split_complete_disjoint_and_fixed_hash(self):
        p = self.prepared
        self.assertEqual(p['orbit_count'], 66)
        self.assertEqual(p['need_counts'], {'train': 3888, 'heldout': 1488})
        rows = p['need_orbits']; domain = set()
        self.assertEqual(sum(r['split'] == 'heldout' for r in rows), 17)
        for r in rows:
            members = set(map(tuple, r['members']))
            self.assertFalse(members & domain); domain.update(members)
            self.assertEqual(tuple(r['canonical']), min(members))
            self.assertEqual(members, set(d.demand_orbit(r['canonical'])))
            payload = 'action_dependency_need_split_v1|' + json.dumps(r['canonical'], separators=(',', ':'))
            self.assertEqual(r['ranking_sha256'], hashlib.sha256(payload.encode()).hexdigest())
        self.assertEqual(domain, set(e.support()))
        self.assertEqual([r['ranking_sha256'] for r in rows], sorted(r['ranking_sha256'] for r in rows))
        for part in d.PARTITIONS:
            for who in range(3):
                self.assertEqual({n[who] for n in p['partitions'][part]['needs']}, set(range(24)))

    def test_four_world_partitions_and_independent_layout_hash(self):
        p = self.prepared['partitions']
        self.assertEqual([p[k]['world_count'] for k in d.PARTITIONS], [419904, 160704, 139968, 53568])
        train = set(map(tuple, p['train']['layouts'])); held = set(map(tuple, p['new_layouts']['layouts']))
        self.assertFalse(train & held)
        self.assertEqual(train | held, set(permutations(range(4))))
        scored = []
        for layout in permutations(range(4)):
            text = 'action_dependency_layout_split_v1|' + json.dumps(layout, separators=(',', ':'))
            scored.append((hashlib.sha256(text.encode()).hexdigest(), layout))
        self.assertEqual(held, {v for h, v in sorted(scored)[:6]})
        self.assertEqual(sum(v['world_count'] for v in p.values()), 774144)

    def test_monitors_preserve_all_pair_endpoints(self):
        for part, spec in self.prepared['partitions'].items():
            backgrounds = spec['monitor_backgrounds']
            self.assertEqual(len({b['layout_index'] for b in backgrounds}), 2)
            expected = sorted((n * len(spec['layouts']) + b['layout_index']) * 6 + b['owner_index']
                              for n in range(len(spec['needs'])) for b in backgrounds)
            self.assertEqual(spec['monitor_indices'], expected)
            self.assertEqual(len(expected), len(set(expected)))
            self.assertEqual(len(expected), 2 * len(spec['needs']))
            ranks = []
            for li, layout in enumerate(spec['layouts']):
                for oi, owner in enumerate(spec['private_sites']):
                    payload = 'action_dependency_monitor_background_v1|' + json.dumps(layout, separators=(',', ':')) + '|' + json.dumps(owner, separators=(',', ':'))
                    ranks.append((hashlib.sha256(payload.encode()).hexdigest(), layout, owner, li, oi))
            chosen = []
            for h, l, o, li, oi in sorted(ranks):
                if li not in {x[0] for x in chosen}: chosen.append((li, oi))
                if len(chosen) == 2: break
            self.assertEqual(chosen, [(b['layout_index'], b['owner_index']) for b in backgrounds])

    def test_pack_and_uniform_sampler_boundaries(self):
        for spec in self.prepared['partitions'].values():
            ids = d.sample_indices(spec, np.array([[0., 0., 0.], [np.nextafter(1., 0.)] * 3]))
            self.assertEqual(ids.tolist(), [0, spec['world_count'] - 1])
            rows = d.pack_states(spec, ids)
            self.assertEqual(rows[0].tolist(), spec['needs'][0] + spec['layouts'][0] + spec['private_sites'][0])
            self.assertEqual(rows[1].tolist(), spec['needs'][-1] + spec['layouts'][-1] + spec['private_sites'][-1])
            for u in ([[1., 0, 0]], [[0, 0]], [[np.nan, 0, 0]]):
                with self.assertRaises(ValueError): d.sample_indices(spec, u)

    def test_all_need_features_and_permission_masks(self):
        for n in range(24):
            s = e.State((n, (n + 1) % 24, (n + 2) % 24), (0, 1, 2, 3))
            for mode in e.INFORMATION:
                views = {a: e.observe(s, a, information=mode) for a in e.AGENTS}
                x = d.encode_observations([views])
                self.assertEqual(x.shape, (1, 3, 54))
                self.assertTrue(np.all((x == 0) | (x == 1)))
                for i in range(3):
                    for j in range(3):
                        block = x[0, i, j * 7:(j + 1) * 7]
                        if mode != 'FI' and i != j:
                            self.assertTrue(np.all(block == 0)); continue
                        nv = e.need_view(s.needs[j])
                        expected = [1] + [int(k in nv['kinds']) for k in ('wood', 'fiber')] + [int(l in nv['lengths']) for l in ('short', 'long')] + [int(t in nv['destinations']) for t in ('L', 'R')]
                        np.testing.assert_array_equal(block, expected)
                    self.assertEqual(x[0, i, 53], mode == 'FI')
                    for site in range(4):
                        block = x[0, i, 21 + site * 5:26 + site * 5]
                        self.assertEqual(bool(block[0]), mode != 'LL' or site in (0, s.private_sites[i]))
                        if not block[0]: self.assertTrue(np.all(block == 0))
                views['A']['researcher'] = {'target': 1}
                with self.assertRaises(ValueError): d.encode_observations([views])

    def test_arrays_native_reward_terms_without_targets_in_features(self):
        spec = self.prepared['partitions']['new_needs_and_layouts']
        ids = np.array([0, 17, 55, spec['world_count'] - 1])
        arrays = d.make_arrays(spec, indices=ids)
        self.assertEqual(arrays['rewards'].shape, (4, 24))
        for b, state in enumerate(arrays['states']):
            for t, action_ids in enumerate(d.JOINT_ACTIONS):
                actions = {a: e.all_actions(a)[action_ids[i]] for i, a in enumerate(e.AGENTS)}
                self.assertEqual(arrays['rewards'][b, t], e.settle(state, actions)['reward'])
        self.assertTrue(np.all((arrays['rewards'] == 1).sum(axis=1) == 1))
        selected = d.make_arrays(spec, information='LL', indices=ids)
        self.assertNotIn('x_FI', selected); self.assertNotIn('x_PL', selected)
        np.testing.assert_array_equal(selected['x_LL'], arrays['x_LL'])

    def test_all_semantic_pair_labels_and_same_listener_observation(self):
        for part in ('train', 'new_needs'):
            spec = self.prepared['partitions'][part]; pair_data = d.content_pairs(spec)
            self.assertTrue(all(pair_data['summary']['content_each_axis_all_six_sender_listener_strata_present'].values()))
            totals = {a: 0. for a in d.AXES}
            for row in pair_data['rows']:
                before, after = row['needs']; sender, listener = row['sender'], row['listener']
                self.assertEqual([i for i in range(3) if before[i] != after[i]], [sender])
                choices = row['listener_correct_actions']
                expected = 'role' if (choices[0] == 0) != (choices[1] == 0) else 'content' if all(choices) and choices[0] != choices[1] else 'descriptive'
                self.assertEqual(row['classification'], expected)
                self.assertEqual(row['success_action_masks'], [1 << c for c in choices])
                if expected != 'content': continue
                self.assertEqual(row['listener_partners'], [sender, sender])
                totals[row['axis']] += 1 / row['within_axis_case_weight_denominator']
                for mode in ('PL', 'LL'):
                    views = [e.observe(e.State(n, (2, 0, 3, 1), (3, 1, 2)), e.AGENTS[listener], information=mode) for n in (before, after)]
                    self.assertEqual(*views)
            for total in totals.values(): self.assertAlmostEqual(total, 1., places=12)


if __name__ == '__main__': unittest.main()
