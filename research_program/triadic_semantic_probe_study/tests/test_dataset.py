"""Independent finite-set/index tests. No neural model or trained outputs."""
from fractions import Fraction
from itertools import product, permutations
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from research_program.triadic_semantic_probe_study import dataset as d
from research_program.triadic_task import environment as env


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data = d.build_dataset()

    def test_all_source_pairs_and_listener_metadata_retained(self):
        self.assertEqual(len(self.data['source_pairs']), 2244)
        self.assertEqual(len(self.data['cases']), 4488)
        self.assertEqual(len(self.data['unique_case_indices']), 768)
        self.assertEqual(len({c['case_id'] for c in self.data['cases']}), 4488)
        self.assertEqual(self.data['summary']['unique_cases_by_class'], {'content': 120, 'role': 408, 'descriptive': 240})
        multiple = [c for c in self.data['cases'] if c['ecology'] == 'multiple']
        self.assertEqual(len(multiple), 3720)
        self.assertTrue(all(not c['disjoint'] and not c['expanded'] for c in multiple))

    def test_menus_and_all24_physical_plans_match_native_engine(self):
        self.assertEqual({a: d.action_menu(a) for a in d.AGENTS}, {a: env.all_actions(a) for a in env.AGENTS})
        plans = d.structural_plans()
        self.assertEqual(len({tuple(p['indices']) for p in plans}), 24)
        needs = self.data['endpoint_index_spaces']['train']['needs']
        for n in needs:
            state = env.State(tuple(n), (0, 1, 2, 3), (1, 2, 3))
            won = [i for i, p in enumerate(plans) if env.settle(state, p['actions'], require_match=True)['reward'] == 1]
            self.assertEqual(won, d.winning_plan_ids(n))

    def test_content_and_role_sets_have_correct_semantic_scope(self):
        for c in self.data['cases']:
            sets = [set(x) for x in c['canonical_success_actions']]
            self.assertEqual([d.mask_members(m) for m in c['canonical_success_action_masks']], sets)
            self.assertEqual(c['disjoint'], not bool(sets[0] & sets[1]))
            if c['classification'] == 'content':
                self.assertEqual(c['roles'], ['always_participate']*2)
                self.assertEqual(c['listener_partners'], [[c['sender']]]*2)
            elif c['classification'] == 'role':
                self.assertEqual(set(c['roles']), {'always_participate', 'always_wait'})
                self.assertEqual(sum(s == {0} for s in sets), 1)
                self.assertTrue(c['role_switch'])
            else: self.assertTrue(sets[0] & sets[1])

    def test_all_original_endpoint_indices_match_explicit_product_order(self):
        cases = self.data['cases']
        for part, a in self.data['partitions'].items():
            spec = self.data['endpoint_index_spaces'][part]
            packed = np.asarray([n+l+o for n, l, o in product(spec['needs'], spec['layouts'], spec['private_sites'])], dtype=np.int16)
            states = d.endpoint_states(spec, a['endpoint_indices'])
            self.assertTrue(np.array_equal(states, packed[a['endpoint_indices']]))
            expected_needs = np.asarray([cases[i]['needs'] for i in a['case_index']], dtype=np.int16)
            self.assertTrue(np.array_equal(states[:, :, :3], expected_needs))
            self.assertTrue(np.array_equal(states[:, 0, 3:], states[:, 1, 3:]))
            who = a['listener']
            self.assertTrue(np.array_equal(states[np.arange(len(who)), 0, who], states[np.arange(len(who)), 1, who]))
            self.assertTrue(np.array_equal(a['endpoint_indices'][:, 0]//(len(spec['layouts'])*6), a['need_indices'][:, 0]))
            self.assertEqual(len(states), 82944 if part == 'train' else 27648)

    def test_all_expanded_masks_follow_location_bijection(self):
        for part, a in self.data['partitions'].items():
            spec = self.data['endpoint_index_spaces'][part]
            for ci in self.data['unique_case_indices']:
                c = self.data['cases'][ci]
                indices = np.flatnonzero(a['case_index'] == ci)
                for li, layout in enumerate(spec['layouts']):
                    selected = indices[a['layout_indices'][indices] == li]
                    self.assertEqual(len(selected), 6)
                    expected = []
                    for actions in c['canonical_success_actions']:
                        transformed = set()
                        for action in actions:
                            if action == 0: transformed.add(0)
                            else:
                                old_site, remainder = divmod(action-1, 4)
                                new_site = next(i for i, material in enumerate(layout) if material == old_site)
                                transformed.add(1+4*new_site+remainder)
                        expected.append(sum(2**x for x in transformed))
                    self.assertTrue(np.array_equal(a['success_action_masks'][selected], np.tile(expected, (6, 1))))

    def test_native_success_masks_for_every_axis_class_and_all24_layouts(self):
        chosen = {}
        for c in self.data['cases']:
            if c['ecology'] == 'unique': chosen.setdefault((c['axis'], c['classification']), c)
        plans = d.structural_plans()
        for c in chosen.values():
            for layout in permutations(range(4)):
                for needs in c['needs']:
                    state = env.State(needs, layout, (1, 2, 3))
                    won = {p['indices'][c['listener']] for p in plans if env.settle(state, p['actions'], require_match=True)['reward'] == 1}
                    self.assertEqual(won, d.full_action_set(needs, c['listener'], layout))

    def test_content_and_role_weights_normalize_separately(self):
        for part, a in self.data['partitions'].items():
            for label, axes in (('content', range(3)), ('role', range(2))):
                w = a[label+'_within_axis_weight']
                self.assertFalse(w[a['classification'] != d.CLASSES.index(label)].any())
                for axis in axes:
                    selected = a['axis'] == axis
                    self.assertAlmostEqual(float(w[selected].sum()), 1., places=14)
                    for sender, listener in permutations(range(3), 2):
                        group = selected & (a['sender'] == sender) & (a['listener'] == listener)
                        self.assertAlmostEqual(float(w[group].sum()), 1/6, places=14)
                positive = w > 0
                self.assertTrue(np.array_equal(w[positive], 1/a['active_weight_denominator'][positive]))
                self.assertEqual(sum(Fraction(c[label+'_within_axis_weight_fraction']) for c in self.data['cases']), len(axes))
            self.assertAlmostEqual(a['content_within_axis_weight'].sum(), 3.)
            self.assertAlmostEqual(a['role_within_axis_weight'].sum(), 2.)

    def test_strict_within_partition_mapping_and_impossibility(self):
        proposals = self.data['layout_donor_mapping_proposal']
        train = proposals['train']; self.assertEqual(train['status'], 'perfect_matching_exists')
        self.assertEqual(sorted(train['donor_layout_indices']), list(range(18)))
        held = proposals['heldout_layouts']; self.assertEqual(held['status'], 'no_perfect_matching')
        self.assertEqual(held['adjacency'], [[2], [2], [0, 1], [], [], []])
        maximum = max(sum(p[i] in held['adjacency'][i] for i in range(6)) for p in permutations(range(6)))
        self.assertEqual(maximum, held['matched_edges']); self.assertEqual(maximum, 2)
        witness = held['Hall_deficiency']
        union = set().union(*(set(held['adjacency'][i]) for i in witness['left_indices']))
        self.assertEqual(union, set(witness['neighbor_right_indices']))
        self.assertLess(len(union), len(witness['left_indices']))

    def test_remote_mapping_changes_all_background_sites_and_preserves_needs_owners(self):
        for part, a in self.data['partitions'].items():
            states = d.endpoint_states(self.data['endpoint_index_spaces'][part], a['endpoint_indices'])
            remote = np.empty_like(states)
            for pi, target in enumerate(d.PARTITIONS):
                mask = a['remote_donor_partition'] == pi
                remote[mask] = d.endpoint_states(self.data['endpoint_index_spaces'][target], a['remote_donor_endpoint_indices'][mask])
            self.assertTrue(np.array_equal(remote[:, :, :3], states[:, :, :3]))
            self.assertTrue(np.array_equal(remote[:, :, 7:], states[:, :, 7:]))
            self.assertTrue(np.array_equal(remote[:, :, 3:7], (states[:, :, 3:7]+1) % 4))
            self.assertTrue(np.all(remote[:, :, 3:7] != states[:, :, 3:7]))
        flow = self.data['remote_background_mapping']['recipient_to_donor_flows']
        self.assertEqual([flow[p][q]['layouts'] for p, q in product(d.PARTITIONS, repeat=2)], [13, 5, 5, 1])
        all_layouts = [tuple(l) for s in self.data['endpoint_index_spaces'].values() for l in s['layouts']]
        self.assertEqual({tuple((m+1) % 4 for m in l) for l in all_layouts}, set(all_layouts))

    def test_prepare_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'static'
            receipt = d.prepare(out)
            self.assertEqual(receipt['status'], 'prepared_static_no_model')
            manifest = json.loads((out/'manifest.json').read_text())
            for name, metadata in manifest['outputs'].items(): self.assertEqual(d.sha(out/name), metadata['sha256'])
            for part in d.PARTITIONS:
                with np.load(out/(part+'.npz'), allow_pickle=False) as z:
                    self.assertEqual(set(z.files), set(self.data['partitions'][part]))
                    for name in z.files: self.assertTrue(np.array_equal(z[name], self.data['partitions'][part][name]))
            with self.assertRaises(AssertionError): d.prepare(out)


if __name__ == '__main__': unittest.main()
