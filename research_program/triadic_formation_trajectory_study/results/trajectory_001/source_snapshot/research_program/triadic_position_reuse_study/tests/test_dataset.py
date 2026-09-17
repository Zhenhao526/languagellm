"""Pure static/source tests; never run preparation or access model output."""
from hashlib import sha256
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import numpy as np
from research_program.triadic_position_reuse_study import dataset as d


def independent_hash(salt, value):
    return sha256((salt + '|' + json.dumps(value, ensure_ascii=False, separators=(',', ':'))).encode()).hexdigest()


def case_identity(case):
    return [case['axis_index'], case['sender'], case['listener'], case['needs'][0], case['needs'][1]]


class StaticDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = d.load_sources()
        cls.result = d.make_prepared(cls.bundle)
        cls.ds = cls.result['discovery']
        cls.val = cls.result['validation']

    def test_source_scope_and_complete_discovery(self):
        self.assertTrue(all('/execution/' not in path and 'checkpoint_' not in path for path in self.bundle['source_sha256']))
        a = self.ds['arrays']; old = self.bundle['parts']['train']['arrays']
        self.assertEqual(len(a['case_index']), 209952)
        np.testing.assert_array_equal(a['source_spec_row'], np.arange(209952))
        for key in ('case_index', 'axis', 'sender', 'listener', 'endpoint_indices', 'correct_actions', 'base_within_axis_weight'):
            np.testing.assert_array_equal(a[key], old[key])
        self.assertEqual(len(self.ds['metadata']['content_cases']), 1944)

    def test_independent_case_rank_exact_four_per_stratum(self):
        source = self.bundle['parts'][d.VALIDATION_PARTITION]['metadata']['content_cases']
        actual = self.val['metadata']['selected_cases']
        for axis in range(3):
            for sender, listener in permutations(range(3), 2):
                candidates = [i for i, r in enumerate(source) if (r['axis_index'], r['sender'], r['listener']) == (axis, sender, listener)]
                candidates.sort(key=lambda i: (independent_hash(d.CASE_SALT, case_identity(source[i])), case_identity(source[i])))
                selected = [r['source_case_index'] for r in actual if r['identity'][:3] == [axis, sender, listener]]
                self.assertEqual(selected, candidates[:4])
        self.assertEqual(len(actual), 72)

    def test_background_rank_two_distinct_layouts(self):
        metadata = self.val['metadata']; spec = metadata['source_part_spec']
        for row in metadata['selected_cases']:
            ident = case_identity(row['case'])
            candidates = [(i, j) for i in range(6) for j in range(6)]
            candidates.sort(key=lambda x: (independent_hash(d.BACKGROUND_SALT, [ident, spec['layouts'][x[0]], spec['private_sites'][x[1]]]), spec['layouts'][x[0]], spec['private_sites'][x[1]]))
            first = candidates[0]
            second = next(x for x in candidates if x[0] != first[0])
            selected = [r for r in metadata['selected_backgrounds'] if r['selected_case_index'] == row['selected_case_index']]
            self.assertEqual([(r['recipient_layout_index'], r['owner_index']) for r in selected], [first, second])

    def test_independent_eligible_donor_and_hash_rank(self):
        spec = self.val['metadata']['source_part_spec']; old = self.bundle['parts'][d.VALIDATION_PARTITION]['arrays']
        cases = self.val['metadata']['selected_cases']
        for row in self.val['metadata']['selected_backgrounds']:
            case = cases[row['selected_case_index']]['case']; ident = case_identity(case)
            li, oi = row['recipient_layout_index'], row['owner_index']; layout = spec['layouts'][li]
            materials = [d.env.full_success_plans(needs)[0][2] for needs in case['needs']]
            eligible = [di for di, donor in enumerate(spec['layouts']) if di != li and all(layout.index(m) != donor.index(m) for m in materials)]
            eligible.sort(key=lambda di: (independent_hash(d.DONOR_SALT, [ident, layout, spec['private_sites'][oi], spec['layouts'][di]]), spec['layouts'][di]))
            self.assertEqual(row['donor_layout_index'], eligible[0])
            self.assertEqual(row['eligible_donor_count'], len(eligible))
            self.assertEqual(int(old['donor_layout_index'][row['donor_shift_index'], li]), eligible[0])

    def test_all_indices_correct_materials_and_native_plans(self):
        arrays = self.val['arrays']; spec = self.val['metadata']['source_part_spec']
        old = self.bundle['parts'][d.VALIDATION_PARTITION]['arrays']
        for row in range(144):
            sr, shift = int(arrays['source_spec_row'][row]), int(arrays['donor_shift_index'][row])
            np.testing.assert_array_equal(arrays['endpoint_indices'][row], old['endpoint_indices'][sr])
            np.testing.assert_array_equal(arrays['donor_endpoint_indices'][row], old['donor_endpoint_indices'][shift, sr])
            for label in ('', 'donor_'):
                states = d.task.pack_states(spec, arrays[label + 'endpoint_indices'][row])
                for endpoint, packed in enumerate(states):
                    state = d.env.State(packed[:3], packed[3:7], packed[7:])
                    acts = arrays[label + 'correct_actions'][row, endpoint]
                    chosen = {agent: d.env.all_actions(agent)[int(acts[i])] for i, agent in enumerate(d.env.AGENTS)}
                    self.assertEqual(d.env.settle(state, chosen)['reward'], 1)
                    listener = int(arrays['listener'][row]); action = chosen[d.env.AGENTS[listener]]
                    site = d.env.SITES.index(action['site'])
                    self.assertEqual(state.layout[site], arrays['target_materials'][row, endpoint])
                    self.assertEqual(action['partner'], d.env.AGENTS[int(arrays['sender'][row])])

    def test_weights_by_axis_sender_listener_background_and_direction(self):
        a = self.val['arrays']
        self.assertEqual(len(a['axis']), 144)
        self.assertEqual(len(np.unique(a['selected_case_index'])), 72)
        np.testing.assert_array_equal(np.bincount(a['selected_case_index']), np.full(72, 2))
        for axis in range(3):
            take = a['axis'] == axis
            self.assertAlmostEqual(a['base_within_axis_weight'][take].sum(), 1, places=12)
            self.assertAlmostEqual((a['base_within_axis_weight'][take] / 2).sum(), .5, places=12)
            for sender, listener in permutations(range(3), 2):
                sl = take & (a['sender'] == sender) & (a['listener'] == listener)
                self.assertEqual(int(sl.sum()), 8)
                self.assertAlmostEqual(a['base_within_axis_weight'][sl].sum(), 1 / 6, places=12)
            discovery = self.ds['arrays']; take = (discovery['axis'] == axis)
            for sender in range(3):
                self.assertAlmostEqual(discovery['base_within_axis_weight'][take & (discovery['sender'] == sender)].sum() * 3, 1, places=12)

    def test_joint_needs_orbits_and_layouts_disjoint_not_individual_values(self):
        proof = self.result['separation_proof']
        self.assertEqual(proof['selected_validation_unique_need_triples'], 140)
        self.assertEqual(proof['selected_validation_orbit_count'], 10)
        for name in ('need', 'orbit', 'layout'):
            self.assertEqual(proof[f'discovery_validation_{name}_intersection'], [])
        ds = self.ds['metadata']['source_part_spec']['needs']
        vs = self.val['metadata']['source_part_spec']['needs']
        for actor in range(3):
            self.assertEqual({row[actor] for row in ds}, set(range(24)))
            self.assertEqual({row[actor] for row in vs}, set(range(24)))

    def test_changed_field_masks_are_pure_and_not_filtered(self):
        for data in (self.ds, self.val):
            a = data['arrays']; attrs = a['truth_attributes']
            np.testing.assert_array_equal(attrs[:, :, 0], a['target_materials'] // 2)
            np.testing.assert_array_equal(attrs[:, :, 1], a['target_materials'] % 2)
            np.testing.assert_array_equal(attrs[:, :, 2], a['target_destinations'])
            np.testing.assert_array_equal(attrs[:, :, 3], np.broadcast_to(a['sender'][:, None], (len(a['axis']), 2)))
            np.testing.assert_array_equal(a['truth_field_change_mask'], np.eye(4, dtype=bool)[a['axis']])
            self.assertEqual(data['metadata']['truth_field_change_counts']['other_change_pattern_rows'], 0)
            self.assertFalse(data['metadata']['truth_field_change_counts']['changes_used_for_case_filtering'])

    def test_no_input_mutation_or_prepare_side_effect(self):
        old = self.bundle['parts'][d.VALIDATION_PARTITION]['arrays']
        before = {k: sha256(v.tobytes()).hexdigest() for k, v in old.items()}
        second = d.make_prepared(self.bundle)
        for key, value in old.items():
            self.assertEqual(sha256(value.tobytes()).hexdigest(), before[key])
        for key in self.val['arrays']:
            np.testing.assert_array_equal(second['validation']['arrays'][key], self.val['arrays'][key])
        with TemporaryDirectory() as temp, patch.object(d, 'load_sources', side_effect=AssertionError('must not be called')):
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                d.prepare(Path(temp))


if __name__ == '__main__':
    unittest.main()
