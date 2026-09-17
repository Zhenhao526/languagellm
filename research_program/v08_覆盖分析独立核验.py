"""Independent post-hoc coverage arithmetic; no model or training imports.

Synthetic checks are executable before the separately owned v0.8 analysis exists.
Real-data checks must wait for the complete 60-run protocol artifact.
"""
from collections import Counter
from copy import deepcopy
from itertools import product
import importlib.util
import hashlib
import json
from pathlib import Path
import unittest


def require(ok, text):
    if not ok:
        raise AssertionError(text)


def histogram(rows):
    result = Counter()
    for row in rows:
        require(type(row['count']) is int and row['count'] >= 0, 'integer counts')
        result[tuple(row['message'])] += row['count']
    return dict(result)


def coverage_reference(decoder, target, delivered_counts, legal_codes=None):
    """Legal-message existential coverage and empirical natural joint success."""
    legal = set(decoder) if legal_codes is None else set(legal_codes)
    require(bool(legal) and legal <= set(decoder), 'nonempty legal decoded code set')
    require(set(delivered_counts) <= legal, 'natural delivered code must be legal')
    require(all(type(n) is int and n >= 0 for n in delivered_counts.values()), 'natural counts')
    target = tuple(target)
    food = {m for m in legal if decoder[m][0] == target[0]}
    water = {m for m in legal if decoder[m][1] == target[1]}
    joint = food & water
    n = sum(delivered_counts.values())
    natural = sum(count for code, count in delivered_counts.items() if code in joint)
    require(bool(joint) or natural == 0, 'U=0 implies N=0')
    category = ('joint' if joint else 'both_marginals_without_joint' if food and water
                else 'one_marginal_only' if food or water else 'neither_marginal')
    return {'U': int(bool(joint)), 'N': natural, 'N_denominator': n,
            'food_covered': bool(food), 'water_covered': bool(water),
            'category': category, 'correct_joint_codes': sorted(joint)}


def saved_decoder_reference(direction):
    audit = direction['menu_audit']
    require(audit['all_menu_permutations_physically_equivalent'] is True,
            'menu-dependent policy cannot be collapsed')
    require((audit['enumerated_messages'], audit['goals'], audit['menus']) == (49, 2, 720),
            'complete menu enumeration')
    rows = direction['receiver_decoder_table']
    require(len(rows) == 49, '49 decoder rows')
    decoder = {}
    for row in rows:
        code = tuple(row['message'])
        actions = tuple(row['actions_by_goal'])
        require(code not in decoder and len(actions) == 2 and
                all(type(a) is int and 0 <= a < 6 for a in actions), 'unique valid decoder row')
        counts = row['menu_action_counts_by_goal']
        require(counts == [[720 if p == action else 0 for p in range(6)] for action in actions],
                'stored full-menu counts must confirm exact physical-action equivalence')
        decoder[code] = actions
    require(set(decoder) == set(product(range(7), repeat=2)), 'complete code domain')
    return decoder


def unique_codebook_reference(codebook):
    """Remove the compatible hidden sender-goal axis only after checking it."""
    maps = {}
    for row in codebook:
        maps.setdefault(row['map_id'], []).append(row)
    unique = []
    for map_id, rows in sorted(maps.items()):
        require(len(rows) == 2 and {r['sender_goal'] for r in rows} == {0, 1},
                'exact two compatible sender-goal rows')
        a, b = sorted(rows, key=lambda r: r['sender_goal'])
        for key in ('food_location', 'water_location', 'both'):
            require(a[key] == b[key], 'hidden-axis target and joint result agree')
        for key in ('emitted_messages', 'delivered_messages'):
            require(histogram(a[key]) == histogram(b[key]), 'hidden sender goal cannot change message')
        unique.append(a)
    return unique


def paired_world_reference(positions, goals, delivered, places, decoder, legal_codes):
    """One row is one world; query order is not assumed to be food then water."""
    require(sorted(goals) == [0, 1], 'exact one query per resource')
    code = tuple(delivered)
    require(code in legal_codes, 'world uses legal delivered code')
    expected_places = [decoder[code][g] for g in goals]
    require(list(places) == expected_places, 'world actions match stored greedy decoder')
    success = [int(places[q] == positions[g]) for q, g in enumerate(goals)]
    out = coverage_reference(decoder, positions, {code: 1}, legal_codes)
    require(out['N'] == success[0] * success[1], 'joint success from same-world paired queries')
    return {'successes': success, 'both': out['N'], 'U': out['U']}


def independent_real_recount(batch):
    """Recount all key numerators independently; never call the audited analyzer."""
    import numpy as np
    batch = Path(batch)
    protocol_path = batch / 'protocol_analysis.json'
    protocol = json.loads(protocol_path.read_text())
    seeds = (27101, 27102, 27103, 27104)
    kinds = ('additive', 'mixed', 'joint')
    conditions = [f'split{s}_{k}' for s in (1, 2, 3) for k in kinds]
    conditions += [f'{p}_{k}' for p in ('full', 'blocked') for k in kinds]
    require(protocol['status'] == 'complete' and not protocol['missing_runs'], 'wait for complete protocol')
    require(len(protocol['runs']) == 60 and {(r['seed'], r['condition']) for r in protocol['runs']} ==
            set(product(seeds, conditions)), 'complete fixed 60-run grid')
    aggregate, individual_seeds = {}, {}
    hashes = {str(protocol_path): hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    total_worlds = 0
    for run in protocol['runs']:
        name, seed = run['condition'], run['seed']
        family = 'split' if name.startswith('split') else 'blocked' if name.startswith('blocked') else 'full'
        kind = name.rsplit('_', 1)[1]
        blocked = family == 'blocked'
        require(len(run['directions']) == 2 and {d['scout'] for d in run['directions']} == {0, 1}, 'two receiver tables')
        decoder = {d['scout']: saved_decoder_reference(d) for d in run['directions']}
        npz = batch / f's{seed}_{name}' / 'final_normal.npz'
        hashes[str(npz)] = hashlib.sha256(npz.read_bytes()).hexdigest()
        with np.load(npz, allow_pickle=False) as z:
            a = {key: z[key] for key in z.files}
        require(a['positions'].shape == a['goals'].shape == a['action'].shape == (9600, 2), '9600 paired worlds')
        require(np.array_equal(np.sort(a['goals'], axis=1), np.tile([0, 1], (9600, 1))), 'one query each')
        require(np.array_equal(np.sort(a['menu'], axis=2), np.tile(np.arange(6), (9600, 2, 1))), 'complete query menus')
        places = a['menu'][np.arange(9600)[:, None], np.arange(2)[None, :], a['action']]
        targets = a['positions'][np.arange(9600)[:, None], a['goals']]
        correct = places == targets
        joint = correct[:, 0] & correct[:, 1]
        require(np.array_equal(places, a['place']) and np.array_equal(correct, a['successes']), 'independent paired settlement')
        require(np.array_equal(a['delivered'], np.zeros_like(a['sent']) if blocked else a['sent']), 'legal normal delivery')
        total_worlds += 9600
        for d in run['directions']:
            scout, table = d['scout'], decoder[d['scout']]
            allowed = {(0, 0)} if blocked else set(table)
            ix = np.flatnonzero(a['scout'] == scout)
            require(len(ix) == 4800, '4800 worlds per direction')
            cells = Counter((tuple(a['positions'][i]), int(a['goals'][i, 0])) for i in ix)
            require(cells == Counter({((f, w), g): 80 for f in range(6) for w in range(6) if f != w for g in (0, 1)}), 'balanced per-direction world denominator')
            actual_decoded = np.array([[table[tuple(a['delivered'][i])][g] for g in a['goals'][i]] for i in ix])
            require(np.array_equal(actual_decoded, places[ix]), 'same saved receiver function for both actual queries')
            unique = unique_codebook_reference(d['phases']['validation']['codebook'])
            require(len(unique) == 30, '30 deduplicated maps per direction')
            subset_ids = [('train', set(run['train_map_ids'])), ('heldout', set(run['heldout_map_ids']))] if family == 'split' else [('all', set(range(30)))]
            for subset, ids in subset_ids:
                counts = Counter()
                for row in unique:
                    if row['map_id'] not in ids:
                        continue
                    target = (row['food_location'], row['water_location'])
                    hist = histogram(row['delivered_messages'])
                    require(sum(hist.values()) == 16, '16 unique validation photos')
                    full = coverage_reference(table, target, hist)
                    legal = coverage_reference(table, target, hist, allowed)
                    require(full['N'] == row['both']['numerator'], 'protocol natural message count')
                    worlds = ix[np.all(a['positions'][ix] == target, axis=1)]
                    require(len(worlds) == 160, '160 endpoint worlds per map/direction')
                    nj = int(joint[worlds].sum())
                    require(full['U'] or nj == 0, 'world success requires a shared correct code')
                    require(legal['U'] or nj == 0, 'world success requires a legal shared correct code')
                    counts.update({'map_units': 1, 'U': full['U'], 'U_channel_allowed': legal['U'],
                        'protocol_N': full['N'], 'protocol_n': 16,
                        'protocol_failure_no_code': 16 * (1 - full['U']),
                        'protocol_failure_no_allowed_code': 16 * (1 - legal['U']),
                        'endpoint_J': nj, 'endpoint_n': len(worlds),
                        'endpoint_failure_no_code': len(worlds) * (1 - full['U']),
                        'endpoint_failure_no_allowed_code': len(worlds) * (1 - legal['U']),
                        'both_marginals_without_joint_maps': int(full['category'] == 'both_marginals_without_joint'),
                        'both_marginals_without_joint_allowed_maps': int(legal['category'] == 'both_marginals_without_joint'),
                        'food_marginal_reachable': int(full['food_covered']),
                        'water_marginal_reachable': int(full['water_covered']),
                        'both_marginals_no_joint': int(full['category'] == 'both_marginals_without_joint'),
                        'food_only': int(full['food_covered'] and not full['water_covered']),
                        'water_only': int(full['water_covered'] and not full['food_covered']),
                        'neither': int(not full['food_covered'] and not full['water_covered'])})
                key = (family, kind, subset)
                aggregate.setdefault(key, Counter()).update(counts)
                individual_seeds.setdefault((*key, seed), Counter()).update(counts)
    rows = []
    for key, counts in sorted(aggregate.items()):
        per_seed = [{'seed': s, **dict(individual_seeds[(*key, s)])} for s in seeds]
        require(len({r['map_units'] for r in per_seed}) == len({r['protocol_n'] for r in per_seed}) == len({r['endpoint_n'] for r in per_seed}) == 1, 'equal per-seed denominators')
        if key[0] == 'split' and key[2] == 'heldout':
            require((counts['map_units'], counts['protocol_n'], counts['endpoint_n']) == (144, 2304, 23040), 'heldout distinct denominators')
        rows.append({'family': key[0], 'reward_kind': key[1], 'subset': key[2], 'counts': dict(counts), 'per_seed': per_seed})
    return {'source_sha256': hashes, 'worlds_checked': total_worlds, 'groups': rows,
            'method': 'Direct independent set-membership/count arithmetic and paired saved actions; no import or call of v08_receiver_coverage.'}


class SyntheticCoverageTests(unittest.TestCase):
    def test_no_cover_implies_no_natural_success(self):
        r = coverage_reference({(0, 0): (4, 5)}, (0, 1), {(0, 0): 16})
        self.assertEqual((r['U'], r['N'], r['category']), (0, 0, 'neither_marginal'))

    def test_correct_code_exists_but_natural_message_does_not_use_it(self):
        r = coverage_reference({(0, 0): (4, 5), (0, 1): (0, 1)}, (0, 1), {(0, 0): 16})
        self.assertEqual((r['U'], r['N']), (1, 0))

    def test_two_marginal_codes_need_not_have_a_joint_code(self):
        r = coverage_reference({(0, 0): (0, 5), (0, 1): (4, 1)}, (0, 1), {(0, 0): 8, (0, 1): 8})
        self.assertEqual((r['food_covered'], r['water_covered'], r['U'], r['N']), (True, True, 0, 0))
        self.assertEqual(r['category'], 'both_marginals_without_joint')

    def test_blocked_only_zero_code_is_legal(self):
        decoder = {(0, 0): (4, 5), (0, 1): (0, 1)}
        legal = {(0, 0)}
        r = coverage_reference(decoder, (0, 1), {(0, 0): 16}, legal)
        self.assertEqual((r['U'], r['N']), (0, 0))
        self.assertEqual(coverage_reference(decoder, (0, 1), {(0, 0): 16})['U'], 1)
        with self.assertRaisesRegex(AssertionError, 'natural delivered code must be legal'):
            coverage_reference(decoder, (0, 1), {(0, 1): 16}, legal)

    def test_query_order_keeps_same_world_pair(self):
        r = paired_world_reference([2, 3], [1, 0], [0, 0], [3, 2], {(0, 0): (2, 3)}, {(0, 0)})
        self.assertEqual(r['both'], 1)

    def test_hidden_sender_goal_is_deduplicated_only_when_equal(self):
        row = {'map_id': 0, 'sender_goal': 0, 'food_location': 0, 'water_location': 1,
               'emitted_messages': [{'message': [0, 0], 'count': 16}],
               'delivered_messages': [{'message': [0, 0], 'count': 16}],
               'both': {'numerator': 0, 'denominator': 16, 'rate': 0.}}
        twin = deepcopy(row); twin['sender_goal'] = 1
        self.assertEqual(len(unique_codebook_reference([row, twin])), 1)
        twin['emitted_messages'][0]['message'] = [0, 1]
        with self.assertRaisesRegex(AssertionError, 'hidden sender goal'):
            unique_codebook_reference([row, twin])

    def test_menu_claim_requires_all_saved_counts(self):
        d = {'menu_audit': {'all_menu_permutations_physically_equivalent': True,
                           'enumerated_messages': 49, 'goals': 2, 'menus': 720},
             'receiver_decoder_table': [{'message': list(code), 'actions_by_goal': [0, 1],
                 'menu_action_counts_by_goal': [[720, 0, 0, 0, 0, 0], [0, 720, 0, 0, 0, 0]]}
                 for code in product(range(7), repeat=2)]}
        self.assertEqual(len(saved_decoder_reference(d)), 49)
        d['receiver_decoder_table'][0]['menu_action_counts_by_goal'][0] = [719, 1, 0, 0, 0, 0]
        with self.assertRaisesRegex(AssertionError, 'full-menu counts'):
            saved_decoder_reference(d)


class AuditedImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name('v08_receiver_coverage.py')
        spec = importlib.util.spec_from_file_location('coverage_under_review', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    @staticmethod
    def full_decoder(overrides):
        return {code: tuple(overrides.get(code, (4, 5))) for code in product(range(7), repeat=2)}

    @staticmethod
    def phase(decoder, code=(0, 0), blocked=False):
        maps = [(a, b) for a in range(6) for b in range(6) if a != b]
        rows = []
        delivered = (0, 0) if blocked else code
        for map_id, target in enumerate(maps):
            for g in (0, 1):
                score = lambda n: {'numerator': n, 'denominator': 16, 'rate': n / 16}
                rows.append({'map_id': map_id, 'food_location': target[0], 'water_location': target[1],
                    'sender_goal': g, 'emitted_messages': [{'message': list(code), 'count': 16}],
                    'delivered_messages': [{'message': list(delivered), 'count': 16}],
                    'both': score(16 * int(decoder[delivered] == target)),
                    'native': score(16 * int(decoder[delivered][g] == target[g])),
                    'switched': score(16 * int(decoder[delivered][1-g] == target[1-g]))})
        return {'photo_pairs': [[i, j] for i in range(4) for j in range(4, 8)], 'codebook': rows,
            'cross_goal': {'message_unchanged_across_sender_goal': {'numerator': 480, 'denominator': 480, 'rate': 1.},
                'same_message_correct_for_both_goals': {'numerator': sum(r['both']['numerator'] for r in rows), 'denominator': 960}}}

    def test_protocol_no_code_unused_code_and_marginal_only_cases(self):
        for name, overrides in [('none', {}), ('unused_joint', {(0, 1): (0, 1)}),
                                ('marginal_only', {(0, 0): (0, 5), (0, 1): (4, 1)})]:
            with self.subTest(name=name):
                decoder = self.full_decoder(overrides)
                reference = coverage_reference(decoder, (0, 1), {(0, 0): 16})
                rows = self.module.protocol_map_rows(self.phase(decoder), decoder, False)
                self.assertEqual((rows[0]['U'], rows[0]['protocol_N'], rows[0]['protocol_n']),
                                 (reference['U'], reference['N'], reference['N_denominator']))
                self.assertEqual(sum(r['protocol_n'] for r in rows), 30 * 16)

    def test_protocol_blocked_delivered_domain(self):
        decoder = self.full_decoder({(0, 1): (0, 1)})
        phase = self.phase(decoder, code=(0, 1), blocked=True)
        rows = self.module.protocol_map_rows(phase, decoder, True)
        self.assertEqual((rows[0]['U'], rows[0]['U_channel_allowed'], rows[0]['protocol_N']), (1, 0, 0))
        phase['codebook'][0]['delivered_messages'][0]['message'] = [0, 1]
        with self.assertRaises(AssertionError):
            self.module.protocol_map_rows(phase, decoder, True)

    def test_hidden_axis_emission_check_survives_blocked_constant_delivery(self):
        decoder = self.full_decoder({})
        phase = self.phase(decoder, code=(0, 1), blocked=True)
        phase['codebook'][1]['emitted_messages'][0]['message'] = [1, 1]
        with self.assertRaisesRegex(AssertionError, 'Emitted frequency differs'):
            self.module.protocol_map_rows(phase, decoder, True)

    def test_five_marginal_classes_against_independent_sets(self):
        examples = [('joint', {(0, 0): (0, 1)}),
                    ('both_marginals_no_joint', {(0, 0): (0, 5), (0, 1): (4, 1)}),
                    ('food_only', {(0, 0): (0, 5)}),
                    ('water_only', {(0, 0): (4, 1)}), ('neither', {})]
        for name, overrides in examples:
            with self.subTest(name=name):
                decoder = self.full_decoder(overrides)
                reference = coverage_reference(decoder, (0, 1), {(0, 0): 16})
                actual = self.module.map_coverage((0, 1), decoder)
                self.assertEqual(actual['marginal_class'], name)
                self.assertEqual((actual['U'], actual['food_marginal_reachable'], actual['water_marginal_reachable']),
                    (reference['U'], reference['food_covered'], reference['water_covered']))
                self.assertEqual(actual['U'] + sum(actual[k] for k in ('both_marginals_no_joint', 'food_only', 'water_only', 'neither')), 1)

    def test_producer_claim_does_not_replace_menu_counts(self):
        direction = {'menu_audit': {'all_menu_permutations_physically_equivalent': True,
                     'enumerated_messages': 49, 'goals': 2, 'menus': 720, 'cases': 70560, 'menus_used_after_check': 1},
             'receiver_decoder_table': [{'message': list(code), 'actions_by_goal': [0, 1],
                 'menu_action_counts_by_goal': [[720, 0, 0, 0, 0, 0], [0, 720, 0, 0, 0, 0]]}
                 for code in product(range(7), repeat=2)]}
        self.assertEqual(self.module.decoder_table(direction), saved_decoder_reference(direction))
        direction['receiver_decoder_table'][0]['menu_action_counts_by_goal'][0] = [719, 1, 0, 0, 0, 0]
        with self.assertRaises(AssertionError):
            self.module.decoder_table(direction)

    def test_final_world_pairs_and_reverse_query_order(self):
        import numpy as np
        decoder = self.full_decoder({(0, 0): (2, 3)})
        goals = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        positions = np.array([[2, 3], [2, 4], [1, 3], [1, 4]])
        places = np.array([[decoder[(0, 0)][g] for g in row] for row in goals])
        menus = np.array([[list(range(6)), list(range(5, -1, -1))] for _ in range(4)])
        actions = np.array([[list(menus[i, j]).index(places[i, j]) for j in range(2)] for i in range(4)])
        successes = places == np.take_along_axis(positions, goals, axis=1)
        a = {'scout': np.array([0, 1, 0, 1]), 'episode': np.array([0, 0, 1, 1]), 'positions': positions, 'photo_ids': np.zeros((4, 2), int),
             'goals': goals, 'menu': menus, 'inventory': np.zeros((4, 2)), 'history': np.zeros((4, 18)),
             'sent': np.zeros((4, 2), int), 'delivered': np.zeros((4, 2), int), 'action': actions,
             'place': places, 'successes': successes, 'reward': successes.mean(1)}
        out = self.module.audit_normal_arrays(a, {0: decoder, 1: decoder}, False, expected_n=4)
        refs = [paired_world_reference(positions[i], goals[i], a['delivered'][i], places[i], decoder, set(decoder)) for i in range(4)]
        self.assertEqual(out['joint'].tolist(), [r['both'] for r in refs])
        self.assertEqual(out['joint'].tolist(), [True, False, False, False])
        self.assertEqual(out['U'].tolist(), [r['U'] for r in refs])
        a['sent'][:] = [0, 1]
        self.module.audit_normal_arrays(a, {0: decoder, 1: decoder}, True, expected_n=4)
        a['delivered'][0] = [0, 1]
        with self.assertRaises(AssertionError):
            self.module.audit_normal_arrays(a, {0: decoder, 1: decoder}, True, expected_n=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
