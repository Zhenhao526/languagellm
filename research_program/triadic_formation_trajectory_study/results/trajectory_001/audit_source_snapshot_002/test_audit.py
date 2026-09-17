"""Independent-auditor preflight with synthetic callbacks only.

Imports verify historical source hashes; no checkpoint, stored message, or result
is loaded. Handwritten logits/probabilities replace all network computation.
"""
from fractions import Fraction
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

from research_program.triadic_formation_trajectory_study import audit_execution as a
from research_program.triadic_formation_trajectory_study import channel as channel
from research_program.triadic_formation_trajectory_study import dataset, metrics, runner
from research_program.triadic_position_reuse_study import metrics as pm

NETS = tuple(range(9))


def logits(net, x):
    v = np.rint(x @ (1+np.arange(x.shape[-1]) % 5)).astype(int)
    if net % 3 == 2:
        z = np.full((len(x), 17), -3.)
        z[np.arange(len(x)), (v+net) % 17] = 3.
    else:
        z = np.full((len(x), 4, 8), -3.)
        for p in range(4):
            z[np.arange(len(x)), p, (v+net+p) % 8] = 3.
    return z


def probability(net, x):
    z = logits(net, x)
    exp = np.exp(z-z.max(-1, keepdims=True))
    return exp/exp.sum(-1, keepdims=True)


def fake_forward(net, x):
    z = logits(net, x)
    return (z.reshape(len(x), 32) if net % 3 != 2 else z), None


def literal_routes(tokens, live, senders=None, replacements=None):
    n = len(tokens); out = np.zeros((n, 3, 99))
    for row, viewer, speaker in product(range(n), range(3), range(3)):
        if live or viewer == speaker:
            for position in range(4):
                z = tokens[row, speaker, position]
                if live and senders is not None and speaker == senders[row] and viewer != speaker:
                    z = replacements[row, position]
                out[row, viewer, 32*speaker+8*position+z] = 1
            out[row, viewer, 96+speaker] = 1
    return out


def measure_fixture():
    # Exactly six ordered sender/listener strata per axis, eight rows per stratum.
    rows = [(q, s, l) for q, s, l in product(range(3), repeat=3) if s != l for _ in range(8)]
    q, s, l = map(np.asarray, zip(*rows))
    spec = dict(axis=q, sender=s, listener=l, base_within_axis_weight=np.full(144, 1/48))
    rng = np.random.default_rng(152)
    baseline = rng.integers(0, 2, (2, 144)).astype(bool)
    positions = {}
    for u in range(8):
        arms = {arm:dict(target_apt=rng.integers(0, 2, (2, 144)).astype(bool),
                         current_apt=rng.integers(0, 2, (2, 144)).astype(bool),
                         natural_current_apt=baseline.copy(),
                         packet_in_train_endpoint_greedy_set=rng.integers(0, 2, (2, 144)).astype(bool))
                for arm in ('same', 'opposite')}
        arms['contrast'] = pm.contrast(arms['same'], arms['opposite'])
        positions[u] = arms
    rates = rng.random((3, 3, 8)); scores = rates-rates.mean(axis=1, keepdims=True)
    profile = dict(selected_positions=np.asarray([[0, 1, 2], [4, 4, 4], [7, 6, 5]], dtype=np.int8),
                   response_rates=rates, scores=scores)
    final_profile = dict(profile, selected_positions=np.asarray([[7, 6, 5], [1, 2, 3], [0, 0, 0]], dtype=np.int8))
    return spec, profile, final_profile, positions


class AuditPreflightTests(TestCase):
    def test_natural_six_sender_heads_then_action_heads_with_independent_routes(self):
        x = np.random.default_rng(12).integers(0, 2, (4, 3, 54)).astype(float)
        for live, actions in product((True, False), repeat=2):
            with self.subTest(live=live, actions=actions):
                callback = mock.Mock(side_effect=probability)
                got = a.natural_trace(NETS, x, live, actions=actions, probability=callback)
                expected_heads = [0, 3, 6, 1, 4, 7]+([2, 5, 8] if actions else [])
                self.assertEqual([c.args[0] for c in callback.call_args_list], expected_heads)
                first = literal_routes(got['messages'][:, 0], live)
                second = literal_routes(got['messages'][:, 1], live)
                np.testing.assert_array_equal(got['first_routes'], first)
                np.testing.assert_array_equal(got['second_routes'], second)
                for who in range(3):
                    np.testing.assert_array_equal(callback.call_args_list[3+who].args[1],
                                                  np.concatenate((x[:, who], first[:, who]), -1))
                self.assertEqual(got['independent_module_samples'], (9 if actions else 6)*len(x))
                with mock.patch.object(channel.core.base, 'actor_forward', side_effect=fake_forward):
                    main = (channel.natural_rollout(NETS, x, live) if actions else
                            channel.generate_messages(NETS, x, live, include_routes=True))
                for name in ('messages', 'first_routes', 'second_routes'):
                    np.testing.assert_array_equal(got[name], main[name])
                if actions:
                    for name in ('action_inputs', 'action_indices', 'action_probabilities'):
                        np.testing.assert_array_equal(got[name], main[name])
                else:
                    self.assertNotIn('action_indices', got)

    def test_whole_outward_patch_all_senders_keeps_self_and_matches_main(self):
        x = np.random.default_rng(817).integers(0, 2, (3, 3, 54)).astype(float)
        natural = a.natural_trace(NETS, x, True, probability=probability)
        m = natural['messages']; sender = np.arange(3)
        for sham in (False, True):
            with self.subTest(sham=sham):
                donor = m[np.arange(3), :, sender].copy()
                if not sham:
                    donor = (donor+3) % 8
                callback = mock.Mock(side_effect=probability)
                out = a.whole_trace(NETS, x, m, sender, donor, True, probability=callback)
                np.testing.assert_array_equal(out['first_routes'], literal_routes(m[:, 0], True, sender, donor[:, 0]))
                np.testing.assert_array_equal(out['second_routes'], literal_routes(out['messages'][:, 1], True, sender, donor[:, 1]))
                np.testing.assert_array_equal(out['messages'][np.arange(3), :, sender], m[np.arange(3), :, sender])
                self.assertEqual([c.args[0] for c in callback.call_args_list], [1, 4, 7, 2, 5, 8])
                self.assertEqual(out['independent_module_samples'], 18)
                with mock.patch.object(channel.core.base, 'actor_forward', side_effect=fake_forward):
                    main = channel.cross_time_whole(NETS, x, m, sender, donor)
                for name in ('messages', 'patched_outward_packets', 'first_routes', 'second_routes',
                             'action_inputs', 'action_probabilities', 'action_indices'):
                    np.testing.assert_array_equal(out[name], main[name])
                if sham:
                    for name in ('messages', 'action_inputs', 'action_indices', 'action_probabilities'):
                        np.testing.assert_array_equal(out[name], natural[name])

    def test_cross_time_early_feedback_can_change_focal_action_but_not_own_w2(self):
        def receiver(net, x):
            if net % 3 == 0:
                p = np.zeros((len(x), 4, 8)); p[:, :, 0] = 1
            elif net % 3 == 1:
                z = x[:, 54:62].argmax(-1)
                p = np.zeros((len(x), 4, 8)); p[:, 1:, 0] = 1
                p[np.arange(len(x)), 0, z] = 1
            else:
                z = x[:, 185:193].argmax(-1)  # B's W2 token0, not donor A's W2
                p = np.zeros((len(x), 17)); p[np.arange(len(x)), z] = 1
            return p
        x = np.zeros((1, 3, 54)); m = a.natural_trace(NETS, x, True, probability=receiver)['messages']
        donor = np.zeros((1, 2, 4), dtype=np.int8); donor[0, 0, 0] = 1; donor[0, 1, 0] = 7
        got = a.whole_trace(NETS, x, m, 0, donor, True, probability=receiver)
        np.testing.assert_array_equal(got['messages'][0, :, 0], 0)
        np.testing.assert_array_equal(got['messages'][0, 1, 1:, 0], 1)
        np.testing.assert_array_equal(got['action_indices'], [[1, 1, 1]])
        self.assertFalse(np.array_equal(got['action_inputs'][:, 0], np.concatenate((x, literal_routes(m[:, 0], True), literal_routes(m[:, 1], True)), -1)[:, 0]))

    def test_silent_whole_uses_no_callback_or_network_and_no_invented_actions(self):
        x = np.zeros((3, 3, 54)); m = np.arange(72).reshape(3, 2, 3, 4).astype(np.int8) % 8
        donor = np.full((3, 2, 4), 7, dtype=np.int8)
        callback = mock.Mock(side_effect=AssertionError('Invisible splice must not forward'))
        got = a.whole_trace(None, x, m, [0, 1, 2], donor, False, probability=callback)
        callback.assert_not_called()
        np.testing.assert_array_equal(got['messages'], m)
        np.testing.assert_array_equal(got['action_inputs'], np.concatenate((x, literal_routes(m[:, 0], False), literal_routes(m[:, 1], False)), -1))
        self.assertEqual(got['independent_module_samples'], 0)
        self.assertNotIn('action_indices', got)

    def test_sorted_static_pool_preserves_repeats_directions_and_full_ids(self):
        validation = dict(endpoint_indices=np.array([[8, 2], [2, 8], [5, 1]], dtype=np.int64),
                          donor_endpoint_indices=np.array([[4, 9], [9, 4], [1, 5]], dtype=np.int64))
        full = np.arange(120, dtype=np.int16).reshape(12, 10)
        got = a.static_pool(validation, full, 17)
        ids = np.array([1, 2, 4, 5, 8, 9], dtype=np.int32)
        np.testing.assert_array_equal(got['validation_pool_endpoint_indices'], ids)
        np.testing.assert_array_equal(got['recipient_pool_rows'], [[4, 1], [1, 4], [3, 0]])
        np.testing.assert_array_equal(got['donor_pool_rows'], [[2, 5], [5, 2], [0, 3]])
        with mock.patch.object(dataset.task, 'pack_states', side_effect=lambda spec, index: full[index]):
            official = dataset.validation_pool(validation, {'world_count': len(full)})['arrays']
        official['train_world_indices'] = np.arange(17, dtype=np.int32)
        a.arrays_equal(got, official, 'Static synthetic schema')
        mapped = runner.local_spec(validation, got)
        for b in (0, 1):
            for arm in ('sham', 'same', 'opposite'):
                r, cf, d = a.indices(mapped, arm, b)
                np.testing.assert_array_equal(ids[r], validation['endpoint_indices'][:, b])
                np.testing.assert_array_equal(ids[cf], validation['endpoint_indices'][:, 1-b])
                wanted = validation['endpoint_indices'][:, b] if arm == 'sham' else validation['donor_endpoint_indices'][:, b if arm == 'same' else 1-b]
                np.testing.assert_array_equal(ids[d], wanted)

    def test_exact_irregular_time_weights_and_signed_auc(self):
        expected = [Fraction(1, 120), Fraction(1, 24), Fraction(7, 60),
                    Fraction(5, 24), Fraction(3, 8), Fraction(1, 4)]
        self.assertEqual(a.exact_time_weights(), expected)
        for i, weight in enumerate(expected):
            basis = np.eye(6)[i]
            self.assertEqual(a.area(basis), float(weight))
            self.assertAlmostEqual(a.area(basis), metrics.normalized_area(basis), places=15)
        self.assertEqual(a.area([0, 0, 0, 0, 0, 1]), .25)
        self.assertAlmostEqual(a.area(-np.asarray(a.STEPS)/6000), -.5)
        with self.assertRaises(AssertionError):
            a.area([0]*5)

    def test_point_summary_schema_agrees_with_main_for_all_positions_and_strata(self):
        spec, profile, final, positions = measure_fixture()
        natural = positions[0]['same']; shams = {0:natural, 4:natural}
        got = a.point_summary(spec, profile, final, positions, natural, shams)
        current = pm.selection_summary(spec, profile, positions)
        current['all_positions'] = {str(u):{arm:pm.aggregate_rows(spec, values) for arm, values in arms.items()} for u, arms in positions.items()}
        expected = dict(current=current, retrospective_final=pm.selection_summary(spec, final, positions),
            discovery=metrics.discovery_description(dict(profile, specificity_scores=profile['scores'])),
            natural=pm.aggregate_rows(spec, natural), shams={str(u):pm.aggregate_rows(spec, values) for u, values in shams.items()})
        counts = dict(scalars=0, max_error=0.)
        a.numeric_tree(got, expected, counts)
        self.assertGreater(counts['scalars'], 2500)
        self.assertLess(counts['max_error'], 1e-14)
        self.assertIn('all_positions', got['current'])
        self.assertNotIn('all_positions', got['retrospective_final'])

    def test_trajectory_has_receiver_rows_and_not_column_subtraction(self):
        points = {t:dict(current={k:(i-2)/10 for k in a.CURVE_KEYS},
                         retrospective_final={k:(5-i)/20 for k in a.CURVE_KEYS})
                  for i, t in enumerate(a.STEPS)}
        cross = {(r, s):{'contrast':{'macro':{'target_apt':i*.1+j*.013}}}
                 for i, r in enumerate(a.STEPS) for j, s in enumerate(a.STEPS)}
        got = a.trajectory_summary(points, cross)
        main = metrics.trajectory(points, cross)
        counts = dict(scalars=0, max_error=0.); a.numeric_tree(got, main, counts)
        matrix = np.asarray(got['cross_time_target_effect'])
        adjusted = np.asarray(got['cross_time_minus_receiver_same_time'])
        np.testing.assert_array_equal(np.diag(adjusted), 0)
        np.testing.assert_allclose(adjusted[4], .013*(np.arange(6)-4), atol=1e-15)
        self.assertFalse(np.allclose(adjusted, matrix-np.diag(matrix)[None, :]))
        missing = dict(cross); missing.pop((100, 3000))
        with self.assertRaises(AssertionError):
            a.trajectory_summary(points, missing)

    def test_temporal_boolean_schema_regression_and_exact_boolean_mismatch(self):
        before = np.zeros((419904, 2, 3, 4), dtype=np.int8); after = before.copy()
        after[:, 1, 2, 3] = 1
        left = {'selected_positions':np.zeros((3, 3), dtype=np.int8)}
        right = {'selected_positions':np.eye(3, dtype=np.int8)}
        got = a.temporal_change(before, after, left, right)
        main = metrics.temporal_change(before, after, left, right)
        counts = dict(scalars=0, max_error=0.); a.numeric_tree(got, main, counts)
        self.assertAlmostEqual(got['train_token_change_rate'], 1/24)
        self.assertEqual(got['sender_whole_packet_change_rate'], [0, 0, 1])
        for actual in (False, 1):
            with self.subTest(actual=actual):
                with self.assertRaisesRegex(AssertionError, 'boolean'):
                    a.numeric_tree({'flag':True}, {'flag':actual}, dict(scalars=0, max_error=0.))

    def test_saved_comparison_rejects_dtype_fields_and_nonfinite_changes(self):
        expected = {'messages':np.zeros((1, 2, 3, 4), dtype=np.int8),
                    'action_probabilities':np.full((1, 3, 17), 1/17)}
        self.assertEqual(a.arrays_equal(expected, {k:v.copy() for k,v in expected.items()}, 'fixture'), 0)
        for actual in ({'messages':expected['messages']},
                       dict(expected, messages=expected['messages'].astype(np.int16)),
                       dict(expected, action_probabilities=np.full((1, 3, 17), np.nan))):
            with self.subTest(fields=list(actual)):
                with self.assertRaises(AssertionError):
                    a.arrays_equal(expected, actual, 'negative fixture')

    def test_expected_budget_matches_full_matrix_reuse_and_producer_schema(self):
        counts = a.configured_counts()
        for key, value in counts.items():
            self.assertEqual(value, runner.CONFIG[key])
        self.assertEqual(counts['new_train_message_worlds'], 16*5*419904)
        self.assertEqual(counts['new_validation_natural_module_samples'], 16*5*554*9)
        self.assertEqual(counts['new_cross_time_module_samples'], 8*(6*6-1)*2*2*144*6)
        self.assertEqual(counts['new_position_module_samples'], 8*5*144*2*(2*(4*6+4*3)+6+3))
        self.assertEqual(counts['parameter_loads'], 8*6+8*5)

    def test_legacy_cell_accepts_new_local_pool_metadata_including_silent_6000(self):
        """Run the unchanged legacy checker on producer-shaped synthetic files.

        Only the content measurement is mocked: these rows deliberately do not
        claim to be the formal content dataset. Routing, saved fields, physics,
        counts, source packet hashes, and silent non-forwarding are checked.
        """
        p, context, old = a.references()
        n = 144
        states = np.tile(np.array([0, 0, 0, 0, 1, 2, 3, 1, 2, 3], dtype=np.int16), (2*n, 1))
        states[n:, 0] = 1
        spec = dict(sender=np.arange(n) % 3,
                    endpoint_indices=np.column_stack((np.arange(n), n+np.arange(n))),
                    donor_endpoint_indices=np.column_stack((n+np.arange(n)[::-1], np.arange(n)[::-1])))
        full_ids = 1000+7*np.arange(2*n, dtype=np.int32)

        def settle_synthetic(data, pool, case, direction):
            for ids, prefix in ((case['endpoint_indices'][:, direction], ''),
                                (case['endpoint_indices'][:, 1-direction], 'counterfactual_')):
                reward, executed, satisfied = context.native(pool['states'][ids], data['action_indices'])
                data[prefix+'reward' if prefix else 'greedy_reward'] = reward
                data[prefix+'satisfied'] = satisfied
                if not prefix:
                    data['executed'] = executed
            return data

        with TemporaryDirectory() as folder:
            for live, unit, update in ((True, 0, 100), (True, 4, 100),
                                       (False, 0, 6000), (False, 4, 6000)):
                with self.subTest(live=live, unit=unit, update=update):
                    x = context.features(states, 'LL')
                    natural = a.natural_trace(NETS, x, live, probability=probability)
                    pool = dict(states=states, **{k:natural[k] for k in
                                ('messages', 'action_indices', 'action_probabilities')})
                    policy = {'prior_position_records':[]}
                    production_callback = mock.Mock(side_effect=fake_forward)
                    with mock.patch.object(channel.core.base, 'actor_forward', production_callback), \
                         mock.patch.object(runner.pr, 'settle_data', side_effect=settle_synthetic):
                        if not live:
                            _, previous = runner.pr.execute_cell(None, x[:n], pool, spec, unit, 'opposite', 0, False)
                            policy['prior_position_records'] = [previous]
                        data, record = runner.position_data(policy, update, NETS if live else None,
                                                            x[:n], pool, spec, full_ids, unit, 'opposite', 0, live)
                    path = Path(folder)/f'{live}_{unit}_{update}.npz'
                    if live:
                        np.savez(path, **data)
                    else:
                        production_callback.assert_not_called()
                    record.update(condition='LL_live' if live else 'LL_silent',
                                  receiver_natural_source='synthetic_only',
                                  path=str(path) if live else None,
                                  data_sha256=p.sha(path) if live else None)
                    self.assertEqual(record['index_scope'], 'local554_pool')
                    self.assertEqual(record['is_reused'], update == 6000)
                    independent_callback = mock.Mock(side_effect=probability if live else
                                                     AssertionError('Silent must not call a network'))
                    with mock.patch.object(old, 'probabilities', independent_callback), \
                         mock.patch.object(p, 'measure', return_value={'synthetic_only':True}):
                        _, scope = p.audit_cell(record, path, pool, spec, NETS if live else None, 'LL', [])
                    self.assertEqual(scope['independent_worlds'], n if live else 0)
                    self.assertEqual(scope['independent_modules'], n*(6 if unit < 4 else 3) if live else 0)
                    if not live:
                        independent_callback.assert_not_called()
                    bad = dict(record, new_network_samples=record['new_network_samples']+1)
                    with mock.patch.object(old, 'probabilities', side_effect=probability), \
                         mock.patch.object(p, 'measure', return_value={}):
                        with self.assertRaisesRegex(AssertionError, 'Actual forward samples'):
                            p.audit_cell(bad, path, pool, spec, NETS if live else None, 'LL', [])

    def test_live_6000_preserves_old_metadata_and_global_indices_without_forward(self):
        n = 144
        rows = np.arange(n)
        spec = dict(sender=rows % 3, endpoint_indices=np.column_stack((rows, rows+n)),
                    donor_endpoint_indices=np.column_stack((rows+n, rows)))
        full_ids = 2000+3*np.arange(2*n, dtype=np.int32)
        messages = (np.arange(2*n*24).reshape(2*n, 2, 3, 4) % 8).astype(np.int8)
        data = dict(recipient_indices=full_ids[rows], donor_indices=full_ids[rows+n],
                    donor_packets=messages[rows+n, :, spec['sender'], :])
        old = dict(unit=4, window=1, position=0, arm='same', direction=0,
                   new_forward_worlds=n, new_network_samples=3*n, neural_forward_calls=3,
                   seed=51101, condition='PL_live', natural_source='prior_natural_source',
                   path='synthetic_old_file.npz', data_sha256='synthetic_digest',
                   routing_sha256={'first_routes':'f', 'second_routes':'s', 'action_inputs':'a'},
                   worlds=n, is_silent_alias=False, max_identity_error=0.,
                   donor_packets_sha256='synthetic_packet_digest', outward_patch_visible=True)
        with mock.patch.object(runner, 'load_npz', return_value=data), \
             mock.patch.object(runner.pr, 'execute_cell', side_effect=AssertionError('Anchor cannot forward')):
            got, rec = runner.position_data({'prior_position_records':[old]}, 6000, None, None,
                    {'messages':messages}, spec, full_ids, 4, 'same', 0, True)
        # These are exactly the legacy keys tested by audit() before reusing the
        # previous audited file. New provenance names do not overwrite them.
        for key in set(old)-{'new_forward_worlds', 'new_network_samples', 'neural_forward_calls'}:
            self.assertEqual(rec[key], old[key])
        self.assertIs(got, data)
        self.assertEqual(rec['index_scope'], 'prior_full_domain')
        self.assertEqual(rec['old_actual_forward_worlds'], n)
        self.assertEqual(rec['new_network_samples'], 0)
        self.assertEqual(rec['new_forward_worlds'], 0)
        self.assertEqual(rec['neural_forward_calls'], 0)
        self.assertEqual(rec['receiver_update'], 6000)
        self.assertEqual(rec['donor_update'], 6000)
        self.assertEqual(rec['kind'], 'position')
        np.testing.assert_array_equal(data['recipient_indices'], full_ids[spec['endpoint_indices'][:, 0]])
