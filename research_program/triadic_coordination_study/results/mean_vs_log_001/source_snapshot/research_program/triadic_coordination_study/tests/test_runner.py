"""No training: frozen boundaries, stable objective semantics and pairing."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from research_program.triadic_coordination_study import runner as r


class ObjectiveBoundaryTests(unittest.TestCase):
    def test_prepared_domain_and_baseline_config_unchanged(self):
        before = deepcopy(r.base.CONFIG)
        original = r.base.make_prepared()
        prepared = r.make_prepared()
        self.assertEqual(r.base.CONFIG, before)
        self.assertEqual(prepared['partitions'], original['partitions'])
        self.assertEqual(prepared['actions'], original['actions'])
        self.assertEqual([(x['seed'], x['objective']) for x in prepared['runs']],
                         [(s, o) for s in r.SEEDS for o in r.OBJECTIVES])
        self.assertEqual(prepared['training_state_samples_total'], 12288000)
        self.assertEqual(prepared['weighted_structural_action_contributions'], 294912000)
        self.assertEqual(prepared['complete_final_world_evaluations'], 1147392)

    def test_prepare_never_initializes_actor_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'prepared'
            with (patch.object(r.base, 'make_actor', side_effect=AssertionError('no actor')),
                  patch.object(r, 'train_run', side_effect=AssertionError('no training'))):
                result = r.prepare(p)
                r.verify(p)
                self.assertEqual(result['status'], 'prepared_not_trained')
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    r.prepare(p)
            self.assertFalse((p / 'execution').exists())

    def test_tamper_rejected_before_execution_or_data_build(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'prepared'
            r.prepare(p)
            data = r.read(p / 'prepared.json')
            data['runs'][0]['seed'] += 1
            (p / 'prepared.json').write_text(json.dumps(data))
            with patch.object(r.base, 'build_arrays', side_effect=AssertionError('no arrays')):
                with self.assertRaisesRegex(ValueError, 'Prepared input changed'):
                    r.execute(p)
            self.assertFalse((p / 'execution').exists())

    def test_frozen_baseline_snapshot_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'prepared'
            r.prepare(p)
            source = p / 'source_snapshot/research_program/triadic_learning_baseline/runner.py'
            source.write_text(source.read_text() + '\n# fixture corruption\n')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                r.verify(p)

    def test_started_batch_cannot_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'prepared'
            r.prepare(p)
            (p / 'execution').mkdir()
            with patch.object(r.base, 'build_arrays', side_effect=AssertionError('no arrays')):
                with self.assertRaisesRegex(ValueError, 'Never resume'):
                    r.execute(p)

    def test_mean_log_not_log_mean_and_zero_R_retained_in_softmax(self):
        logits = np.zeros((2, 3, 17))
        rewards = np.zeros((2, 24))
        rewards[0, 0] = 1
        rewards[1, 0] = .5
        terms = r.objective_terms(logits, rewards)
        np.testing.assert_allclose(terms['J'], [1/17**3, .5/17**3], atol=0, rtol=1e-15)
        self.assertLess(terms['log_J'].mean(), np.log(terms['J'].mean()))
        self.assertEqual(np.count_nonzero(terms['posterior_weights']), 2)
        self.assertTrue((terms['probabilities'] > 0).all())

    def test_underflow_J_does_not_destroy_log_objective_gradient(self):
        logits = np.full((1, 3, 17), -1000.0)
        logits[:, :, 0] = 0
        rewards = np.zeros((1, 24)); rewards[0, 0] = 1
        terms = r.objective_terms(logits, rewards)
        self.assertEqual(terms['J'].tolist(), [0.0])
        self.assertEqual(terms['log_J'].tolist(), [-2000.0])
        selected = r.loss_and_derivative(terms, 'mean_log_J', 6000)
        self.assertTrue(np.isfinite(selected['derivative']).all())
        self.assertGreater(np.linalg.norm(selected['derivative']), 0)
        self.assertEqual(selected['loss'], 2000)
        ordinary = r.loss_and_derivative(terms, 'mean_J', 6000)
        np.testing.assert_array_equal(ordinary['derivative'], np.zeros((1,3,17)))

    def test_no_positive_native_support_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive outcome'):
            r.objective_terms(np.zeros((1,3,17)), np.zeros((1,24)))

    def test_primary_retains_all_paired_directions_and_requires_eight_runs(self):
        records = []
        changes = (-1, 3, 0, 3)
        for i, seed in enumerate(r.SEEDS):
            for obj in r.OBJECTIVES:
                n = 100 + (changes[i] if obj == 'mean_log_J' else 0)
                records.append({'seed': seed, 'objective': obj,
                    'final': {'new_needs_and_layouts': {'worlds': 8244, 'greedy_full_success_rate': n/8244}}})
        result = r.primary_comparison(records)
        self.assertEqual(len(result['seed_pairs']), 4)
        self.assertLess(result['seed_pairs'][0]['paired_difference_log_minus_mean'], 0)
        self.assertAlmostEqual(result['equal_weight_mean_paired_difference'], 1.25/8244, places=15)
        with self.assertRaisesRegex(ValueError, 'eight runs'):
            r.primary_comparison(records[:-1])
        duplicate = deepcopy(records); duplicate[-1] = deepcopy(duplicate[0])
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
            r.primary_comparison(duplicate)

    def test_pairing_checks_full_log_tail_and_batch_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            records = []
            lines = [json.dumps({'update': u, 'batch_indices_sha256': f'b{u}',
                'batch_states_sha256': f's{u}', 'entropy_coefficient': r.base.entropy_coefficient(u)})+'\n' for u in range(1,6001)]
            for seed in r.SEEDS:
                for obj in r.OBJECTIVES:
                    directory = p / f'seed_{seed}_{obj}'; directory.mkdir()
                    (directory / 'training.jsonl').write_text(''.join(lines))
                    records.append({'seed': seed, 'objective': obj, 'initial_parameter_sha256': f'i{seed}',
                        'monitor': [{'monitor': {'fixed_fixture': seed}}]})
            self.assertEqual(len(r.check_pairing(p, records)), 4)
            path = p / f'seed_{r.SEEDS[0]}_mean_J/training.jsonl'
            with path.open('a') as stream: stream.write(lines[-1])
            with self.assertRaisesRegex(ValueError, 'Unequal paired training log length'):
                r.check_pairing(p, records)
            path.write_text(''.join(lines))
            changed = list(lines); changed[500] = changed[500].replace('"b501"', '"not_same_batch"')
            path.write_text(''.join(changed))
            with self.assertRaisesRegex(ValueError, 'Paired world stream'):
                r.check_pairing(p, records)


if __name__ == '__main__':
    unittest.main()
