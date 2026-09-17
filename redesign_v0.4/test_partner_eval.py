"""Engineering-only tests; artificial protocols are not experimental evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from partner_eval import (PAIRS, _seed, checkpoint_population,
                          evaluate_population, sender_alignment)
from test_curriculum_eval import ProtocolAgent, ToyBank


class RecordingAgent(ProtocolAgent):
    def observe(self, features, public):
        self.last_input = (features.clone(), public.clone())
        return super().observe(features, public)


class PermutedSender(RecordingAgent):
    def send(self, local):
        logits = super().send(local)
        return logits[:, [0, 2, 1, 3, 4]]


class PopulationEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.agents = [RecordingAgent(i % 2) for i in range(4)]
        self.bank = ToyBank()

    def test_checkpoint_all_pairs_modes_groups_and_repeated_seed(self):
        a = checkpoint_population(self.agents, self.bank, 5821, n=128)
        b = checkpoint_population(self.agents, self.bank, 5821, n=128)
        self.assertEqual(a, b)
        self.assertEqual([tuple(p['agents']) for p in a['pairs']], list(PAIRS))
        self.assertEqual(a['aggregates']['original']['pair_count'], 2)
        self.assertEqual(a['aggregates']['cross']['pair_count'], 4)
        self.assertEqual(a['aggregates']['all']['pair_count'], 6)
        self.assertNotIn('sender_alignment', a)
        seeds, hashes = [], []
        for pair in a['pairs']:
            self.assertEqual(set(pair['tasks']), {'full'})
            self.assertEqual(set(pair['tasks']['full']), {'normal', 'shuffle', 'stochastic'})
            self.assertNotIn('intervention', pair)
            self.assertNotIn('traces', pair)
            outputs = pair['tasks']['full']
            self.assertEqual(len({v['external_cases_sha256'] for v in outputs.values()}), 1)
            hashes.append(outputs['normal']['external_cases_sha256'])
            seeds.append(pair['evaluation_seeds']['full'])
            for d in outputs['normal']['by_direction']:
                self.assertEqual(d['population_sender'], pair['agents'][d['restricted_sender']])
                self.assertEqual(d['population_receiver'], pair['agents'][d['mixed_receiver']])
        self.assertEqual(len(set(seeds)), 6)
        self.assertEqual(len(set(hashes)), 6)
        for group in ('original', 'cross', 'all'):
            selected = [p for p in a['pairs'] if group == 'all' or p['group'] == group]
            expected = np.mean([p['tasks']['full']['normal']['balanced_gathering'] for p in selected])
            self.assertEqual(a['aggregates'][group]['tasks']['full']['normal']['mean_reward_per_step'], expected)
        json.dumps(a, allow_nan=False)

    def test_final_traces_pair_external_inputs_and_global_directions(self):
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate_population(self.agents, self.bank, 7691, n=128, trace_dir=folder,
                                         intervention_n=128, alignment_n=96)
            self.assertEqual(len(list(Path(folder).glob('*.jsonl'))), 48)
            for pair in result['pairs']:
                self.assertEqual(set(pair['tasks']), {'full', 'curriculum'})
                for task in ('full', 'curriculum'):
                    values = pair['tasks'][task]
                    self.assertEqual(set(values), {'normal', 'shuffle', 'blank', 'stochastic'})
                    self.assertEqual(len({v['external_cases_sha256'] for v in values.values()}), 1)
                    paths = [pair['traces'][f'{task}_{mode}']['path'] for mode in ('normal', 'shuffle', 'blank')]
                    rows = [[json.loads(line) for line in Path(path).read_text().splitlines()] for path in paths]
                    for triple in zip(*rows):
                        for key in ('case', 'remaining', 'inventory', 'kinds', 'image_ids', 'sent'):
                            self.assertEqual(triple[0][key], triple[1][key])
                            self.assertEqual(triple[0][key], triple[2][key])
                for d in pair['intervention']['directions']:
                    self.assertEqual(d['population_sender'], pair['agents'][d['sender']])
                    self.assertEqual(d['population_receiver'], pair['agents'][d['receiver']])
                for task, task_seed in pair['evaluation_seeds'].items():
                    self.assertNotEqual(task_seed, pair['intervention_seed'])
            self.assertEqual(result['sender_alignment']['cases'], 96)
            json.dumps(result, allow_nan=False)

    def test_alignment_uses_same_photos_and_public_without_symbol_relabeling(self):
        result = sender_alignment(self.agents, self.bank, 901, n=192)
        self.assertEqual(result['all_four_greedy_agreement'], 1)
        self.assertFalse(result['symbol_permutation_applied'])
        for pair in result['pairs']:
            self.assertEqual(pair['mean_total_variation'], 0)
        for agent in self.agents[1:]:
            torch.testing.assert_close(agent.last_input[0], self.agents[0].last_input[0], rtol=0, atol=0)
            torch.testing.assert_close(agent.last_input[1], self.agents[0].last_input[1], rtol=0, atol=0)
        public = self.agents[0].last_input[1]
        kinds = self.agents[0].last_input[0][..., 0].sum(1).numpy()
        for time in torch.unique(public[:, 2]):
            counts = np.bincount(kinds[(public[:, 2] == time).numpy()].astype(int), minlength=3)
            self.assertEqual(counts.tolist(), [4, 4, 4])
        self.agents[3] = PermutedSender(1)
        changed = sender_alignment(self.agents, self.bank, 901, n=192)
        self.assertEqual(result['external_cases_sha256'], changed['external_cases_sha256'])
        self.assertAlmostEqual(changed['all_four_greedy_agreement'], 1 / 3)
        p03 = next(p for p in changed['pairs'] if p['agents'] == [0, 3])
        self.assertAlmostEqual(p03['greedy_agreement'], 1 / 3)
        self.assertAlmostEqual(p03['mean_total_variation'], 2 / 3, places=6)

    def test_input_validation_and_stream_separation(self):
        with self.assertRaises(ValueError):
            checkpoint_population(self.agents[:3], self.bank, 3)
        with self.assertRaises(ValueError):
            checkpoint_population([self.agents[0]] * 4, self.bank, 3)
        with self.assertRaises(ValueError):
            checkpoint_population(self.agents, self.bank, 3, n=0)
        with self.assertRaises(ValueError):
            evaluate_population(self.agents, self.bank, 3, intervention_n=0)
        self.assertEqual(_seed(102, 100, 1), _seed(102, 100, 1))
        self.assertEqual(len({_seed(102, s, p) for s in (100, 101, 200, 300) for p in range(6)}), 24)


if __name__ == '__main__':
    unittest.main()
