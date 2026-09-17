"""Independently enumerate photo policies and derive frozen-function bounds."""
from pathlib import Path
import argparse
import importlib.util
import itertools
import json

import numpy as np
import torch

from audit_partner_communication import SavedBank

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


@torch.no_grad()
def audit(folder):
    bank = SavedBank(folder)
    spec = importlib.util.spec_from_file_location('bounds_audit_saved_agents', folder / 'source/agents.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pools = [np.flatnonzero((bank.splits == 'test') & (bank.labels == q)) for q in (0, 1)]
    same = [torch.as_tensor([(i, j) for i in pool for j in pool]) for pool in pools]
    mixed = torch.as_tensor([(i, j) for i in pools[0] for j in pools[1]] + [(i, j) for i in pools[1] for j in pools[0]])
    kinds = torch.from_numpy(bank.labels[mixed.numpy()])
    graph = {'original': [(0, 1), (2, 3)], 'cross': [(0, 2), (0, 3), (1, 2), (1, 3)],
             'all': list(itertools.combinations(range(4), 2))}
    output = []
    for seed in read(folder / 'config.json')['seeds']:
        saved = read(folder / f'frozen_function_bounds_s{seed}.json')
        origin = read(folder / f'both_learn_s{seed}/origin.json')['origin_path']
        states = torch.load(origin, weights_only=True)
        agents = [module.ResourceAgent() for _ in range(4)]
        for agent, state in zip(agents, states):
            agent.load_state_dict(state)
        p = {mode: np.empty((4, 16, 2, 5)) for mode in ('normal', 'stochastic')}
        a = {mode: np.empty((4, 16, 5, 2)) for mode in ('normal', 'stochastic')}
        for who, agent in enumerate(agents):
            for t in range(16):
                for q in (0, 1):
                    public = torch.zeros(len(same[q]), 3)
                    public[:, 2] = (t + 1) / 16
                    observation = agent.observe(bank.features[same[q]], public)
                    probabilities = agent.send(observation[1]).softmax(-1)
                    choices = probabilities.argmax(-1).numpy()
                    p['normal'][who, t, q] = np.bincount(choices, minlength=5) / len(choices)
                    p['stochastic'][who, t, q] = probabilities.double().numpy().mean(0)
                public = torch.zeros(len(mixed), 3)
                public[:, 2] = (t + 1) / 16
                observation = agent.observe(bank.features[mixed], public)
                for symbol in range(5):
                    probabilities = agent.act(*observation, torch.full((len(mixed),), symbol, dtype=torch.int64)).softmax(-1)
                    selected = kinds[torch.arange(len(mixed)), probabilities.argmax(-1)].numpy()
                    a['normal'][who, t, symbol] = np.bincount(selected, minlength=2) / len(selected)
                    for q in (0, 1):
                        a['stochastic'][who, t, symbol, q] = (probabilities * (kinds == q)).sum(-1).double().mean().item()
        for d in (p, a):
            for mode in d:
                d[mode] /= d[mode].sum(-1, keepdims=True)
        discrepancies = []
        for mode in p:
            saved_p = np.asarray(saved['sender_distributions_agent_time_resource_symbol'][mode])
            saved_a = np.asarray(saved['listener_resource_probabilities_agent_time_symbol_resource'][mode])
            discrepancies += [float(np.max(np.abs(p[mode] - saved_p))), float(np.max(np.abs(a[mode] - saved_a)))]
            np.testing.assert_allclose(p[mode], saved_p, atol=1e-8, rtol=0)
            np.testing.assert_allclose(a[mode], saved_a, atol=1e-8, rtol=0)
            for name, pairs in graph.items():
                receiving, sending = [], []
                for target in range(4):
                    peers = [y if x == target else x for x, y in pairs if target in (x, y)]
                    for t in range(16):
                        distributions = p[mode][peers, t].mean(0)
                        # Bayes error under equal resource priors equals half the
                        # distribution overlap, a separate form of the TV bound.
                        receiving.append(1 - .5 * np.minimum(distributions[0], distributions[1]).sum())
                        for restricted in (0, 1):
                            code_values = [a[mode][peers, t, symbol, 1 - restricted].mean() for symbol in range(5)]
                            sending.append(max(code_values))
                fixed_sender = (6 + 8 * np.mean(receiving)) / 14
                fixed_listener = (6 + 8 * np.mean(sending)) / 14
                assert abs(fixed_sender - saved['fixed_sender'][mode][name]['full_success_upper_bound']) < 1e-8
                assert abs(fixed_listener - saved['fixed_listener'][mode][name]['full_success_upper_bound']) < 1e-8
        output.append({'seed': seed, 'max_probability_discrepancy': max(discrepancies),
                       'all_test_photo_pairs_and_public_times_reenumerated': True,
                       'greedy_and_sampling_bounds_for_three_partner_graphs_recomputed': True})
        print(json.dumps({'bounds_audited': seed}), flush=True)
    report = {'status': 'passed', 'populations': len(output), 'runs': output,
              'scope': 'Independent finite-photo probability enumeration and Bayes-overlap / code-oracle derivations. Bounds are expectations, not sample-score ceilings.'}
    (folder / '固定函数上界独立核查.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=ROOT / 'results/mechanism_001')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.directory)
    print(json.dumps({'status': result['status'], 'populations': result['populations']}))
