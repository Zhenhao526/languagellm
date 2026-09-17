"""Engineering-only fixed-code receiver control; never an emergence result.

Uses the original ResourceAgent and cached frozen DINO photo features. Human
symbol 1 means collect food; symbol 2 means collect water. The recipient gets
one option of each, with randomized left/right position, and public inventory
[0, 0]. A scalar 0/1 consequence is the sole learning signal in the primary
control. Ground-truth category IDs are used by the simulator/reward and scorer,
never passed directly into the policy. This diagnoses the recipient path; it
does not validate sender learning or a two-agent emergent convention.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents import ResourceAgent, draw
from run_pilot import ImageBank, write_json


def batch_data(bank, rng, n, split):
    first = rng.integers(2, size=n)
    kinds = np.column_stack([first, 1 - first])
    target = np.tile(np.arange(2), (n + 1) // 2)[:n]
    rng.shuffle(target)
    own, ids = bank.sample(kinds, split, rng)
    public = torch.zeros((n, 3), dtype=torch.float32)
    public[:, 2] = torch.from_numpy(rng.integers(1, 17, size=n).astype(np.float32) / 16)
    symbols = torch.from_numpy(target + 1)
    return own, public, symbols, kinds, target, ids


@torch.no_grad()
def evaluate(agent, bank, seed, n=4096):
    rng = np.random.default_rng(seed)
    own, public, symbols, kinds, target, _ = batch_data(bank, rng, n, 'test')
    options, local = agent.observe(own, public)
    probs = agent.act(options, local, symbols).softmax(-1)
    swapped_probs = agent.act(options, local, 3 - symbols).softmax(-1)
    silent_probs = agent.act(options, local, torch.zeros_like(symbols)).softmax(-1)
    choices = probs.argmax(-1).numpy()
    swaps = swapped_probs.argmax(-1).numpy()
    silent = silent_probs.argmax(-1).numpy()
    got = kinds[np.arange(n), choices]
    flipped = kinds[np.arange(n), swaps]
    silent_got = kinds[np.arange(n), silent]
    correct_option = torch.from_numpy((kinds[:, 0] != target).astype(np.int64))
    return {
        'n': n,
        'greedy_target_accuracy': float((got == target).mean()),
        'accuracy_by_target_kind': [float((got[target == k] == k).mean()) for k in (0, 1)],
        'probability_mass_on_target': float(probs.gather(1, correct_option[:, None]).mean()),
        'same_observation_message_swap_resource_flip': float((got != flipped).mean()),
        'swapped_message_original_target_accuracy': float((flipped == target).mean()),
        'silenced_original_target_accuracy': float((silent_got == target).mean()),
        'message_swap_mean_action_total_variation': float((.5 * (probs - swapped_probs).abs().sum(-1)).mean()),
        'food_choice_fraction': float((got == 0).mean()),
    }


def initialize(seed, condition):
    torch.manual_seed(seed * 1000)
    agent = ResourceAgent()
    if condition == 'prepared':
        weights = torch.load(ROOT / 'results/pilot_001' / f'prepared_s{seed}.pt', map_location='cpu', weights_only=True)
        agent.load_state_dict(weights[0], strict=True)
    return agent


def run_condition(bank, seed, condition, lr, config, method='reward_only'):
    agent = initialize(seed, condition)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    rng = np.random.default_rng(seed * 10000 + 801)
    policy_rng = np.random.default_rng(seed * 10000 + 802)
    evaluation_seed = seed + 820000
    record = {'seed': seed, 'initialization': condition, 'learning_rate': lr,
              'method': method, 'initial': evaluate(agent, bank, evaluation_seed),
              'curve': [], 'model_emergence_evidence': False}
    start = time.monotonic()
    for update in range(1, config['updates'] + 1):
        n = config['batch']
        own, public, symbols, kinds, target, _ = batch_data(bank, rng, n, 'train')
        options, local = agent.observe(own, public)
        logits = agent.act(options, local, symbols)
        choice, logp, entropy = draw(logits, policy_rng)
        got = kinds[np.arange(n), choice.numpy()]
        reward = torch.from_numpy((got == target).astype(np.float32))
        if method == 'reward_only':
            loss = -((reward - .5) * logp).mean() - config['entropy'] * entropy.mean()
        elif method == 'supervised_ce_capacity_only':
            labels = torch.from_numpy((kinds[:, 0] != target).astype(np.int64))
            loss = F.cross_entropy(logits, labels)
        else:
            raise ValueError(method)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), config['clip_norm'])
        optimizer.step()
        if update % 100 == 0:
            state = {'update': update, 'train_sampled_reward': float(reward.mean()),
                     'heldout': evaluate(agent, bank, evaluation_seed)}
            record['curve'].append(state)
    record['final'] = evaluate(agent, bank, evaluation_seed)
    record['seconds'] = time.monotonic() - start
    record['trained_parameters'] = [name for name, p in agent.named_parameters() if p.grad is not None]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent / 'positive_control_results'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = {'purpose': 'engineering receiver-path positive control, not language emergence',
              'seeds': [101, 202, 303], 'recipient': 0,
              'initializations': ['prepared', 'random'], 'learning_rates': [1e-3, 3e-4],
              'updates': 500, 'batch': 128, 'entropy': .01, 'clip_norm': 2,
              'evaluation_n': 4096, 'test_split_only_for_scoring': True,
              'symbol_codebook': {'1': 'food', '2': 'water'},
              'target_sampling': 'exact 50/50 each batch, shuffled',
              'local_options': 'one food plus one water, randomized positions',
              'public': '[0,0,remaining/16], remaining uniform integer 1..16',
              'supervision_primary': 'scalar reward only, fixed baseline 0.5',
              'conditional_secondary': 'If no primary final accuracy >=0.9, run supervised CE 500x128 at lr1e-3 in all seeds/initializations; capacity control only',
              'no_success_based_budget_extension': True,
              'torch': torch.__version__, 'python': platform.python_version(), 'device': 'cpu',
              'source_sha256': {filename: hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
                                for filename in ('agents.py', 'run_pilot.py')},
              'prepared_sha256': {str(seed): hashlib.sha256((ROOT / 'results/pilot_001' / f'prepared_s{seed}.pt').read_bytes()).hexdigest() for seed in (101, 202, 303)}}
    write_json(out / 'config.json', config)
    print(json.dumps({'config_before_execution': config}, ensure_ascii=False), flush=True)
    bank = ImageBank()
    records = []
    for seed in config['seeds']:
        for condition in config['initializations']:
            for lr in config['learning_rates']:
                record = run_condition(bank, seed, condition, lr, config)
                records.append(record)
                write_json(out / 'results.json', records)
                print(json.dumps({k: record[k] for k in ('seed', 'initialization', 'learning_rate', 'method', 'initial', 'final', 'seconds')}, ensure_ascii=False), flush=True)
    if not any(row['final']['greedy_target_accuracy'] >= .9 for row in records):
        for seed in config['seeds']:
            for condition in config['initializations']:
                record = run_condition(bank, seed, condition, 1e-3, config, 'supervised_ce_capacity_only')
                records.append(record)
                write_json(out / 'results.json', records)
                print(json.dumps({k: record[k] for k in ('seed', 'initialization', 'learning_rate', 'method', 'final', 'seconds')}, ensure_ascii=False), flush=True)
    write_json(out / 'completed.json', {'status': 'complete', 'runs': len(records),
                                        'model_emergence_evidence': False})


if __name__ == '__main__':
    main()
