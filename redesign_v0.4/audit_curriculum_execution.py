"""Read-only checks of the fixed curriculum_001 execution and paired records."""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/curriculum_001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def states(path):
    return torch.load(path, weights_only=True, map_location='cpu')


def equal_states(left, right):
    return len(left) == len(right) and all(
        a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        for a, b in zip(left, right))


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def main():
    config = read(OUT / 'config.json')
    recorded = read(OUT / 'source_hashes.json')
    sources = {name: {'recorded': expected, 'archived': sha(OUT / 'source' / name),
                      'current': sha(ROOT / name)} for name, expected in recorded.items()}
    assert all(len(set(row.values())) == 1 for row in sources.values())
    # Import the archived definitions for configuration comparison only. Data
    # locations are checked explicitly below; no training function is called.
    sys.path.insert(0, str(OUT / 'source'))
    import run_curriculum as saved
    assert all(config[k] == v for k, v in saved.CONFIG.items())
    assert config['seeds'] == [101, 202, 303]
    assert config['conditions'] == ['baseline', 'signalling']
    assert (config['course_updates'], config['transition_updates'], config['pure_reward_updates']) == (600, 300, 300)
    assert config['batch_size'] == 1024 and config['horizon'] == 16
    assert config['sender_max_entropy_coefficient'] == 0

    data_checks = {}
    for name in ['manifest.json', 'features.npz', 'encoder_report.json']:
        data_checks[name] = {'archived': sha(OUT / ('data_' + name)), 'current': sha(ROOT / 'data' / name)}
        assert len(set(data_checks[name].values())) == 1
    scaling = np.load(OUT / 'feature_scaling.npz')
    z = np.load(OUT / 'data_features.npz')['features'].astype(np.float32)
    entries = read(OUT / 'data_manifest.json')['images']
    train = np.asarray([entry['split'] == 'train' for entry in entries])
    center = z[train].mean(0)
    scale = float(np.sqrt(((z[train] - center) ** 2).mean()))
    assert np.array_equal(scaling['center'], center) and np.array_equal(scaling['scale'], scale)
    prepared_hashes = read(OUT / 'initial_checkpoint_hashes.json')
    assert all(prepared_hashes[str(s)] == sha(ROOT / f'results/pilot_001/prepared_s{s}.pt') for s in config['seeds'])
    aggregate = read(OUT / 'results.json')
    completion = read(OUT / 'completed.json')
    assert completion['status'] == 'completed' and completion['runs'] == 6 and len(aggregate) == 6

    runs = {}
    for condition in config['conditions']:
        for seed in config['seeds']:
            name = f'{condition}_s{seed}'
            path = OUT / name
            rows = [json.loads(line) for line in (path / 'training_metrics.jsonl').read_text().splitlines()]
            result = read(path / 'result.json')
            assert result == next(x for x in aggregate if x['condition'] == condition and x['seed'] == seed)
            assert len(rows) == 1200 and [r['update'] for r in rows] == list(range(1, 1201))
            assert all(finite(r) for r in rows)
            assert result['updates'] == 1200 and result['joint_training_steps'] == 1228800
            assert result['final_aux_weight'] == result['final_action_entropy_weight'] == 0
            for index, row in enumerate(rows, 1):
                expected_stage = 'course' if index <= 600 else ('transition' if index <= 900 else 'pure_reward')
                expected_task = 'curriculum' if index <= 600 else 'full'
                expected_aux = (.1 if index <= 600 else (.1 * (1 - (index - 600) / 300) if index <= 900 else 0)) if condition == 'signalling' else 0
                assert row['stage'] == expected_stage and row['task'] == expected_task
                assert row['aux_weight'] == expected_aux
                assert row['action_entropy_weight'] == (.05 if index <= 900 else 0)
                assert 0 <= row['sampled_training_success'] <= 1
                assert [a['agent'] for a in row['agents']] == [0, 1]
            initial = states(path / 'initial.pt')
            assert equal_states(initial, states(ROOT / f'results/pilot_001/prepared_s{seed}.pt'))
            curve = read(path / 'learning_curve.json')
            assert [r['update'] for r in curve] == config['checkpoints']
            for checkpoint in config['checkpoints']:
                checkpoint_states = states(path / f'checkpoint_{checkpoint:04d}.pt')
                assert all(torch.isfinite(v).all() for state in checkpoint_states for v in state.values())
                if checkpoint == 0:
                    assert equal_states(initial, checkpoint_states)
            final = states(path / 'checkpoint_1200.pt')
            changes = [{module: sum(float((old[k] - new[k]).square().sum()) for k in old if k.startswith(module + '.')) ** .5
                        for module in ['project', 'sender', 'actor', 'value']} for old, new in zip(initial, final)]
            assert all(value > 0 for a in changes for value in a.values())
            for task in ['curriculum', 'full']:
                assert len({result[task][m]['external_cases_sha256'] for m in ['normal', 'shuffle', 'blank', 'stochastic']}) == 1
            runs[name] = {
                'updates': len(rows), 'stage_counts': {s: sum(r['stage'] == s for r in rows) for s in ['course', 'transition', 'pure_reward']},
                'finite_metrics': True, 'finite_all_checkpoints': True, 'initial_matches_pre_social_prepared': True,
                'all_final_300_aux_and_action_entropy_zero': True, 'parameter_l2_changes_by_agent': changes,
                'gradient_zero_counts_by_agent': [{module: sum(row['agents'][a]['gradient_norms'][module] == 0 for row in rows)
                                                  for module in ['project', 'sender', 'actor', 'value']} for a in [0, 1]],
                'final_course': result['curriculum']['normal']['balanced_gathering'],
                'final_full': result['full']['normal']['balanced_gathering'],
                'final_full_shuffle': result['full']['shuffle']['balanced_gathering'],
                'final_full_stochastic': result['full']['stochastic']['balanced_gathering'],
                'course_at_600': next(r['curriculum']['normal']['balanced_gathering'] for r in curve if r['update'] == 600),
                'world_hashes': [r['world_input_sha256'] for r in rows]}

    pairs = {}
    for seed in config['seeds']:
        a, b = [OUT / f'{c}_s{seed}' for c in config['conditions']]
        ra, rb = [read(p / 'result.json') for p in [a, b]]
        ca, cb = [read(p / 'learning_curve.json') for p in [a, b]]
        assert equal_states(states(a / 'initial.pt'), states(b / 'initial.pt'))
        assert runs[a.name]['world_hashes'] == runs[b.name]['world_hashes']
        for task in ['curriculum', 'full']:
            for mode in ['normal', 'shuffle', 'blank', 'stochastic']:
                assert ra[task][mode]['external_cases_sha256'] == rb[task][mode]['external_cases_sha256']
        for r1, r2 in zip(ca, cb):
            for task in ['curriculum', 'full']:
                for mode in ['normal', 'shuffle', 'stochastic']:
                    assert r1[task][mode]['external_cases_sha256'] == r2[task][mode]['external_cases_sha256']
        pairs[str(seed)] = {
            'same_initial_tensors': True, 'same_all_1200_external_training_batches': True,
            'same_all_checkpoint_and_final_external_evaluation_cases': True,
            'full_greedy_signalling_minus_baseline': rb['full']['normal']['balanced_gathering'] - ra['full']['normal']['balanced_gathering'],
            'full_stochastic_signalling_minus_baseline': rb['full']['stochastic']['balanced_gathering'] - ra['full']['stochastic']['balanced_gathering']}
    for row in runs.values():
        row.pop('world_hashes')
    audit = {'batch': 'curriculum_001', 'status': 'passed', 'source_hashes': sources,
             'archived_config_matches_source_CONFIG': True, 'archived_data_matches_current': data_checks,
             'archived_feature_scaling_matches_training_only_scaling': True, 'initial_checkpoint_hashes_match_original': True,
             'predeclared_fixed_budget_complete': True, 'runs': runs, 'paired_checks': pairs,
             'scope': 'Source/configuration/parameter/record consistency audit; no retraining, hyperparameter search or trajectory re-evaluation.'}
    target = ROOT / '课程实验_正式运行核查.json'
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'status': audit['status'], 'pairs': pairs, 'audit': str(target)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
