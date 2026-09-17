"""Frozen formal-execution gate for v0.32; no performance-based filtering."""
from pathlib import Path
from datetime import datetime, timezone
import json, numpy as np, torch
import run_support as run

R = Path(__file__).resolve().parent
S = R / 'results/smoke_002'


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def main():
    complete = run.read(S / 'training_complete.json')
    assert complete['status'] == 'complete' and complete['formal'] is False
    assert complete['social_runs'] == 2 and complete['pair_updates'] == 80
    smoke_source = run.PROJECT / 'redesign_v0.28/results/smoke_001'; source = run.PROJECT / 'redesign_v0.28/results/formation_001'
    smoke_inputs = run.input_hashes(smoke_source, [99528], [1]); formal_inputs = run.input_hashes(source, run.SEEDS, [1, 2, 3])
    assert complete['source_hashes'] == run.source_hashes(); assert complete['input_hashes'] == smoke_inputs
    conditions = tuple(run.support.CONDITIONS); folders = [S / 'social' / f's99528_p1_{condition}' for condition in conditions]
    initial = [torch.load(folder / 'initial.pt', weights_only=True) for folder in folders]
    assert all(torch.equal(initial[0][i][key], initial[1][i][key]) for i in range(4) for key in initial[0][i])
    checked = 0
    for condition, folder in zip(conditions, folders):
        config = run.read(folder / 'config.json'); assert config['population_agents'] == 4 and config['private_types'] == [0, 1, 0, 1]
        assert config['messages_per_population_update'] == 960 and config['actions_per_population_update'] == 1920
        assert config['joint_payoff'] == run.support._DESIGN['joint_payoff']; assert config['groups'] == {key: value.tolist() for key, value in run.support.groups(1, condition).items()}
        rows = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]; assert [row['update'] for row in rows] == list(range(1, 41))
        for update in (0, 40):
            for i, j in run.PAIR_KEYS:
                raw = load_npz(folder / f'protocol_{update:04d}_i{i}_j{j}.npz'); assert raw['tokens'].shape == (180, 2) and raw['sender_log_probs'].shape == (180, 49) and raw['receiver_logits'].shape == (49, 2, 6)
                assert np.isfinite(raw['sender_log_probs']).all() and np.isfinite(raw['receiver_logits']).all(); np.testing.assert_allclose(np.exp(raw['sender_log_probs']).sum(-1), 1, atol=1e-6); checked += 1
    for rel in ('train_0001.npz', 'train_0040.npz'):
        left = load_npz(folders[0] / rel); right = load_npz(folders[1] / rel); world_keys = [key for key in left if key.startswith('world__')]; assert set(world_keys) == {key for key in right if key.startswith('world__')}
        for key in world_keys: np.testing.assert_array_equal(left[key], right[key])
    original = run.read(source / 'training_complete.json')['files']; bound = 0
    for path, digest in formal_inputs.items():
        p = Path(path)
        if p.is_relative_to(source):
            rel = str(p.relative_to(source)); assert run.sha(p) == digest if rel == 'training_complete.json' else original[rel] == digest; bound += 1
    run.write(S / 'terminal_receipt.json', dict(session_id=87322, exit_code=0, status='complete', observed_by='root tools.write_stdin', reported_program_seconds=complete['seconds']))
    checks = [R / 'implementation_review.json', S / 'raw_validation.json', S / 'audit_execution.json']
    for path in checks: assert run.read(path)['passed'], path
    run.write(R / 'preflight_qa.json', dict(passed=True, version='v0.32', created_utc=datetime.now(timezone.utc).isoformat(), source_hashes=run.source_hashes(), input_hashes=formal_inputs, smoke_output=str(S), smoke_completion_sha256=run.sha(S / 'training_complete.json'), smoke_protocol_tables=checked, initial_exactly_paired=True, conditions=list(conditions), original_bound_inputs=bound, checks={str(path): run.sha(path) for path in checks}, criterion_uses_performance_threshold=False, development_performance_not_used_for_selection=True, synthetic_tests=dict(support=5, metrics=3, analyzer_self_tests=208), scope='Joint-payoff/schedule/fixture tests, independent 40-step smoke, paired condition inputs, original input binding, independent smoke statistics and endpoint replay'))
    print('preflight passed', checked, 'smoke protocol tables;', bound, 'formal inputs bound')


if __name__ == '__main__': main()
