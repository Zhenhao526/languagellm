"""Describe every natural saved code stream; no network loading or forward."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import argparse
import json
import time
import numpy as np

SEEDS = (49101, 49102, 49103, 49104)
ECOS = ('unique', 'multiple')
CONDS = ('FI_silent', 'FI_live', 'PI_silent', 'PI_live')
PARTS = ('train', 'heldout_layouts')
STEPS = (0, 100, 500, 1500, 3000, 6000)
CONTRACT = dict(
    scope='Complete natural generated-message description, not semantics or a new causal measure',
    frequency_records=448, adjacent_checkpoint_records=320, fixed_example_records=1920,
    examples='First five ascending monitor state indices at every checkpoint, with all outcomes retained',
    weighting='Full endpoint target weights; monitor equal weights on fixed 672 worlds',
    silent='Generated own codes included; not claimed to be delivered across people',
    neural_forward_calls=0, weight_loads=0, training_updates=0,
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def array_digest(a):
    # The runner hashes contiguous array bytes, including shape and dtype.
    a = np.ascontiguousarray(a)
    header = (json.dumps(dict(shape=list(a.shape), dtype=a.dtype.str),
                        ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    h = sha256(header)
    h.update(a.tobytes())
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def frequencies(messages, weights):
    assert messages.ndim == 4 and messages.shape[1:] == (2, 3, 4)
    assert messages.dtype.kind in 'iu' and np.all((messages >= 0) & (messages < 8))
    assert weights.shape == (len(messages),) and np.all(weights > 0)
    assert abs(weights.sum() - 1) < 1e-12
    codes = (messages.astype(np.int64) * np.array([512, 64, 8, 1])).sum(-1)
    counts = np.zeros((2, 3, 4096), dtype=np.int64)
    masses = np.zeros((2, 3, 4096), dtype=np.float64)
    tokens = np.zeros((2, 3, 4, 8), dtype=np.float64)
    for window in range(2):
        for agent in range(3):
            counts[window, agent] = np.bincount(codes[:, window, agent], minlength=4096)
            masses[window, agent] = np.bincount(codes[:, window, agent], weights=weights, minlength=4096)
            for position in range(4):
                tokens[window, agent, position] = np.bincount(
                    messages[:, window, agent, position], weights=weights, minlength=8)
    assert np.all(counts.sum(-1) == len(messages))
    assert np.allclose(masses.sum(-1), 1, rtol=0, atol=1e-11)
    assert np.allclose(tokens.sum(-1), 1, rtol=0, atol=1e-11)
    logp = np.zeros_like(masses)
    np.log2(masses, out=logp, where=masses > 0)
    stats = dict(string_types=np.count_nonzero(counts, axis=-1).tolist(),
                 modal_string_mass=masses.max(-1).tolist(),
                 string_entropy_bits=(-(masses * logp).sum(-1)).tolist())
    return counts, masses, tokens, stats


def analyze(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    assert not out.exists()
    status = read(run / 'execution/status.json')
    root = read(run / 'execution/results.json')
    assert status['status'] == 'completed' and status['completed_runs'] == 32
    assert root['status'] == 'completed' and root['completed_run_count'] == 32
    expected = [(s, e, c) for s in SEEDS for e in ECOS for c in CONDS]
    assert [(r['seed'], r['ecology'], r['condition']) for r in root['runs']] == expected
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources = {str(Path(__file__).resolve()): digest(__file__)}
    for rel in ('plan.json', 'prepared.json', 'freeze.json', 'execution/status.json', 'execution/results.json'):
        sources[str(run / rel)] = digest(run / rel)
    records, changes, examples, counts_all, mass_all, tokens_all = [], [], [], [], [], []
    try:
        for policy in root['runs']:
            key = {k: policy[k] for k in ('seed', 'ecology', 'condition')}
            monitoring = {row['update']: row['monitor'] for row in policy['monitor']}
            assert set(monitoring) == set(STEPS)
            previous = {}
            for stage, step in [('monitor', u) for u in STEPS] + [('final', 6000)]:
                for part in PARTS:
                    score = (monitoring[step][part] if stage == 'monitor' else policy['final'][part])['natural']
                    path = run / 'execution' / score['data_file']
                    wp = run / 'execution' / score['weights_file']
                    sources[str(path)] = digest(path)
                    sources[str(wp)] = digest(wp)
                    assert sources[str(path)] == score['data_sha256']
                    assert sources[str(wp)] == score['weights_file_sha256']
                    with np.load(wp, allow_pickle=False) as z:
                        weights = z[score['weight_array']]
                    assert array_digest(weights) == score['weights_sha256']
                    with np.load(path, allow_pickle=False) as z:
                        ids, states, msg, actions, reward = [z[k] for k in
                            ('state_indices', 'states', 'messages', 'action_indices', 'greedy_reward')]
                    assert np.all(np.diff(ids) > 0)
                    assert array_digest(msg) == score['uniform']['messages_sha256']
                    assert array_digest(states) == score['uniform']['packed_states_sha256']
                    assert array_digest(actions) == score['uniform']['action_indices_sha256']
                    assert len(weights) == len(ids) == score['uniform']['worlds']
                    assert abs(float(weights @ (reward == 1)) - score['weighted']['full_success_rate']) < 1e-12
                    counts, mass, token_mass, stats = frequencies(msg, weights)
                    row = dict(**key, stage=stage, update=step, partition=part, worlds=len(ids),
                               frequency_array_row=len(records), source_file=str(path), **stats)
                    records.append(row)
                    counts_all.append(counts); mass_all.append(mass); tokens_all.append(token_mass)
                    if stage == 'monitor':
                        assert len(ids) == 672 and np.allclose(weights, 1 / 672, rtol=0, atol=1e-15)
                        if part in previous:
                            old_step, old_ids, old_states, old_msg = previous[part]
                            assert np.array_equal(ids, old_ids) and np.array_equal(states, old_states)
                            different = old_msg != msg
                            changes.append(dict(**key, partition=part, before_update=old_step, after_update=step,
                                worlds=672, changed_strings_by_window_agent=different.any(-1).sum(0).tolist(),
                                changed_token_counts=different.sum(0).tolist(),
                                worlds_with_any_generated_message_change=int(different.any(axis=(1, 2, 3)).sum())))
                        previous[part] = (step, ids, states, msg)
                        for i in range(5):
                            examples.append(dict(**key, partition=part, update=step, state_index=int(ids[i]),
                                researcher_packed_state=states[i].tolist(), generated_codes=msg[i].tolist(),
                                action_indices=actions[i].tolist(), reward=float(reward[i]),
                                cross_person_delivery=key['condition'].endswith('_live'),
                                selection='First five fixed monitor indices; no success or string selection'))
        assert (len(records), len(changes), len(examples)) == (448, 320, 1920)
        np.savez_compressed(out / 'message_frequencies.npz', raw_string_counts=np.stack(counts_all),
                            weighted_string_mass=np.stack(mass_all), weighted_position_symbol_mass=np.stack(tokens_all))
        with (out / 'fixed_examples.jsonl').open('x', encoding='utf-8') as f:
            for row in examples:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        assert all(digest(p) == h for p, h in sources.items())
        result = dict(status='completed', completed_at_utc=datetime.now(timezone.utc).isoformat(),
            contract=CONTRACT, input_sha256=sources, records=records, adjacent_changes=changes,
            output_sha256={name: digest(out / name) for name in ('message_frequencies.npz', 'fixed_examples.jsonl')},
            elapsed_seconds=time.perf_counter() - started)
        write(out / 'summary.json', result)
        return dict(status='completed', records=448, adjacent_changes=320, examples=1920,
                    summary_sha256=digest(out / 'summary.json'), elapsed_seconds=result['elapsed_seconds'])
    except BaseException as exc:
        write(out / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.out), ensure_ascii=False))
