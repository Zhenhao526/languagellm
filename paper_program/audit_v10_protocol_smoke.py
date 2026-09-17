"""Independent NumPy-only review of saved v0.10 development probes.

No model imports, forward passes, training, or formal result reads.
"""
from pathlib import Path
from itertools import permutations
import ast
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / 'redesign_v0.10'
MAPS = list(permutations(range(6), 2))
MATCHINGS = {1: ((0, 1), (2, 3), (4, 5)),
             2: ((0, 2), (1, 4), (3, 5)),
             3: ((0, 3), (1, 5), (2, 4))}


def partition(p):
    def ids(k):
        pairs = {x for a, b in MATCHINGS[k] for x in ((a, b), (b, a))}
        return np.array([i for i, pair in enumerate(MAPS) if pair in pairs])
    added, sealed = ids(p), ids(p % 3 + 1)
    return dict(old=np.array([i for i in range(30) if i not in set(added) | set(sealed)]),
                added=added, sealed=sealed)


def probabilities(row):
    # Separate scalar-vector implementation, evaluated once per distribution.
    row = np.asarray(row, dtype=np.float64)
    weights = np.exp(row - max(row))
    return weights / sum(weights)


def independent_summary(z, p):
    receiver = np.array([[probabilities(z['receiver_logits'][m, g])
                          for g in range(2)] for m in range(49)])
    decoder = z['receiver_logits'].argmax(-1)
    values, counts, different_argmax = [], np.zeros((30, 49), dtype=int), 0
    for i, ((food, water), greedy) in enumerate(zip(z['positions'], z['greedy_message'])):
        first = probabilities(z['sender_first_logits'][i])
        second = np.array([probabilities(row) for row in z['sender_second_logits'][i]])
        distribution = np.array([first[a] * second[a, b] for a in range(7) for b in range(7)])
        assert np.isclose(sum(distribution), 1, atol=1e-12, rtol=0)
        a = int(first.argmax())
        native = (a, int(second[a].argmax()))
        assert tuple(greedy) == native
        code = native[0] * 7 + native[1]
        different_argmax += int(distribution.argmax() != code)
        products = [receiver[m, 0, food] * receiver[m, 1, water] for m in range(49)]
        n = int(tuple(decoder[code]) == (food, water))
        u = int(any(tuple(row) == (food, water) for row in decoder))
        q = sum(distribution[m] * products[m] for m in range(49))
        b = max(products)
        reward = sum(distribution[m] * (
            .25 * (receiver[m, 0, food] + receiver[m, 1, water]) + .5 * products[m])
            for m in range(49))
        assert 0 <= q <= b + 1e-12 <= 1 + 1e-12 and n <= u
        mid = MAPS.index((int(food), int(water)))
        counts[mid, code] += 1
        values.append((mid, n, u, q, b, reward))
    values = np.asarray(values)
    result = {}
    for group, pool in dict(partition(p), all=np.arange(30)).items():
        selected = values[np.isin(values[:, 0], pool)]
        result[group] = dict(n=len(selected), natural_correct=int(sum(selected[:, 1])))
        result[group].update(zip(('N', 'U', 'Q', 'B', 'expected_mixed_reward'),
                                selected[:, 1:].mean(0).tolist()))
        result[group]['C_S'] = sum(max(counts[mid, m] for mid in pool) for m in range(49)) / len(selected)
        assert result[group]['N'] <= result[group]['C_S'] + 1e-12
    return result, different_argmax


def main():
    source = V10 / 'probe_protocol.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads((V10 / 'protocol_fixed_manifest.json').read_text())
    assert manifest['hashes'][str(source)] == digest
    archive = V10 / 'results/generalization_001/protocol_sources'
    for filename in ('probe_protocol.py', '协议探针方案.md'):
        assert (archive / filename).read_bytes() == (V10 / filename).read_bytes()
    # Extract only pure analysis functions; importing the module would load model code.
    tree = ast.parse(source.read_text())
    pure = ast.Module(body=[node for node in tree.body
                           if isinstance(node, ast.FunctionDef) and node.name in ('softmax', 'summarize')],
                      type_ignores=[])
    namespace = {'np': np, 'run': type('PartitionOnly', (), {'partition': staticmethod(partition)})}
    exec(compile(pure, str(source), 'exec'), namespace)
    records = []
    for path in sorted((V10 / 'results/smoke_001/probe').glob('*.npz')):
        scout = int(path.stem[-1])
        with np.load(path) as data:
            z = {k: data[k] for k in data.files}
        assert z['positions'].shape == (480, 2)
        assert len(set(map(tuple, np.column_stack((z['positions'], z['photo_ids']))))) == 480
        independent, argmax_differences = independent_summary(z, 1)
        actual = namespace['summarize'](z, 1)
        for group in independent:
            for metric in independent[group]:
                assert np.isclose(independent[group][metric], actual[group][metric], rtol=0, atol=1e-12)
        assert [independent[x]['n'] for x in ('old', 'added', 'sealed', 'all')] == [288, 96, 96, 480]
        # Verify the frozen decoder table against all four deterministic smoke traces.
        decoder = z['receiver_logits'].argmax(-1)
        lookup = {tuple(np.r_[pos, photos]): msg for pos, photos, msg in
                  zip(z['positions'], z['photo_ids'], z['greedy_message'])}
        trace_folder = V10 / 'results/smoke_001' / path.stem[:-3]
        traces = []
        for mode in ('normal', 'shuffle', 'blank', 'erase_memory'):
            with np.load(trace_folder / f'final_{mode}.npz') as data:
                t = {k: data[k][data['scout'] == scout] for k in data.files}
            code = 7 * t['delivered'][:, 0] + t['delivered'][:, 1]
            expected_place = decoder[code[:, None], t['goals']]
            assert np.array_equal(expected_place, t['place'])
            expected_action = (t['menu'] == expected_place[:, :, None]).argmax(-1)
            assert np.array_equal(expected_action, t['action'])
            matched = 0
            if mode != 'erase_memory':
                for pos, photos, sent in zip(t['positions'], t['photo_ids'], t['sent']):
                    key = tuple(np.r_[pos, photos])
                    if key in lookup:
                        assert np.array_equal(sent, lookup[key])
                        matched += 1
            traces.append(dict(mode=mode, world_rows=len(code), actions=2 * len(code),
                               matching_normal_input_sender_rows=matched))
        records.append(dict(file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            source_logits_worlds=480, bounds=independent,
                            joint_argmax_differs_from_native_rows=argmax_differences, trace_checks=traces))
    assert len(records) == 2
    # Analytic calibration cases, independent of model output.
    p = np.full(49, 1 / 49)
    uniform_q = sum(p * (1 / 6) ** 2)
    uniform_reward = sum(p * (.25 * (1 / 6 + 1 / 6) + .5 / 36))
    assert np.isclose(uniform_q, 1 / 36, atol=1e-15)
    assert np.isclose(uniform_reward, 7 / 72, atol=1e-15)
    result = dict(status='passed', review_type='read-only saved development arrays; no forward pass',
                  probe_sha256=digest, archive_matches=True,
                  comparisons_absolute_tolerance=1e-12, smoke_directions=records,
                  total_saved_visual_inputs=960,
                  total_trace_world_rows=sum(t['world_rows'] for r in records for t in r['trace_checks']),
                  total_trace_actions=sum(t['actions'] for r in records for t in r['trace_checks']),
                  total_sender_matching_rows=sum(t['matching_normal_input_sender_rows'] for r in records for t in r['trace_checks']),
                  uniform_receiver=dict(Q=uniform_q, mixed_reward=uniform_reward),
                  formal_endpoint_trace_replay='not performed by this smoke-only review')
    output = ROOT / 'paper_program/v10协议审查_smoke.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'smoke_directions'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
