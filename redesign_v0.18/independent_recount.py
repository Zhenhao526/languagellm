"""Root NumPy recount from raw probabilities, receiver actions and score norms.

Does not import production moments or runner. Full mu and score derivatives are
independently audited on a declared subset, not reconstructed by this script.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def softmax(a):
    a = np.asarray(a, np.float64)
    a = np.exp(a - np.max(a, axis=-1, keepdims=True))
    return a / np.sum(a, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = args.out.resolve()
    done = read(root / 'measurement_complete.json')
    assert done['status'] == 'complete'
    checks = 0
    max_error = 0.
    def close(x, y, atol=2e-11, rtol=2e-10):
        nonlocal checks, max_error
        x, y = np.asarray(x, float), np.asarray(y, float)
        assert x.shape == y.shape
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert np.allclose(x, y, atol=atol, rtol=rtol), (x, y)
        checks += 1
        max_error = max(max_error, float(np.max(np.abs(x-y), initial=0.)))
    policies = []
    for entry in read(root / 'policy_results.json'):
        assert sha(root / entry['file']) == entry['sha256']
        result = read(root / entry['file'])
        A, B, C, mu2, baselines, means, ez2, dot = [[] for _ in range(8)]
        for w in result['world_summaries']:
            path = root / w['file']
            assert sha(path) == w['sha256']
            with np.load(path, allow_pickle=False) as z:
                first = softmax(z['first_logits'])
                conditional = softmax(z['second_logits'])
                pi = (first[:, None] * conditional).reshape(-1)
                close(pi, z['probability'])
                close(pi.sum(), 1.)
                q = softmax(z['receiver_logits'])
                close(q, z['receiver_probabilities'])
                ET = np.zeros(49)
                ET2 = np.zeros(49)
                for f, water in itertools.product(range(6), repeat=2):
                    f_ok = float(f == z['positions'][0])
                    w_ok = float(water == z['positions'][1])
                    target = (f_ok+w_ok)/4 + f_ok*w_ok/2 - 1
                    weight = q[:, 0, f]*q[:, 1, water]
                    ET += target*weight
                    ET2 += target*target*weight
                close(ET, z['expected_target'])
                close(ET2, z['expected_target_sq'])
                assert np.all(ET2 >= ET*ET-2e-12)
                norm = z['score_norm_sq']
                assert norm.shape == (49,) and np.isfinite(norm).all() and (norm >= 0).all()
                a, b, c = np.sum(pi*norm), np.sum(pi*norm*ET), np.sum(pi*norm*ET2)
                close([a, b, c], [w['A'], w['B'], w['C']])
                close(float(z['baseline']), w['baseline'])
                close(pi @ ET, w['expected_target'])
                close(pi @ ET2, w['expected_target_sq'])
                A.append(a); B.append(b); C.append(c); mu2.append(w['mu_norm_sq'])
                baselines.append(float(z['baseline'])); means.append(float(pi @ ET))
                ez2.append(w['expected_score_norm_sq']); dot.append(w['mu_dot_expected_score'])
        A, B, C, mu2, baseline, means, ez2, dot = map(np.asarray, (A, B, C, mu2, baselines, means, ez2, dot))
        n = len(A)
        assert n == 18
        matrix = C[:, None] - 2*B[:, None]*baseline[None, :] + A[:, None]*baseline[None, :]**2 - mu2[:, None]
        # Each world receives each one of n baseline values equally often over
        # all permutations. This explicit n-by-n table independently recounts it.
        v_match = np.trace(matrix) / n**2
        v_perm = np.sum(matrix) / n**3
        optimal = np.divide(B, A, out=np.zeros_like(B), where=A > 0)
        v_opt = np.sum(C-2*optimal*B+optimal**2*A-mu2)/n**2
        weighted = np.sum(A*(baseline-optimal)**2)/n**2
        close(v_match-v_opt, weighted)
        expected = {'variance_matched':v_match, 'variance_permuted':v_perm, 'variance_optimal':v_opt,
            'permutation_minus_matched':v_perm-v_match, 'weighted_optimal_distance':weighted,
            'role_variance_matched':v_match/4, 'role_variance_permuted':v_perm/4,
            'role_variance_optimal':v_opt/4, 'role_permutation_minus_matched':(v_perm-v_match)/4}
        for k, value in expected.items():
            close(value, result['aggregate'][k])
            close(value, entry[k])
        # Bound numerical corrections if finite-gradient E[z] is retained.
        match_correction = np.sum(2*baseline*dot-baseline**2*ez2)/n**2
        perm_abs_bound = np.sum(2*abs(baseline.mean())*abs(dot)+np.mean(baseline**2)*ez2)/n**2
        perm_abs_bound += np.var(baseline)*ez2.sum()/(n*(n-1))
        row = {k:result[k] for k in ('seed','partition','direction','checkpoint')}
        row.update({k:float(v) for k,v in expected.items()})
        row.update(relative_reduction=float((v_perm-v_match)/v_perm) if v_perm>0 else None,
            value_mse_against_mean_target=float(np.mean((baseline-means)**2)),
            value_mse_against_optimal=float(np.mean((baseline-optimal)**2)),
            finite_precision_match_variance_correction=float(match_correction),
            finite_precision_perm_variance_absolute_correction_bound=float(perm_abs_bound))
        policies.append(row)
    metrics = ['variance_matched','variance_permuted','variance_optimal','permutation_minus_matched',
        'role_variance_matched','role_variance_permuted','role_variance_optimal','role_permutation_minus_matched',
        'value_mse_against_mean_target','value_mse_against_optimal','weighted_optimal_distance']
    seed_rows = []
    for seed in sorted({r['seed'] for r in policies}):
        for t in [0,100,600]:
            rows = [r for r in policies if r['seed']==seed and r['checkpoint']==t]
            assert len(rows)==(2 if not done['formal'] else 6)
            out = {k:float(np.mean([r[k] for r in rows])) for k in metrics}
            out.update(seed=seed, checkpoint=t)
            out['relative_reduction'] = out['permutation_minus_matched']/out['variance_permuted']
            seed_rows.append(out)
    aggregate = []
    for t in [0,100,600]:
        rows = [r for r in seed_rows if r['checkpoint']==t]
        out = {k:float(np.mean([r[k] for r in rows])) for k in metrics}
        out.update(checkpoint=t, sources=len(rows), positive_sources=sum(r['permutation_minus_matched']>0 for r in rows),
            negative_sources=sum(r['permutation_minus_matched']<0 for r in rows))
        out['relative_reduction'] = out['permutation_minus_matched']/out['variance_permuted']
        aggregate.append(out)
    payload = dict(passed=True, source_sha256=sha(__file__), measurement_receipt_sha256=sha(root/'measurement_complete.json'),
        checks=checks, max_absolute_comparison_error=max_error, policies=policies, seed_rows=seed_rows, aggregate=aggregate,
        independent_scope='All raw pi, receiver 36-action ET/ET2, stored score-norm A/B/C, n-by-n baseline table and source aggregation. mu norm and score derivatives are audited separately on a declared subset.',
        max_abs_finite_precision_match_correction=max(abs(r['finite_precision_match_variance_correction']) for r in policies),
        max_finite_precision_perm_correction_bound=max(r['finite_precision_perm_variance_absolute_correction_bound'] for r in policies))
    dest = root/'independent_recount.json'
    assert not dest.exists()
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:payload[k] for k in ('passed','checks','max_absolute_comparison_error','aggregate')}, ensure_ascii=False))


if __name__=='__main__': main()
