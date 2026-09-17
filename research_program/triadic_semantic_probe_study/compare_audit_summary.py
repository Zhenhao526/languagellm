"""Compare completed independent replays to summary JSON; no model/producer import."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

AXES = ('kind_wood_fiber', 'length_short_long', 'destination_L_R')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def compare(audit_path, summary_path, out):
    audit_path, summary_path, out = map(Path, (audit_path, summary_path, out))
    assert not out.exists(), 'Refuse overwrite'
    a, s = read(audit_path), read(summary_path)
    assert a['status'] == 'passed' and s['status'] == 'completed_read_only_summary'
    for path, digest in s['source_sha256'].items():
        assert sha(path) == digest, 'Summary source changed '+path
    index = {}
    for c in s['catalog']:
        key = (c['kind'], c['seed'], c['condition'], c['checkpoint'], c['partition'], c['mode'])
        assert key not in index
        path = summary_path.parent/c['report_file']
        assert sha(path) == c['report_sha256'], 'Report SHA differs'
        report = read(path)
        for field, value in zip(('kind', 'seed', 'condition', 'checkpoint', 'partition', 'mode'), key):
            assert report['metadata'][field] == value
        index[key] = report['metrics']
    assert len(index) == 400
    scalar_checks = metric_checks = 0
    max_error = 0.0
    visited = set()
    def check(actual, expected):
        nonlocal scalar_checks, max_error
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
        error = float(np.max(np.abs(actual-expected))) if expected.size else 0.0
        assert error <= 2e-12, f'Summary number differs by {error}'
        scalar_checks += expected.size; max_error = max(max_error, error)
    def check_group(actual, expected, axes):
        nonlocal metric_checks
        for metric, values in expected.items():
            target = 'both_team_full_success' if metric == 'both_teams_full' else metric
            check(actual['macro'][target], values['macro'])
            for i, axis in enumerate(axes):
                check(actual['by_axis'][axis]['metrics'][target]['mean'], values['by_axis'][i])
            metric_checks += 1
    for r in a['natural_and_intervention_statistics']:
        if 'natural' in r:
            key = ('natural', r['seed'], r['condition'], r['checkpoint'], r['partition'], 'natural')
            for group, axes in (('content', AXES), ('role', AXES[:2])):
                check_group(index[key][group], r['natural'][group], axes)
        else:
            key = ('intervention', r['seed'], r['condition'], r['checkpoint'], r['partition'], r['mode'])
            check_group(index[key]['content'], r['intervention'], AXES)
        visited.add(key)
    for r in a['remote_contrasts']:
        key = ('remote_contrast', r['seed'], r['condition'], 6000, r['partition'], r['window'])
        check_group(index[key]['content'], r['remote'], AXES); visited.add(key)
    assert visited == set(index), 'Missing an entire summary record'
    assert [r['seed'] for r in a['primary_seed_pairs']] == [r['seed'] for r in s['primary']['seeds']]
    for own, official in zip(a['primary_seed_pairs'], s['primary']['seeds']):
        for old, new in (('PI_silent', 'PI_silent_macro'), ('PI_live', 'PI_live_macro'), ('live_minus_silent', 'PI_live_minus_silent')):
            check(official[new], own[old])
    check(s['primary']['equal_seed_mean']['PI_live_minus_silent'], a['primary_equal_seed_mean'])
    remote = {}
    for window in ('w1', 'w2', 'both'):
        selected = [r for r in a['remote_contrasts'] if r['condition'] == 'PI_live'
                    and r['partition'] == 'heldout_layouts' and r['window'] == window]
        assert [r['seed'] for r in selected] == [49101, 49102, 49103, 49104]
        values = [r['remote']['direction_mean_target_apt_opposite_minus_same']['macro'] for r in selected]
        remote[window] = dict(seed_values=values, equal_seed_mean=float(np.mean(values)))
    result = dict(status='passed', compared_at=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(Path(__file__).resolve()): sha(__file__), str(audit_path.resolve()): sha(audit_path),
                       str(summary_path.resolve()): sha(summary_path)},
        reports_checked=len(visited), natural_reports=192, two_direction_intervention_reports=160,
        remote_window_contrasts=48, checked_metric_fields=metric_checks, checked_scalar_entries=int(scalar_checks),
        max_absolute_difference=max_error, absolute_tolerance=2e-12,
        primary_seed_pairs=a['primary_seed_pairs'], primary_equal_seed_mean=a['primary_equal_seed_mean'],
        PI_live_heldout_remote_three_windows=remote, new_network_forwards=0, new_training_updates=0,
        limits=['Compared every400 report at independent common per-axis and macro metrics, not every auxiliary stratum or row_metrics NPZ.',
                'Natural both_teams_full maps to official both_team_full_success.',
                'Independent weights sum directly; official metrics normalize within-axis weights, allowing floating-point last-bit differences.',
                'Raw512 natural/intervention record identities, route SHA and full replay were checked in the separate audit.'])
    with out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status', 'reports_checked', 'checked_scalar_entries', 'max_absolute_difference')}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True); p.add_argument('--summary', required=True); p.add_argument('--out', required=True)
    args = p.parse_args()
    compare(args.audit, args.summary, args.out)
