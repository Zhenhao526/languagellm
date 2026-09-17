"""Post hoc correction of the frozen factorial response statistic.

The first production result used a faulty NumPy broadcast in the researcher-side
factor subset statistic.  This module leaves every trained parameter, NPZ and
production JSON untouched, then rebuilds only ``factor_response`` from the
saved native executed-pair arrays with the intended elementwise comparison.
It is deliberately JSON/NPZ-only: no model forward, gradient or optimizer
state is accessed.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from . import metrics, runner


HERE = Path(__file__).resolve().parent
PARTS = metrics.PARTS
GROUPS = ('heldout_changed_actor', 'seen_changed_actor')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def corrected_subset(case_spec, values, edge_indices):
    """Recompute the nine-stratum Q with elementwise endpoint comparison."""
    values = np.asarray(values)
    require(values.shape == (case_spec['world_count'],), 'Complete pair vector')
    require(values.dtype.kind in 'iu' and ((values >= -1) & (values <= 2)).all(),
            'Executed-pair domain')
    n = int(case_spec['need_worlds'])
    bcount = int(case_spec['n_backgrounds'])
    domains = values.astype(np.int8, copy=False).reshape(n, bcount)
    edges_all = np.asarray(case_spec['edge_need_indices'], dtype=np.int64)
    targets_all = np.asarray(case_spec['target_pairs'], dtype=np.int8)
    counts_by_bg = [np.bincount(domains[:, j] + 1, minlength=4)[1:] for j in range(bcount)]
    rows = []
    total = {'both_correct': 0, 'edge_count': 0}
    for stratum, subset in enumerate(edge_indices):
        ids = np.asarray(subset, dtype=np.int64)
        require(ids.ndim == 1 and len(ids) > 0 and ((ids >= 0) & (ids < len(edges_all))).all(),
                'Nonempty valid edge subset')
        edges = edges_all[ids]
        targets = targets_all[ids]
        qs = []
        shuffles = []
        both = 0
        for j in range(bcount):
            actual = domains[edges, j]
            # The production bug used actual[:, None] == targets, which
            # compared each endpoint against both target columns.  Each row is
            # a two-endpoint pair and must be compared element by element.
            hits = np.all(actual == targets, axis=1)
            qs.append(float(hits.mean()))
            count = counts_by_bg[j]
            numerator = int(np.sum(count[targets[:, 0]] * count[targets[:, 1]], dtype=np.int64))
            shuffles.append(numerator / (n * (n - 1) * len(ids)))
            both += int(hits.sum())
        rows.append(dict(changed_person=stratum // 3, axis_index=stratum % 3,
                         edge_count=int(len(ids)), state_edge_count=int(len(ids) * bcount),
                         Q=float(np.mean(qs)), Q_shuffle=float(np.mean(shuffles)),
                         Q_excess=float(np.mean(qs) - np.mean(shuffles)),
                         raw_both_correct=both))
        total['both_correct'] += both
        total['edge_count'] += int(len(ids) * bcount)
    q = float(np.mean([row['Q'] for row in rows]))
    s = float(np.mean([row['Q_shuffle'] for row in rows]))
    return dict(schema='factorial_edge_subset_metrics_v1', group_worlds=case_spec['world_count'],
                complete_nine_strata=True, strata=rows, Q=q, Q_shuffle=s,
                Q_excess=q - s, raw_counts=total,
                weighting='Equal nine changed-person×axis strata; within each stratum equal edges and backgrounds.',
                scope='Edges are researcher-labelled after evaluation. Held-out group means the changed actor has resource 5 or 6 at one endpoint; no label enters observations or training.')


def update_evaluation(record, path, case_spec, groups):
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        require('actual_pair_index' in saved.files, 'Saved native pair index missing')
        pair_index = saved['actual_pair_index']
    out = deepcopy(record)
    out['factor_response'] = {
        group: (corrected_subset(case_spec, pair_index, groups[group])
                if all(groups[group]) else None)
        for group in GROUPS
    }
    out['path'] = str(path.resolve())
    out['data_sha256'] = sha(path)
    return out


def reanalyse(run, output='correction_001'):
    run = Path(run).resolve()
    destination = Path(output)
    destination = destination if destination.is_absolute() else run / destination
    require(not destination.exists(), 'Never overwrite correction output')
    main_path = run / 'execution' / 'results.json'
    main = read(main_path)
    require(main['status'] == 'completed' and len(main['runs']) == 128, 'Complete factorial run required')
    static = runner.prepared()
    corrected_runs = []
    changed_evaluations = 0
    npz_reads = 0
    for source_run in main['runs']:
        out_run = deepcopy(source_run)
        regime = source_run['regime']
        cases = static['need_response_cases'][regime]
        groups = static['factor_edge_groups'][regime]
        for row in out_run['trajectory']:
            old = row['evaluation']
            row['evaluation'] = update_evaluation(old, old['path'], cases[metrics.TARGET], groups[metrics.TARGET])
            changed_evaluations += int(old['factor_response'] != row['evaluation']['factor_response'])
            npz_reads += 1
        for part in PARTS:
            old = out_run['final'][part]
            if part == metrics.TARGET:
                # Preserve the production alias identity while correcting its
                # copied response field.
                updated = update_evaluation(old, old['path'], cases[part], groups[part])
                updated['alias_of'] = old['alias_of']
                updated['additional_forward_module_samples'] = old['additional_forward_module_samples']
            else:
                updated = update_evaluation(old, old['path'], cases[part], groups[part])
            out_run['final'][part] = updated
            changed_evaluations += int(old['factor_response'] != updated['factor_response'])
            npz_reads += 1
        corrected_runs.append(out_run)
    corrected = deepcopy(main)
    corrected['runs'] = corrected_runs
    corrected['primary'] = metrics.primary(corrected_runs)
    corrected['status'] = 'completed_corrected_factor_response'
    corrected['correction'] = dict(schema='factorial_factor_response_correction_v1',
                                   reason='Production factor subset used a two-dimensional broadcast instead of elementwise endpoint comparison.',
                                   original_results_sha256=sha(main_path),
                                   original_primary=main['primary'],
                                   corrected_formula='np.all(actual == targets, axis=1)',
                                   changed_evaluation_records=changed_evaluations,
                                   npz_reads=npz_reads, model_forwards=0,
                                   gradients=0, optimizer_updates=0,
                                   at=datetime.now(timezone.utc).isoformat(),
                                   code_sha256=sha(HERE / 'correction.py'))
    destination.mkdir(parents=True)
    corrected_path = destination / 'results.json'
    write(corrected_path, corrected)
    receipt = dict(status='passed', correction_status='completed_json_npz_only',
                   inputs_sha256={str(main_path): sha(main_path), str(HERE / 'correction.py'): sha(HERE / 'correction.py'),
                                  str(HERE / 'metrics.py'): sha(HERE / 'metrics.py')},
                   outputs_sha256={str(corrected_path): sha(corrected_path)},
                   changed_evaluation_records=changed_evaluations, npz_reads=npz_reads,
                   model_forwards=0, gradients=0, optimizer_updates=0)
    write(destination / 'receipt.json', receipt)
    return dict(status='passed', corrected_results=str(corrected_path),
                primary=corrected['primary'], receipt=str(destination / 'receipt.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', default='correction_001')
    args = parser.parse_args()
    print(json.dumps(reanalyse(args.run, args.output), ensure_ascii=False))
