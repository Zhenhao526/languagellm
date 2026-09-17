"""JSON-only aggregation for the FI capacity pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import design, runner

T7_975 = 2.364624251


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (len(design.SEEDS),) or not np.isfinite(x).all():
        raise ValueError('Expected eight finite seed values')
    mean = float(x.mean()); sd = float(x.std(ddof=1)); se = sd / math.sqrt(len(x)); half = T7_975 * se
    return dict(n=len(x), mean=mean, sample_sd=sd, standard_error=se, df=len(x) - 1,
                ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T7_975,
                positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()),
                negative_count=int((x < 0).sum()))


def auc(values, updates):
    return float(np.trapz(np.asarray(values, dtype=np.float64), np.asarray(updates, dtype=np.float64)) /
                 (updates[-1] - updates[0]))


def summarize(runs):
    by = {(run['seed'], run['condition']): run for run in runs}
    primary = []; final_interactions = []
    metrics = ('target_pair_legal_rate', 'physical_execution_rate', 'q_rate',
               'conditional_q_rate', 'proposal_legal_rate')
    secondary = {key: [] for key in metrics}
    per_seed = []
    for seed in design.SEEDS:
        cell = {}
        for schedule in design.SCHEDULES:
            for live in (False, True):
                run = by[seed, f'{schedule}_altpair_factorized_strict_PL_{"live" if live else "silent"}']
                trajectory = run['trajectory']
                cell[schedule, live] = {
                    key: auc([row['target_trajectory'][key] for row in trajectory], design.UPDATES)
                    for key in metrics
                }
                cell[schedule, live]['final_target_pair_legal_rate'] = run['final']['new_layouts']['target_pair_legal_rate']
        interaction = ((cell['rematched', True]['target_pair_legal_rate'] - cell['rematched', False]['target_pair_legal_rate'])
                       - (cell['static', True]['target_pair_legal_rate'] - cell['static', False]['target_pair_legal_rate']))
        primary.append(interaction)
        final_interaction = ((cell['rematched', True]['final_target_pair_legal_rate'] - cell['rematched', False]['final_target_pair_legal_rate'])
                             - (cell['static', True]['final_target_pair_legal_rate'] - cell['static', False]['final_target_pair_legal_rate']))
        final_interactions.append(final_interaction)
        row = dict(seed=seed, target_pair_legal_auc_by_condition={f'{s}_{"live" if l else "silent"}': cell[s, l]['target_pair_legal_rate']
                  for s in design.SCHEDULES for l in (False, True)}, target_pair_legal_interaction=interaction,
                  target_pair_legal_final_interaction=final_interaction)
        for key in metrics:
            value = ((cell['rematched', True][key] - cell['rematched', False][key])
                     - (cell['static', True][key] - cell['static', False][key]))
            secondary[key].append(value); row[f'{key}_interaction'] = value
        per_seed.append(row)
    return dict(
        name='fi_capacity_rematching_by_communication_target_pair_legal_auc',
        primary=dict(mean=float(np.mean(primary)), statistics=stats(primary), by_seed=per_seed,
                     scope='For each seed, integrate target-pair legal rate conditional on physical execution over six checkpoints, then form rematched×communication interaction.',
                     no_posthoc_selection=True),
        secondary={key: dict(mean=float(np.mean(values)), statistics=stats(values))
                   for key, values in secondary.items()},
        final_target_pair_interaction=stats(final_interactions), independent_seeds=len(design.SEEDS),
        inference='Approximate Student-t intervals across eight independent FI seed blocks; trajectory checkpoints are repeated measures.',
        unit='Rate; multiply by 100 for percentage points.', information=design.INFORMATION,
        pilot=True, no_language_claim=True,
    )


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    raw = json.loads((source / 'execution/results.json').read_text())
    if raw.get('status') != 'completed' or raw.get('information') != design.INFORMATION:
        raise ValueError('Incomplete FI execution')
    runs = raw['runs']
    if len(runs) != len(design.SEEDS) * len(design.CONDITIONS):
        raise ValueError('Unexpected FI run count')
    result = dict(status='completed_json_only_aggregation', source=str(source), information=design.INFORMATION,
                  runs=runs, summary=summarize(runs), no_new_model_calls=True,
                  note='Reads completed FI pilot rows; no additional forward or optimizer update.')
    path = output / 'results.json'; path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_runs=len(runs), output_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   model_forwards=raw['measured_budget']['compact_evaluation_forward_module_samples'],
                   optimizer_updates=0, information=design.INFORMATION)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.source, args.output)
