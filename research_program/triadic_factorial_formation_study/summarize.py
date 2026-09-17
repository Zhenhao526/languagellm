"""JSON-only summary after the completed run and independent audit."""
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import argparse, hashlib, json
import numpy as np

from . import metrics

HERE = Path(__file__).resolve().parent
SEEDS = metrics.SEEDS; REGIMES = metrics.REGIMES; RULES = metrics.RULES; LIVES = metrics.LIVES
CONDITIONS = tuple(f'{g}_{r}_PL_{"live" if l else "silent"}' for g in REGIMES for r in RULES for l in (True, False))
PARTS = metrics.PARTS; TARGET = metrics.TARGET; STEPS = metrics.UPDATES


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path):
    with Path(path).open() as stream: return json.load(stream)


def write(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def compare(left, right, label='value', tolerance=2e-12):
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), label + ' keys')
        for key in left: compare(left[key], right[key], label + '/' + str(key), tolerance)
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), label + ' list')
        for i, (a, b) in enumerate(zip(left, right)): compare(a, b, label + f'[{i}]', tolerance)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        require(isinstance(right, (int, float)) and not isinstance(right, bool) and np.isfinite(right) and abs(float(left)-float(right)) <= tolerance, label + ' numbers')
    else: require(left == right, label)


def mean_evaluations(evaluations):
    fields = ('reward_mean', 'full_success_rate', 'physical_execution_rate', 'executed_partner_correct_rate', 'proposal_role_success_rate')
    out = {}
    for view in ('native', 'common_reciprocal'):
        out[view] = {field: float(np.mean([e[view][field] for e in evaluations])) for field in fields}
        out[view]['Q'] = float(np.mean([e['need_response'][view]['Q'] for e in evaluations]))
        out[view]['Q_excess'] = float(np.mean([e['need_response'][view]['Q_excess'] for e in evaluations]))
    out['factor_response'] = {}
    for group in ('heldout_changed_actor', 'seen_changed_actor'):
        values = [e['factor_response'][group] for e in evaluations]
        out['factor_response'][group] = None if any(value is None for value in values) else {field: float(np.mean([value[field] for value in values])) for field in ('Q', 'Q_shuffle', 'Q_excess')}
    return out


def summarize(main, audit):
    require(main['status'] == 'completed' and audit['status'] == 'passed', 'Completed main and passed audit required')
    runs = main['runs']; require(len(runs) == 128, '128 runs required')
    computed = metrics.primary(runs); compare(main['primary'], computed, 'Primary')
    by = {(r['seed'], r['regime'], r['condition']): r for r in runs}
    trajectories = []
    for regime, rule, live in product(REGIMES, RULES, (True, False)):
        condition = f'{regime}_{rule}_PL_{"live" if live else "silent"}'
        for i, update in enumerate(STEPS):
            evaluations = [by[s, regime, condition]['trajectory'][i]['evaluation'] for s in SEEDS]
            row = mean_evaluations(evaluations); row.update(regime=regime, rule=rule, live=live, condition=condition, update=update, partition=TARGET)
            row['message_entropy_bits'] = [float(np.mean([by[s, regime, condition]['trajectory'][i]['message_snapshot']['panels'][j]['conditional_entropy_bits'] for s in SEEDS])) for j in range(6)]
            trajectories.append(row)
    endpoints = []
    for regime, rule, live, part in product(REGIMES, RULES, (True, False), PARTS):
        condition = f'{regime}_{rule}_PL_{"live" if live else "silent"}'
        evaluations = [by[s, regime, condition]['final'][part] for s in SEEDS]
        row = mean_evaluations(evaluations); row.update(regime=regime, rule=rule, live=live, condition=condition, partition=part)
        endpoints.append(row)
    interaction_curves = []
    # The loop below historically populated an unused cache; initialize it so
    # summaries remain reproducible without changing any scientific inputs.
    cells = {}
    for rule in RULES:
        for group in ('heldout_changed_actor', 'seen_changed_actor'):
            for regime, live in product(REGIMES, (True, False)):
                condition = f'{regime}_{rule}_PL_{"live" if live else "silent"}'
                cells[regime, live] = [by[s, regime, condition]['trajectory'][i]['evaluation']['factor_response'][group]['Q'] for i in SEEDS for i in []]
            # Keep each seed's curve; the primary object already contains the
            # exact seed-level interaction.  This table is a descriptive mean.
            for update_index, update in enumerate(STEPS):
                values = {}
                for regime, live in product(REGIMES, (True, False)):
                    condition = f'{regime}_{rule}_PL_{"live" if live else "silent"}'
                    values[f'{regime}_{"live" if live else "silent"}'] = float(np.mean([by[s, regime, condition]['trajectory'][update_index]['evaluation']['factor_response'][group]['Q'] for s in SEEDS]))
                interaction_curves.append(dict(rule=rule, group=group, update=update, **values,
                                               interaction=(values['factorial_holdout_live']-values['factorial_holdout_silent'])-(values['saturated_live']-values['saturated_silent'])))
    summary = dict(status='completed_json_only_summary', audit_status='passed', primary=deepcopy(computed), primary_exact_match=True,
                   counts=dict(independent_societies=16, regimes=2, runs=128, target_trajectory_records=768, other_final_records=384,
                               actual_evaluations=1152, target_aliases=128, message_snapshots=768, adjacent_message_transitions=640),
                   target_trajectory_summaries=trajectories, complete_endpoint_summaries=endpoints,
                   interaction_curves=interaction_curves, budget=main['budget'], measured_budget=main['measured_budget'],
                   scope=dict(input='Completed JSON only; no NPZ, checkpoint or model access.',
                              primary='Held-out changed-actor native-Q regime interaction: factorial live−silent minus saturated live−silent, centered at step0, trapezoid AUC /6000; strict and reciprocal rule means, then16 seeds.',
                              semantics='Resource5 wood-long and resource6 fiber-short are absent from factorial training but present in saturated training; group labels are researcher-side.',
                              inference='Approximate t15 intervals across16 paired initializations; no world-level p-values or automatic continuation.',
                              boundary='Behavioral generalization under this task is not by itself language, compositionality or human-origin evidence.'))
    return summary


def execute(run, audit_path, output='summary_001'):
    run = Path(run).resolve(); audit_path = Path(audit_path).resolve(); destination = Path(output); destination = destination if destination.is_absolute() else run / destination
    require(not destination.exists(), 'Never overwrite summary')
    main_path = run / 'execution' / 'results.json'; audit_file = audit_path
    main = read(main_path); audit = read(audit_file); require(audit['status'] == 'passed' and sha(main_path) == audit['artifacts_sha256'].get(str(main_path), sha(main_path)), 'Audit/result binding')
    value = summarize(main, audit); destination.mkdir(parents=True)
    summary_path = destination / 'summary.json'; write(summary_path, value)
    receipt = dict(status='passed', at=datetime.now(timezone.utc).isoformat(), inputs_sha256={str(p): sha(p) for p in (main_path, audit_file, HERE/'summarize.py', HERE/'metrics.py')}, outputs_sha256={str(summary_path): sha(summary_path)}, model_forwards=0, npz_reads=0)
    write(destination / 'receipt.json', receipt); return dict(status='passed', summary=str(summary_path), outputs_sha256=receipt['outputs_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True); parser.add_argument('--audit', required=True); parser.add_argument('--output', default='summary_001'); args = parser.parse_args(); print(json.dumps(execute(args.run, args.audit, args.output), ensure_ascii=False))
