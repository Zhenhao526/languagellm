"""JSON-only summary for the global reward-scale null control."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import design

UPDATES = np.asarray(design.UPDATES, dtype=np.float64)
T7_975 = 2.364624251
KEYS = ('q_rate', 'reward_rate', 'conditional_q_rate',
        'physical_execution_rate', 'target_pair_legal_rate',
        'proposal_legal_rate', 'conflict_world_rate')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (len(design.SEEDS),) and np.isfinite(x).all(),
            'Expected one finite value per independent seed')
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    half = T7_975 * sd / math.sqrt(len(x))
    return dict(n=len(x), mean=mean, sample_sd=sd,
                standard_error=sd / math.sqrt(len(x)), df=len(x) - 1,
                ci95_lower=mean - half, ci95_upper=mean + half,
                positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()),
                negative_count=int((x < 0).sum()), values=[float(v) for v in x])


def centered_auc(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == UPDATES.shape, 'Checkpoint shape mismatch')
    return float(np.trapezoid(x - x[0], UPDATES) / UPDATES[-1])


def trajectory(run, key):
    rows = sorted(run['trajectory'], key=lambda row: int(row['update']))
    require([int(r['update']) for r in rows] == list(map(int, UPDATES)),
            'Checkpoint grid mismatch')
    return np.asarray([r['target_trajectory'][key] for r in rows], dtype=np.float64)


def final(run, key):
    return float(run['final']['new_layouts'][key])


def by_run(runs):
    result = {(int(r['seed']), r['rule'], r['information'], r['schedule'], bool(r['live'])): r
              for r in runs}
    expected = {(s, rule, info, sched, live)
                for s in design.SEEDS for rule in design.RULES
                for info in design.INFORMATIONS for sched in design.SCHEDULES
                for live in design.LIVES}
    require(set(result) == expected, 'Incomplete run grid')
    return result


def gain(by, seed, rule, info, schedule, key, endpoint=False):
    live = by[seed, rule, info, schedule, True]
    silent = by[seed, rule, info, schedule, False]
    if endpoint:
        return final(live, key) - final(silent, key)
    return centered_auc(trajectory(live, key) - trajectory(silent, key))


def schedule_gain(by, seed, rule, info, key, endpoint=False):
    return float(np.mean([gain(by, seed, rule, info, s, key, endpoint)
                          for s in design.SCHEDULES]))


def contrast(by, left, right, info, key, endpoint=False):
    values = [schedule_gain(by, seed, left, info, key, endpoint)
              - schedule_gain(by, seed, right, info, key, endpoint)
              for seed in design.SEEDS]
    return stats(values)


def absolute(by, rule, info, schedule):
    return {channel: {key: stats([final(by[s, rule, info, schedule, channel], key)
                                  for s in design.SEEDS])
                      for key in KEYS} for channel in (True, False)}


def global_gradient_probe():
    """Analytical/numerical null checks on fresh logits and native rewards."""
    from . import kernel, runner
    static = design.make_prepared()
    arrays = runner.make_arrays(static['partitions']['train'], 'PL')
    rng = np.random.default_rng(67099001)
    logits = rng.normal(size=(7, 3, 18))
    rewards = arrays['rewards'][:7]
    zero = kernel.objective_terms(logits, rewards, 'c0')
    scaled = kernel.objective_terms(logits, rewards, 'global25')
    keys = ('intent_log_gradient', 'proposal_log_gradient')
    gradient_errors = {k: float(np.max(np.abs(zero[k] - scaled[k]))) for k in keys}
    log_shift = float(np.max(np.abs(scaled['log_J'] - zero['log_J'] - np.log(.25))))
    reward_scale_error = float(np.max(np.abs(scaled['J'] - .25 * zero['J'])))
    eps = 1e-6
    finite_errors = []
    for row, actor, head, coord in ((0, 0, 'intent', 0), (1, 1, 'proposal', 5)):
        plus = logits.copy(); minus = logits.copy()
        index = coord if head == 'intent' else coord + 2
        plus[row, actor, index] += eps; minus[row, actor, index] -= eps
        field = 'intent_log_gradient' if head == 'intent' else 'proposal_log_gradient'
        target = coord
        expected = zero[field][row, actor, target]
        num0 = (kernel.objective_terms(plus, rewards, 'c0')['log_J'][row]
                - kernel.objective_terms(minus, rewards, 'c0')['log_J'][row]) / (2 * eps)
        num1 = (kernel.objective_terms(plus, rewards, 'global25')['log_J'][row]
                - kernel.objective_terms(minus, rewards, 'global25')['log_J'][row]) / (2 * eps)
        finite_errors.append({'c0': float(abs(num0 - expected)),
                              'global25': float(abs(num1 - expected))})
    return dict(gradient_max_abs_error=gradient_errors, log_shift_max_abs_error=log_shift,
                expected_reward_scale_max_abs_error=reward_scale_error,
                finite_difference_errors=finite_errors, no_model_calls=True)


def summarize(runs):
    by = by_run(runs)
    contrasts = {info: {key: {
        'c75_minus_global25_AUC': contrast(by, 'c75', 'global25', info, key, False),
        'c75_minus_global25_endpoint': contrast(by, 'c75', 'global25', info, key, True),
        'global25_minus_c0_AUC': contrast(by, 'global25', 'c0', info, key, False),
        'global25_minus_c0_endpoint': contrast(by, 'global25', 'c0', info, key, True),
    } for key in KEYS} for info in design.INFORMATIONS}
    absolute_final = {rule: {info: {schedule: absolute(by, rule, info, schedule)
                                    for schedule in design.SCHEDULES}
                             for info in design.INFORMATIONS}
                      for rule in design.RULES}
    by_seed = []
    for seed in design.SEEDS:
        row = {'seed': seed}
        for info in design.INFORMATIONS:
            for key in KEYS:
                row[f'{info}_{key}_c75_minus_global25_AUC'] = schedule_gain(by, seed, 'c75', info, key) - schedule_gain(by, seed, 'global25', info, key)
                row[f'{info}_{key}_global25_minus_c0_AUC'] = schedule_gain(by, seed, 'global25', info, key) - schedule_gain(by, seed, 'c0', info, key)
        by_seed.append(row)
    return dict(
        name='triadic_global_reward_scale_null_control',
        contrasts=contrasts,
        absolute_final=absolute_final,
        by_seed=by_seed,
        global_gradient_probe=global_gradient_probe(),
        independent_seeds=len(design.SEEDS),
        paired_unit='source seed; static/rematched schedules averaged within seed',
        inference='Student-t intervals across eight independent initializations; no result-dependent stopping',
        design='1560 two-alternative-partner needs; c0/c75/global25 × PL/FI × static/rematched × live/silent',
        interpretation=('A near-zero global25-minus-c0 contrast is the predicted null for a '
                        'state-independent multiplicative reward factor under mean log expected '
                        'reward. A c75-minus-global25 contrast isolates conflict structure only '
                        'to the extent that numerical and optimizer effects are negligible.'),
        claim_boundary=('Task-level protocol and optimization control only. This does not establish '
                        'lexical meaning, grammar, cultural transmission, or human language origin.'),
        no_external_model=True, no_llm=True, no_vision_model=True)


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite summary output')
    data = json.loads((source / 'execution' / 'results.json').read_text())
    summary = summarize(data['runs'])
    output.mkdir(parents=False)
    path = output / 'results.json'
    path.write_text(json.dumps(dict(status='completed_after_json_only_aggregation', source=str(source),
                                    summary=summary, no_model_calls=True),
                                ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_runs=len(data['runs']), model_forwards=0,
                   output_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.source, args.output)
