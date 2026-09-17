"""JSON-only dose-response summary for the conflict-penalty experiment."""
from __future__ import annotations

import argparse
import hashlib
import math
import json
from pathlib import Path

import numpy as np

from . import design

T7_975 = 2.364624251
UPDATES = np.asarray(design.UPDATES, dtype=np.float64)
DOSE_RULES = ('c0', 'c25', 'c50', 'c75', 'c100')
DOSES = np.asarray([design.PENALTIES[r] for r in DOSE_RULES], dtype=np.float64)
METRICS = (
    'q_rate', 'reward_rate', 'conditional_q_rate', 'physical_execution_rate',
    'target_pair_legal_rate', 'proposal_legal_rate',
    'executed_pair_proposal_legal_rate', 'engagement_rate', 'neutral_rate',
    'third_agent_neutral_rate', 'ignored_proposal_rate',
    'ignored_proposal_world_rate', 'conflict_world_rate', 'unexecuted_proposal_rate',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (len(design.SEEDS),) and np.isfinite(x).all(),
            'Expected eight finite seed values')
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(len(x))
    half = T7_975 * se
    return dict(n=len(x), mean=mean, sample_sd=sd, standard_error=se, df=len(x) - 1,
                ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T7_975,
                positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()),
                negative_count=int((x < 0).sum()), values=[float(v) for v in x])


def centered_auc(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (len(UPDATES),), 'Expected five checkpoints')
    return float(np.trapezoid(x - x[0], UPDATES) / UPDATES[-1])


def trajectory(run, key):
    rows = sorted(run['trajectory'], key=lambda row: int(row['update']))
    require([int(row['update']) for row in rows] == list(map(int, UPDATES)),
            'Checkpoint mismatch')
    return np.asarray([row['target_trajectory'][key] for row in rows], dtype=np.float64)


def final(run, key, partition='new_layouts'):
    return float(run['final'][partition][key])


def _by_run(runs):
    by = {(r['seed'], r['rule'], r['information'], r['schedule'], bool(r['live'])): r
          for r in runs}
    expected = {(seed, rule, info, schedule, bool(live))
                for seed in design.SEEDS for rule in design.RULES
                for info in design.INFORMATIONS for schedule in design.SCHEDULES
                for live in design.LIVES}
    require(len(by) == len(expected) and set(by) == expected, 'Incomplete run grid')
    return by


def gain_values(by, seed, rule, information, schedule, key, endpoint=False):
    live = by[seed, rule, information, schedule, True]
    silent = by[seed, rule, information, schedule, False]
    if endpoint:
        return final(live, key) - final(silent, key)
    return centered_auc(trajectory(live, key) - trajectory(silent, key))


def schedule_average_gain(by, seed, rule, information, key, endpoint=False):
    return float(np.mean([gain_values(by, seed, rule, information, schedule, key, endpoint)
                          for schedule in design.SCHEDULES]))


def dose_values(by, information, key, endpoint=False):
    result = {}
    for rule in DOSE_RULES:
        values = [schedule_average_gain(by, seed, rule, information, key, endpoint)
                  for seed in design.SEEDS]
        result[rule] = stats(values)
    return result


def dose_slope(by, information, key, endpoint=False):
    slopes = []
    for seed in design.SEEDS:
        values = np.asarray([schedule_average_gain(by, seed, rule, information, key, endpoint)
                             for rule in DOSE_RULES], dtype=np.float64)
        slopes.append(float(np.polyfit(DOSES, values, 1)[0]))
    return stats(slopes)


def dose_delta(by, information, key, endpoint=False):
    values = []
    for seed in design.SEEDS:
        values.append(schedule_average_gain(by, seed, 'c100', information, key, endpoint)
                      - schedule_average_gain(by, seed, 'c0', information, key, endpoint))
    return stats(values)


def hard_anchor(by, information, key, endpoint=False):
    values = [schedule_average_gain(by, seed, 'strict', information, key, endpoint)
              for seed in design.SEEDS]
    return stats(values)


def absolute_final(by, rule, information, schedule):
    return {
        channel: {key: stats([final(by[seed, rule, information, schedule, channel], key)
                              for seed in design.SEEDS]) for key in METRICS}
        for channel in (True, False)
    }


def information_interaction(by, rule, key, endpoint=False):
    values = []
    for seed in design.SEEDS:
        pl = schedule_average_gain(by, seed, rule, 'PL', key, endpoint)
        fi = schedule_average_gain(by, seed, rule, 'FI', key, endpoint)
        values.append(pl - fi)
    return stats(values)


def summarize(runs):
    by = _by_run(runs)
    dose = {info: {key: {
        'slope_endpoint': dose_slope(by, info, key, True),
        'slope_centered_AUC': dose_slope(by, info, key, False),
        'c100_minus_c0_endpoint': dose_delta(by, info, key, True),
        'c100_minus_c0_centered_AUC': dose_delta(by, info, key, False),
        'by_level_endpoint': dose_values(by, info, key, True),
        'by_level_centered_AUC': dose_values(by, info, key, False),
    } for key in METRICS} for info in design.INFORMATIONS}
    anchors = {info: {key: {
        'endpoint': hard_anchor(by, info, key, True),
        'centered_AUC': hard_anchor(by, info, key, False),
    } for key in METRICS} for info in design.INFORMATIONS}
    absolute = {rule: {info: {schedule: absolute_final(by, rule, info, schedule)
                              for schedule in design.SCHEDULES}
                       for info in design.INFORMATIONS}
                for rule in design.RULES}
    info_interactions = {rule: {key: {
        'endpoint': information_interaction(by, rule, key, True),
        'centered_AUC': information_interaction(by, rule, key, False),
    } for key in METRICS} for rule in design.RULES}
    by_seed = []
    for seed in design.SEEDS:
        row = {'seed': seed}
        for info in design.INFORMATIONS:
            for key in ('q_rate', 'reward_rate', 'physical_execution_rate',
                        'target_pair_legal_rate'):
                row[f'{info}_{key}_dose_slope_endpoint'] = float(
                    np.polyfit(DOSES, [schedule_average_gain(by, seed, r, info, key, True)
                                       for r in DOSE_RULES], 1)[0])
                row[f'{info}_{key}_dose_slope_auc'] = float(
                    np.polyfit(DOSES, [schedule_average_gain(by, seed, r, info, key, False)
                                       for r in DOSE_RULES], 1)[0])
        by_seed.append(row)
    return dict(
        name='triadic_conflict_penalty_dose_response',
        primary={
            'dose_response': dose,
            'strict_hard_anchor': anchors,
            'interpretation': ('Positive target-pair slope means the live−silent gain '
                               'increases as the engaged-third conflict cost increases.'),
        },
        information_interactions=info_interactions, absolute_final=absolute,
        by_seed=by_seed, independent_seeds=len(design.SEEDS),
        dose_levels={r: design.PENALTIES[r] for r in DOSE_RULES},
        paired_unit='source seed; schedules averaged within seed',
        inference='Student-t intervals across eight independent initializations; dose slopes are fitted per seed.',
        design='1560 two-alternative-partner need triples; five reciprocal penalty levels plus strict × PL/FI × static/rematched × live/silent.',
        claim_boundary=('Task-level conflict-cost and communication effects only. These results do not establish lexical '
                        'meaning, grammar, conventionality, cultural transmission or human language origin.'),
        no_external_model=True, no_llm=True, no_vision_model=True,
    )


def main(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite summary output')
    output.mkdir(parents=False)
    data = json.loads((source / 'execution' / 'results.json').read_text())
    summary = summarize(data['runs'])
    path = output / 'results.json'
    path.write_text(json.dumps(dict(status='completed_after_json_only_aggregation',
                                     source=str(source), summary=summary,
                                     no_model_calls=True), ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_runs=len(data['runs']), model_forwards=0,
                   output_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.source, args.output)
