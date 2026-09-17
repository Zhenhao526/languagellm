"""JSON-only statistical summary for the frozen rule experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import design

T7_975 = 2.364624251
UPDATES = np.asarray(design.UPDATES, dtype=np.float64)
METRICS = (
    'q_rate', 'conditional_q_rate', 'physical_execution_rate',
    'target_pair_legal_rate', 'proposal_legal_rate',
    'executed_pair_proposal_legal_rate', 'engagement_rate', 'neutral_rate',
    'third_agent_neutral_rate', 'ignored_proposal_rate',
    'ignored_proposal_world_rate', 'unexecuted_proposal_rate',
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
    require(x.shape == (len(UPDATES),), 'Expected six checkpoints')
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


def pair_summary(by, rule, information, schedule):
    result = {}
    for key in METRICS:
        auc = [gain_values(by, seed, rule, information, schedule, key) for seed in design.SEEDS]
        endpoint = [gain_values(by, seed, rule, information, schedule, key, True)
                    for seed in design.SEEDS]
        live = [final(by[seed, rule, information, schedule, True], key) for seed in design.SEEDS]
        silent = [final(by[seed, rule, information, schedule, False], key) for seed in design.SEEDS]
        result[key] = dict(centered_AUC=stats(auc), endpoint=stats(endpoint),
                           live_endpoint=stats(live), silent_endpoint=stats(silent),
                           unit='probability; multiply by 100 for percentage points')
    return result


def schedule_average(by, rule, information):
    result = {}
    for key in METRICS:
        auc, endpoint = [], []
        for seed in design.SEEDS:
            auc.append(float(np.mean([gain_values(by, seed, rule, information, schedule, key)
                                      for schedule in design.SCHEDULES])))
            endpoint.append(float(np.mean([gain_values(by, seed, rule, information, schedule, key, True)
                                           for schedule in design.SCHEDULES])))
        result[key] = dict(centered_AUC=stats(auc), endpoint=stats(endpoint),
                           unit='probability; schedule average within seed')
    return result


def interaction(by, information, schedule, key):
    endpoint, auc = [], []
    for seed in design.SEEDS:
        strict = gain_values(by, seed, 'strict', information, schedule, key)
        reciprocal = gain_values(by, seed, 'reciprocal', information, schedule, key)
        strict_e = gain_values(by, seed, 'strict', information, schedule, key, True)
        reciprocal_e = gain_values(by, seed, 'reciprocal', information, schedule, key, True)
        auc.append(reciprocal - strict)
        endpoint.append(reciprocal_e - strict_e)
    return dict(endpoint=stats(endpoint), centered_AUC=stats(auc),
                interpretation='Positive values mean reciprocal settlement has a larger routed-message gain than strict settlement.',
                unit='probability; multiply by 100 for percentage points')


def information_interaction(by, rule, schedule, key):
    endpoint, auc = [], []
    for seed in design.SEEDS:
        pl = gain_values(by, seed, rule, 'PL', schedule, key)
        fi = gain_values(by, seed, rule, 'FI', schedule, key)
        pl_e = gain_values(by, seed, rule, 'PL', schedule, key, True)
        fi_e = gain_values(by, seed, rule, 'FI', schedule, key, True)
        auc.append(pl - fi)
        endpoint.append(pl_e - fi_e)
    return dict(endpoint=stats(endpoint), centered_AUC=stats(auc),
                interpretation='Positive values mean routed-message gain is larger under PL than FI.',
                unit='probability; multiply by 100 for percentage points')


def schedule_average_interaction(by, information, key):
    endpoint, auc = [], []
    for seed in design.SEEDS:
        endpoint.append(float(np.mean([
            gain_values(by, seed, 'reciprocal', information, schedule, key, True)
            - gain_values(by, seed, 'strict', information, schedule, key, True)
            for schedule in design.SCHEDULES])))
        auc.append(float(np.mean([
            gain_values(by, seed, 'reciprocal', information, schedule, key)
            - gain_values(by, seed, 'strict', information, schedule, key)
            for schedule in design.SCHEDULES])))
    return dict(endpoint=stats(endpoint), centered_AUC=stats(auc),
                interpretation='Positive values mean reciprocal settlement has a larger routed-message gain than strict settlement, averaged over schedules.',
                unit='probability; multiply by 100 for percentage points; schedule average within seed')


def absolute_final(by, rule, information, schedule):
    return {
        channel: {key: stats([final(by[seed, rule, information, schedule, channel], key)
                              for seed in design.SEEDS]) for key in METRICS}
        for channel in (True, False)
    }


def checkpoint_rows(by, rule, information, key):
    rows = []
    for index, update in enumerate(UPDATES.astype(int)):
        values = []
        for seed in design.SEEDS:
            gains = []
            for schedule in design.SCHEDULES:
                live = trajectory(by[seed, rule, information, schedule, True], key)[index]
                silent = trajectory(by[seed, rule, information, schedule, False], key)[index]
                gains.append(float(live - silent))
            values.append(float(np.mean(gains)))
        rows.append(dict(update=int(update), live_minus_silent=stats(values)))
    return rows


def summarize(runs):
    by = _by_run(runs)
    gains = {
        rule: {info: {schedule: pair_summary(by, rule, info, schedule)
                      for schedule in design.SCHEDULES}
               for info in design.INFORMATIONS}
        for rule in design.RULES
    }
    overall = {rule: {info: schedule_average(by, rule, info)
                      for info in design.INFORMATIONS} for rule in design.RULES}
    interactions = {info: {schedule: {key: interaction(by, info, schedule, key)
                                      for key in METRICS}
                           for schedule in design.SCHEDULES}
                    for info in design.INFORMATIONS}
    averaged_interactions = {info: {key: schedule_average_interaction(by, info, key)
                                    for key in METRICS}
                             for info in design.INFORMATIONS}
    info_interactions = {rule: {schedule: {key: information_interaction(by, rule, schedule, key)
                                           for key in METRICS}
                                for schedule in design.SCHEDULES}
                         for rule in design.RULES}
    absolute = {rule: {info: {schedule: absolute_final(by, rule, info, schedule)
                              for schedule in design.SCHEDULES}
                       for info in design.INFORMATIONS}
                for rule in design.RULES}
    checkpoints = {rule: {info: {key: checkpoint_rows(by, rule, info, key)
                                 for key in METRICS}
                          for info in design.INFORMATIONS}
                   for rule in design.RULES}
    by_seed = []
    for seed in design.SEEDS:
        row = {'seed': seed}
        for rule in design.RULES:
            for info in design.INFORMATIONS:
                for schedule in design.SCHEDULES:
                    row[f'{rule}_{info}_{schedule}_q_endpoint'] = gain_values(
                        by, seed, rule, info, schedule, 'q_rate', True)
                    row[f'{rule}_{info}_{schedule}_target_pair_endpoint'] = gain_values(
                        by, seed, rule, info, schedule, 'target_pair_legal_rate', True)
                    row[f'{rule}_{info}_{schedule}_target_pair_auc'] = gain_values(
                        by, seed, rule, info, schedule, 'target_pair_legal_rate')
        by_seed.append(row)
    return dict(
        name='rule_information_communication_factorial',
        primary={
            'pair_gains': gains,
            'schedule_averaged': overall,
            'rule_interactions': interactions,
            'schedule_averaged_rule_interactions': averaged_interactions,
            'interpretation': ('The primary estimand is reciprocal−strict on the routed '
                               'message gain in target-pair legality conditional on physical execution.'),
        },
        information_interactions=info_interactions, absolute_final=absolute,
        checkpoint_gains=checkpoints, by_seed=by_seed,
        independent_seeds=len(design.SEEDS),
        paired_unit='source seed; schedules are paired strata and averaged within seed for overall estimates',
        inference='Student-t intervals across eight independent initializations; all factorial contrasts are paired within seed.',
        design='1560 two-alternative-partner need triples; strict/reciprocal × PL/FI × static/rematched × live/silent.',
        claim_boundary=('Task-level rule and communication effects only. The results do not establish lexical '
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
