"""Read-only cross-split semantic-edge probe.

The primary confirmation evaluates edges whose two endpoints are both in the
held-out need partition.  This probe asks the complementary question: can a
policy respond when one endpoint is a seen training need and the other is a
new object×attribute combination, with the same layout and private-site
background?  It reuses saved final train and heldout-resource action arrays;
no model or optimizer is invoked.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import argparse
import json
import math
import numpy as np

from research_program.triadic_action_dependency_study import environment as native
from . import design, metrics, runner


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def target_pair(need):
    plans = native.full_success_plans(tuple(need))
    require(len(plans) == 1, 'Each need must have one unique plan')
    return ((0, 1), (0, 2), (1, 2)).index(tuple(plans[0][:2]))


def build_edges(static):
    train = static['partitions']['train']; held = static['partitions']['heldout_resource']
    train_lookup = {tuple(map(int, row)): i for i, row in enumerate(train['needs'])}
    held_lookup = {tuple(map(int, row)): i for i, row in enumerate(held['needs'])}
    rows = []
    from research_program.triadic_need_response_study import cases
    for before in train_lookup:
        for actor in range(3):
            # Reuse the frozen semantic flip definition from the response
            # builder; only edges crossing train -> heldout are retained.
            for axis, changed_id in cases.flips(before[actor]):
                after = list(before); after[actor] = changed_id; after = tuple(after)
                if after not in held_lookup:
                    continue
                axis_index = cases.AXES.index(axis)
                before_pair = target_pair(before); after_pair = target_pair(after)
                if before_pair == after_pair:
                    continue
                rows.append(dict(before_need=list(before), after_need=list(after), before_index=train_lookup[before],
                                 after_index=held_lookup[after], changed_person=actor, axis=axis, axis_index=axis_index,
                                 target_pairs=[before_pair, after_pair]))
    rows.sort(key=lambda row: (row['changed_person'], row['axis_index'], row['before_need'], row['after_need']))
    groups = {f'{actor}_{axis}': [] for actor in range(3) for axis in range(3)}
    for i, row in enumerate(rows):
        row['edge_index'] = i; groups[f"{row['changed_person']}_{row['axis_index']}"].append(i)
    required = [groups[f'{actor}_{axis}'] for actor in range(3) for axis in (0, 1)]
    require(len(rows) > 0 and all(required) and all(not groups[f'{actor}_2'] for actor in range(3)),
            'Cross-split probe needs the six kind/length strata and no destination cross-edge')
    return rows, groups


def edge_metrics(train_saved, held_saved, edges, groups, backgrounds):
    train_actual = np.asarray(train_saved['actual_pair_index'], dtype=np.int8)
    held_actual = np.asarray(held_saved['actual_pair_index'], dtype=np.int8)
    train_target = np.asarray(train_saved['target_pair_index'], dtype=np.int8)
    held_target = np.asarray(held_saved['target_pair_index'], dtype=np.int8)
    values = []; partner_rates = []
    active_groups = {key: value for key, value in groups.items() if value}
    for key in sorted(active_groups, key=lambda text: tuple(map(int, text.split('_')))):
        selected = active_groups[key]; q_rows = []; p_rows = []
        for edge_index in selected:
            edge = edges[edge_index]; before = edge['before_index']; after = edge['after_index']
            first = []; second = []
            for background in range(backgrounds):
                first.append(int(train_actual[before * backgrounds + background] == train_target[before * backgrounds + background]))
                second.append(int(held_actual[after * backgrounds + background] == held_target[after * backgrounds + background]))
            q_rows.append(float(np.mean(np.asarray(first, dtype=bool) & np.asarray(second, dtype=bool))))
            p_rows.append(float(np.mean(np.asarray(second, dtype=bool))))
        values.append(float(np.mean(q_rows))); partner_rates.append(float(np.mean(p_rows)))
    return dict(Q=float(np.mean(values)), partner_endpoint_rate=float(np.mean(partner_rates)), strata_Q=values,
                strata_partner_endpoint_rate=partner_rates, edge_count=len(edges), stratum_count=len(active_groups),
                weighting='Equal six changed-person×{kind,length} strata; within each stratum equal cross-split edges and backgrounds.')


def stats(values):
    values = np.asarray(values, dtype=np.float64); require(values.shape == (len(design.SEEDS),), 'Sixteen seed values required')
    mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd / math.sqrt(len(values)); half = metrics.T15_975 * se
    return dict(n=len(values), mean=mean, sample_sd=sd, standard_error=se, df=15,
                ci95_lower=mean-half, ci95_upper=mean+half, t_critical=metrics.T15_975)


def endpoint_interaction(cells):
    interaction = (cells['all_or_nothing', True] - cells['all_or_nothing', False]) - \
                  (cells['partial', True] - cells['partial', False])
    return float(interaction)


def probe(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); require(not output.exists(), 'Never overwrite probe')
    plan, static = runner.verify(source); edges, groups = build_edges(static)
    backgrounds = len(static['partitions']['train']['layouts']) * len(static['partitions']['train']['private_sites'])
    by = {}; npz_reads = 0
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            result_path = source / 'execution' / runner.name(seed, condition) / 'result.json'; result = read(result_path)
            train_path = Path(result['final']['train']['path']); held_path = Path(result['trajectory'][-1]['evaluation']['path'])
            require(train_path.is_file() and held_path.is_file(), 'Missing endpoint NPZ')
            require(runner.sha(train_path) == result['final']['train']['data_sha256'] and runner.sha(held_path) == result['trajectory'][-1]['evaluation']['data_sha256'], 'Endpoint NPZ hash mismatch')
            with np.load(train_path, allow_pickle=False) as train_saved, np.load(held_path, allow_pickle=False) as held_saved:
                value = edge_metrics(train_saved, held_saved, edges, groups, backgrounds); npz_reads += 2
            payoff, rule, live = design.parse_condition(condition); by[seed, payoff, rule, live] = value
    require(len(by) == 128, 'Incomplete endpoint probe grid')
    rows = []
    for seed in design.SEEDS:
        q_by_rule = {}; p_by_rule = {}
        for rule in design.RULES:
            qcells = {(payoff, live): by[seed, payoff, rule, live]['Q'] for payoff, live in product(design.PAYOFFS, design.LIVES)}
            pcells = {(payoff, live): by[seed, payoff, rule, live]['partner_endpoint_rate'] for payoff, live in product(design.PAYOFFS, design.LIVES)}
            q_by_rule[rule] = dict(cells={f'{payoff}_{"live" if live else "silent"}': value for (payoff, live), value in qcells.items()}, interaction=endpoint_interaction(qcells))
            p_by_rule[rule] = dict(cells={f'{payoff}_{"live" if live else "silent"}': value for (payoff, live), value in pcells.items()}, interaction=endpoint_interaction(pcells))
        rows.append(dict(seed=seed, by_rule=q_by_rule, partner_by_rule=p_by_rule,
                         rule_mean_interaction=float(np.mean([q_by_rule[r]['interaction'] for r in design.RULES])),
                         partner_rule_mean_interaction=float(np.mean([p_by_rule[r]['interaction'] for r in design.RULES]))))
    q_values = np.asarray([row['rule_mean_interaction'] for row in rows]); p_values = np.asarray([row['partner_rule_mean_interaction'] for row in rows])
    value = dict(status='passed', source=str(source), plan_sha256=runner.sha(source / 'plan.json'), prepared_sha256=runner.sha(source / 'prepared.json'),
                 cross_split_edges=len(edges), edge_groups={key: len(value) for key, value in groups.items()}, backgrounds=backgrounds,
                 conditions=128, npz_reads=npz_reads, model_forwards=0, optimizer_updates=0,
                 primary=dict(name='cross_split_heldout_Q_payoff_communication_endpoint_interaction', target='train_to_heldout_resource',
                              mean_interaction=float(q_values.mean()), statistics=stats(q_values), by_seed=rows,
                              definition='For each seed and rule, [(all_or_nothing live−silent)−(partial live−silent)] on cross-split seen→heldout semantic-edge Q at update6000; average strict/reciprocal and sixteen seeds.',
                              scope='One endpoint is a seen training need and the other is a heldout object×attribute need; equal six changed-person×{kind,length} strata.'),
                 partner_secondary=dict(mean_interaction=float(p_values.mean()), statistics=stats(p_values)),
                 no_model_calls=True, no_training_updates=True)
    output.mkdir(parents=True); write(output / 'probe.json', value); write(output / 'receipt.json', dict(status='passed', probe_sha256=runner.sha(output / 'probe.json'), npz_reads=npz_reads, model_forwards=0, optimizer_updates=0))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(probe(args.source, args.output), ensure_ascii=False, indent=2))
