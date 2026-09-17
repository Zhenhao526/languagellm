"""Independent32-run execution audit. Preparing/importing does not run inference.

Only an explicit CLI invocation audits completed runs, forwards final networks,
and checks saved intermediate records. No optimizer or intermediate forward.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path
import platform
import re
import time
import traceback
import numpy as np

from research_program.triadic_message_study import audit_execution as previous

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (49101, 49102, 49103, 49104)
ECOLOGIES = ('unique', 'multiple')
CONDITIONS = ('FI_silent', 'FI_live', 'PI_silent', 'PI_live')
PARTITIONS = ('train', 'heldout_layouts')
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
PAIR_NAMES = ('AB', 'AC', 'BC')
PAIRS = tuple(combinations(range(3), 2))
require, read, sha, array_sha, json_bytes, close = previous.require, previous.read, previous.sha, previous.array_sha, previous.json_bytes, previous.close
TOL = 2e-12


def compare(actual, expected, path=''):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), 'Wrong mapping '+path)
        for key, value in expected.items():
            require(key in actual, 'Missing '+path+key)
            compare(actual[key], value, path+key+'/')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'Wrong list '+path)
        for i, value in enumerate(expected): compare(actual[i], value, path+str(i)+'/')
    elif isinstance(expected, float):
        close(actual, expected, 'Float differs '+path)
    else:
        require(actual == expected, 'Value differs '+path)


def legal_edges(needs):
    needs = np.asarray(needs)
    resources = ((0, 1), (2, 3), (0, 2), (1, 3))
    destinations = ((0,), (1,), (0, 1))
    table = np.array([[bool(set(resources[a//3]) & set(resources[b//3]))
        and bool(set(destinations[a % 3]) & set(destinations[b % 3])) for b in range(12)] for a in range(12)])
    return np.stack([table[needs[:, i], needs[:, j]] for i, j in PAIRS], axis=1)


def independent_specs():
    layers, excluded = [], []
    for d in product(range(3), repeat=3):
        resources = list(product(range(4), repeat=3))
        needs = np.array([[3*r[a]+d[a] for a in range(3)] for r in resources], dtype=np.int16)
        counts = legal_edges(needs).sum(1)
        support = {'unique': needs[counts == 1].tolist(), 'multiple': needs[counts >= 2].tolist()}
        row = dict(destinations=list(d), supports=support)
        (layers if all(support.values()) else excluded).append(row)
    require(len(layers) == 21 and len(excluded) == 6, 'Independent common D differs')
    old_specs, _, _ = previous.pure.split_specs()
    layouts = {'train': old_specs['train']['layouts'], 'heldout_layouts': old_specs['new_layouts']['layouts']}
    owners = [list(p) for p in permutations((1, 2, 3))]
    result = {}
    for ecology in ECOLOGIES:
        needs, strata = [], []
        for layer in layers:
            start = len(needs); needs.extend(layer['supports'][ecology])
            strata.append(dict(destinations=layer['destinations'], need_indices=list(range(start, len(needs)))))
        result[ecology] = {}
        for pi, part in enumerate(PARTITIONS):
            physical = len(layouts[part])*6
            monitor = []
            for di, stratum in enumerate(strata):
                rng = np.random.default_rng(np.random.SeedSequence([61917001, pi, di]))
                arrangements = rng.choice(physical, 32, replace=False)
                draws = rng.random(32)
                chosen = np.array(stratum['need_indices'])[np.floor(draws*len(stratum['need_indices'])).astype(int)]
                monitor.extend((chosen*physical+arrangements).tolist())
            result[ecology][part] = dict(ecology=ecology, partition=part, needs=needs, layouts=layouts[part], private_sites=owners,
                demand_strata=strata, world_count=len(needs)*physical, monitor_indices=sorted(monitor),
                population_weighting='equal destination strata, uniform conditional resources, layouts and owners',
                monitor_weighting='equal strata; each32 frozen sampled worlds equal weight')
    return result, layers, excluded


def weights_for(spec, monitor=False):
    if monitor: return np.full(672, 1/672, dtype=np.float64)
    per_need = {}
    for layer in spec['demand_strata']:
        for n in layer['need_indices']: per_need[n] = 1/(21*len(layer['need_indices'])*len(spec['layouts'])*6)
    return np.repeat(np.array([per_need[i] for i in range(len(spec['needs']))]), len(spec['layouts'])*6)


def sample(spec, uniform):
    di = np.floor(uniform[:, 0]*21).astype(np.int64)
    ni = np.empty(len(uniform), dtype=np.int64)
    for k, layer in enumerate(spec['demand_strata']):
        positions = np.flatnonzero(di == k)
        ni[positions] = np.asarray(layer['need_indices'])[np.floor(uniform[positions, 1]*len(layer['need_indices'])).astype(np.int64)]
    li = np.floor(uniform[:, 2]*len(spec['layouts'])).astype(np.int64)
    oi = np.floor(uniform[:, 3]*6).astype(np.int64)
    ids = (ni*len(spec['layouts'])+li)*6+oi
    return ids, np.asarray(spec['needs'], dtype=np.int16)[ni] % 3, li, oi


def weighted_measures(values, weights):
    """Independent subsets of the saved weighted dictionary, including the primary."""
    states, actions = values['states'], values['action_indices']
    require(len(weights) == len(states) and np.isfinite(weights).all() and (weights >= 0).all()
        and abs(weights.sum()-1) <= 1e-12, 'Bad probability weights')
    r, execution, satisfied = previous.native_settlement(states, actions)
    require(np.array_equal(r, values['greedy_reward']), 'Weighted source score differs')
    active = actions != 0; count = active.sum(1)
    raw = np.maximum(actions-1, 0)
    sites, dests = raw//4, (raw//2) % 2
    partners = np.stack([np.asarray([j for j in range(3) if j != i])[raw[:, i] % 2] for i in range(3)], axis=1)
    edge = legal_edges(states[:, :3])
    topology = np.stack([(count == 2) & active[:, i] & active[:, j] & (partners[:, i] == j) & (partners[:, j] == i) for i, j in PAIRS], axis=1)
    role = topology & edge
    physical = np.stack([execution[:, i] & execution[:, j] for i, j in PAIRS], axis=1)
    W = lambda v: float(np.sum(weights*np.asarray(v, dtype=float)))
    C = lambda event, group: W(event & group)/W(group) if W(group) else None
    any_role, any_topology, any_physical, full = role.any(1), topology.any(1), physical.any(1), r == 1
    gamma = [W(edge[:, k]) for k in range(3)]; best = max(gamma)
    unique_groups = {}
    for k, name in enumerate(PAIR_NAMES):
        mask = (edge.sum(1) == 1) & edge[:, k]; i, j = PAIRS[k]
        outsider = 3-i-j
        unique_groups[name] = dict(population_weight=W(mask), raw_worlds=int(mask.sum()),
            compatible_role_rate=C(any_role, mask), topology_consistency_rate=C(any_topology, mask),
            physical_match_rate=C(any_physical, mask), full_success_rate=C(full, mask),
            mean_reward=W(r*mask)/W(mask) if W(mask) else None,
            isolated_wait_rate=C(~active[:, outsider], mask), required_agents_active_rate=C(active[:, i]&active[:, j], mask),
            topology_confusion_conditional={p: C(topology[:, q], mask) for q, p in enumerate(PAIR_NAMES)},
            no_mutual_pair_rate=C(~any_topology, mask))
    weighted = dict(compatible_role_rate=W(any_role), topology_consistency_rate=W(any_topology), physical_match_rate=W(any_physical),
        full_success_rate=W(full), reward_mean=W(r), compatible_role_but_not_physical_rate=W(any_role & ~any_physical),
        topology_but_incompatible_rate=W(any_topology & ~any_role),
        active_count_weights={str(k): W(count == k) for k in range(4)}, native_reward_weights={str(v): W(r == v) for v in (0., .5, 1.)},
        two_active_not_mutual_rate=W((count == 2)&~any_topology),
        site_mismatch_given_topology=C(np.stack([topology[:, k]&(sites[:, i] != sites[:, j]) for k, (i, j) in enumerate(PAIRS)], axis=1).any(1), any_topology),
        destination_mismatch_given_topology=C(np.stack([topology[:, k]&(dests[:, i] != dests[:, j]) for k, (i, j) in enumerate(PAIRS)], axis=1).any(1), any_topology),
        pair_weights={name: dict(active_pair_weight=W((count == 2)&active[:, i]&active[:, j]),
            proposal_weight=W(topology[:, k]), compatible_proposal_weight=W(role[:, k]), executed_weight=W(physical[:, k]),
            full_success_weight=W(physical[:, k]&full)) for k, (name, (i, j)) in enumerate(zip(PAIR_NAMES, PAIRS))},
        agent_rates={a: dict(wait_rate=W(~active[:, i]), participant_rate=W(active[:, i]),
            executed_participant_rate=W(execution[:, i]), own_need_satisfied_rate=W(satisfied[:, i])) for i, a in enumerate('ABC')},
        fixed_pair_oracles={p: dict(gamma=g, full_success_oracle=g, mean_reward_oracle=(1+g)/2,
            mean_log_J_oracle=-(1-g)*math.log(2)) for p, g in zip(PAIR_NAMES, gamma)},
        best_fixed_pair_gamma=best, compatible_role_excess_best_fixed_pair=W(any_role)-best,
        full_success_excess_best_fixed_pair=W(full)-best, best_fixed_pair_reward_oracle=(1+best)/2,
        reward_excess_best_fixed_pair=W(r)-(1+best)/2, best_fixed_pair_mean_log_J_oracle=-(1-best)*math.log(2),
        full_information_unrestricted_oracle=dict(compatible_role_rate=1., full_success_rate=1., reward_mean=1., mean_log_J=0.),
        unique_edge_subgroups=unique_groups)
    for key in ('expected_reward', 'full_success_probability', 'physical_execution_probability'):
        array_key = 'conditional_exact_'+('execution_probability' if key == 'physical_execution_probability' else key)
        weighted['conditional_exact_'+key+'_mean'] = W(values[array_key])
    unique, inverse, counts = np.unique(actions, axis=0, return_inverse=True, return_counts=True)
    raw = dict(worlds=len(states), weights_sum=float(weights.sum()), unique_need_tables=len(np.unique(states[:, :3], axis=0)),
        joint_action_domain_size=4913, observed_joint_actions=len(unique), omitted_joint_actions_have_zero_count_and_weight=True,
        compatible_role_worlds=int(any_role.sum()), topology_consistent_worlds=int(any_topology.sum()), physical_matched_worlds=int(any_physical.sum()),
        full_success_worlds=int(full.sum()), joint_action_distribution=[dict(action_indices=a.tolist(), raw_worlds=int(n),
            probability_weight=W(inverse == i)) for i, (a, n) in enumerate(zip(unique, counts))])
    return weighted, raw


def primary_from_summaries(summaries):
    index = {(r['seed'], r['ecology'], r['condition']): r for r in summaries}
    rows = []
    keys = ('compatible_role_rate', 'topology_consistency_rate', 'physical_match_rate', 'full_success_rate', 'reward_mean',
            'best_fixed_pair_gamma', 'compatible_role_excess_best_fixed_pair')
    for seed in SEEDS:
        cells = {e: {c: {k: index[seed, e, c]['final']['heldout_layouts']['natural'][k] for k in keys} for c in CONDITIONS} for e in ECOLOGIES}
        pi = {e: cells[e]['PI_live']['compatible_role_rate']-cells[e]['PI_silent']['compatible_role_rate'] for e in ECOLOGIES}
        fi = {e: cells[e]['FI_live']['compatible_role_rate']-cells[e]['FI_silent']['compatible_role_rate'] for e in ECOLOGIES}
        full = {e: cells[e]['PI_live']['full_success_rate']-cells[e]['PI_silent']['full_success_rate'] for e in ECOLOGIES}
        rows.append(dict(seed=seed, eight_cells=cells, PI_live_minus_silent_by_ecology=pi, FI_live_minus_silent_by_ecology=fi,
            primary_DiD_unique_minus_multiple=pi['unique']-pi['multiple'], FI_DiD_unique_minus_multiple=fi['unique']-fi['multiple'],
            full_success_PI_live_minus_silent_by_ecology=full, full_success_DiD_unique_minus_multiple=full['unique']-full['multiple']))
    return dict(partition='heldout_layouts', primary_metric='compatible_role_rate', independent_paired_seeds=4,
        seed_pairs=rows, equal_seed_means={k: sum(r[k] for r in rows)/4 for k in
            ('primary_DiD_unique_minus_multiple', 'FI_DiD_unique_minus_multiple', 'full_success_DiD_unique_minus_multiple')}, significance_test=None)


def audit(run):
    started = time.perf_counter()
    run = Path(run).resolve(); execution = run/'execution'
    plan, prepared, freeze = [read(run/name) for name in ('plan.json', 'prepared.json', 'freeze.json')]
    results, status = [read(execution/name) for name in ('results.json', 'status.json')]
    require(results['status'] == status['status'] == 'completed' and results['completed_run_count'] == status['completed_runs'] == 32,
        'Wait for complete32-run execution')
    require(not list(execution.rglob('failure.json')), 'Source failure record exists')
    require(sha(run/'plan.json') == freeze['plan_sha256'] == results['plan_sha256'], 'Plan anchor differs')
    require(sha(run/'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Prepared anchor differs')
    require(plan['runtime'] == dict(python=platform.python_version(), numpy=np.__version__), 'Numerical runtime differs')
    require(prepared['config'] == plan['config'], 'Prepared config differs')
    compare(plan['config'], dict(seeds=list(SEEDS), ecologies=list(ECOLOGIES), conditions=list(CONDITIONS),
        partitions=list(PARTITIONS), updates=6000, batch_size=256, checkpoints=list(CHECKPOINTS),
        monitor_worlds_per_partition=672, trajectories_per_state=2, dtype='float64', evaluation_batch_size=1024,
        sender_windows=2, sender_tokens_per_window=4, alphabet=['@', '#', '$', '%', '&', '*', '+', '~'],
        learning_rate=.001, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8, global_gradient_clip=5.,
        sender_entropy_coefficient=0, entropy_initial=.001, entropy_zero_after_updates=1000,
        same_initialization_and_message_uniforms_across_eight_arms=True,
        same_world_uniforms_destinations_layout_owners_across_eight_arms=True), 'config/')
    artifacts, sources = {}, {}
    for relative, digest in plan['sources'].items():
        for path in (ROOT/relative, run/'source_snapshot'/relative):
            require(sha(path) == digest, 'Frozen source changed '+str(path)); sources[str(path)] = digest
    old_receipt = ROOT/'research_program/triadic_message_study/results/messages_001/audit_execution_001/verification.json'
    require(sha(previous.__file__) == read(old_receipt)['audit_source_sha256'], 'Independent old audit source changed')
    sources[str(Path(previous.__file__))] = sha(previous.__file__)
    artifacts[str(old_receipt)] = sha(old_receipt)
    pure_receipt = ROOT/'research_program/triadic_learning_baseline/results/learning_001/audit_execution_001/verification.json'
    require(sha(previous.pure.__file__) == read(pure_receipt)['audit_source_sha256'], 'Independent baseline helper changed')
    sources[str(Path(previous.pure.__file__))] = sha(previous.pure.__file__)
    artifacts[str(pure_receipt)] = sha(pure_receipt)
    design_path = HERE/'design_audit_002/verification.json'; design_audit = read(design_path)
    require(design_audit['status'] == 'passed' and design_audit['design_sha256'] == sha(HERE/'design.py'), 'Design audit002 does not bind current design')
    for path, digest in design_audit['source_sha256'].items():
        require(sha(path) == digest, 'Design-audited source changed'); sources[path] = digest
    artifacts[str(design_path)] = sha(design_path)
    specs, layers, excluded = independent_specs()
    require(specs == prepared['partitions'] and layers == prepared['common_destination_layers']
        and excluded == prepared['excluded_destination_layers'], 'Independent21D support/split/monitor differs')
    expected_runs = [dict(seed=s, ecology=e, condition=c, directory=f'seed_{s}_{e}_{c}') for s, e, c in product(SEEDS, ECOLOGIES, CONDITIONS)]
    require(prepared['runs'] == expected_runs and [(r['seed'], r['ecology'], r['condition']) for r in results['runs']]
        == [(r['seed'], r['ecology'], r['condition']) for r in expected_runs], 'Complete ordered32-run grid differs')
    require({p.name for p in execution.glob('seed_*')} == {r['directory'] for r in expected_runs}, 'Extra/missing run folder')
    require({p.name for p in execution.glob('worker_seed_*')} == {f'worker_seed_{s}' for s in SEEDS}, 'Extra/missing worker folder')
    compare(results, dict(paired_seed_count=4, updates_total=192000, training_world_samples_total=49152000,
        sampled_complete_message_trajectories_total=98304000, weighted_structural_action_contributions=2359296000,
        offline_reward_table_entries_actual=18247680, language_or_convention_claim_automatically_supported=False), 'totals/')
    compare(prepared, dict(planned_checkpoint_files=192, planned_natural_monitor_files=384,
        planned_closed_monitor_forward_files=192, planned_natural_final_files=64, planned_closed_final_forward_files=32,
        full_natural_worlds_total=3041280, full_closed_worlds_total=1520640, all_final_forward_worlds_total=4561920), 'prepared_counts/')
    previous.finite_tree(results)
    top = read(execution/'started.json')
    require(top['plan_sha256'] == freeze['plan_sha256'] and top['device'] == 'cpu_numpy'
        and all(v == '1' for v in top['thread_environment'].values()), 'Top execution source/device/thread differs')
    processes = results['worker_processes']
    require(len(processes) == len({p['pid'] for p in processes}) == 4 and all(p['exitcode'] == 0 for p in processes), 'Four workers did not complete')
    states = {e: {p: previous.pure.packed(specs[e][p]) for p in PARTITIONS} for e in ECOLOGIES}
    expected_inputs, expected_weights = {}, {}
    for e, part in product(ECOLOGIES, PARTITIONS):
        x = states[e][part]
        expected_inputs[e, part] = dict(worlds=len(x),
            full_information_features_sha256=array_sha(previous.observed_features(x, True)),
            private_information_features_sha256=array_sha(previous.observed_features(x, False)),
            native_rewards_sha256=array_sha(previous.pure.rewards(x)), states_sha256=array_sha(x))
        expected_weights[e, part] = dict(full_weights=weights_for(specs[e][part]), monitor_weights=weights_for(specs[e][part], True),
            monitor_indices=np.asarray(specs[e][part]['monitor_indices'], dtype=np.int64))
    index = {(r['seed'], r['ecology'], r['condition']): r for r in results['runs']}
    counts = Counter(); max_p = max_stat = 0.; summaries = []; pairing = []
    evaluated_paths = set(); weight_paths = set()

    def evaluate(entry, ecology, part, indices, networks=None):
        nonlocal max_p, max_stat
        path = execution/entry['data_file']; require(path not in evaluated_paths, 'Unexpected duplicate forward record')
        require(entry['data_sha256'] == entry['uniform']['data_sha256'] == sha(path), 'Outer/uniform data hash differs')
        expected_state = states[ecology][part][indices]
        values, _, receipt = previous.check_evaluation(path, expected_state, indices, entry['uniform'], entry['actual_rollout_condition'], networks)
        evaluated_paths.add(path); artifacts[str(path)] = receipt['sha256']
        max_p = max(max_p, receipt['max_probability_error']); max_stat = max(max_stat, receipt['max_conditional_statistic_error'])
        wpath = execution/entry['weights_file']
        require(wpath == execution/f'worker_seed_{seed}'/f'weights_{ecology}_{part}.npz', 'Wrong evaluation weight provenance')
        require(sha(wpath) == entry['weights_file_sha256'], 'Weight file hash changed')
        mode = 'full_weights' if networks is not None else 'monitor_weights'
        require(entry['weight_array'] == mode, 'Wrong full/monitor weighting')
        weights = expected_weights[ecology, part][mode]
        require(array_sha(weights) == entry['weights_sha256'], 'Evaluation weight hash differs')
        weighted, raw = weighted_measures(values, weights)
        compare(entry['weighted'], weighted, str(path)+'/weighted/'); compare(entry['raw'], raw, str(path)+'/raw/')
        counts['final_npz' if networks is not None else 'monitor_npz'] += 1
        counts['saved_evaluation_worlds'] += len(indices)
        counts['network_forward_batches'] += receipt['actual_forward_network_batches']
        counts['network_forward_samples'] += receipt['actual_forward_network_samples']
        if networks is not None: counts['final_forward_worlds'] += len(indices)
        return values, weighted

    def modes(entries, ecology, part, condition, indices, networks=None):
        require(entries['natural']['actual_rollout_condition'] == condition and entries['natural']['reused_natural'] is False, 'Natural mode differs')
        value, natural = evaluate(entries['natural'], ecology, part, indices, networks)
        del value
        if condition.endswith('_silent'):
            expected = dict(entries['natural'], reused_natural=True, control='already_silent_identical_route_no_duplicate_forward')
            require(entries['closed'] == expected, 'Silent closed should be exact natural reference')
            counts['silent_closed_references'] += 1
            return dict(natural=natural, closed=natural)
        require(entries['closed']['actual_rollout_condition'] == condition.replace('_live', '_silent')
            and entries['closed']['reused_natural'] is False
            and entries['closed']['control'] == 'same_parameters_from_window1_cross_channels_closed_own_messages_retained', 'Live closed route differs')
        value, closed = evaluate(entries['closed'], ecology, part, indices, networks)
        del value
        return dict(natural=natural, closed=closed)

    aggregate_inputs = read(execution/'input_arrays.json')
    require(aggregate_inputs['worker_count'] == 4 and aggregate_inputs['all_worker_arrays_and_weights_identical'] is True
        and aggregate_inputs['actual_offline_reward_table_entries'] == 18247680, 'Global input receipt differs')
    for seed in SEEDS:
        worker = execution/f'worker_seed_{seed}'
        ws, wr, statusw = [read(worker/n) for n in ('started.json', 'results.json', 'status.json')]
        process = next(p for p in processes if p['name'] == f'partner_ecology_seed_{seed}')
        require(ws['seed'] == seed and ws['pid'] == process['pid'] and ws['parent_pid'] == top['pid']
            and ws['plan_sha256'] == freeze['plan_sha256'] and all(v == '1' for v in ws['thread_environment'].values()), 'Worker identity/source differs')
        require(wr['status'] == statusw['status'] == 'completed' and wr['seed'] == seed
            and wr['runs'] == [index[seed, e, c] for e, c in product(ECOLOGIES, CONDITIONS)]
            and wr['offline_reward_table_entries'] == 4561920, 'Worker result aggregation differs')
        inputs = read(worker/'input_arrays.json')
        require(inputs == aggregate_inputs['arrays_by_worker'][str(seed)], 'Aggregate worker input mismatch')
        for e, p in product(ECOLOGIES, PARTITIONS):
            compare(inputs[e][p], expected_inputs[e, p], 'worker_arrays/')
            receipt = inputs[e][p]['weights']; wpath = execution/receipt['file']
            require(wpath == worker/f'weights_{e}_{p}.npz' and sha(wpath) == receipt['file_sha256'], 'Weight location/hash differs')
            with np.load(wpath, allow_pickle=False) as loaded:
                require(set(loaded.files) == set(expected_weights[e, p]), 'Wrong weight arrays')
                for key, value in expected_weights[e, p].items():
                    require(np.array_equal(loaded[key], value), 'Independent full/monitor weights differ')
                    require(array_sha(value) == receipt[key+'_sha256'], 'Weight array receipt differs')
            artifacts[str(wpath)] = sha(wpath); weight_paths.add(wpath)
        for path in worker.glob('*.json'): artifacts[str(path)] = sha(path)
        checkpoint_rng, initial_hashes, final_networks, monitor_rows = {}, {}, {}, {}
        for e, c in product(ECOLOGIES, CONDITIONS):
            key = (e, c); directory = execution/f'seed_{seed}_{e}_{c}'; row = index[seed, e, c]
            require(read(directory/'result.json') == row, 'Per-run aggregate differs')
            compare(row, dict(updates=6000, training_world_samples=1536000, sampled_complete_message_trajectories=3072000), 'run_budget/')
            monitor = [json.loads(line) for line in (directory/'monitor.jsonl').read_text().splitlines()]
            require(monitor == row['monitor'] and [m['update'] for m in monitor] == list(CHECKPOINTS), 'Checkpoint schedule differs')
            monitor_rows[key] = monitor; checkpoint_rng[key] = {}
            require({p.name for p in directory.glob('checkpoint_*.npz')} == {f'checkpoint_{u:04d}.npz' for u in CHECKPOINTS}, 'Checkpoint inventory differs')
            modes_expected = ('natural', 'closed') if c.endswith('_live') else ('natural',)
            require({p.name for p in directory.glob('monitor_*.npz')} == {f'monitor_{u:04d}_{p}_{m}.npz' for u, p, m in product(CHECKPOINTS, PARTITIONS, modes_expected)}, 'Monitor inventory differs')
            require({p.name for p in directory.glob('final_*.npz')} == {f'final_{p}_{m}.npz' for p, m in product(PARTITIONS, modes_expected)}, 'Final inventory differs')
            for entry in monitor:
                update = entry['update']; path = directory/f'checkpoint_{update:04d}.npz'
                require(sha(path) == entry['checkpoint_sha256'], 'Checkpoint SHA differs')
                nets, w_rng, m_rng = previous.checkpoint(path, update, seed)
                checkpoint_rng[key][update] = (w_rng, m_rng); artifacts[str(path)] = sha(path); counts['checkpoint_files'] += 1
                if update == 0:
                    initial_hashes[key] = previous.network_hash(nets)
                    require(initial_hashes[key] == row['initial_parameter_sha256'], 'Initial parameter hash differs')
                if update == 6000:
                    final_networks[key] = nets
                    require(previous.network_hash(nets) == row['final_parameter_sha256'] and sha(path) == row['final_checkpoint_sha256'], 'Final parameter hash differs')
                for part in PARTITIONS:
                    ids = np.asarray(specs[e][part]['monitor_indices'], dtype=np.int64)
                    modes(entry['monitor'][part], e, part, c, ids)
            for filename in ('result.json', 'monitor.jsonl', 'training.jsonl'): artifacts[str(directory/filename)] = sha(directory/filename)
            require(sha(directory/'training.jsonl') == row['training_log_sha256'], 'Training log hash differs')
        require(len(set(initial_hashes.values())) == 1, 'Eight-arm initial parameters differ')
        world = np.random.default_rng(np.random.SeedSequence([seed, 200]))
        message = {(t, w, a): np.random.default_rng(np.random.SeedSequence([seed, a, t, w, 300])) for t, w, a in product(range(2), range(2), range(3))}
        def rng_check(update):
            states_m = {f't{t}_w{w}_a{a}': rng.bit_generator.state for (t, w, a), rng in message.items()}
            require(all(v[update] == (world.bit_generator.state, states_m) for v in checkpoint_rng.values()), 'Rebuilt checkpoint RNG differs')
        rng_check(0)
        logs = {(e, c): (execution/f'seed_{seed}_{e}_{c}'/'training.jsonl').open() for e, c in product(ECOLOGIES, CONDITIONS)}
        previous_time = {key: -1. for key in logs}
        try:
            for update in range(1, 6001):
                U = world.random((256, 4))
                M = np.empty((2, 256, 2, 3, 4), dtype=np.float64)
                for (t, w, a), rng in message.items(): M[t, :, w, a] = rng.random((256, 4))
                draws = {e: sample(specs[e]['train'], U) for e in ECOLOGIES}
                require(all(np.array_equal(draws['unique'][k], draws['multiple'][k]) for k in (1, 2, 3)), 'Cross-ecology D/layout/owner mismatch')
                for (e, c), stream in logs.items():
                    line = stream.readline(); require(bool(line), 'Truncated log')
                    row = json.loads(line); ids, d, li, oi = draws[e]
                    compare(row, dict(seed=seed, ecology=e, condition=c, update=update,
                        world_uniforms_sha256=array_sha(U), batch_indices_sha256=array_sha(ids),
                        batch_states_sha256=array_sha(states[e]['train'][ids]), destinations_sha256=array_sha(d),
                        layout_indices_sha256=array_sha(li), owner_indices_sha256=array_sha(oi), sample_uniforms_sha256=array_sha(M),
                        entropy_coefficient=.001*max(0, 1-(update-1)/1000)), 'training/')
                    previous.finite_tree(row)
                    require(0 <= row['mean_J'] <= 1 and row['min_log_J'] <= row['mean_log_J'] <= row['max_log_J'] <= TOL,
                        'Training objective bounds differ')
                    require(0 <= row['mean_actor_entropy'] <= math.log(17)+TOL and 0 <= row['zero_float_J_states'] <= 512, 'Training entropy/J counts')
                    close(row['mean_F'], row['mean_log_J']+row['entropy_coefficient']*row['mean_actor_entropy'], 'Training F algebra')
                    close(row['receiver_loss'], -row['mean_F'], 'Receiver loss sign')
                    close(row['sender_advantage_mean'], 0., 'PairedLOO sum differs')
                    require(row['sender_advantage_abs_mean'] >= 0 and row['sender_advantage_max_abs'] >= row['sender_advantage_abs_mean']-TOL
                        and row['sender_advantage_squared_mean'] >= row['sender_advantage_abs_mean']**2-TOL, 'LOO bounds')
                    require(row['sender_mean_complete_log_score'] <= TOL and re.fullmatch(r'[0-9a-f]{64}', row['sampled_messages_sha256']), 'Recorded sampled message hash or score invalid')
                    require(row['gradient_norm'] >= 0 and row['elapsed_seconds'] >= previous_time[e, c], 'Gradient norm/time invalid')
                    close(row['gradient_clip_scale'], min(1., 5/max(row['gradient_norm'], 1e-300)), 'Gradient clip rule')
                    previous_time[e, c] = row['elapsed_seconds']; counts['training_records'] += 1
                if update in CHECKPOINTS: rng_check(update)
            require(all(not f.read() for f in logs.values()), 'Extra training rows')
        finally:
            for f in logs.values(): f.close()
        pairing.append(dict(seed=seed, same_initial_parameters=True, common_uniform_context_updates=6000,
            actual_states_matched_within_ecology=True, actual_states_not_required_equal_across_ecologies=True))
        for e, c in product(ECOLOGIES, CONDITIONS):
            directory = execution/f'seed_{seed}_{e}_{c}'; row = index[seed, e, c]; finals = {}; actions = Counter()
            for part in PARTITIONS:
                ids = np.arange(len(states[e][part]), dtype=np.int64)
                finals[part] = modes(row['final'][part], e, part, c, ids, final_networks[e, c])
                for mode in ('natural', 'closed') if c.endswith('_live') else ('natural',):
                    final_path = execution/row['final'][part][mode]['data_file']
                    monitor_path = execution/monitor_rows[e, c][-1]['monitor'][part][mode]['data_file']
                    selected = np.asarray(specs[e][part]['monitor_indices'], dtype=np.int64)
                    with np.load(final_path, allow_pickle=False) as fv, np.load(monitor_path, allow_pickle=False) as mv:
                        for field in fv.files:
                            if fv[field].dtype.kind == 'f': close(mv[field], fv[field][selected], 'Final monitor subset '+field)
                            else: require(np.array_equal(mv[field], fv[field][selected]), 'Final monitor subset differs '+field)
                for item in row['final'][part]['natural']['uniform']['joint_argmax_action_counts']:
                    actions[tuple(item['action_indices'])] += item['worlds']
            domain = dict(worlds=sum(actions.values()), distinct_joint_argmax_actions=len(actions),
                joint_argmax_action_counts=[dict(action_indices=list(a), worlds=n) for a, n in sorted(actions.items())],
                weighting='raw_complete_domain_counts_only_not_target_distribution')
            require(domain == row['full_domain_actions'], 'Full-domain action counts differ')
            candidate = finals['heldout_layouts']['natural']['full_success_rate'] >= .99
            require(candidate == row['candidate_threshold_met'], 'Preserved candidate threshold differs')
            summaries.append(dict(seed=seed, ecology=e, condition=c, final=finals, candidate_threshold_met=candidate))
        del final_networks
    require(counts['training_records'] == 192000 and counts['checkpoint_files'] == 192 and counts['monitor_npz'] == 576
        and counts['final_npz'] == 96 and counts['final_forward_worlds'] == 4561920
        and counts['network_forward_samples'] == 41057280 and counts['saved_evaluation_worlds'] == 4948992
        and counts['silent_closed_references'] == 224 and len(weight_paths) == 16, 'Audit coverage/budget differs')
    require(results['pairing_checks'] == pairing, 'Saved pairing checks differ')
    primary = primary_from_summaries(summaries)
    compare(results['primary_comparison'], primary, 'primary/')
    require(results['all_seed_candidates_by_ecology_condition'] == {e: {c: all(r['candidate_threshold_met'] for r in summaries
        if r['ecology'] == e and r['condition'] == c) for c in CONDITIONS} for e in ECOLOGIES}, 'Candidate aggregate differs')
    for path in (run/'plan.json', run/'prepared.json', run/'freeze.json', execution/'results.json', execution/'status.json', execution/'started.json', execution/'input_arrays.json'):
        artifacts[str(path)] = sha(path)
    require(all(sha(p) == digest for p, digest in sources.items()), 'Sources changed during audit')
    require(all(sha(p) == digest for p, digest in artifacts.items()), 'Records changed during audit')
    return dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-started,
        audit_source_sha256=sha(__file__), sources_sha256=sources, artifacts_sha256=artifacts,
        counts=dict(counts), weight_npz=len(weight_paths), paired_seeds=4, primary_comparison=primary, runs=summaries,
        max_action_probability_abs_error=max_p, max_conditional_statistic_abs_error=max_stat,
        limits=['Actual final saved weights forwarded; not a zero-forward audit.',
            'No training, optimizer replay, per-step gradient recomputation or intermediate-checkpoint neural forward.',
            'Intermediate monitoring records/weights/native rewards and conditional action statistics checked; token tie counts only bounded.',
            '192000 training logs reproduce external U and world ids; sampled token hashes bound but not regenerated from historical weights.',
            'Independent weighted metrics do not call official metrics.summarize or primary_comparison.',
            'Monitor gamma is only an empirical bound on the fixed672 sampled worlds; complete endpoints carry target-distribution bounds.',
            'Closed silent rows reuse natural records and do not count as additional forward evaluations.',
            'Large finite world counts are not independent training replications or language evidence.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); require(not args.out.exists(), 'No overwrite or automatic retry')
    args.out.mkdir(parents=True)
    try:
        result = audit(args.run)
        (args.out/'verification.json').write_bytes(json_bytes(result))
        (args.out/'独立核验.md').write_text('# 伙伴生态执行独立核验\n\n核验通过。32运行的192000批外生随机数与实际世界、192检查点、576监测和96完整终点相符。独立重算4561920末点世界的两窗消息与行动；全部新的分层权重、角色指标和四种子差中差另用独立逻辑核对。\n\n没有重放优化器或中间模型。silent闭信引用没有新增前向。监测gamma仅适用于固定672世界；不作为总体严格界。完整来源、数字、误差与限制见verification.json。\n')
        print(json.dumps({k: result[k] for k in ('status', 'counts', 'max_action_probability_abs_error', 'max_conditional_statistic_abs_error')}, ensure_ascii=False))
    except BaseException as error:
        (args.out/'failure.json').write_bytes(json_bytes(dict(status='failed', error=str(error), traceback=traceback.format_exc(),
            audit_source_sha256=sha(__file__), automatic_retry=False)))
        raise
