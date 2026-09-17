"""Independent audit of the completed semantic probe; no training.

The module is being prepared before execution. Importing it does not load any
checkpoint or read new diagnostic results. Reused mathematical primitives are
verified against the previous completed execution audit before use.
"""
from __future__ import annotations
import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'
from pathlib import Path
import hashlib
import json
import argparse
from collections import Counter
from datetime import datetime, timezone
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'research_program/triadic_partner_ecology_study/results/partner_001'
SEEDS = (49101, 49102, 49103, 49104)
CONDITIONS = ('FI_silent', 'FI_live', 'PI_silent', 'PI_live')
PARTITIONS = ('train', 'heldout_layouts')
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
MODES = ('sham', 'local_opposite_w1', 'local_opposite_w2', 'local_opposite_both',
         'remote_same_w1', 'remote_same_w2', 'remote_same_both',
         'remote_opposite_w1', 'remote_opposite_w2', 'remote_opposite_both')
TOLERANCE = 2e-12


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    header = (json.dumps(dict(shape=list(a.shape), dtype=a.dtype.str), sort_keys=True,
                         separators=(',', ':'), allow_nan=False)+'\n').encode()
    h = hashlib.sha256(header); h.update(a.tobytes())
    return h.hexdigest()


def historical_primitives():
    """Do not trust an unanchored current helper merely because its name matches."""
    receipt_path = OLD/'audit_execution_001/verification.json'
    receipt = read(receipt_path)
    require(receipt['status'] == 'passed', 'Historical execution audit did not pass')
    names = ('research_program/triadic_message_study/audit_execution.py',
             'research_program/triadic_learning_baseline/audit_execution.py')
    sources = {}
    for name in names:
        path = ROOT/name
        require(sha(path) == receipt['sources_sha256'][str(path)], 'Historical helper source changed: '+name)
        sources[str(path)] = sha(path)
    # These modules implement independent features/MLP/score. No producer call.
    from research_program.triadic_message_study import audit_execution as previous
    return previous, dict(receipt_path=str(receipt_path), receipt_sha256=sha(receipt_path), sources_sha256=sources)


def recipient_views(messages, live, senders=None, replacements=None):
    """Construct each receiver's onehots directly; edit only foreign packets."""
    messages = np.asarray(messages)
    n = len(messages)
    require(messages.shape == (n, 3, 4) and messages.dtype.kind in 'iu'
            and np.all((messages >= 0) & (messages < 8)), 'Invalid actual generated messages')
    override = replacements is not None
    if override:
        senders, replacements = np.asarray(senders), np.asarray(replacements)
        require(senders.shape == (n,) and senders.dtype.kind in 'iu'
                and np.all((senders >= 0) & (senders < 3)), 'Invalid designated senders')
        require(replacements.shape == (n, 4) and replacements.dtype.kind in 'iu'
                and np.all((replacements >= 0) & (replacements < 8)), 'Invalid donor packet')
    inputs = np.zeros((n, 3, 99), dtype=np.float64)
    rows = np.arange(n)
    for receiver in range(3):
        for sender in range(3):
            if live or receiver == sender:
                packet = messages[:, sender].copy()
                if override and receiver != sender:
                    use = senders == sender
                    packet[use] = replacements[use]
                inputs[:, receiver, 96+sender] = 1.0
                for position in range(4):
                    inputs[rows, receiver, 32*sender+8*position+packet[:, position]] = 1.0
    return inputs


def intervention_forward(networks, features, natural_messages, senders, donor_messages,
                         windows, live, probability_function):
    """Independent six-module replay; production intervention is never imported.

    probability_function is the SHA-anchored independent MLP/softmax in real
    execution. Tests may substitute a deterministic artificial function.
    """
    x, natural, donor = map(np.asarray, (features, natural_messages, donor_messages))
    n = len(x)
    require(x.shape == (n, 3, 54) and natural.shape == (n, 2, 3, 4)
            and donor.shape == (n, 2, 4), 'Invalid full intervention dimensions')
    require(tuple(windows) in ((0,), (1,), (0, 1)) and len(networks) == 9, 'Invalid windows/modules')
    first = recipient_views(natural[:, 0], live, senders, donor[:, 0]) if 0 in windows else recipient_views(natural[:, 0], live)
    second_inputs = np.concatenate((x, first), axis=2)
    # All first-window receiver inputs are complete before any second window.
    probabilities = [probability_function(networks[3*a+1], second_inputs[:, a]) for a in range(3)]
    require(all(p.shape == (n, 4, 8) for p in probabilities), 'Second sender shape')
    generated_second = np.stack([p.argmax(-1) for p in probabilities], axis=1).astype(np.int8)
    second = recipient_views(generated_second, live, senders, donor[:, 1]) if 1 in windows else recipient_views(generated_second, live)
    action_inputs = np.concatenate((x, first, second), axis=2)
    policies = np.stack([probability_function(networks[3*a+2], action_inputs[:, a]) for a in range(3)], axis=1)
    require(policies.shape == (n, 3, 17) and np.isfinite(policies).all(), 'Action policy shape/finite')
    return dict(messages=np.stack((natural[:, 0], generated_second), axis=1),
                first_routes=first, second_routes=second, action_inputs=action_inputs,
                action_probabilities=policies, action_indices=policies.argmax(-1).astype(np.int16))


def demand_accepts(needs, material, destination):
    """Boolean acceptance from original kind/length and destination sets."""
    resource_membership = np.array([[m in s for m in range(4)] for s in ((0, 1), (2, 3), (0, 2), (1, 3))])
    destination_membership = np.array([[d in s for d in range(2)] for s in ((0,), (1,), (0, 1))])
    return resource_membership[np.asarray(needs)//3, material] & destination_membership[np.asarray(needs) % 3, destination]


def success_masks(states, listeners):
    """Enumerate every matched plan and project native full success, no teacher."""
    states, listeners = np.asarray(states), np.asarray(listeners)
    n = len(states)
    require(states.shape == (n, 10) and listeners.shape == (n,), 'Bad full-success mask input')
    masks = np.zeros(n, dtype=np.uint32)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        outsider = 3-left-right
        for site in range(4):
            for dest in range(2):
                good = demand_accepts(states[:, left], states[:, 3+site], dest) & demand_accepts(states[:, right], states[:, 3+site], dest)
                actions = {outsider: 0, left: 1+4*site+2*dest+[a for a in range(3) if a != left].index(right),
                           right: 1+4*site+2*dest+[a for a in range(3) if a != right].index(left)}
                for who, action in actions.items():
                    masks[good & (listeners == who)] |= np.uint32(1 << action)
    require(np.all(masks != 0), 'A diagnostic world lacks a full-success plan')
    return masks


def expected_budgets():
    natural = 5*16*46656
    interventions = 8*17280*2*10
    return dict(new_natural_npz=160, new_intervention_npz=320, reused_natural_npz=32,
                new_natural_worlds=natural, new_intervention_worlds=interventions,
                new_forward_worlds=natural+interventions,
                new_network_samples=9*natural+6*interventions,
                referenced_old_natural_worlds=16*46656)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def packed_states(space, indices):
    indices = np.asarray(indices)
    layouts, owners = len(space['layouts']), len(space['private_sites'])
    return np.concatenate((np.asarray(space['needs'], dtype=np.int16)[indices//(layouts*owners)],
        np.asarray(space['layouts'], dtype=np.int16)[(indices//owners) % layouts],
        np.asarray(space['private_sites'], dtype=np.int16)[indices % owners]), axis=-1)


def close(actual, expected, name):
    require(np.shape(actual) == np.shape(expected), 'Shape differs: '+name)
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), 'Nonfinite '+name)
    error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)))) if np.size(actual) else 0.0
    require(np.allclose(actual, expected, atol=TOLERANCE, rtol=TOLERANCE), 'Numerical mismatch: '+name)
    return error


def checked_dataset(directory):
    """Check saved labels by original full-success formula, not producer labels."""
    directory = Path(directory)
    manifest = read(directory/'manifest.json')
    for name, record in manifest['outputs'].items():
        require(sha(directory/name) == record['sha256'], 'Dataset file SHA '+name)
    for path, digest in manifest['source_sha256'].items():
        require(sha(path) == digest, 'Dataset source SHA '+path)
    spaces = read(directory/'endpoint_index_spaces.json')
    original = read(OLD/'prepared.json')
    require(sha(OLD/'prepared.json') == read(OLD/'freeze.json')['prepared_sha256'], 'Old endpoint universe changed')
    cases = read(directory/'cases.json')
    require(len(cases) == 4488, 'Case metadata missing')
    unique = [c for c in cases if c['ecology'] == 'unique']
    require(len(unique) == 768 and Counter(c['classification'] for c in unique) == dict(content=120, role=408, descriptive=240), 'Unique cases differ')
    counts = Counter((c['classification_index'], c['axis_index'], c['sender'], c['listener']) for c in unique)
    specs, universes = {}, {}
    remote_index = {}
    for part in PARTITIONS:
        space = spaces[part]; old = original['partitions']['unique'][part]
        require(all(space[key] == old[key] for key in ('needs', 'layouts', 'private_sites')), 'Dataset universe differs from old freeze')
        for index, layout in enumerate(space['layouts']):
            remote_index[tuple(layout)] = (PARTITIONS.index(part), index)
        universes[part] = packed_states(space, np.arange(space['endpoint_world_count']))
    for part in PARTITIONS:
        z = load_npz(directory/f'{part}.npz'); specs[part] = z
        space = spaces[part]; n = len(z['case_index']); physical = len(space['layouts'])*6
        require(n == (82944 if part == 'train' else 27648), 'Wrong complete pair count')
        require(np.array_equal(z['case_index'], np.repeat([c['case_index'] for c in unique], physical)), 'Case order/coverage')
        require(np.array_equal(z['layout_indices'], np.tile(np.repeat(np.arange(len(space['layouts'])), 6), 768)), 'Layout coverage')
        require(np.array_equal(z['owner_indices'], np.tile(np.arange(6), n//6)), 'Owner coverage')
        endpoints = packed_states(space, z['endpoint_indices'])
        require(np.array_equal(endpoints[:, :, :3], np.asarray([cases[int(i)]['needs'] for i in z['case_index']])), 'Endpoint needs/case alignment')
        for direction in (0, 1):
            masks = success_masks(endpoints[:, direction], z['listener'])
            require(np.array_equal(masks, z['success_action_masks'][:, direction]), 'Native full-success action set')
        changed = endpoints[:, 0, :3] != endpoints[:, 1, :3]
        require(np.all(changed.sum(-1) == 1) and np.array_equal(changed.argmax(-1), z['sender']), 'Changed private subject differs')
        require(np.all(endpoints[:, 0, 3:] == endpoints[:, 1, 3:]), 'Natural pair background changed')
        before = endpoints[np.arange(n), 0, z['sender']]
        after = endpoints[np.arange(n), 1, z['sender']]
        axis_expected = np.where(before//3 == after//3, 2, np.where(before//3 < 2, 0, 1))
        require(np.array_equal(axis_expected, z['axis']), 'Demand axis meaning differs')
        resource_flip = z['axis'] != 2
        require(np.all((before[resource_flip]//3 ^ after[resource_flip]//3) == 1)
                and np.all(before[resource_flip] % 3 == after[resource_flip] % 3), 'Invalid resource flip')
        require(np.all((before[~resource_flip] % 3 + after[~resource_flip] % 3) == 1), 'Invalid destination flip')
        masks = z['success_action_masks']; roles = (masks != 1).astype(np.int8)
        require(np.array_equal(roles, z['roles']), 'Role labels differ')
        disjoint = (masks[:, 0] & masks[:, 1]) == 0
        classes = np.where(roles[:, 0] != roles[:, 1], 2, np.where(disjoint, 1, 0)).astype(np.int8)
        require(np.array_equal(classes, z['classification']), 'Content/role classification differs')
        actual_partners = np.full((n, 2), -1, dtype=np.int8)
        for who in range(3):
            own = z['listener'] == who
            for peer_id, peer in enumerate([a for a in range(3) if a != who]):
                peer_mask = sum(1 << (1+4*site+2*dest+peer_id) for site in range(4) for dest in range(2))
                use = own[:, None] & ((masks & peer_mask) != 0)
                require(not np.any(use & (actual_partners != -1)), 'Unique case has multiple participant partners')
                actual_partners[use] = peer
        require(np.array_equal(actual_partners, z['listener_partners']), 'Required partner labels differ')
        for key, code in (('content_within_axis_weight', 1), ('role_within_axis_weight', 2)):
            weights = np.zeros(n)
            for row, case_id in enumerate(z['case_index']):
                c = cases[int(case_id)]
                require(all(int(z[field][row]) == int(c[casefield]) for field, casefield in
                    (('sender', 'sender'), ('listener', 'listener'), ('axis', 'axis_index'), ('classification', 'classification_index'))), 'Static case fields')
                if c['classification_index'] == code:
                    weights[row] = 1/(6*counts[(code, c['axis_index'], c['sender'], c['listener'])]*physical)
            close(z[key], weights, key)
        for row in range(n):
            layout = tuple((endpoints[row, 0, 3:7]+1) % 4)
            dp, li = remote_index[layout]
            require(int(z['remote_donor_partition'][row]) == dp and int(z['remote_donor_layout_indices'][row]) == li, 'Remote plus1 mapping')
            donor_space = spaces[PARTITIONS[dp]]
            expected = packed_states(donor_space, z['remote_donor_endpoint_indices'][row])
            require(np.array_equal(expected[:, :3], endpoints[row, :, :3]) and np.all(expected[:, 3:7] == layout)
                    and np.array_equal(expected[:, 7:], endpoints[row, :, 7:]), 'Remote source content/owner mapping')
    return specs, universes, spaces, manifest


def metric_summary(values, spec, code):
    """Independent within-axis weighted sums; do not import new metrics."""
    axes = (0, 1, 2) if code == 1 else (0, 1)
    weights = spec['content_within_axis_weight' if code == 1 else 'role_within_axis_weight']
    result = {}
    for key, array in values.items():
        by_axis = []
        for axis in axes:
            mask = (spec['classification'] == code) & (spec['axis'] == axis)
            w = weights[mask]
            require(abs(float(w.sum())-1) <= 1e-10, 'Metric weights do not sum to1')
            by_axis.append(np.sum(array[mask]*w.reshape((-1,)+(1,)*(array.ndim-1)), axis=0))
        means = np.asarray(by_axis)
        result[key] = dict(by_axis=means.tolist(), macro=np.mean(means, axis=0).tolist())
    return result


def listener_values(actions, probabilities, masks, listener):
    n = len(listener)
    selected = actions[np.arange(n)[:, None], np.arange(2)[None, :], listener[:, None]]
    p = probabilities[np.arange(n)[:, None], np.arange(2)[None, :], listener[:, None]]
    apt = ((masks >> selected.astype(np.uint32)) & 1).astype(bool)
    mass = np.sum(p*((masks[:, :, None] >> np.arange(17, dtype=np.uint32)) & 1), axis=-1)
    return selected, apt, mass


def natural_statistics(data, spec):
    ids = spec['endpoint_indices']; actions = data['action_indices'][ids]; p = data['action_probabilities'][ids]
    selected, apt, mass = listener_values(actions, p, spec['success_action_masks'], spec['listener'])
    values = dict(both_endpoints_apt=apt.all(1), endpoint_apt=apt, mean_endpoint_apt=apt.mean(1),
        endpoint_probability_mass=mass, mean_endpoint_probability_mass=mass.mean(1),
        listener_action_changed=selected[:, 0] != selected[:, 1],
        any_team_action_changed=np.any(actions[:, 0] != actions[:, 1], axis=-1),
        both_teams_full=(data['greedy_reward'][ids] == 1).all(1), native_reward=data['greedy_reward'][ids])
    return {name: metric_summary(values, spec, code) for name, code in (('content', 1), ('role', 2))}


def intervention_statistics(directions, pool, spec, rows):
    subset = {key: value[rows] for key, value in spec.items()}
    actions = np.stack([d['action_indices'] for d in directions], axis=1)
    probabilities = np.stack([d['action_probabilities'] for d in directions], axis=1)
    _, apt, mass = listener_values(actions, probabilities, subset['success_action_masks'], subset['listener'])
    _, counterfactual, targetmass = listener_values(actions, probabilities, subset['success_action_masks'][:, ::-1], subset['listener'])
    ids = subset['endpoint_indices']
    _, naturalapt, naturalmass = listener_values(pool['action_indices'][ids], pool['action_probabilities'][ids], subset['success_action_masks'], subset['listener'])
    reward = np.stack([d['greedy_reward'] for d in directions], axis=1)
    values = dict(current_apt=apt, counterfactual_apt=counterfactual, current_probability_mass=mass,
        counterfactual_probability_mass=targetmass, native_reward=reward, native_full_success=reward == 1,
        natural_current_apt=naturalapt, natural_current_probability_mass=naturalmass,
        current_apt_minus_natural=apt.astype(float)-naturalapt,
        current_probability_mass_minus_natural=mass-naturalmass)
    values.update({'direction_mean_'+key: value.mean(1) for key, value in list(values.items())})
    return metric_summary(values, subset, 1), values


def check_saved_outcomes(data, states, previous):
    n = len(states)
    require(data['messages'].shape == (n, 2, 3, 4) and data['messages'].dtype == np.int8
            and np.all((data['messages'] >= 0) & (data['messages'] < 8)), 'Saved message domain/type')
    actions, probabilities = data['action_indices'], data['action_probabilities']
    require(actions.shape == (n, 3) and actions.dtype == np.int16 and np.all((actions >= 0) & (actions < 17)), 'Full17 action domain/type')
    require(probabilities.shape == (n, 3, 17) and probabilities.dtype == np.float64
            and np.isfinite(probabilities).all() and np.all(probabilities >= 0), 'Saved probability domain/type')
    close(probabilities.sum(-1), np.ones((n, 3)), 'Normalization')
    require(np.array_equal(actions, probabilities.argmax(-1)), 'Saved greedy actions differ')
    reward, executed, satisfied = previous.native_settlement(states, actions)
    require(np.array_equal(reward, data['greedy_reward']) and np.array_equal(executed, data['executed'])
            and np.array_equal(satisfied, data['satisfied']), 'Independent native D1 settlement mismatch')
    return n


def audit(run):
    """Reproduce all newly saved results only after the producer completed."""
    run = Path(run).resolve(); execution = run/'execution'
    require(read(execution/'status.json')['status'] == 'completed', 'Producer not completed')
    result = read(execution/'results.json'); require(result['status'] == 'completed', 'Incomplete results')
    require(not (execution/'failure.json').exists(), 'Producer has a failure record')
    plan = read(run/'plan.json'); freeze = read(run/'freeze.json')
    require(sha(run/'plan.json') == freeze['plan_sha256'] == result['plan_sha256'], 'Plan SHA mismatch')
    require(plan['config'] == result['config'], 'Runtime config differs from frozen plan')
    for field, expected in (('seeds', SEEDS), ('conditions', CONDITIONS), ('partitions', PARTITIONS), ('checkpoints', CHECKPOINTS), ('modes', MODES)):
        require(tuple(plan['config'][field]) == expected, 'Fixed matrix differs '+field)
    require(plan['config']['batch_size'] == 1024 and plan['config']['training_updates'] == 0, 'Batch/budget changed')
    require(result['new_network_samples'] == 50181120 and result['model_parameter_loads'] == 88, 'Declared forward/load budget')
    sources = {}
    for path, digest in plan['sources_sha256'].items():
        require(sha(path) == digest, 'Frozen source changed '+path)
        require(sha(run/'source_snapshot'/Path(path).relative_to(ROOT)) == digest, 'Snapshot source changed '+path)
        sources[path] = digest
    for path, digest in plan['inputs_sha256'].items():
        require(sha(path) == digest, 'Frozen input changed '+path)
    previous, history = historical_primitives()
    old_receipt = read(history['receipt_path'])
    specs, worlds, spaces, manifest = checked_dataset(plan['dataset_directory'])
    feature_hashes = read(execution/'feature_hashes.json')
    features = {part: {full: previous.observed_features(states, full) for full in (False, True)} for part, states in worlds.items()}
    for part in PARTITIONS:
        for full in (False, True):
            require(array_sha(features[part][full]) == feature_hashes[part][str(full)], 'Independent observation hash differs')
    require(feature_hashes == result['feature_hashes'], 'Result observation hashes differ')
    for part in PARTITIONS:
        spec = specs[part]; ids = spec['endpoint_indices']; listener = spec['listener']
        require(np.array_equal(features[part][False][ids[:, 0], listener], features[part][False][ids[:, 1], listener]), 'Listener private observations differ')
    policies = plan['policies']
    require([(p['seed'], p['condition']) for p in policies] == [(s, c) for s in SEEDS for c in CONDITIONS], 'Policies not complete/canonical')
    require(policies == result['policies'], 'Policy source mapping changed')
    for policy in policies:
        folder = OLD/'execution'/f"seed_{policy['seed']}_unique_{policy['condition']}"
        require(policy['checkpoints'] == {str(u): str(folder/f'checkpoint_{u:04d}.npz') for u in CHECKPOINTS}, 'Checkpoint identity/ecology map')
        require(policy['endpoints'] == {p: str(folder/f'final_{p}_natural.npz') for p in PARTITIONS}, 'Natural endpoint condition/ecology map')
        for path in policy['checkpoints'].values():
            require(sha(path) == old_receipt['artifacts_sha256'][path], 'Full checkpoint historical SHA closure')
    natural = result['natural_records']; interventions = result['intervention_records']
    nindex = {(r['seed'], r['condition'], r['checkpoint'], r['partition']): r for r in natural}
    iindex = {(r['seed'], r['condition'], r['partition'], r['mode'], r['direction']): r for r in interventions}
    require(len(natural) == len(nindex) == 192 and set(nindex) == {(s, c, u, p) for s in SEEDS for c in CONDITIONS for u in CHECKPOINTS for p in PARTITIONS}, 'Natural record matrix')
    require(len(interventions) == len(iindex) == 320 and set(iindex) == {(s, c, p, m, d) for s in SEEDS for c in ('PI_silent', 'PI_live') for p in PARTITIONS for m in MODES for d in (0, 1)}, 'Intervention record matrix')
    actual_files = {str(p.resolve()) for p in execution.glob('seed_*/*.npz')}
    expected_files = {r['path'] for r in natural+interventions if not r['reused_saved_endpoint']}
    require(len(expected_files) == 480 and actual_files == expected_files, 'Unexpected/missing new result files')
    artifacts = {str(execution/'results.json'): sha(execution/'results.json'), str(run/'plan.json'): sha(run/'plan.json')}
    for record in natural+interventions:
        path = Path(record['path'])
        require(path.is_absolute() and sha(path) == record['data_sha256'], 'Result file SHA mismatch')
        require(record['kind'] == ('natural' if record in natural else 'intervention'), 'Wrong record kind')
        artifacts[str(path)] = record['data_sha256']
    checked, statistics, contrasts = [], [], []
    counts = dict(new_natural_npz=0, new_intervention_npz=0, reused_natural_npz=0,
        new_natural_worlds=0, new_intervention_worlds=0, referenced_old_natural_worlds=0,
        independent_network_samples=0, independent_native_settlements=0, checked_routing_batches=0,
        loaded_checkpoint_files=0)
    max_error = 0.0
    for policy in policies:
        seed, condition = policy['seed'], policy['condition']
        full, live = condition.startswith('FI_'), condition.endswith('_live')
        for step in CHECKPOINTS[:-1]:
            path = policy['checkpoints'][str(step)]
            require(sha(path) == old_receipt['artifacts_sha256'][path], 'Old checkpoint not historically anchored')
            networks, _, _ = previous.checkpoint(path, step, seed); counts['loaded_checkpoint_files'] += 1
            for part in PARTITIONS:
                record = nindex[seed, condition, step, part]; data = load_npz(record['path']); states = worlds[part]
                require(record['reused_saved_endpoint'] is False and record['worlds'] == len(states), 'New natural reference flag/count')
                require(np.array_equal(data['states'], states) and np.array_equal(data['state_indices'], np.arange(len(states))), 'New natural state identity')
                counts['independent_native_settlements'] += check_saved_outcomes(data, states, previous)
                for start in range(0, len(states), 1024):
                    end = min(start+1024, len(states)); sl = slice(start, end)
                    messages, p, _ = previous.final_forward(networks, states[sl], condition)
                    require(np.array_equal(messages, data['messages'][sl]), 'New natural messages differ')
                    require(np.array_equal(p.argmax(-1), data['action_indices'][sl]), 'New natural actions differ')
                    max_error = max(max_error, close(p, data['action_probabilities'][sl], 'Natural action probabilities'))
                    counts['independent_network_samples'] += 9*(end-start)
                counts['new_natural_npz'] += 1; counts['new_natural_worlds'] += len(states)
                statistics.append(dict(seed=seed, condition=condition, checkpoint=step, partition=part,
                                       natural=natural_statistics(data, specs[part])))
                checked.append(dict(kind='natural', seed=seed, condition=condition, checkpoint=step, partition=part, worlds=len(states)))
        pool = {}
        for part in PARTITIONS:
            record = nindex[seed, condition, 6000, part]; path = record['path']; data = load_npz(path); pool[part] = data
            require(record['reused_saved_endpoint'] is True and path == policy['endpoints'][part]
                    and sha(path) == old_receipt['artifacts_sha256'][path], 'Endpoint reference/history SHA differs')
            require(np.array_equal(data['states'], worlds[part]) and np.array_equal(data['state_indices'], np.arange(len(worlds[part]))), 'Endpoint state indexing differs')
            counts['independent_native_settlements'] += check_saved_outcomes(data, worlds[part], previous)
            counts['reused_natural_npz'] += 1; counts['referenced_old_natural_worlds'] += len(worlds[part])
            statistics.append(dict(seed=seed, condition=condition, checkpoint=6000, partition=part,
                                   natural=natural_statistics(data, specs[part])))
        if not full:
            path = policy['checkpoints']['6000']
            require(sha(path) == old_receipt['artifacts_sha256'][path], 'Final checkpoint SHA')
            networks, _, _ = previous.checkpoint(path, 6000, seed); counts['loaded_checkpoint_files'] += 1
            for part in PARTITIONS:
                spec = specs[part]; rows = np.flatnonzero(spec['classification'] == 1); n = len(rows)
                require(np.array_equal(rows, np.flatnonzero(spec['content_within_axis_weight'] > 0)), 'All content rows required')
                remote_values = {}
                for mode in MODES:
                    directions = []
                    for direction in (0, 1):
                        record = iindex[seed, condition, part, mode, direction]; data = load_npz(record['path']); directions.append(data)
                        require(record['checkpoint'] == 6000 and record['reused_saved_endpoint'] is False and record['worlds'] == n, 'Intervention identity/count')
                        recipients = spec['endpoint_indices'][rows, direction]; senders = spec['sender'][rows]
                        require(np.array_equal(data['dataset_rows'], rows) and np.array_equal(data['recipient_indices'], recipients), 'Intervention row alignment')
                        endpoint = direction if mode == 'sham' or mode.startswith('remote_same') else 1-direction
                        if mode.startswith('remote_'):
                            dp = spec['remote_donor_partition'][rows]; di = spec['remote_donor_endpoint_indices'][rows, endpoint]
                        else:
                            dp = np.full(n, PARTITIONS.index(part), dtype=np.int8); di = spec['endpoint_indices'][rows, endpoint]
                        require(np.array_equal(dp, data['donor_partition']) and np.array_equal(di, data['donor_indices']), 'Donor identity mismatch')
                        packets = np.empty((n, 2, 4), dtype=np.int8)
                        for pi, name in enumerate(PARTITIONS):
                            use = np.flatnonzero(dp == pi)
                            packets[use] = pool[name]['messages'][di[use], :, senders[use], :]
                        require(np.array_equal(packets, data['donor_packets']), 'Actual donor packet mismatch')
                        windows = (0, 1) if mode == 'sham' or mode.endswith('both') else (0,) if mode.endswith('w1') else (1,)
                        batches = record['routing_batches']
                        require([(b['start'], b['end']) for b in batches] == [(i, min(i+1024, n)) for i in range(0, n, 1024)], 'Routing batch bounds')
                        for batch in batches:
                            start, end = batch['start'], batch['end']; sl = slice(start, end); ids = recipients[sl]
                            trace = intervention_forward(networks, features[part][False][ids], pool[part]['messages'][ids],
                                senders[sl], packets[sl], windows, live, previous.probabilities)
                            for key in ('first_routes', 'second_routes', 'action_inputs'):
                                require(array_sha(trace[key]) == batch[key+'_sha256'], 'Complete routed input SHA differs '+key)
                            require(np.array_equal(trace['messages'], data['messages'][sl]) and np.array_equal(trace['action_indices'], data['action_indices'][sl]), 'Intervention generated tokens/actions differ')
                            max_error = max(max_error, close(trace['action_probabilities'], data['action_probabilities'][sl], 'Intervention probability'))
                            own = senders[sl]; batchrows = np.arange(end-start)
                            require(np.array_equal(trace['messages'][batchrows, 1, own], pool[part]['messages'][ids, 1, own]), 'Sender own generated W2 changed')
                            if mode == 'sham' or not live:
                                require(np.array_equal(trace['messages'], pool[part]['messages'][ids]) and np.array_equal(trace['action_indices'], pool[part]['action_indices'][ids]), 'Sham/silent discrete identity')
                                close(trace['action_probabilities'], pool[part]['action_probabilities'][ids], 'Sham/silent distribution identity')
                            if mode == 'local_opposite_both' and live:
                                expected = np.concatenate((features[part][False][di[sl]], recipient_views(pool[part]['messages'][di[sl], 0], True), recipient_views(pool[part]['messages'][di[sl], 1], True)), axis=2)
                                other = np.arange(3)[None, :] != own[:, None]
                                require(np.array_equal(trace['action_inputs'][other], expected[other]), 'Local-both structural input identity')
                                close(trace['action_probabilities'][other], pool[part]['action_probabilities'][di[sl]][other], 'Local-both distribution identity')
                            counts['checked_routing_batches'] += 1; counts['independent_network_samples'] += 6*(end-start)
                        counts['independent_native_settlements'] += check_saved_outcomes(data, worlds[part][recipients], previous)
                        counts['new_intervention_npz'] += 1; counts['new_intervention_worlds'] += n
                        checked.append(dict(kind='intervention', seed=seed, condition=condition, partition=part, mode=mode, direction=direction, worlds=n))
                    summary, values = intervention_statistics(directions, pool[part], spec, rows)
                    statistics.append(dict(seed=seed, condition=condition, checkpoint=6000, partition=part, mode=mode, intervention=summary))
                    if mode.startswith('remote_'):
                        remote_values[mode] = values
                for window in ('w1', 'w2', 'both'):
                    same, opposite = remote_values['remote_same_'+window], remote_values['remote_opposite_'+window]
                    values = dict(target_apt_opposite_minus_same=opposite['counterfactual_apt'].astype(float)-same['counterfactual_apt'],
                        target_probability_mass_opposite_minus_same=opposite['counterfactual_probability_mass']-same['counterfactual_probability_mass'],
                        current_apt_loss_natural_minus_same=same['natural_current_apt'].astype(float)-same['current_apt'],
                        native_reward_opposite_minus_same=opposite['native_reward']-same['native_reward'])
                    values.update({'direction_mean_'+k: v.mean(1) for k, v in list(values.items())})
                    contrasts.append(dict(seed=seed, condition=condition, partition=part, window=window,
                        remote=metric_summary(values, {k: v[rows] for k, v in spec.items()}, 1)))
        print(json.dumps(dict(audited_seed=seed, condition=condition, counts=counts)), flush=True)
    budget = expected_budgets()
    for key in ('new_natural_npz', 'new_intervention_npz', 'reused_natural_npz', 'new_natural_worlds', 'new_intervention_worlds', 'referenced_old_natural_worlds'):
        require(counts[key] == budget[key], 'Final audit count '+key)
    require(counts['independent_network_samples'] == 50181120 and counts['loaded_checkpoint_files'] == 88, 'Independent total replay count')
    require(result['counts'] == dict(new_natural_worlds=3732480, new_intervention_worlds=2764800, reused_endpoint_worlds=746496), 'Producer final budget count')
    primary = []
    for seed in SEEDS:
        values = {c: next(r['natural']['content']['both_endpoints_apt']['macro'] for r in statistics
            if r.get('natural') and (r['seed'], r['condition'], r['checkpoint'], r['partition']) == (seed, c, 6000, 'heldout_layouts')) for c in ('PI_silent', 'PI_live')}
        require(values['PI_silent'] == 0, 'PI silent structural zero failed')
        primary.append(dict(seed=seed, **values, live_minus_silent=values['PI_live']-values['PI_silent']))
    # Recheck frozen files and outputs after the full replay.
    for path, digest in {**plan['sources_sha256'], **plan['inputs_sha256'], **artifacts}.items():
        require(sha(path) == digest, 'File changed during independent audit '+path)
    return dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(), audit_source_sha256=sha(__file__),
        producer_plan_sha256=sha(run/'plan.json'), historical_primitive_anchors=history,
        sources_sha256=sources, artifacts_sha256=artifacts, counts=counts,
        max_action_probability_abs_error=max_error, natural_and_intervention_statistics=statistics, remote_contrasts=contrasts,
        primary_seed_pairs=primary, primary_equal_seed_mean=float(np.mean([r['live_minus_silent'] for r in primary])),
        checked_new_files=checked, training_updates=0,
        limits=['Old6000 natural NPZ is SHA anchored to prior independent full forward; not forwarded again.',
                'No optimizer replay or new training. Independent replay is additional CPU inference, counted separately.',
                'Primary/remote values independently recalculated; comparison to a later producer analysis artifact is not yet included.',
                'Sham/silent/local_both identities are implementation checks, not additional semantic evidence.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); run, out = Path(args.run).resolve(), Path(args.out).resolve()
    require(read(run/'execution/status.json')['status'] == 'completed', 'Wait for all completed before audit')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        result = audit(run); result['elapsed_seconds'] = time.perf_counter()-started
        with (out/'verification.json').open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        lines = ['# 独立执行核验', '', '状态：通过。全部新自然记录与干预记录已用独立观察、MLP、路由及原生结算重放；32个旧端点按历史独立审计SHA引用。', '',
            f"新自然 160 NPZ、新干预 320 NPZ；本次独立网络样本 {result['counts']['independent_network_samples']:,}，原生结算 {result['counts']['independent_native_settlements']:,}；没有训练。",
            f"最大动作概率绝对误差 {result['max_action_probability_abs_error']:.8g}。", '',
            '四个种子的主要局部观察开放−静默差：'+str([r['live_minus_silent'] for r in result['primary_seed_pairs']]),
            '', '本轮为既有政策的开发诊断；实现恒等不构成额外语义或组合性证据。完整数值、逐记录SHA和限制见 verification.json。']
        (out/'独立核验.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(dict(status='passed', output=str(out), elapsed_seconds=result['elapsed_seconds'])) )
    except BaseException as error:
        failure = dict(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                       elapsed_seconds=time.perf_counter()-started, audit_source_sha256=sha(__file__))
        with (out/'failure.json').open('x') as f: json.dump(failure, f, ensure_ascii=False, indent=2)
        raise


if __name__ == '__main__':
    main()
