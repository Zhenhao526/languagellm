"""Post-hoc saved-array check of the PI remote-both endpoint-swap identity.

No network imports, parameter loads, forward calls, optimization or settlement.
Reconstruct actual routed inputs from saved states/messages/donor packets and
bind them to the producer's saved batch input hashes.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
import numpy as np


def require(test, message):
    if not test: raise AssertionError(message)


def sha(path): return sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def array_sha(a):
    a = np.ascontiguousarray(a)
    header = (json.dumps(dict(shape=list(a.shape), dtype=a.dtype.str), sort_keys=True,
                         separators=(',', ':'), allow_nan=False)+'\n').encode()
    h = sha256(header); h.update(a.tobytes()); return h.hexdigest()


def load(path, sources, expected=None):
    path = Path(path).resolve(); digest = sha(path)
    require(expected is None or digest == expected, 'Changed input '+str(path))
    sources[str(path)] = digest
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def states(space, ids):
    l, o = len(space['layouts']), len(space['private_sites'])
    return np.concatenate((np.asarray(space['needs'])[ids//(l*o)],
        np.asarray(space['layouts'])[(ids//o) % l],
        np.asarray(space['private_sites'])[ids % o]), axis=-1)


def private_features(state):
    """Independent 54-column PI encoder: own need, public/private item, owners."""
    n = len(state); x = np.zeros((n, 3, 54), dtype=np.float64); rows = np.arange(n)
    for a in range(3):
        need = state[:, a]; resource, destination = need//3, need % 3
        x[:, a, 7*a] = 1; x[rows, a, 7*a+1+resource] = 1
        for d in (0, 1): x[:, a, 7*a+5+d] = (destination == d) | (destination == 2)
        for site in range(4):
            visible = np.ones(n, dtype=bool) if site == 0 else state[:, 7+a] == site
            use = rows[visible]; material = state[visible, 3+site]
            x[use, a, 21+5*site] = 1
            x[use, a, 22+5*site+material//2] = 1
            x[use, a, 24+5*site+material % 2] = 1
        for owner in range(3): x[rows, a, 41+3*(state[:, 7+owner]-1)+owner] = 1
        x[:, a, 50+a] = 1
    return x


def route(tokens, senders, donor, live):
    n = len(tokens); rows = np.arange(n); out = np.zeros((n, 3, 99), dtype=np.float64)
    for viewer in range(3):
        for speaker in range(3):
            if live or viewer == speaker:
                packet = tokens[:, speaker].copy()
                if viewer != speaker:
                    chosen = senders == speaker; packet[chosen] = donor[chosen]
                out[:, viewer, 96+speaker] = 1
                for position in range(4):
                    out[rows, viewer, 32*speaker+8*position+packet[:, position]] = 1
    return out


def inputs(record, data, state, senders, live):
    x = private_features(state)
    routes = [route(data['messages'][:, w], senders, data['donor_packets'][:, w], live) for w in (0, 1)]
    full = np.concatenate((x, *routes), axis=-1)
    for b in record['routing_batches']:
        sl = slice(b['start'], b['end'])
        for key, a in (('first_routes', routes[0]), ('second_routes', routes[1]), ('action_inputs', full)):
            require(array_sha(a[sl]) == b[key+'_sha256'], 'Reconstructed producer input hash '+key)
    return x, full


def endpoint_values(directions, masks, listener):
    rows = np.arange(len(listener))
    chosen = np.stack([d['action_indices'][rows, listener] for d in directions], axis=1)
    p = np.stack([d['action_probabilities'][rows, listener] for d in directions], axis=1)
    accepted = ((masks[:, :, None] >> np.arange(17, dtype=np.uint32)) & 1)
    current = ((masks >> chosen.astype(np.uint32)) & 1).astype(bool)
    target = ((masks[:, ::-1] >> chosen.astype(np.uint32)) & 1).astype(bool)
    return dict(current_apt=current, target_apt=target,
                current_mass=(p*accepted).sum(-1), target_mass=(p*accepted[:, ::-1]).sum(-1))


def weighted(values, axis, weights):
    by_axis = []
    for a in range(3):
        use = axis == a; w = weights[use]
        require(abs(float(w.sum())-1) < 1e-10, 'Axis weights')
        by_axis.append(float(np.dot(w/w.sum(), values[use].mean(1))))
    return dict(by_axis=by_axis, macro=float(np.mean(by_axis)))


def audit(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    require(not out.exists(), 'Never overwrite endpoint-swap check')
    out.mkdir(parents=True)
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    result_path = run/'execution/results.json'; result = read(result_path)
    require(result['status'] == 'completed', 'Wait for complete')
    sources[str(result_path)] = sha(result_path)
    data_root = Path(result['dataset_directory'])
    manifest_path, spaces_path = data_root/'manifest.json', data_root/'endpoint_index_spaces.json'
    manifest, spaces = read(manifest_path), read(spaces_path)
    sources[str(manifest_path)] = sha(manifest_path); sources[str(spaces_path)] = sha(spaces_path)
    require(sha(spaces_path) == manifest['outputs']['endpoint_index_spaces.json']['sha256'], 'State index SHA')
    specs = {p: load(data_root/(p+'.npz'), sources, manifest['outputs'][p+'.npz']['sha256'])
             for p in ('train', 'heldout_layouts')}
    selected = [r for r in result['intervention_records'] if r['mode'] in ('remote_same_both', 'remote_opposite_both')]
    index = {(r['seed'], r['condition'], r['partition'], r['mode'], r['direction']): r for r in selected}
    expected = {(s, c, p, m, b) for s in (49101, 49102, 49103, 49104)
                for c in ('PI_silent', 'PI_live') for p in specs
                for m in ('remote_same_both', 'remote_opposite_both') for b in (0, 1)}
    require(len(selected) == len(index) == 64 and set(index) == expected, 'Complete64 saved files')
    checks = []; summaries = []; paired_worlds = 0; paired_agents = 0; max_error = 0.; batch_count = 0
    for seed in (49101, 49102, 49103, 49104):
        for condition in ('PI_silent', 'PI_live'):
            live = condition == 'PI_live'
            for part, spec in specs.items():
                rows = np.flatnonzero(spec['classification'] == 1); n = len(rows)
                senders, listeners = spec['sender'][rows], spec['listener'][rows]
                masks, axis, weights = spec['success_action_masks'][rows], spec['axis'][rows], spec['content_within_axis_weight'][rows]
                data = {}; records = {}
                for mode in ('remote_same_both', 'remote_opposite_both'):
                    data[mode] = []; records[mode] = []
                    for b in (0, 1):
                        rec = index[seed, condition, part, mode, b]
                        d = load(rec['path'], sources, rec['data_sha256'])
                        require(np.array_equal(d['dataset_rows'], rows), 'Preserved case order')
                        require(np.array_equal(d['recipient_indices'], spec['endpoint_indices'][rows, b]), 'Recipient direction')
                        records[mode].append(rec); data[mode].append(d)
                for b in (0, 1):
                    opposite, same = data['remote_opposite_both'][b], data['remote_same_both'][1-b]
                    require(np.array_equal(opposite['donor_partition'], same['donor_partition'])
                        and np.array_equal(opposite['donor_indices'], same['donor_indices'])
                        and np.array_equal(opposite['donor_packets'], same['donor_packets']), 'Swapped donor identity')
                    left_state = states(spaces[part], opposite['recipient_indices'])
                    right_state = states(spaces[part], same['recipient_indices'])
                    changed = left_state != right_state
                    want = np.zeros((n, 10), dtype=bool); want[np.arange(n), senders] = True
                    require(np.array_equal(changed, want), 'Exactly sender need changes')
                    lx, li = inputs(records['remote_opposite_both'][b], opposite, left_state, senders, live)
                    rx, ri = inputs(records['remote_same_both'][1-b], same, right_state, senders, live)
                    nonsender = np.arange(3)[None, :] != senders[:, None]
                    require(np.array_equal(lx[nonsender], rx[nonsender]), 'Non-sender own observations')
                    require(np.array_equal(li[nonsender], ri[nonsender]), 'Non-sender full input exchange')
                    require(np.array_equal(opposite['action_indices'][nonsender], same['action_indices'][nonsender]), 'Non-sender saved action exchange')
                    error = float(np.max(np.abs(opposite['action_probabilities'][nonsender]-same['action_probabilities'][nonsender])))
                    require(error == 0., 'Non-sender probability exchange not exact')
                    max_error = max(max_error, error); paired_worlds += n; paired_agents += 2*n
                    batch_count += len(records['remote_opposite_both'][b]['routing_batches'])+len(records['remote_same_both'][1-b]['routing_batches'])
                    checks.append(dict(seed=seed, condition=condition, partition=part, opposite_recipient_direction=b,
                        same_recipient_direction=1-b, case_worlds=n, nonsender_agents=2*n,
                        full_input_exact=True, saved_actions_exact=True, saved_probability_max_error=error,
                        sender_action_difference_count=int(np.count_nonzero(opposite['action_indices'][np.arange(n),senders]
                            != same['action_indices'][np.arange(n),senders])),
                        recipient_reward_difference_count=int(np.count_nonzero(opposite['greedy_reward'] != same['greedy_reward']))))
                sv = endpoint_values(data['remote_same_both'], masks, listeners)
                ov = endpoint_values(data['remote_opposite_both'], masks, listeners)
                for cur, tar in (('current_apt', 'target_apt'), ('current_mass', 'target_mass')):
                    require(np.array_equal(ov[tar], sv[cur][:, ::-1]), 'Current/target directional identity')
                    require(np.array_equal(ov[cur], sv[tar][:, ::-1]), 'Reverse directional identity')
                calculated = {prefix+'_'+name: weighted(v, axis, weights)
                    for prefix, values in (('same',sv),('opposite',ov)) for name,v in values.items()}
                calculated['contrast_target_apt'] = weighted(ov['target_apt'].astype(float)-sv['target_apt'], axis, weights)
                calculated['equivalent_same_current_minus_target'] = weighted(sv['current_apt'].astype(float)-sv['target_apt'], axis, weights)
                require(abs(calculated['contrast_target_apt']['macro']-calculated['equivalent_same_current_minus_target']['macro']) < 1e-15, 'Contrast equality')
                summaries.append(dict(seed=seed, condition=condition, partition=part, **calculated))
    require(all(sha(path) == h for path,h in sources.items()), 'Source changed during check')
    held = [r for r in summaries if r['condition']=='PI_live' and r['partition']=='heldout_layouts']
    result = dict(status='passed_saved_array_identity', completed_at=datetime.now(timezone.utc).isoformat(),
        posthoc=True, sources_sha256=sources, saved_intervention_files=64, paired_direction_checks=len(checks),
        paired_case_worlds=paired_worlds, paired_nonsender_inputs=paired_agents,
        full_producer_batch_input_hashes_checked=batch_count, max_nonsender_probability_error=max_error,
        checks=checks, statistics=summaries,
        PI_live_heldout_equal_seed_means={k:float(np.mean([r[k]['macro'] for r in held])) for k in held[0]
            if isinstance(held[0][k],dict)},
        model_parameter_loads=0, network_forward_calls=0, optimizer_updates=0,
        input_reconstruction='Independent PI encoding + saved generated messages + saved outbound donor overrides, bound to complete producer batch hashes.',
        limits=['No new generation or natural-remote-donor reproduction claim.',
            'Equality concerns the two non-senders. Sender action and recipient native R need not exchange.',
            'The check establishes algebraic redundancy of both-window marginal statistics, not compositionality.'])
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return {k:result[k] for k in ('status','saved_intervention_files','paired_case_worlds','paired_nonsender_inputs','full_producer_batch_input_hashes_checked','PI_live_heldout_equal_seed_means')}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    try: print(json.dumps(audit(a.run,a.out),ensure_ascii=False))
    except BaseException as error:
        out=Path(a.out)
        if out.is_dir() and not (out/'failure.json').exists():
            (out/'failure.json').write_text(json.dumps(dict(status='failed',error=repr(error)),ensure_ascii=False)+'\n')
        raise
