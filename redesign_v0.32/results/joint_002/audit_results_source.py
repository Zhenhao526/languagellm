"""Independent bounded audit for the v0.32 joint-payoff population run."""
from __future__ import annotations
import argparse, copy, itertools, json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT)); import analyze_results as stats
sys.path.insert(0, str(PROJECT / 'redesign_v0.21'))
from audit_execution import enumerate_sender, enumerate_receiver
sys.path.insert(0, str(PROJECT / 'redesign_v0.8')); from camp import remake_agents

SEND = ('send_context', 'send_embedding', 'send_recur', 'send_out')
RECV = ('receive_embedding', 'actor')
RESET = SEND + RECV + ('send_value', 'receive_value')
PRIVATE_TYPES = (0, 1, 0, 1)
AGENTS = 4
PAIR_KEYS = stats.PAIR_KEYS
MATCHINGS = {
    'fixed_partners': (((0, 1), (2, 3)),),
    'rotating_partners': (((0, 1), (2, 3)), ((0, 3), (2, 1))),
}
read, write, sha, npz = stats.read, stats.write, stats.sha, stats.npz
load = lambda path: torch.load(path, weights_only=True, map_location='cpu')


def seed_value(seed, panel, agent):
    return int(np.random.SeedSequence([32032, int(seed), int(panel), int(agent), 1]).generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def reset(agent, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for name in RESET:
            for module in getattr(agent, name).modules():
                if hasattr(module, 'reset_parameters'): module.reset_parameters()
    agent.requires_grad_(False)
    for name, value in agent.named_parameters(): value.requires_grad_(name.split('.')[0] in SEND + RECV)
    return agent


def frozen(state, reset_stage=False):
    excluded = RESET if reset_stage else SEND + RECV
    return {key: value for key, value in state.items() if key.split('.')[0] not in excluded}


def fixture(seed, panel, slot, step, condition):
    groups = stats.groups(panel, condition)
    rng = lambda stream: np.random.default_rng(np.random.SeedSequence([32032, int(seed), int(panel), int(slot), int(step), int(stream)]))
    maps = np.tile(groups['train12'], 10)[rng(0).permutation(120)]
    photos = rng(1).random((120, 2))
    base = maps * 12 + (photos[:, 0] * 6).astype(np.int64) * 2 + (photos[:, 1] * 2).astype(np.int64)
    return dict(indices=np.r_[base, base + 360], uniforms=rng(2).random((240, 8)).astype(np.float32))


def apply_joint_trace(left, right, entropy_weight):
    joint = (left['success'].all(1) & right['success'].all(1)).astype(np.float32)
    reward = (.125 * (left['success'].sum(1) + right['success'].sum(1)) + .5 * joint).astype(np.float32)
    advantage = reward - np.float32(.5)
    for trace in (left, right):
        trace['reward'] = reward.copy(); trace['advantage'] = advantage.copy(); trace['joint_success'] = joint.copy(); trace['baseline'] = .5; trace['entropy_weight'] = float(entropy_weight)
    def loss(trace, role):
        values = torch.from_numpy(trace['sender_logp' if role == 'sender' else 'receiver_logp'])
        entropy = torch.from_numpy(trace['sender_entropy' if role == 'sender' else 'receiver_entropy'])
        return float((-(values * torch.from_numpy(advantage)).mean() - entropy_weight * entropy.mean()))
    left['sender_loss'] = loss(left, 'sender'); left['receiver_loss'] = loss(left, 'receiver')
    right['sender_loss'] = loss(right, 'sender'); right['receiver_loss'] = loss(right, 'receiver')


def sample_from_probabilities(probabilities, uniforms):
    return np.minimum((np.cumsum(probabilities, axis=-1) < uniforms[..., None]).sum(-1), probabilities.shape[-1] - 1)


def check_fixture(seed, panel, step, condition, world_table, raw, log, cache, c):
    saved_world = {key.removeprefix('world__'): value for key, value in raw.items() if key.startswith('world__')}
    saved_trace = {key.removeprefix('trace__'): value for key, value in raw.items() if key.startswith('trace__')}
    expected_world = {}; expected_trace = {}
    matching = MATCHINGS[condition][step % len(MATCHINGS[condition])]
    for slot, (i, j) in enumerate(matching):
        f = fixture(seed, panel, slot, step, condition); prefix = f'slot{slot}__'; idx = f['indices']; ew = .02 if step < 2100 else 0.
        expected_world.update({prefix + key: value for key, value in f.items()})
        left_key = prefix + f'i{i}_j{j}__'; right_key = prefix + f'i{j}_j{i}__'
        left = {key.removeprefix(left_key): value for key, value in saved_trace.items() if key.startswith(left_key)}
        right = {key.removeprefix(right_key): value for key, value in saved_trace.items() if key.startswith(right_key)}
        c.exact(left['h'], cache['train', PRIVATE_TYPES[i]][idx], f'{left_key}frozen h')
        c.exact(right['h'], cache['train', PRIVATE_TYPES[j]][idx], f'{right_key}frozen h')
        c.exact(left['uniforms'], f['uniforms'][:, :4], f'{left_key}uniforms')
        c.exact(right['uniforms'], f['uniforms'][:, 4:], f'{right_key}uniforms')
        for key, trace in ((left_key, left), (right_key, right)):
            c.exact(trace['positions'], world_table['positions'][idx], f'{key}positions')
            sampled0 = sample_from_probabilities(trace['token_probabilities'][:, 0], trace['uniforms'][:, 0])
            sampled1 = sample_from_probabilities(trace['token_probabilities'][:, 1], trace['uniforms'][:, 1])
            sampled = np.stack((sampled0, sampled1), axis=1).astype(trace['messages'].dtype)
            c.exact(trace['messages'], sampled, f'{key}messages')
            action0 = sample_from_probabilities(trace['action_probabilities'][:, 0], trace['uniforms'][:, 2])
            action1 = sample_from_probabilities(trace['action_probabilities'][:, 1], trace['uniforms'][:, 3])
            actions = np.stack((action0, action1), axis=1).astype(trace['actions'].dtype)
            c.exact(trace['actions'], actions, f'{key}actions')
            success = (actions == world_table['positions'][idx]).astype(np.float32)
            c.exact(trace['success'], success, f'{key}success')
        apply_joint_trace(left, right, ew)
        expected_trace.update({left_key + key: value for key, value in left.items()})
        expected_trace.update({right_key + key: value for key, value in right.items()})
        c.coverage['sampled_direction_fixtures'] += 2
        c.coverage['sampled_message_trajectories'] += 2 * len(idx)
    c.exact(expected_world, saved_world, 'full paired fixture')
    c.exact(expected_trace, saved_trace, 'full joint trace')
    c.exact(stats.arrays_sha(expected_world), log['world_sha256'], 'world hash')
    c.exact(stats.arrays_sha(expected_trace), log['trace_sha256'], 'trace hash')


def audit(out, c):
    inv = read(out / 'invocation.json'); done = read(out / 'training_complete.json')
    c.exact(done['status'], 'complete', 'terminal'); c.exact(inv['threads'], 1, 'threads')
    c.exact(inv['source_hashes'], done['source_hashes'], 'source map'); c.exact(inv['input_hashes'], done['input_hashes'], 'input map')
    for path, digest in inv['source_hashes'].items():
        c.exact(sha(path), digest, 'current source'); c.exact(sha(out / 'frozen_sources' / Path(path).relative_to(PROJECT)), digest, 'frozen source')
    source = Path(inv['source']); source_done = read(source / 'training_complete.json'); c.exact(source_done['status'], 'complete', 'inherited source')
    source_qa = read(source / 'audit_execution.json'); c.exact(source_qa['passed'], True, 'inherited audit'); c.exact(source_qa['training_complete_sha256'], sha(source / 'training_complete.json'), 'inherited terminal binding')
    for path, digest in inv['input_hashes'].items(): c.exact(sha(path), digest, 'input unchanged'); c.coverage['consumed_inputs_hashed'] += 1
    if inv['formal']:
        seal = read(source / 'completion_manifest.json'); artifacts = {item['path']: item['sha256'] for item in seal['artifacts']}
        for path, digest in inv['input_hashes'].items(): c.exact(artifacts[path], digest, 'input in source seal')
    for rel, digest in done['files'].items(): c.exact(sha(out / rel), digest, 'completion file'); c.coverage['completion_bound_files'] += 1
    tables = {split: npz(out / f'{split}_worlds.npz') for split in ('train', 'test')}
    for split, world in tables.items():
        c.exact(world, npz(source / f'{split}_worlds.npz'), f'{split} world table'); c.exact(world['positions'], stats.MAPS[world['map_id']], f'{split} coordinates')
        half = len(world['map_id']) // 2; photos = np.unique(world['photo_ids'][:half], axis=0); c.exact(world['shown'], np.repeat(np.arange(2), half), f'{split} masks'); c.exact(world['map_id'], np.tile(np.repeat(np.arange(30), len(photos)), 2), f'{split} layouts')
    c.require(not set(tables['train']['photo_ids'].flat) & set(tables['test']['photo_ids'].flat), 'disjoint photos')
    seeds, panels = inv['seeds'], inv['partitions']; times = [0, 100, 600, 1200, 2100, 2400] if inv['formal'] else [0, 40]
    c.exact(seeds, [34101, 34102, 34103, 34104] if inv['formal'] else [99528], 'seeds'); c.exact(panels, [1, 2, 3] if inv['formal'] else [1], 'panels'); c.exact(inv['updates'], times[-1], 'budget'); c.exact(inv['social_conditions'], list(stats.CONDITIONS), 'conditions'); c.exact(inv['population_agents'], 4, 'population size'); c.exact(inv['private_types'], list(PRIVATE_TYPES), 'private types')
    for seed, panel in itertools.product(seeds, panels):
        prepared = load(source / f'prepared_{seed}.pt'); templates = remake_agents(seed, prepared, 7, 2, 'identity'); agents = []
        for i, d in enumerate(PRIVATE_TYPES):
            agent = copy.deepcopy(templates[d]); private = load(source / 'private' / f's{seed}_p{panel}_d{d}_all' / 'final.pt')['agent']; agent.load_state_dict(private); reset(agent, seed_value(seed, panel, i)); agents.append(agent)
            c.exact(frozen(agent.state_dict(), True), frozen(private, True), 'private frozen tensors'); c.exact([name for name, value in agent.named_parameters() if value.requires_grad], [name for name, value in agent.named_parameters() if name.split('.')[0] in SEND + RECV], 'trainable communication only')
        cachefolder = source / 'cache' / f's{seed}_p{panel}_all'; manifest = read(cachefolder / 'manifest.json'); c.exact(manifest['private_head_used'], False, 'private head excluded')
        cache = {(split, d): np.load(cachefolder / f'{split}_d{d}.npy') for split, d in itertools.product(('train', 'test'), (0, 1))}
        for condition in stats.CONDITIONS:
            folder = out / 'social' / f's{seed}_p{panel}_{condition}'; cfg = read(folder / 'config.json'); result = read(folder / 'result.json'); initial = load(folder / 'initial.pt')
            c.exact(initial, [agent.state_dict() for agent in agents], 'paired independent initial states'); c.exact(cfg['population_agents'], 4, 'config population'); c.exact(cfg['private_types'], list(PRIVATE_TYPES), 'config private types'); c.exact(cfg['groups'], {key: value.tolist() for key, value in stats.groups(panel, condition).items()}, 'config groups'); c.exact((cfg['seed'], cfg['partition'], cfg['condition'], cfg['updates']), (seed, panel, condition, times[-1]), 'run identity'); c.exact(cfg['checkpoints'], times, 'checkpoints'); c.exact((cfg['learning_rate'], cfg['clip']), (.0007, 2.), 'optimizer'); c.exact(cfg['joint_payoff'], stats.read(ROOT / 'support_design.json')['joint_payoff'], 'joint payoff config')
            opts = load(folder / 'initial_optimizer.pt')
            for opt in opts: c.exact(opt['state'], {}, 'fresh Adam'); c.exact(opt['param_groups'][0]['lr'], .0007, 'Adam lr')
            for t in times:
                states = load(folder / f'checkpoint_{t:04d}.pt')
                for d in range(AGENTS): c.exact(frozen(states[d]), frozen(initial[d]), 'frozen checkpoint')
                if t == 0: c.exact(states, initial, 'checkpoint zero')
                c.coverage['pair_checkpoints_verified'] += 1
            final = load(folder / 'final.pt'); c.exact(final, states, 'final checkpoint')
            for i, j in PAIR_KEYS:
                agent_i, agent_j = agents[i], agents[j]; agent_i.load_state_dict(final[i]); agent_j.load_state_dict(final[j]); raw = npz(folder / f'protocol_{times[-1]:04d}_i{i}_j{j}.npz'); lp, tok = enumerate_sender(agent_i, torch.from_numpy(cache['test', PRIVATE_TYPES[i]])); c.exact(lp.detach().numpy(), raw['sender_log_probs'], f'i{i}j{j} sender'); c.exact(tok.detach().numpy(), raw['tokens'], f'i{i}j{j} greedy'); c.exact(enumerate_receiver(agent_j).detach().numpy(), raw['receiver_logits'], f'i{i}j{j} receiver'); c.coverage['endpoint_sender_worlds_replayed'] += len(lp); c.coverage['endpoint_receiver_tables_replayed'] += 1
            logs = [json.loads(line) for line in (folder / 'training.jsonl').read_text().splitlines()]; c.exact([item['update'] for item in logs], list(range(1, times[-1] + 1)), 'log rows'); c.coverage['training_log_rows_counted'] += len(logs)
            for step in sorted({0, times[-1] - 1} | ({2100} if times[-1] > 2100 else set())):
                check_fixture(seed, panel, step, condition, tables['train'], npz(folder / f'train_{step + 1:04d}.npz'), logs[step], cache, c)
            c.exact((result['status'], result['updates'], result['messages'], result['actions'], result['frozen_verified'], result['joint_payoff']), ('complete', times[-1], times[-1] * 960, times[-1] * 1920, True, True), 'run terminal'); c.coverage['social_runs'] += 1
            for agent, state in zip(agents, initial): agent.load_state_dict(state)
    runs = len(seeds) * len(panels) * len(stats.CONDITIONS); c.exact(len(list((out / 'social').glob('*/result.json'))), runs, 'run inventory')
    for key, value in dict(social_runs=runs, pair_updates=runs * times[-1], messages=runs * times[-1] * 960, actions=runs * times[-1] * 1920, new_private_fits=0, new_dino_inferences=0).items(): c.exact(done[key], value, 'complete ' + key)
    c.exact(c.coverage['social_runs'], runs, 'all population runs')
    return inv, dict(source_audit_sha256=sha(source / 'audit_execution.json'), source_training_complete_sha256=sha(source / 'training_complete.json'), source_completion_manifest_sha256=sha(source / 'completion_manifest.json') if inv['formal'] else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); args = ap.parse_args(); out = args.out.resolve(); torch.set_num_threads(1); c = stats.Checks(); start = time.monotonic()
    try:
        inv, provenance = audit(out, c); deps = {str(Path(module.__file__).resolve()): sha(module.__file__) for module in list(sys.modules.values()) if getattr(module, '__file__', None) and str(Path(module.__file__).resolve()).startswith(str(PROJECT)) and Path(module.__file__).is_file()}; result = dict(passed=True, status='passed_bounded_joint_payoff_audit_and_endpoint_replay', formal=inv['formal'], checks=c.count, coverage=dict(c.coverage), source_hashes=inv['source_hashes'], input_hashes=inv['input_hashes'], source_provenance=provenance, audit_source_sha256=sha(__file__), audit_dependencies=deps, training_complete_sha256=sha(out / 'training_complete.json'), seconds=time.monotonic() - start, exclusions=['No complete gradient/Adam replay; endpoint sender/receiver replay and selected joint payoff traces are checked.','Inherited visual/private preparation and DINO encoding are bound by input/source hashes.','The joint payoff is a controlled same-world cooperation pressure, not a natural ecology.']); write(out / 'audit_execution.json', result); shutil.copy2(__file__, out / 'audit_results_source.py'); print(json.dumps({key: result[key] for key in ('passed', 'checks', 'coverage', 'seconds')}))
    except Exception as exc:
        stamp = time.time_ns(); write(out / f'audit_failure_{stamp}.json', dict(passed=False, error=repr(exc), traceback=traceback.format_exc(), audit_source_sha256=sha(__file__))); shutil.copy2(__file__, out / f'audit_results_failure_{stamp}.py'); raise


if __name__ == '__main__': main()
