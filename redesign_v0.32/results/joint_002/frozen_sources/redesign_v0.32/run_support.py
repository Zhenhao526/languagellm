"""Train the four-agent population under a same-world joint payoff."""
import argparse, copy, hashlib, itertools, json, platform, shutil, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent; PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'redesign_v0.8')); from camp import remake_agents
sys.path.insert(0, str(PROJECT / 'redesign_v0.21')); import social_model as communication
sys.path.insert(0, str(ROOT)); import support, joint, metrics

SEEDS = [34101, 34102, 34103, 34104]; PARTITIONS = [1, 2, 3]
AGENTS = 4; PRIVATE_TYPES = (0, 1, 0, 1)
CHECKPOINTS = (0, 100, 600, 1200, 2100, 2400)
PAIR_KEYS = tuple((i, j) for i in range(AGENTS) for j in range(AGENTS) if i != j and PRIVATE_TYPES[i] != PRIVATE_TYPES[j])


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays_sha(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        a = np.ascontiguousarray(value); h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def state_sha(state): return arrays_sha({key: value.detach().numpy() for key, value in state.items()})
def checkpoint_times(updates): return sorted({0, updates, *[t for t in CHECKPOINTS if t < updates]})


def source_hashes():
    files = [ROOT / name for name in ('run_support.py', 'support.py', 'joint.py', 'metrics.py', 'code_relabelings.npy', 'support_design.json', '固定执行方案.md')]
    files += [PROJECT / name for name in ('redesign_v0.28/world.py', 'redesign_v0.20/temporal_model.py', 'redesign_v0.20/temporal_world.py', 'redesign_v0.21/social_model.py', 'redesign_v0.8/camp.py', 'redesign_v0.4/run_pilot.py', 'redesign_v0.4/agents.py', 'redesign_v0.4/resource_env.py')]
    return {str(path): sha(path) for path in files}


def input_hashes(source, seeds, parts):
    files = [source / name for name in ('training_complete.json', 'train_worlds.npz', 'test_worlds.npz', 'private_applicability.json')]
    files += [PROJECT / 'redesign_v0.28/data' / name for name in ('feature_cache.pt', 'selection.json', 'encoder_receipt.json')]
    for seed in seeds:
        files.append(source / f'prepared_{seed}.pt')
        for part in parts:
            files.append(source / 'cache' / f's{seed}_p{part}_all' / 'manifest.json')
            for d in (0, 1):
                files += [source / 'cache' / f's{seed}_p{part}_all' / f'{split}_d{d}.npy' for split in ('train', 'test')]
                files += [source / 'private' / f's{seed}_p{part}_d{d}_all' / name for name in ('final.pt', 'result.json')]
    return {str(path): sha(path) for path in files}


def restore_population(seed, part, prepared, source):
    templates = remake_agents(seed, prepared, 7, 2, 'identity'); agents = []; resets = []
    for i, private_type in enumerate(PRIVATE_TYPES):
        agent = copy.deepcopy(templates[private_type])
        blob = torch.load(source / 'private' / f's{seed}_p{part}_d{private_type}_all' / 'final.pt', weights_only=True)
        agent.load_state_dict(blob['agent']); resets.append(communication.reset_communication(agent, support.seed_value(seed, part, i, 1))); agents.append(agent)
    return agents, resets


@torch.no_grad()
def social_eval_population(agents, cache, worlds, part, condition, folder, step, endpoint):
    scores = {}
    for i, j in PAIR_KEYS:
        h = cache['test', PRIVATE_TYPES[i]]
        raw = dict(worlds, sender_log_probs=communication.message_log_probs(agents[i], h).numpy(), tokens=communication.greedy_messages(agents[i], h).numpy(), receiver_logits=communication.receiver_logits(agents[j]).numpy())
        key = f'i{i}_j{j}'; np.savez_compressed(folder / f'protocol_{step:04d}_{key}.npz', **raw); scores[key] = metrics.social(raw, part, condition)
        if step == endpoint: metrics.save_null(raw, part, condition, folder / f'recombination_null_{key}.npz')
    return scores


def update_population(agents, opts, cache, tables, seed, part, step, condition):
    losses = [{'sender': [], 'receiver': []} for _ in agents]; worlds = {}; traces = {}
    for slot, (i, j) in enumerate(support.matching(condition, step)):
        fixture = support.fixture(seed, part, slot, step, condition, tables['train']); idx = fixture['indices']; uniforms = fixture['uniforms']
        li, rj, lj, ri, pair_traces = joint.pair_loss(agents[i], agents[j], cache['train', PRIVATE_TYPES[i]][idx], cache['train', PRIVATE_TYPES[j]][idx], uniforms[:, :4], uniforms[:, 4:], tables['train']['positions'][idx], .02 if step < 2100 else 0.)
        losses[i]['sender'].append(li); losses[j]['receiver'].append(rj); losses[j]['sender'].append(lj); losses[i]['receiver'].append(ri)
        worlds.update({f'slot{slot}__{key}': value for key, value in fixture.items()})
        traces.update({f'slot{slot}__i{i}_j{j}__{key}': np.asarray(value) for key, value in pair_traces['i_to_j'].items()})
        traces.update({f'slot{slot}__i{j}_j{i}__{key}': np.asarray(value) for key, value in pair_traces['j_to_i'].items()})
    logs = []
    for i, (agent, opt) in enumerate(zip(agents, opts)):
        loss = (sum(losses[i]['sender']) + sum(losses[i]['receiver'])) / 2
        opt.zero_grad(set_to_none=True); loss.backward()
        norms = {role: float(torch.nn.utils.clip_grad_norm_(params, 2.)) for role, params in communication.trainable_groups(agent).items()}
        assert all(np.isfinite(value) for value in norms.values()) and all(param.grad is None for param in agent.parameters() if not param.requires_grad)
        logs.append(dict(loss=float(loss.detach()), norms=norms))
    for opt in opts: opt.step()
    return worlds, traces, logs


def fit_population(seed, part, condition, prepared, cache, tables, source, out, args):
    folder = out / 'social' / f's{seed}_p{part}_{condition}'; folder.mkdir(parents=True)
    agents, resets = restore_population(seed, part, prepared, source)
    opts = [torch.optim.Adam([param for group in communication.trainable_groups(agent).values() for param in group], lr=.0007) for agent in agents]
    prefixes = communication.SENDER_MODULES + communication.RECEIVER_MODULES
    frozen = lambda agent: state_sha({key: value for key, value in agent.state_dict().items() if not key.startswith(prefixes)})
    frozen_hashes = [frozen(agent) for agent in agents]; save = lambda: [copy.deepcopy(agent.state_dict()) for agent in agents]
    torch.save(save(), folder / 'initial.pt'); torch.save([opt.state_dict() for opt in opts], folder / 'initial_optimizer.pt')
    write(folder / 'config.json', dict(seed=seed, partition=part, condition=condition, updates=args.updates, checkpoints=checkpoint_times(args.updates), reset=resets, frozen_hashes=frozen_hashes, cache=str((source / 'cache' / f's{seed}_p{part}_all').resolve()), source=str(source), population_agents=AGENTS, private_types=list(PRIVATE_TYPES), matching_schedule=support._DESIGN['population'], groups={key: value.tolist() for key, value in support.groups(part, condition).items()}, learning_rate=.0007, clip=2., messages_per_population_update=960, actions_per_population_update=1920, joint_payoff=support._DESIGN['joint_payoff'], source_private={str(source / 'private' / f's{seed}_p{part}_d{d}_all' / 'final.pt'): sha(source / 'private' / f's{seed}_p{part}_d{d}_all' / 'final.pt') for d in (0, 1)}))
    curve = []; started = time.monotonic()
    with (folder / 'training.jsonl').open('w') as log:
        for step in range(args.updates + 1):
            if step in checkpoint_times(args.updates):
                scores = social_eval_population(agents, cache, tables['test'], part, condition, folder, step, args.updates); curve.append(dict(update=step, scores=scores)); write(folder / 'curve.json', curve); torch.save(save(), folder / f'checkpoint_{step:04d}.pt'); torch.save([opt.state_dict() for opt in opts], folder / f'optimizer_{step:04d}.pt'); print(json.dumps(dict(phase='joint_social', run=folder.name, update=step, seconds=round(time.monotonic() - started, 2))), flush=True)
            if step == args.updates: break
            worlds, trace, people = update_population(agents, opts, cache, tables, seed, part, step, condition)
            if step in (0, 2100, args.updates - 1):
                np.savez_compressed(folder / f'train_{step + 1:04d}.npz', **{f'world__{key}': value for key, value in worlds.items()}, **{f'trace__{key}': value for key, value in trace.items()}); torch.save(save(), folder / f'after_{step + 1:04d}.pt'); torch.save([opt.state_dict() for opt in opts], folder / f'after_{step + 1:04d}_optimizer.pt')
            log.write(json.dumps(dict(update=step + 1, world_sha256=arrays_sha(worlds), trace_sha256=arrays_sha(trace), people=people)) + '\n')
            if (step + 1) % 100 == 0: log.flush()
    assert frozen_hashes == [frozen(agent) for agent in agents]
    torch.save(save(), folder / 'final.pt'); write(folder / 'result.json', dict(status='complete', scores=curve[-1]['scores'], updates=args.updates, seconds=time.monotonic() - started, messages=args.updates * 960, actions=args.updates * 1920, frozen_verified=True, joint_payoff=True))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True, type=Path); ap.add_argument('--dev', action='store_true'); ap.add_argument('--updates', type=int, default=2400); args = ap.parse_args(); torch.set_num_threads(1)
    seeds = [99528] if args.dev else SEEDS; parts = [1] if args.dev else PARTITIONS; conditions = list(support.CONDITIONS); source = PROJECT / 'redesign_v0.28/results' / ('smoke_001' if args.dev else 'formation_001')
    hashes = source_hashes(); inputs = input_hashes(source, seeds, parts); preflight = None
    if args.dev: assert args.updates == 40
    else:
        gate = read(ROOT / 'preflight_qa.json'); assert gate['passed'] and gate['source_hashes'] == hashes and gate['input_hashes'] == inputs and args.updates == 2400; preflight = dict(path=str(ROOT / 'preflight_qa.json'), sha256=sha(ROOT / 'preflight_qa.json'))
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    for file in hashes:
        target = out / 'frozen_sources' / Path(file).relative_to(PROJECT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(file, target)
    write(out / 'invocation.json', dict(formal=not args.dev, seeds=seeds, partitions=parts, updates=args.updates, source=str(source), source_hashes=hashes, input_hashes=inputs, preflight=preflight, torch_version=str(torch.__version__), numpy_version=np.__version__, platform=platform.platform(), threads=1, new_dino_inferences=0, social_conditions=conditions, population_agents=AGENTS, private_types=list(PRIVATE_TYPES), messages_per_population_update=960, actions_per_population_update=1920, primary='population mean target12 J at2400 rotating_partners minus fixed_partners under joint-round payoff', study_scope='joint_round_partner_payoff_multi_agent_multi_partner_target', new_private_fits=0, test_worlds=180, train_worlds=720))
    tables = {split: dict(np.load(source / f'{split}_worlds.npz')) for split in ('train', 'test')}
    for split, worlds in tables.items(): np.savez_compressed(out / f'{split}_worlds.npz', **worlds)
    started = time.monotonic()
    for seed, part in itertools.product(seeds, parts):
        prepared = torch.load(source / f'prepared_{seed}.pt', weights_only=True); cache = {(split, d): torch.from_numpy(np.load(source / 'cache' / f's{seed}_p{part}_all' / f'{split}_d{d}.npy')) for split, d in itertools.product(('train', 'test'), (0, 1))}
        assert all(not h.requires_grad and torch.isfinite(h).all() and h.shape == ({'train': 720, 'test': 180}[split], 96) for (split, d), h in cache.items())
        for condition in conditions: fit_population(seed, part, condition, prepared, cache, tables, source, out, args)
    for file, digest in {**hashes, **inputs}.items(): assert sha(file) == digest, file
    runs = len(seeds) * len(parts) * len(conditions); write(out / 'training_complete.json', dict(status='complete', formal=not args.dev, social_runs=runs, pair_updates=runs * args.updates, messages=runs * args.updates * 960, actions=runs * args.updates * 1920, seconds=time.monotonic() - started, source_hashes=hashes, input_hashes=inputs, new_private_fits=0, new_dino_inferences=0, files={str(file.relative_to(out)): sha(file) for file in sorted(out.rglob('*')) if file.is_file()})); print(json.dumps(dict(status='complete', social_runs=runs, seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__': main()
