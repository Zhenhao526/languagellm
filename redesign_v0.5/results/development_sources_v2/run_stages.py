"""Fixed-budget exploratory stages with individual REINFORCE and exact routing."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from camp import (MAPS, HOLDOUT, TRAIN_MAPS, ImageBank, make_agents, individual_practice,
                  write_json, initial_world, collect, scene_visual, photo_ids, remake_agents,
                  projected_banks, draw, evaluate_prepared_four_sites)

ROOT = Path(__file__).resolve().parent
A = {
    'known_sequence': dict(known=True, vocab=5, length=2, schedule='course'),
    'hidden_sequence': dict(known=False, vocab=5, length=2, schedule='course'),
    'known_atomic': dict(known=True, vocab=25, length=1, schedule='course'),
    'hidden_atomic': dict(known=False, vocab=25, length=1, schedule='course'),
    'hidden_single': dict(known=False, vocab=5, length=1, schedule='course'),
    'hidden_blocked': dict(known=False, vocab=5, length=2, blocked=True, schedule='course'),
    'hidden_sequence_direct': dict(known=False, vocab=5, length=2, schedule='direct'),
    'hidden_sequence_mixed': dict(known=False, vocab=5, length=2, schedule='mixed'),
    'hidden_sequence_holdout': dict(known=False, vocab=5, length=2, holdout=True),
}
B = {
    'immediate_continue': dict(known=False, vocab=5, length=2),
    'delay_memory': dict(known=False, vocab=5, length=2, delay=3),
    'delay_reset': dict(known=False, vocab=5, length=2, delay=3, memory_mode='reset'),
    'delay_replay': dict(known=False, vocab=5, length=2, delay=3, memory_mode='replay'),
}
C = {
    'persistent_communication': dict(known=False, vocab=5, length=2, horizon=3),
    'persistent_blocked': dict(known=False, vocab=5, length=2, horizon=3, blocked=True),
    'persistent_channel_removed': dict(known=False, vocab=5, length=2, horizon=3, blocked=True),
    'persistent_known': dict(known=True, vocab=5, length=2, horizon=3),
    'persistent_delayed': dict(known=False, vocab=5, length=2, horizon=3, delay=3),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(agents):
    h = hashlib.sha256()
    for i, agent in enumerate(agents):
        for k, v in sorted(agent.state_dict().items()):
            h.update(f'{i}/{k}'.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def training_schedule(seed, updates, kind):
    """Same stage-count multiset for ordered and mixed conditions, fixed before training."""
    first, second = min(400, updates // 3), min(400, updates // 3)
    levels = np.asarray([2] * first + [3] * second + [4] * (updates - first - second))
    if kind == 'mixed':
        levels = np.random.default_rng(seed + 7110000).permutation(levels)
    elif kind != 'course':
        levels[:] = 4
    sites = np.random.default_rng(seed + 7120000).permutation(4)
    pools = {k: [i for i, p in enumerate(MAPS) if set(p).issubset(set(sites[:k]))] for k in (2, 3, 4)}
    return levels, pools


def shuffle_conditional(sent, goals, inventory, rng):
    """Shuffle within current goal/inventory and direction, keeping known-goal context."""
    out = sent.copy()
    keys = np.column_stack((goals, inventory))
    for key in np.unique(keys, axis=0):
        rows = np.flatnonzero((keys == key).all(1))
        out[rows] = sent[rng.permutation(rows)]
    return out


def rollout(agents, banks, bank, plan, seed, n, *, training=False, greedy=True,
            mode='normal', trace=False, split='test'):
    # Each direction has n/2 independent episodes; roles remain fixed within episode.
    assert n % 2 == 0
    rng = np.random.default_rng(seed)
    h = plan.get('horizon', 1)
    n2 = n // 2
    all_rewards, records, learning, world_hash = [], [], [[], []], hashlib.sha256()
    for scout in range(2):
        collector = 1 - scout
        policy = [np.random.default_rng(seed + 101 + scout * 1000 + i) for i in range(2)]
        intervention = np.random.default_rng(seed + 10001 + scout)
        positions, inventory = initial_world(rng, n2, plan.get('holdout', False) and training,
                                              plan.get('map_pool') if training else None)
        ids = photo_ids(bank, rng, n2, split)
        history = np.zeros((n2, 14), np.float32)
        steps, rews = [], []
        for t in range(h):
            goals = rng.integers(2, size=n2)
            # A public coordinate frame, private menu order; neither sender condition gets menu.
            menu = np.argsort(rng.random((n2, 4)), axis=1)
            refill = rng.random(n2)
            new_ids = photo_ids(bank, rng, n2, split)
            for x in (positions, ids, goals, menu, refill):
                world_hash.update(np.ascontiguousarray(x).tobytes())
            goal = torch.from_numpy(np.eye(2, dtype=np.float32)[goals])
            inv = torch.from_numpy(inventory.astype(np.float32))
            visual = scene_visual(positions, ids, banks[scout])
            memory_mode = plan.get('memory_mode', 'retain')
            if mode == 'erase_memory':
                memory_mode = 'reset'
            latent = agents[scout].observe(visual, plan.get('delay', 0), memory_mode)
            goal_input = goal if plan['known'] else torch.zeros_like(goal)
            sent, slogp, sentropy, svalue = agents[scout].send(latent, goal_input, inv, policy[0], greedy)
            delivered = sent.detach().numpy().copy()
            if plan.get('blocked', False) or mode == 'blank':
                delivered[:] = 0
            elif mode == 'shuffle':
                delivered = shuffle_conditional(delivered, goals, inventory, intervention)
            logits, rvalue = agents[collector].receive(torch.from_numpy(delivered), goal, inv,
                                                       torch.from_numpy(history), torch.from_numpy(menu))
            act, alogp, aentropy = draw(logits, policy[1], greedy)
            place = menu[np.arange(n2), act.detach().numpy()]
            nxt, after, reward, gathered, overflow = collect(positions, inventory, goals, place, refill,
                                                            replenish=h > 1)
            rews.append(reward)
            steps.append((slogp, sentropy, svalue, alogp, aentropy, rvalue))
            if trace:
                records.append(dict(scout=np.full(n2, scout), step=np.full(n2, t),
                    episode=np.arange(n2), positions=positions.copy(), inventory=inventory.copy(),
                    goals=goals, menu=menu, photo_ids=ids.copy(), sent=sent.detach().numpy(),
                    delivered=delivered, action=act.detach().numpy(), place=place, reward=reward,
                    gathered=gathered, overflow=overflow, next_inventory=after.copy(),
                    next_positions=nxt.copy(), refill_uniform=refill, history=history.copy()))
            # Only experienced action and gathered resource enter collector history.
            event = np.column_stack((np.eye(4)[place], gathered, gathered.sum(1) == 0)).astype(np.float32)
            history = np.column_stack((history[:, 7:], event)).astype(np.float32)
            if h > 1:
                for kind in range(2):
                    taken = gathered[:, kind].astype(bool)
                    ids[taken, kind] = new_ids[taken, kind]
            inventory, positions = after, nxt
        rewards = np.stack(rews)
        all_rewards.append(rewards)
        if training:
            returns = np.flip(np.flip(rewards - 1, axis=0).cumsum(axis=0), axis=0).copy()
            for t, values in enumerate(steps):
                sl, se, sv, al, ae, av = values
                target = torch.from_numpy(returns[t])
                learning[scout].append((sl, se, sv, target))
                learning[collector].append((al, ae, av, target))
    rewards = np.concatenate(all_rewards, axis=1)
    stats = dict(mean_reward=float(rewards.mean()), per_step=rewards.mean(1).tolist(),
                 perfect_episode_rate=float((rewards.sum(0) == h).mean()), episodes=n,
                 horizon=h, direction_means=[float(x.mean()) for x in all_rewards],
                 world_sha256=world_hash.hexdigest())
    return stats, learning, records


@torch.no_grad()
def evaluate(agents, banks, bank, plan, seed, n, out=None, prefix='final'):
    result = {}
    for mode, greedy in [('normal', True), ('shuffle', True), ('blank', True),
                         ('stochastic', False), ('erase_memory', True)]:
        stats, _, records = rollout(agents, banks, bank, plan, seed, n, greedy=greedy,
                                    mode=mode, trace=out is not None)
        result[mode] = stats
        if records:
            arrays = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
            np.savez_compressed(out / f'{prefix}_{mode}.npz', **arrays)
            if plan.get('horizon', 1) == 1:
                mapid = np.array([next(i for i, p in enumerate(MAPS) if np.array_equal(x, p))
                                  for x in arrays['positions']])
                testmask = np.isin(mapid, HOLDOUT)
                result[mode]['heldout_map_reward'] = float(arrays['reward'][testmask].mean())
                result[mode]['seen_map_reward'] = float(arrays['reward'][~testmask].mean())
                result[mode]['maps_were_withheld'] = bool(plan.get('holdout', False))
    return result


def train(seed, name, plan, agents, bank, out, updates, batch, eval_n, checkpoints, source=None):
    out.mkdir(parents=True, exist_ok=False)
    init = state_sha(agents)
    torch.save([a.state_dict() for a in agents], out / 'initial.pt')
    write_json(out / 'config.json', dict(seed=seed, condition=name, plan=plan, updates=updates,
        batch=batch, eval_n=eval_n, checkpoints=checkpoints, learning_rate=.0007,
        entropy_coefficient=.02, entropy_off_after=int(updates * .8), gamma=1,
        source_hashes={p.name: sha(p) for p in (ROOT / 'camp.py', ROOT / 'run_stages.py')},
        initial_sha256=init, warmstart_source=source,
        trainable_parameters=[sum(p.numel() for p in a.parameters() if p.requires_grad)
                                                for a in agents]))
    banks = projected_banks(agents, bank)
    frozen = [sha_tensor(b.numpy()) for b in banks]
    optimizers = [torch.optim.Adam([p for p in a.parameters() if p.requires_grad], lr=.0007) for a in agents]
    curve, started = [], time.monotonic()
    levels, pools = training_schedule(seed, updates, plan.get('schedule', 'direct'))
    write_json(out / 'training_schedule.json', dict(levels=levels.tolist(), map_pools=pools))
    with (out / 'training.jsonl').open('w') as f:
        for update in range(updates + 1):
            if update in checkpoints:
                scores = evaluate(agents, banks, bank, plan, seed + 9100000, min(eval_n, 1024))
                row = dict(update=update, scores=scores)
                curve.append(row); write_json(out / 'curve.json', curve)
                torch.save([a.state_dict() for a in agents], out / f'checkpoint_{update:04d}.pt')
                print(json.dumps(dict(stage=out.parent.name, seed=seed, condition=name, update=update,
                    normal=round(scores['normal']['mean_reward'], 4), shuffle=round(scores['shuffle']['mean_reward'], 4))), flush=True)
            if update == updates:
                break
            train_plan = dict(plan, map_pool=pools[int(levels[update])])
            stats, learning, _ = rollout(agents, banks, bank, train_plan,
                seed * 100000 + update + 1, batch, training=True, greedy=False, split='train')
            weight = .02 if update < int(updates * .8) else 0.
            info = []
            for i in range(2):
                losses, entropies, plosses, vlosses = [], [], [], []
                for logp, ent, value, target in learning[i]:
                    policy_loss = -(logp * (target - value).detach()).mean()
                    value_loss = .5 * F.mse_loss(value, target)
                    losses.append(policy_loss + value_loss - weight * ent.mean())
                    entropies.append(float(ent.mean().detach()))
                    plosses.append(float(policy_loss.detach())); vlosses.append(float(value_loss.detach()))
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                optimizers[i].zero_grad(); loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(agents[i].parameters(), 2.))
                optimizers[i].step()
                info.append(dict(loss=float(loss.detach()), gradient_norm=norm, entropy=entropies,
                                 policy_loss=plosses, value_loss=vlosses))
            f.write(json.dumps(dict(update=update + 1, active_sites=int(levels[update]), reward=stats['mean_reward'],
                 world_sha256=stats['world_sha256'], entropy_weight=weight, agents=info)) + '\n')
            if (update + 1) % 100 == 0:
                f.flush()
    final = evaluate(agents, banks, bank, plan, seed + 9200000, eval_n, out)
    assert frozen == [sha_tensor(b.numpy()) for b in projected_banks(agents, bank)]
    torch.save([a.state_dict() for a in agents], out / 'final.pt')
    torch.save([o.state_dict() for o in optimizers], out / 'final_optimizer.pt')
    result = dict(seed=seed, condition=name, plan=plan, updates=updates, batch=batch, scores=final,
                  initial_sha256=init, final_sha256=state_sha(agents), seconds=time.monotonic() - started,
                  frozen_projection_verified=True)
    write_json(out / 'result.json', result)
    return agents, result


def sha_tensor(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def prepare(seed, bank, out):
    path = out / f'prepared_{seed}.pt'
    if path.exists():
        return torch.load(path, weights_only=True)
    base = make_agents(seed)
    result = individual_practice(base, bank, seed)
    four = evaluate_prepared_four_sites(base, bank, seed + 7000000)
    if min(four) < .90 or min(r['heldout_need_sensitive_choice'] for r in result) < .90:
        write_json(out / f'preparation_{seed}_FAILED.json', dict(two_sites=result, four_sites=four))
        raise RuntimeError('individual preparation gate failed; no seed replacement')
    states = [a.state_dict() for a in base]
    torch.save(states, path)
    write_json(out / f'preparation_{seed}.json', dict(two_sites=result, four_sites=four, seed=seed,
        source='new independent resource-consequence practice; no inherited communication weights'))
    return states


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stages', default='A')
    p.add_argument('--seeds', type=int, nargs='+', default=[24001, 24002, 24003])
    p.add_argument('--updates-a', type=int, default=1800)
    p.add_argument('--updates-b', type=int, default=900)
    p.add_argument('--updates-c', type=int, default=1200)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--eval-n', type=int, default=8192)
    p.add_argument('--conditions', nargs='+')
    args = p.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    bank = ImageBank()
    manifest = dict(seeds=args.seeds, stages=args.stages, updates_a=args.updates_a, updates_b=args.updates_b,
        updates_c=args.updates_c, batch=args.batch, eval_n=args.eval_n,
        torch=str(torch.__version__), python=platform.python_version(), device='cpu',
        backbone='frozen official DINOv2 ViT-L/14, previously cached 60 real photographs',
        image_scope='exploration using existing 44/16 split; not a new visual confirmation set',
        hashes={str(p): sha(p) for p in [ROOT/'camp.py', ROOT/'run_stages.py',
                ROOT.parent/'redesign_v0.4/data/features.npz', ROOT.parent/'redesign_v0.4/data/manifest.json']})
    invocation = args.out / f'invocation_{args.stages}_{int(time.time())}.json'
    write_json(invocation, manifest)
    for stage in args.stages:
        if stage not in 'ABC':
            raise ValueError(stage)
        conditions = {'A': A, 'B': B, 'C': C}[stage]
        updates = getattr(args, 'updates_' + stage.lower())
        checkpoints = sorted(set([0, min(100, updates), min(300, updates), updates // 2, updates]))
        for seed in args.seeds:
            prepared = prepare(seed, bank, args.out)
            for name, plan in conditions.items():
                if args.conditions and name not in args.conditions:
                    continue
                out = args.out / stage / f's{seed}_{name}'
                if (out / 'result.json').exists():
                    print(f'SKIP completed {out}', flush=True); continue
                agents = remake_agents(seed, prepared, plan['vocab'], plan['length'])
                source_info = dict(path=str(args.out / f'prepared_{seed}.pt'),
                                   sha256=sha(args.out / f'prepared_{seed}.pt'), kind='individual preparation only')
                if stage in 'BC':
                    source_name = 'hidden_blocked' if name == 'persistent_blocked' else 'hidden_sequence'
                    source = args.out / 'A' / f's{seed}_{source_name}' / 'final.pt'
                    states = torch.load(source, weights_only=True)
                    for a, state in zip(agents, states):
                        a.load_state_dict(state)
                    source_info = dict(path=str(source), sha256=sha(source), kind='stage A social checkpoint')
                train(seed, name, plan, agents, bank, out, updates, args.batch, args.eval_n, checkpoints, source_info)
    print('COMPLETED ' + args.stages, flush=True)


if __name__ == '__main__':
    main()
