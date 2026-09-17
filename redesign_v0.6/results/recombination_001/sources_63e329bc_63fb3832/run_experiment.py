"""Six-site balanced-combination exploration; independent reward-trained interfaces."""
from __future__ import annotations
import argparse
from functools import lru_cache
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from camp import (SITES, HISTORY, MAPS, MATCHINGS, split_maps, ImageBank, make_agents,
                  individual_practice, write_json, initial_world, collect, scene_visual,
                  photo_ids, remake_agents, projected_banks, draw, evaluate_prepared_six_sites)

ROOT = Path(__file__).resolve().parent
SEEDS = [25101, 25102, 25103, 25104]
CONDITIONS = {f'split{s}_{kind}': dict(split=s, schedule=kind if kind != 'atomic_direct' else 'direct',
    vocab=49 if kind == 'atomic_direct' else 7, length=1 if kind == 'atomic_direct' else 2,
    known=False, blocked=False) for s in MATCHINGS for kind in ('course', 'mixed', 'direct', 'atomic_direct')}
CONDITIONS.update(full_direct=dict(split=0, schedule='direct', vocab=7, length=2, known=False, blocked=False),
                  full_blocked=dict(split=0, schedule='direct', vocab=7, length=2, known=False, blocked=True))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(agents):
    h = hashlib.sha256()
    for i, a in enumerate(agents):
        for k, v in sorted(a.state_dict().items()):
            h.update(f'{i}/{k}'.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def training_schedule(seed, updates, kind):
    first = min(600, updates // 4)
    levels = np.array([2] * first + [4] * first + [6] * (updates - 2 * first))
    order = np.arange(updates)
    if kind == 'mixed':
        boundary = updates - min(300, updates // 6)
        order[:boundary] = np.random.default_rng(seed + 37110000).permutation(order[:boundary])
    elif kind != 'course':
        levels[:] = SITES
    return levels[order], order


@lru_cache(None)
def access_options(split, level):
    train_maps, _ = split_maps(split)
    options = []
    for sites in combinations(range(SITES), level):
        pool = [int(i) for i in train_maps if set(MAPS[i]).issubset(sites)]
        if pool:
            options.append((sites, pool))
    return options


def training_batch_plan(seed, batch_identity, level, plan):
    options = access_options(plan['split'], int(level))
    weights = np.array([len(pool) for _, pool in options], dtype=float)
    rng = np.random.default_rng(seed + 37120000 + int(batch_identity))
    selected = int(rng.choice(len(options), p=weights / weights.sum()))
    sites, pool = options[selected]
    return dict(plan, map_pool=list(pool), allowed_sites=list(sites))


def rollout(agents, banks, bank, plan, seed, n, *, training=False, greedy=True,
            mode='normal', trace=False, split='test'):
    assert n % 2 == 0
    rng = np.random.default_rng(seed)
    n2 = n // 2
    rewards, records, learning, world_hash = [], [], [[], []], hashlib.sha256()
    for scout in range(2):
        collector = 1 - scout
        policy = [np.random.default_rng(seed + 101 + scout * 1000 + i) for i in range(2)]
        intervention = np.random.default_rng(seed + 10001 + scout)
        if training:
            positions, inventory = initial_world(rng, n2, plan['map_pool'])
            goals = rng.integers(2, size=n2)
        else:
            assert n2 % (len(MAPS) * 2) == 0, 'balanced evaluation requires n multiple of 120'
            ix = np.tile(np.asarray(list(product(range(len(MAPS)), range(2)))), (n2 // (len(MAPS)*2), 1))
            ix = ix[rng.permutation(n2)]
            positions, goals = MAPS[ix[:, 0]].copy(), ix[:, 1]
            inventory = np.zeros((n2, 2), np.int64)
        ids = photo_ids(bank, rng, n2, split)
        menu = np.argsort(rng.random((n2, SITES)), axis=1)
        refill = rng.random(n2)
        history = np.zeros((n2, HISTORY), np.float32)
        for x in (positions, ids, goals, menu, refill):
            world_hash.update(np.ascontiguousarray(x).tobytes())
        goal = torch.from_numpy(np.eye(2, dtype=np.float32)[goals])
        inv = torch.from_numpy(inventory.astype(np.float32))
        latent = agents[scout].observe(scene_visual(positions, ids, banks[scout]),
                                       memory_mode='reset' if mode == 'erase_memory' else 'retain')
        sent, sl, se, sv = agents[scout].send(latent, torch.zeros_like(goal), inv, policy[0], greedy)
        delivered = sent.detach().numpy().copy()
        if plan['blocked'] or mode == 'blank':
            delivered[:] = 0
        elif mode == 'shuffle':
            for g in range(2):
                rows = np.flatnonzero(goals == g)
                delivered[rows] = sent.detach().numpy()[intervention.permutation(rows)]
        logits, av = agents[collector].receive(torch.from_numpy(delivered), goal, inv,
                                              torch.from_numpy(history), torch.from_numpy(menu))
        if training:
            legal = torch.from_numpy(np.isin(menu, plan['allowed_sites']))
            logits = logits.masked_fill(~legal, -1e9)
        act, al, ae = draw(logits, policy[1], greedy)
        place = menu[np.arange(n2), act.detach().numpy()]
        nxt, after, reward, gathered, overflow = collect(positions, inventory, goals, place, refill, replenish=False)
        rewards.append(reward)
        if training:
            target = torch.from_numpy(reward - 1)
            learning[scout].append((sl, se, sv, target))
            learning[collector].append((al, ae, av, target))
        if trace:
            records.append(dict(scout=np.full(n2, scout), step=np.zeros(n2, dtype=np.int64),
                episode=np.arange(n2), positions=positions, inventory=inventory, goals=goals,
                menu=menu, photo_ids=ids, sent=sent.detach().numpy(), delivered=delivered,
                action=act.detach().numpy(), place=place, reward=reward, gathered=gathered,
                overflow=overflow, next_inventory=after, next_positions=nxt, refill_uniform=refill, history=history))
    stats = dict(mean_reward=float(np.concatenate(rewards).mean()),
        direction_means=[float(x.mean()) for x in rewards], episodes=n, horizon=1,
        world_sha256=world_hash.hexdigest())
    return stats, learning, records


@torch.no_grad()
def evaluate(agents, banks, bank, plan, seed, n, out=None):
    result = {}
    for mode, greedy in [('normal', True), ('shuffle', True), ('blank', True), ('stochastic', False), ('erase_memory', True)]:
        stats, _, rows = rollout(agents, banks, bank, plan, seed, n, greedy=greedy, mode=mode, trace=True)
        arrays = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        train_ids, held_ids = split_maps(plan['split'])
        mapids = np.array([np.flatnonzero((MAPS == p).all(1))[0] for p in arrays['positions']])
        stats['map_groups'] = {}
        for name, pool in [('seen', train_ids), ('unseen', held_ids)]:
            mask = np.isin(mapids, pool)
            count, correct = int(mask.sum()), int(arrays['reward'][mask].sum())
            stats['map_groups'][name] = dict(n=count, correct=correct, mean_reward=correct/count if count else None)
        stats['maps_were_withheld'] = plan['split'] != 0
        if out is not None:
            np.savez_compressed(out / f'final_{mode}.npz', **arrays)
        result[mode] = stats
    return result


def prepare(seed, bank, out):
    path = out / f'prepared_{seed}.pt'
    if path.exists():
        return torch.load(path, weights_only=True)
    base = make_agents(seed)
    result = individual_practice(base, bank, seed)
    six = evaluate_prepared_six_sites(base, bank, seed + 37000000)
    if min(six) < .90 or min(r['heldout_need_sensitive_choice'] for r in result) < .90:
        write_json(out / f'preparation_{seed}_FAILED.json', dict(two_sites=result, six_sites=six))
        raise RuntimeError('Preparation gate failed; retain this seed and diagnose, do not replace')
    states = [a.state_dict() for a in base]
    torch.save(states, path)
    write_json(out / f'preparation_{seed}.json', dict(seed=seed, two_sites=result, six_sites=six,
        source='New individual scalar-consequence practice, 200x64 per agent; no inherited social weights'))
    return states


def train(seed, name, plan, prepared, bank, out, updates, batch, eval_n):
    out.mkdir(parents=True, exist_ok=False)
    agents = remake_agents(seed, prepared, plan['vocab'], plan['length'])
    banks = projected_banks(agents, bank)
    frozen = [{k: v.clone() for k, v in a.state_dict().items() if k.startswith('project.')} for a in agents]
    init = state_sha(agents)
    torch.save([a.state_dict() for a in agents], out / 'initial.pt')
    boundary = updates - min(300, updates // 6)
    checkpoints = sorted(set(x for x in [0,100,300,600,1200,1800,boundary,updates] if x<=updates))
    train_ids, held_ids = split_maps(plan['split'])
    write_json(out / 'config.json', dict(seed=seed, condition=name, plan=plan, updates=updates, batch=batch,
        eval_n=eval_n, checkpoint_eval_n=1200, checkpoints=checkpoints, sites=SITES, history_dim=HISTORY,
        train_map_ids=train_ids.tolist(), heldout_map_ids=held_ids.tolist(), map_table=MAPS.tolist(),
        learning_rate=.0007, entropy_coefficient=.02, entropy_off_after=boundary, gamma=1,
        training_seed_offset=30000000, initial_sha256=init,
        source_hashes={p.name:sha(p) for p in (ROOT/'camp.py', ROOT/'run_experiment.py', ROOT/'固定执行方案.md')},
        prepared_source=dict(path=str(out.parent/f'prepared_{seed}.pt'), sha256=sha(out.parent/f'prepared_{seed}.pt')),
        trainable_parameters=[sum(p.numel() for p in a.parameters() if p.requires_grad) for a in agents]))
    levels, order = training_schedule(seed, updates, plan['schedule'])
    write_json(out / 'training_schedule.json', dict(levels=levels.tolist(), batch_identities=order.tolist(),
        sampling='access subset weighted by its number of legal training maps, then uniform map; independent access RNG'))
    optimizers = [torch.optim.Adam([p for p in a.parameters() if p.requires_grad], lr=.0007) for a in agents]
    curve, started = [], time.monotonic()
    with (out/'training.jsonl').open('w') as log:
        for update in range(updates+1):
            if update in checkpoints:
                scores = evaluate(agents,banks,bank,plan,seed+39100000,1200)
                curve.append(dict(update=update,scores=scores)); write_json(out/'curve.json',curve)
                torch.save([a.state_dict() for a in agents],out/f'checkpoint_{update:04d}.pt')
                print(json.dumps(dict(seed=seed,condition=name,update=update,
                    normal=round(scores['normal']['mean_reward'],4),
                    unseen=scores['normal']['map_groups']['unseen']['mean_reward'])),flush=True)
            if update==updates: break
            bid=int(order[update]); level=int(levels[update])
            train_plan=training_batch_plan(seed,bid,level,plan)
            stats,learning,_=rollout(agents,banks,bank,train_plan,seed*100000+30000000+bid+1,
                                     batch,training=True,greedy=False,split='train')
            weight=.02 if update<boundary else 0.
            info=[]
            for who in range(2):
                losses=[]; components=[]
                for lp,ent,value,target in learning[who]:
                    ploss=-(lp*(target-value).detach()).mean()
                    vloss=.5*F.mse_loss(value,target)
                    losses.append(ploss+vloss-weight*ent.mean())
                    components.append(dict(policy_loss=float(ploss.detach()),value_loss=float(vloss.detach()),entropy=float(ent.mean().detach())))
                loss=torch.stack(losses).mean()
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                optimizers[who].zero_grad(); loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(agents[who].parameters(),2.))
                optimizers[who].step()
                info.append(dict(loss=float(loss.detach()),gradient_norm=norm,components=components))
            log.write(json.dumps(dict(update=update+1,batch_identity=bid,active_sites=level,
                allowed_sites=train_plan['allowed_sites'],map_pool=train_plan['map_pool'],
                world_sha256=stats['world_sha256'],reward=stats['mean_reward'],entropy_weight=weight,agents=info))+'\n')
            if (update+1)%100==0:log.flush()
    final=evaluate(agents,banks,bank,plan,seed+39200000,eval_n,out)
    for a,snapshot in zip(agents,frozen):
        assert all(torch.equal(v,a.state_dict()[k]) for k,v in snapshot.items())
    torch.save([a.state_dict() for a in agents],out/'final.pt')
    torch.save([o.state_dict() for o in optimizers],out/'final_optimizer.pt')
    result=dict(seed=seed,condition=name,plan=plan,updates=updates,batch=batch,scores=final,
        initial_sha256=init,final_sha256=state_sha(agents),seconds=time.monotonic()-started,frozen_projection_verified=True)
    write_json(out/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    p.add_argument('--updates',type=int,default=2400)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--eval-n',type=int,default=9600)
    p.add_argument('--conditions',nargs='+',choices=list(CONDITIONS))
    args=p.parse_args();torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'camp.py',ROOT/'run_experiment.py',ROOT/'固定执行方案.md',
           ROOT.parent/'redesign_v0.4/agents.py',ROOT.parent/'redesign_v0.4/run_pilot.py',
           ROOT.parent/'redesign_v0.4/resource_env.py',ROOT.parent/'redesign_v0.4/data/manifest.json',
           ROOT.parent/'redesign_v0.4/data/features.npz']
    manifest=dict(seeds=args.seeds,conditions=args.conditions or list(CONDITIONS),updates=args.updates,
        batch=args.batch,eval_n=args.eval_n,python=platform.python_version(),torch=str(torch.__version__),
        device='cpu',backbone='Frozen official DINOv2 ViT-L/14, old cached44/16 photographs; exploratory',
        hashes={str(f):sha(f) for f in files})
    write_json(args.out/f'invocation_{time.time_ns()}.json',manifest)
    snapshot=args.out/('sources_'+sha(ROOT/'camp.py')[:8]+'_'+sha(ROOT/'run_experiment.py')[:8]);snapshot.mkdir(exist_ok=True)
    for f in files:
        if f.suffix in ('.py','.md'):shutil.copy2(f,snapshot/f.name)
    bank=ImageBank()
    for seed in args.seeds:
        prepared=prepare(seed,bank,args.out)
        for name,plan in CONDITIONS.items():
            if args.conditions and name not in args.conditions:continue
            dest=args.out/f's{seed}_{name}'
            if (dest/'result.json').exists():
                cfg=json.loads((dest/'config.json').read_text())
                assert cfg['updates']==args.updates and cfg['batch']==args.batch and cfg['eval_n']==args.eval_n
                assert cfg['plan']==plan and cfg['source_hashes']=={f.name:sha(f) for f in files[:3]}
                print(f'SKIP completed {dest}',flush=True);continue
            train(seed,name,plan,prepared,bank,dest,args.updates,args.batch,args.eval_n)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
