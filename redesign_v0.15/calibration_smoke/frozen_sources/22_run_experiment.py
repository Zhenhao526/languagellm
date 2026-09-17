"""Matched resource-complementarity returns for two independent collection decisions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
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
SEEDS = [27101, 27102, 27103, 27104]
REPRESENTATIONS = ('identity',)
REWARDS = {'additive': 0., 'mixed': .5, 'joint': 1.}
CONDITIONS = {f'split{split}_{kind}': dict(split=split, representation='identity',
    schedule='direct', vocab=7, length=2, known=False, blocked=False,
    reward_kind=kind, complementarity=weight)
    for split in MATCHINGS for kind, weight in REWARDS.items()}
for prefix, blocked in (('full', False), ('blocked', True)):
    CONDITIONS.update({f'{prefix}_{kind}': dict(split=0, representation='identity',
        schedule='direct', vocab=7, length=2, known=False, blocked=blocked,
        reward_kind=kind, complementarity=weight) for kind, weight in REWARDS.items()})
WORLD_KEYS = ('positions', 'photo_ids', 'goals', 'menu')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(agents, trainable_only=False, receiver_only=False):
    h = hashlib.sha256()
    for i, a in enumerate(agents):
        active = {k for k, p in a.named_parameters() if p.requires_grad}
        for k, v in sorted(a.state_dict().items()):
            if trainable_only and k not in active: continue
            if receiver_only and not k.startswith(('receive_embedding.', 'actor.', 'receive_value.')): continue
            h.update(f'{i}/{k}'.encode()); h.update(v.detach().numpy().tobytes())
    return h.hexdigest()


def training_schedule(seed, updates, kind):
    assert kind == 'direct'
    return np.full(updates, SITES, dtype=np.int64), np.arange(updates)


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
    rng = np.random.default_rng(seed + 47120000 + int(batch_identity))
    selected = int(rng.choice(len(options), p=weights / weights.sum()))
    sites, pool = options[selected]
    return dict(plan, map_pool=list(pool), allowed_sites=list(sites))


def fixture(bank, rng, n, plan, training, split):
    """External scene and two goal queries; no query observes the other's action."""
    if training:
        positions, inventory = initial_world(rng, n, plan['map_pool'])
        first_goal = rng.integers(2, size=n)
    else:
        assert n % 60 == 0, 'balanced evaluation needs 60 cells per direction'
        ix = np.tile(np.asarray(list(product(range(30), range(2)))), (n // 60, 1))
        ix = ix[rng.permutation(n)]
        positions, first_goal = MAPS[ix[:, 0]].copy(), ix[:, 1]
        inventory = np.zeros((n, 2), np.int64)
    return dict(positions=positions, inventory=inventory,
        photo_ids=photo_ids(bank, rng, n, split),
        goals=np.column_stack((first_goal, 1-first_goal)),
        menu=np.argsort(rng.random((n, 2, SITES)), axis=2),
        history=np.zeros((n, HISTORY), np.float32))


def utility(successes, complementarity):
    assert successes.ndim == 2 and successes.shape[1] == 2
    return ((1-complementarity)*successes.mean(1)
            + complementarity*successes.prod(1)).astype(np.float32)


def pair_metrics(reward, successes, goals):
    n = len(reward)
    if not n:
        return dict(n=0, decisions=0, reward_sum=0., mean_reward=None, reward_variance=None, positive_rewards=0,
            single_correct=0, single_accuracy=None, both_correct=0, both_accuracy=None,
            food_correct=0, water_correct=0, outcome_counts={k:0 for k in ('00','01','10','11')})
    by_resource = np.take_along_axis(successes, np.argsort(goals, axis=1), axis=1).astype(np.int64)
    codes = 2*by_resource[:, 0]+by_resource[:, 1]
    return dict(n=n, decisions=2*n, reward_sum=float(reward.astype(np.float64).sum()),
        mean_reward=float(reward.astype(np.float64).mean()),
        reward_variance=float(reward.astype(np.float64).var()), positive_rewards=int((reward>0).sum()),
        single_correct=int(successes.sum()), single_accuracy=float(successes.astype(np.float64).mean()),
        both_correct=int(successes.prod(1).sum()), both_accuracy=float(successes.prod(1).astype(np.float64).mean()),
        food_correct=int(by_resource[:, 0].sum()), water_correct=int(by_resource[:, 1].sum()),
        outcome_counts={key:int((codes==i).sum()) for i,key in enumerate(('00','01','10','11'))})


def rollout(agents, banks, bank, plan, seed, n, *, training=False, greedy=True,
            mode='normal', trace=False, split='test'):
    assert n % 2 == 0
    rng = np.random.default_rng(seed)
    n2 = n // 2
    rewards, successes_all, goals_all, records = [], [], [], []
    learning, world_hash = [[], []], hashlib.sha256()
    direction_metrics = []
    for scout in range(2):
        collector = 1-scout
        # Sender and receiver have independent generators. The receiver consumes
        # distinct uniforms for its two independent conditional actions.
        sender_rng = np.random.default_rng(seed+101+scout*1000)
        receiver_rng = np.random.default_rng(seed+102+scout*1000)
        intervention = np.random.default_rng(seed+10001+scout)
        world = fixture(bank, rng, n2, plan, training, split)
        for key in WORLD_KEYS: world_hash.update(np.ascontiguousarray(world[key]).tobytes())
        inv = torch.from_numpy(world['inventory'].astype(np.float32))
        latent = agents[scout].observe(scene_visual(world['positions'], world['photo_ids'], banks[scout]),
            memory_mode='reset' if mode=='erase_memory' else 'retain')
        sent, sl, se, sv = agents[scout].send(latent, torch.zeros(n2, 2), inv, sender_rng, greedy)
        delivered = sent.detach().numpy().copy()
        if plan['blocked'] or mode=='blank': delivered[:] = 0
        elif mode=='shuffle':
            for first in (0, 1):
                rows = np.flatnonzero(world['goals'][:, 0]==first)
                delivered[rows] = sent.detach().numpy()[intervention.permutation(rows)]
        # The same actual delivered message is reused. No feedback, map change,
        # inventory update, recurrent update or first action is passed between queries.
        acts, places, lps, ents, values = [], [], [], [], []
        for query in range(2):
            goal = torch.from_numpy(np.eye(2, dtype=np.float32)[world['goals'][:, query]])
            menu = torch.from_numpy(world['menu'][:, query])
            logits, value = agents[collector].receive(torch.from_numpy(delivered), goal, inv,
                torch.from_numpy(world['history']), menu)
            action, lp, ent = draw(logits, receiver_rng, greedy)
            acts.append(action.detach().numpy())
            places.append(world['menu'][np.arange(n2), query, action.detach().numpy()])
            lps.append(lp); ents.append(ent); values.append(value)
        action, place = np.column_stack(acts), np.column_stack(places)
        targets = np.take_along_axis(world['positions'], world['goals'], axis=1)
        successes = (place==targets).astype(np.float32)
        reward = utility(successes, plan['complementarity'])
        rewards.append(reward); successes_all.append(successes); goals_all.append(world['goals'])
        direction_metrics.append(pair_metrics(reward, successes, world['goals']))
        if training:
            target = torch.from_numpy(reward-1)
            learning[scout].append((sl, se, sv, target))
            # Joint-action score is a SUM, not an average. The baseline is a
            # function of the delivered message/goals only, independent of actions.
            learning[collector].append((sum(lps), sum(ents), torch.stack(values).mean(0), target))
        if trace:
            records.append(dict(scout=np.full(n2, scout), episode=np.arange(n2),
                **world, sent=sent.detach().numpy(), delivered=delivered,
                action=action, place=place, successes=successes, reward=reward))
    total = pair_metrics(np.concatenate(rewards), np.concatenate(successes_all), np.concatenate(goals_all))
    stats = dict(**total, direction_means=[x['mean_reward'] for x in direction_metrics],
        direction_metrics=direction_metrics, episodes=n, horizon=1, world_sha256=world_hash.hexdigest())
    return stats, learning, records


@torch.no_grad()
def evaluate(agents, banks, bank, plan, seed, n, out=None):
    result = {}
    for mode, greedy in [('normal',True),('shuffle',True),('blank',True),('stochastic',False),('erase_memory',True)]:
        stats, _, rows = rollout(agents,banks,bank,plan,seed,n,greedy=greedy,mode=mode,trace=True)
        arrays = {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        train_ids, held_ids = split_maps(plan['split'])
        mapids = np.array([np.flatnonzero((MAPS==p).all(1))[0] for p in arrays['positions']])
        stats['map_groups'] = {}
        for name, pool in [('seen',train_ids),('unseen',held_ids)]:
            mask = np.isin(mapids,pool)
            stats['map_groups'][name] = pair_metrics(arrays['reward'][mask],arrays['successes'][mask],arrays['goals'][mask])
        stats['maps_were_withheld'] = plan['split'] != 0
        if out is not None: np.savez_compressed(out/f'final_{mode}.npz',**arrays)
        result[mode] = stats
    return result


def prepare(seed, bank, out):
    path = out / f'prepared_{seed}.pt'
    if path.exists():
        return torch.load(path, weights_only=True)
    base = make_agents(seed)
    result = individual_practice(base, bank, seed)
    six = evaluate_prepared_six_sites(base, bank, seed + 47000000)
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
    agents = remake_agents(seed, prepared, plan['vocab'], plan['length'], plan['representation'])
    banks = projected_banks(agents, bank)
    frozen = [{k: v.clone() for k, v in a.state_dict().items() if k.startswith('project.') or k == 'input_transform'} for a in agents]
    init = state_sha(agents)
    torch.save([a.state_dict() for a in agents], out / 'initial.pt')
    boundary = updates - min(300, updates // 6)
    checkpoints = sorted(set(x for x in [0,100,300,600,1200,1800,boundary,updates] if x<=updates))
    train_ids, held_ids = split_maps(plan['split'])
    write_json(out / 'config.json', dict(seed=seed, condition=name, plan=plan, updates=updates, batch=batch,
        eval_n=eval_n, checkpoint_eval_n=1200, checkpoints=checkpoints, sites=SITES, history_dim=HISTORY,
        train_map_ids=train_ids.tolist(), heldout_map_ids=held_ids.tolist(), map_table=MAPS.tolist(),
        learning_rate=.0007, entropy_coefficient=.02, entropy_off_after=boundary, gamma=1,
        training_seed_offset=60000000, choices_per_world=2,
        reward_formula='(1-lambda)*mean(successes)+lambda*product(successes)',
        receiver_policy_score='sum of two action log probabilities', trace_schema='paired_world_v1', initial_sha256=init,
        trainable_initial_sha256=state_sha(agents, trainable_only=True),
        receiver_initial_sha256=state_sha(agents, receiver_only=True),
        input_transforms=[a.input_transform.tolist() for a in agents],
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
                scores = evaluate(agents,banks,bank,plan,seed+69100000,1200)
                curve.append(dict(update=update,scores=scores)); write_json(out/'curve.json',curve)
                torch.save([a.state_dict() for a in agents],out/f'checkpoint_{update:04d}.pt')
                print(json.dumps(dict(seed=seed,condition=name,update=update,
                    normal_single=round(scores['normal']['single_accuracy'],4),
                    normal_both=round(scores['normal']['both_accuracy'],4),
                    unseen_both=scores['normal']['map_groups']['unseen']['both_accuracy'])),flush=True)
            if update==updates: break
            bid=int(order[update]); level=int(levels[update])
            train_plan=training_batch_plan(seed,bid,level,plan)
            stats,learning,_=rollout(agents,banks,bank,train_plan,seed*100000+60000000+bid+1,
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
                world_sha256=stats['world_sha256'],reward=stats['mean_reward'],
                single_accuracy=stats['single_accuracy'],both_accuracy=stats['both_accuracy'],
                reward_variance=stats['reward_variance'],positive_rewards=stats['positive_rewards'],
                outcome_counts=stats['outcome_counts'],entropy_weight=weight,agents=info))+'\n')
            if (update+1)%100==0:log.flush()
    final=evaluate(agents,banks,bank,plan,seed+69200000,eval_n,out)
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
    formal = args.seeds == SEEDS and args.updates == 2400 and args.batch == 512 and args.eval_n == 9600
    if formal:
        control_path = args.out/'individual_controls/summary.json'
        controls = json.loads(control_path.read_text())
        assert controls['complete'] and controls['passed'] and controls['runs']==4
        assert {x['seed'] for x in controls['results']}==set(SEEDS)
        for result in controls['results']:
            assert result['representation']=='identity' and result['complete']
            assert all(x['correct']*10>=x['n']*9 for x in result['scores']['normal']['per_agent'])
        for name in ('camp.py','run_experiment.py','run_controls.py','固定执行方案.md'):
            assert sha(args.out/'control_sources'/name)==sha(ROOT/name), 'Control source changed'
        gate_path = args.out/'social_launch_gate.json'
        if gate_path.exists():
            gate=json.loads(gate_path.read_text())
            assert gate['control_summary_sha256']==sha(control_path)
        else:
            write_json(gate_path,dict(started_utc=datetime.now(timezone.utc).isoformat(),
                controls_complete=4,all_passed=True,social_runs_planned=60,
                control_summary_sha256=sha(control_path),diagnostic_weights_transferred=False))
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
