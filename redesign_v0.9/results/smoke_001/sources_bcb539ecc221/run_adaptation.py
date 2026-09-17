"""Continue existing v0.8 conventions after adding six previously withheld maps."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
V8 = ROOT.parent / 'redesign_v0.8'
sys.path.insert(0, str(V8))
import run_experiment as v8
from camp import MAPS, ImageBank, remake_agents, projected_banks, split_maps, write_json

SEEDS = [27101, 27102, 27103, 27104]
ARMS = ['sender_only', 'receiver_only', 'both']
SOURCE = V8 / 'results/complementarity_001'
SENDER = ('slot_phi.', 'memory.', 'send_')
RECEIVER = ('receive_embedding.', 'actor.', 'receive_value.')


def rng_seed(seed, split, purpose, step=0):
    return int(np.random.SeedSequence([9009, seed, split, purpose, step]).generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def select_parameters(agents, arm):
    assert arm in ARMS
    records = []
    for a in agents:
        row = {'active': [], 'frozen': []}
        for name, p in a.named_parameters():
            sender, receiver = name.startswith(SENDER), name.startswith(RECEIVER)
            assert not (sender and receiver)
            assert sender or receiver or name.startswith('project.'), name
            active = ((arm in ('sender_only', 'both') and sender)
                      or (arm in ('receiver_only', 'both') and receiver))
            p.requires_grad_(active)
            row['active' if active else 'frozen'].append(name)
        row['trainable_count'] = sum(p.numel() for p in a.parameters() if p.requires_grad)
        records.append(row)
    return records


def state_subset(agents, active=False):
    result = []
    for a in agents:
        names = {k for k,p in a.named_parameters() if p.requires_grad == active}
        result.append({k:v.clone() for k,v in a.state_dict().items() if k in names or (not active and k == 'input_transform')})
    return result


def subset_verified(agents, snapshot):
    return all(torch.equal(v, a.state_dict()[k]) for a,s in zip(agents,snapshot) for k,v in s.items())


def grouped_stats(stats, rows, split):
    arrays = {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    old, new = split_maps(split)
    ids = np.array([np.flatnonzero((MAPS == p).all(1))[0] for p in arrays['positions']])
    groups = {}
    directions = []
    for name,pool in [('old',old),('new',new)]:
        mask = np.isin(ids,pool)
        groups[name] = v8.pair_metrics(arrays['reward'][mask], arrays['successes'][mask], arrays['goals'][mask])
    for scout in range(2):
        row = {}
        for name,pool in [('old',old),('new',new)]:
            mask = np.isin(ids,pool) & (arrays['scout']==scout)
            row[name] = v8.pair_metrics(arrays['reward'][mask], arrays['successes'][mask], arrays['goals'][mask])
        directions.append(row)
    stats.update(map_groups=groups, direction_groups=directions,
                 new_maps_in_adaptation_training=True)
    return stats, arrays


@torch.no_grad()
def evaluate(agents, banks, bank, plan, seed, n, out=None, modes=('normal',)):
    scores = {}
    for mode in modes:
        stats,_,rows = v8.rollout(agents,banks,bank,plan,seed,n,greedy=mode!='stochastic',
                                 mode=mode,trace=True,split='test')
        stats,arrays = grouped_stats(stats,rows,plan['split'])
        if out is not None: np.savez_compressed(out/f'final_{mode}.npz',**arrays)
        scores[mode] = stats
    return scores


def source_files():
    return [ROOT/'run_adaptation.py', ROOT/'固定执行方案.md', V8/'camp.py', V8/'run_experiment.py',
            ROOT.parent/'redesign_v0.4/agents.py', ROOT.parent/'redesign_v0.4/run_pilot.py',
            ROOT.parent/'redesign_v0.4/resource_env.py', ROOT.parent/'redesign_v0.4/data/manifest.json',
            ROOT.parent/'redesign_v0.4/data/features.npz']


def run(seed, split, arm, bank, out, updates=600, batch=512, eval_n=9600):
    name = f's{seed}_split{split}_{arm}'
    dest = out/name
    source_dir = SOURCE/f's{seed}_split{split}_mixed'
    checkpoint = source_dir/'final.pt'
    plan = dict(v8.CONDITIONS[f'split{split}_mixed'])
    files = source_files()
    hashes = {str(f):v8.sha(f) for f in files}
    if (dest/'result.json').exists():
        cfg = json.loads((dest/'config.json').read_text())
        assert (cfg['updates'],cfg['batch'],cfg['eval_n']) == (updates,batch,eval_n)
        assert cfg['source_hashes']==hashes and cfg['source_checkpoint']['sha256']==v8.sha(checkpoint)
        print(f'SKIP {name}',flush=True)
        return
    dest.mkdir(parents=True,exist_ok=False)
    prepared = torch.load(SOURCE/f'prepared_{seed}.pt',weights_only=True)
    agents = remake_agents(seed,prepared,7,2,'identity')
    original = torch.load(checkpoint,weights_only=True)
    for a,s in zip(agents,original): a.load_state_dict(s)
    assert v8.state_sha(agents)==json.loads((source_dir/'result.json').read_text())['final_sha256']
    partition = select_parameters(agents,arm)
    frozen = state_subset(agents)
    initial_hash = v8.state_sha(agents)
    torch.save([a.state_dict() for a in agents],dest/'initial.pt')
    banks = projected_banks(agents,bank)
    boundary = updates - min(100, max(1, updates//6))
    checkpoints = sorted(set([0,updates]+[x for x in (25,50,100,200,400,500) if x<updates]))
    evaluation_seed = rng_seed(seed,split,2)
    train_plan = dict(plan,map_pool=list(range(30)),allowed_sites=list(range(6)))
    cfg = dict(seed=seed,split=split,arm=arm,plan=plan,training_plan=train_plan,
        updates=updates,batch=batch,eval_n=eval_n,checkpoints=checkpoints,
        learning_rate=.0007,entropy_coefficient=.02,entropy_off_after=boundary,
        optimizer='fresh Adam in all arms',role_loss_reduction='mean of both roles including frozen constant',
        evaluation_seed=evaluation_seed,seed_formula='SeedSequence([9009,seed,split,purpose,step]) uint64 >> 1',
        training_purpose=1,evaluation_purpose=2,partition=partition,source_hashes=hashes,
        source_checkpoint=dict(path=str(checkpoint),sha256=v8.sha(checkpoint)),
        initial_sha256=initial_hash,old_map_ids=split_maps(split)[0].tolist(),new_map_ids=split_maps(split)[1].tolist(),
        inherited_seeds=True,new_maps_are_adaptation_training=True)
    write_json(dest/'config.json',cfg)
    optimizers = [torch.optim.Adam([p for p in a.parameters() if p.requires_grad],lr=.0007) for a in agents]
    torch.save([o.state_dict() for o in optimizers],dest/'initial_optimizer.pt')
    started = time.monotonic()
    curve = []
    with (dest/'training.jsonl').open('w') as log:
        for update in range(updates+1):
            if update in checkpoints:
                assert subset_verified(agents,frozen), 'Frozen module changed'
                scores = evaluate(agents,banks,bank,plan,evaluation_seed,eval_n)
                curve.append(dict(update=update,scores=scores,frozen_modules_verified=True))
                write_json(dest/'curve.json',curve)
                torch.save([a.state_dict() for a in agents],dest/f'checkpoint_{update:04d}.pt')
                print(json.dumps(dict(run=name,update=update,
                    new_both=scores['normal']['map_groups']['new']['both_accuracy'],
                    old_both=scores['normal']['map_groups']['old']['both_accuracy'])),flush=True)
            if update==updates:break
            train_seed = rng_seed(seed,split,1,update)
            stats,learning,_ = v8.rollout(agents,banks,bank,train_plan,train_seed,batch,
                                        training=True,greedy=False,split='train')
            entropy_weight = .02 if update<boundary else 0.
            info=[]
            for who in range(2):
                losses=[];components=[]
                for lp,ent,value,target in learning[who]:
                    policy = -(lp*(target-value).detach()).mean()
                    value_loss = .5*F.mse_loss(value,target)
                    losses.append(policy+value_loss-entropy_weight*ent.mean())
                    components.append(dict(policy_loss=float(policy.detach()),value_loss=float(value_loss.detach()),
                                           entropy=float(ent.mean().detach()),trainable=bool(lp.requires_grad)))
                loss = torch.stack(losses).mean()
                assert torch.isfinite(loss)
                optimizers[who].zero_grad(set_to_none=True)
                loss.backward()
                assert all(p.grad is None for p in agents[who].parameters() if not p.requires_grad)
                norms = {}
                for role,prefixes in [('sender',SENDER),('receiver',RECEIVER)]:
                    parameters=[p for k,p in agents[who].named_parameters() if p.requires_grad and k.startswith(prefixes)]
                    if parameters:
                        norms[role]=float(torch.nn.utils.clip_grad_norm_(parameters,2.))
                assert all(np.isfinite(n) for n in norms.values())
                optimizers[who].step()
                info.append(dict(loss=float(loss.detach()),gradient_norm_by_role=norms,components=components))
            log.write(json.dumps(dict(update=update+1,rng_seed=train_seed,world_sha256=stats['world_sha256'],
                reward=stats['mean_reward'],both_accuracy=stats['both_accuracy'],single_accuracy=stats['single_accuracy'],
                entropy_weight=entropy_weight,agents=info))+'\n')
            if (update+1)%100==0:log.flush()
    final=evaluate(agents,banks,bank,plan,evaluation_seed,eval_n,dest,
                   modes=('normal','shuffle','blank','stochastic','erase_memory'))
    assert final['normal']==curve[-1]['scores']['normal']
    assert subset_verified(agents,frozen)
    torch.save([a.state_dict() for a in agents],dest/'final.pt')
    torch.save([o.state_dict() for o in optimizers],dest/'final_optimizer.pt')
    result=dict(seed=seed,split=split,arm=arm,updates=updates,batch=batch,scores=final,
        initial_sha256=initial_hash,final_sha256=v8.state_sha(agents),source_checkpoint=cfg['source_checkpoint'],
        seconds=time.monotonic()-started,frozen_modules_verified=True)
    write_json(dest/'result.json',result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'results/adaptation_001')
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    p.add_argument('--splits',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--arms',choices=ARMS,nargs='+',default=ARMS)
    p.add_argument('--updates',type=int,default=600)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--eval-n',type=int,default=9600)
    args=p.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    formal=args.updates==600 and args.batch==512 and args.eval_n==9600
    if formal:
        qa=json.loads((ROOT/'preflight_qa.json').read_text())
        assert qa['passed'] and qa['source_sha256']==v8.sha(ROOT/'run_adaptation.py')
    seeds=[rng_seed(s,k,1,u) for s in args.seeds for k in args.splits for u in range(args.updates)]
    seeds += [rng_seed(s,k,2) for s in args.seeds for k in args.splits]
    assert len(seeds)==len(set(seeds)), 'Derived RNG collision'
    files=source_files()
    invocation=dict(started_utc=datetime.now(timezone.utc).isoformat(),seeds=args.seeds,splits=args.splits,arms=args.arms,
        updates=args.updates,batch=args.batch,eval_n=args.eval_n,formal=formal,
        python=platform.python_version(),torch=str(torch.__version__),device='cpu',derived_rng_count=len(seeds),
        unique_derived_rng_count=len(set(seeds)),source_hashes={str(f):v8.sha(f) for f in files})
    write_json(args.out/f'invocation_{time.time_ns()}.json',invocation)
    snapshot=args.out/('sources_'+v8.sha(ROOT/'run_adaptation.py')[:12])
    snapshot.mkdir(exist_ok=True)
    for f in files:
        if f.suffix in ('.py','.md'):shutil.copy2(f,snapshot/f.name)
    bank=ImageBank()
    for seed in args.seeds:
        for split in args.splits:
            for arm in args.arms:run(seed,split,arm,bank,args.out,args.updates,args.batch,args.eval_n)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
