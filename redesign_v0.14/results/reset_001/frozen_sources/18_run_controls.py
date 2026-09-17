"""Personal full-information visual-action controls; their weights are never reused socially."""
from pathlib import Path
import argparse
import json
import shutil
import platform
from itertools import product
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from camp import MAPS,SITES,HISTORY,ImageBank,remake_agents,projected_banks,scene_visual,photo_ids,draw,write_json
from run_experiment import ROOT,SEEDS,REPRESENTATIONS,prepare,state_sha,sha


def fixture(bank,seed,n,training):
    rng=np.random.default_rng(seed)
    if training:
        maps=rng.integers(len(MAPS),size=n); goals=rng.integers(2,size=n)
    else:
        assert n%60==0
        ix=np.tile(np.array(list(product(range(30),range(2)))),(n//60,1));ix=ix[rng.permutation(n)]
        maps,goals=ix.T
    return dict(positions=MAPS[maps].copy(),goals=goals,photo_ids=photo_ids(bank,rng,n,'train' if training else 'test'),
                menu=np.argsort(rng.random((n,6)),axis=1))


def attempt(a,head,projected,world,seed,greedy,erase=False):
    n=len(world['goals'])
    latent=a.observe(scene_visual(world['positions'],world['photo_ids'],projected),
                     memory_mode='reset' if erase else 'retain')
    logits=head(latent).reshape(n,2,6)[torch.arange(n),torch.from_numpy(world['goals'])]
    logits=logits.gather(1,torch.from_numpy(world['menu']))
    action,logp,entropy=draw(logits,np.random.default_rng(seed),greedy)
    place=world['menu'][np.arange(n),action.detach().numpy()]
    reward=(place==world['positions'][np.arange(n),world['goals']]).astype(np.float32)
    return logp,entropy,reward,action.detach().numpy(),place


@torch.no_grad()
def evaluate(agents,heads,banks,bank,seed,n,out=None):
    scores={}
    for mode in ('normal','erase_memory'):
        counts=[];records=[]
        for who in range(2):
            world=fixture(bank,seed+who*1000,n,False)
            _,_,reward,action,place=attempt(agents[who],heads[who],banks[who],world,seed+who+771,True,mode=='erase_memory')
            correct=int(reward.sum());counts.append(dict(correct=correct,n=n,accuracy=correct/n))
            records.append(dict(**world,agent=np.full(n,who),action=action,place=place,reward=reward))
        scores[mode]=dict(per_agent=counts,mean_accuracy=sum(x['accuracy'] for x in counts)/2)
        if out is not None:
            np.savez_compressed(out/f'final_{mode}.npz',**{k:np.concatenate([r[k] for r in records]) for k in records[0]})
    return scores


def run(seed,representation,prepared,bank,root,updates=2400,batch=512,eval_n=9600):
    out=root/'individual_controls'/f's{seed}_{representation}';out.mkdir(parents=True,exist_ok=False)
    agents=remake_agents(seed,prepared,representation=representation)
    initial=state_sha(agents);common=state_sha(agents,trainable_only=True)
    heads=[]
    for who in range(2):
        torch.manual_seed(seed*1000+881+who)
        heads.append(nn.Sequential(nn.Linear(96,96),nn.Tanh(),nn.Linear(96,12)))
    snapshots=[{k:v.clone() for k,v in a.state_dict().items() if not k.startswith(('memory.','slot_phi.'))} for a in agents]
    for a in agents:
        for k,p in a.named_parameters():p.requires_grad_(k.startswith(('memory.','slot_phi.')))
    banks=projected_banks(agents,bank)
    torch.save(dict(agents=[a.state_dict() for a in agents],heads=[h.state_dict() for h in heads]),out/'initial.pt')
    write_json(out/'config.json',dict(seed=seed,representation=representation,updates=updates,batch=batch,eval_n_per_agent=eval_n,
        gate=.90,initial_sha256=initial,common_initial_sha256=common,
        input_transforms=[a.input_transform.tolist() for a in agents],
        fitted_agent_prefixes=['memory.','slot_phi.'],probe_head='96->96 Tanh->12, own demand selects six-site branch',
        optimizer='Adam',learning_rate=.0007,entropy_coefficient=.02,entropy_off_after=updates-300,
        training_world='Full30 maps, scalar reward only, independent personal choices; no communication',
        prepared_source=dict(path=str(root/f'prepared_{seed}.pt'),sha256=sha(root/f'prepared_{seed}.pt')),
        source_hashes={p.name:sha(p) for p in (ROOT/'camp.py',ROOT/'run_experiment.py',ROOT/'run_controls.py',ROOT/'固定执行方案.md')}))
    parameters=[list(h.parameters())+[p for p in a.parameters() if p.requires_grad] for a,h in zip(agents,heads)]
    optimizers=[torch.optim.Adam(params,lr=.0007) for params in parameters]
    curve=[];started=time.monotonic()
    with (out/'training.jsonl').open('w') as log:
        for update in range(updates+1):
            if update in (0,600,1200,1800,updates):
                scores=evaluate(agents,heads,banks,bank,seed+49300000,1200)
                curve.append(dict(update=update,scores=scores));write_json(out/'curve.json',curve)
                print(json.dumps(dict(control=representation,seed=seed,update=update,normal=scores['normal']['mean_accuracy'])),flush=True)
            if update==updates:break
            records=[]
            for who,(a,head,opt,params) in enumerate(zip(agents,heads,optimizers,parameters)):
                ws=seed*100000+50000000+who*10000+update+1
                world=fixture(bank,ws,batch,True)
                lp,ent,reward,_,_=attempt(a,head,banks[who],world,ws+557000,False)
                target=torch.from_numpy(reward-.5)
                loss=-(lp*target).mean()-(.02 if update<updates-300 else 0.)*ent.mean()
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,2.));opt.step()
                h=__import__('hashlib').sha256()
                for key in ('positions','goals','photo_ids','menu'):h.update(np.ascontiguousarray(world[key]).tobytes())
                records.append(dict(agent=who,world_sha256=h.hexdigest(),reward=float(reward.mean()),loss=float(loss.detach()),gradient_norm=norm))
            log.write(json.dumps(dict(update=update+1,agents=records))+'\n')
            if (update+1)%100==0:log.flush()
    scores=evaluate(agents,heads,banks,bank,seed+49400000,eval_n,out)
    assert all(torch.equal(v,a.state_dict()[k]) for a,snap in zip(agents,snapshots) for k,v in snap.items())
    torch.save(dict(agents=[a.state_dict() for a in agents],heads=[h.state_dict() for h in heads]),out/'final.pt')
    torch.save([o.state_dict() for o in optimizers],out/'final_optimizer.pt')
    passed=all(r['correct']*10>=r['n']*9 for r in scores['normal']['per_agent'])
    result=dict(complete=True,passed=passed,seed=seed,representation=representation,scores=scores,
        frozen_unused_parameters_verified=True,seconds=time.monotonic()-started,
        scope='Independent diagnostic copies; no diagnostic encoder or head weights enter social learning')
    write_json(out/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);args.out.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/name for name in ('camp.py','run_experiment.py','run_controls.py','固定执行方案.md')]
    snapshot=args.out/'control_sources';snapshot.mkdir(exist_ok=False)
    for source in sources:shutil.copy2(source,snapshot/source.name)
    write_json(args.out/'control_invocation.json',dict(seeds=SEEDS,representations=REPRESENTATIONS,
        updates=2400,batch_per_person=512,eval_n_per_person=9600,python=platform.python_version(),
        torch=str(torch.__version__),source_hashes={source.name:sha(source) for source in sources}))
    bank=ImageBank();results=[]
    for seed in SEEDS:
        prepared=prepare(seed,bank,args.out)
        for representation in REPRESENTATIONS:
            result=run(seed,representation,prepared,bank,args.out)
            results.append(result)
    summary=dict(complete=True,passed=all(r['passed'] for r in results),runs=len(results),results=results)
    write_json(args.out/'individual_controls/summary.json',summary)
    print(json.dumps(dict(complete=True,passed=summary['passed'],runs=len(results))),flush=True)

if __name__=='__main__':main()
