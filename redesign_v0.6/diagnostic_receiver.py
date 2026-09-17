"""External arbitrary map-code control; never feeds weights into social training."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from camp import MAPS,HISTORY,SITES,ImageBank,remake_agents,draw,collect,write_json
from run_experiment import prepare,state_sha,sha

ROOT=Path(__file__).resolve().parent
SEED=99102
PREFIXES=('receive_embedding.','actor.','receive_value.')


def fixture(seed,n,balanced=False):
    rng=np.random.default_rng(seed)
    if balanced:
        ix=np.tile(np.array([(i,g) for i in range(30) for g in range(2)]),(n//60,1))
        ix=ix[rng.permutation(n)]
        maps,goals=ix.T
    else: maps=rng.integers(30,size=n);goals=rng.integers(2,size=n)
    return maps,goals,np.argsort(rng.random((n,SITES)),axis=1)


def attempt(agent,codes,world,seed,greedy):
    maps,goals,menu=world;n=len(maps)
    logits,v=agent.receive(torch.from_numpy(codes[maps]),torch.from_numpy(np.eye(2,dtype=np.float32)[goals]),
        torch.zeros(n,2),torch.zeros(n,HISTORY),torch.from_numpy(menu))
    act,lp,ent=draw(logits,np.random.default_rng(seed),greedy)
    place=menu[np.arange(n),act.detach().numpy()]
    reward=(place==MAPS[maps,goals]).astype(np.float32)
    return lp,ent,v,reward,place


def main():
    torch.set_num_threads(1)
    out=ROOT/'results/receiver_control_99102';out.mkdir(parents=True,exist_ok=False)
    bank=ImageBank();prepared=prepare(SEED,bank,out)
    mapping=np.random.default_rng(399102).permutation(49)[:30]
    write_json(out/'config.json',dict(seed=SEED,updates=2400,batch=512,gate=.9,
        mapping=mapping.tolist(),interpretation='External arbitrary whole-map codes; no social formation or weight reuse',
        source_hashes={p.name:sha(p) for p in (ROOT/'diagnostic_receiver.py',ROOT/'camp.py',ROOT/'run_experiment.py',ROOT/'固定执行方案.md')}))
    report={}
    for name,vocab,length in [('sequence',7,2),('atomic',49,1)]:
        dest=out/name;dest.mkdir()
        codes=np.column_stack((mapping//7,mapping%7)) if length==2 else mapping[:,None]
        agents=remake_agents(SEED,prepared,vocab,length)
        untouched=[{k:v.clone() for k,v in a.state_dict().items() if not k.startswith(PREFIXES)} for a in agents]
        for a in agents:
            for k,p in a.named_parameters():p.requires_grad_(k.startswith(PREFIXES))
        torch.save([a.state_dict() for a in agents],dest/'initial.pt')
        opts=[torch.optim.Adam([p for p in a.parameters() if p.requires_grad],lr=.0007) for a in agents]
        with (dest/'training.jsonl').open('w') as f:
            for u in range(2400):
                scores=[]
                for who,(a,opt) in enumerate(zip(agents,opts)):
                    ws=SEED*1000000+who*100000+u
                    world=fixture(ws,512)
                    lp,ent,v,r,_=attempt(a,codes,world,ws+20000,False)
                    target=torch.from_numpy(r-1)
                    loss=-(lp*(target-v).detach()).mean()+.5*F.mse_loss(v,target)-(.02 if u<2100 else 0)*ent.mean()
                    assert torch.isfinite(loss)
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),2.);opt.step()
                    scores.append(float(r.mean()))
                f.write(json.dumps(dict(update=u+1,reward=scores))+'\n')
                if (u+1)%600==0:print(name,u+1,scores,flush=True)
        scores=[];traces=[]
        with torch.no_grad():
            for who,a in enumerate(agents):
                world=fixture(9910200+who,9600,True)
                _,_,_,r,place=attempt(a,codes,world,88100+who,True)
                maps,goals,menu=world
                traces.append(dict(agent=np.full(len(r),who),maps=maps,positions=MAPS[maps],goals=goals,menu=menu,
                    delivered=codes[maps],place=place,reward=r))
                scores.append(int(r.sum()) / len(r))
                assert all(torch.equal(v,a.state_dict()[k]) for k,v in untouched[who].items())
        np.savez_compressed(dest/'final_normal.npz',**{k:np.concatenate([r[k] for r in traces]) for k in traces[0]})
        torch.save([a.state_dict() for a in agents],dest/'final.pt')
        report[name]=dict(per_agent=scores,pass_gate=min(scores)>=.9,untouched_parameters_verified=True)
        write_json(out/'result.json',dict(complete=False,conditions=report))
        print('FINAL',name,scores,flush=True)
    result=dict(complete=True,passed=all(r['pass_gate'] for r in report.values()),conditions=report)
    write_json(out/'result.json',result)
    assert result['passed'], 'Positive control failed; diagnose before main training'

if __name__=='__main__':main()
