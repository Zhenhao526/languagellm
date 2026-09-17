"""Reconstruct partner routing, sampled training messages/actions and final behavior.

No import of v11 runner, training_queries, interact or its seed helper.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'redesign_v0.8'))
from camp import ImageBank,remake_agents,projected_banks,scene_visual


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rng_for(seed,p,kind,step,d):
    seq=np.random.SeedSequence([11011,int(seed),int(p),int(kind),int(step),int(d)])
    return np.random.default_rng(int(seq.generate_state(1,np.uint64)[0])//2)


def choose(logits,rng,greedy):
    log=F.log_softmax(logits,dim=-1);prob=torch.exp(log)
    if greedy:index=torch.argmax(prob,dim=-1)
    else:
        u=torch.tensor(rng.random((len(prob),1)).astype(np.float32))
        index=torch.sum(torch.cumsum(prob,dim=-1)<u,dim=-1).clamp_max(prob.shape[-1]-1)
    return index,log[torch.arange(len(index)),index],-(prob*log).sum(-1)


class Audit:
    def __init__(self):self.counts=Counter();self.failures=[]
    def check(self,ok,label,context):
        self.counts[label]+=1
        if not bool(ok):self.failures.append(dict(check=label,context=context))


@torch.no_grad()
def replay(path,senders,receivers,banks,cfg,step,stream,mode,audit):
    with np.load(path) as data:z={key:data[key] for key in data.files}
    terms=[{},{}];training=path.name.startswith('train_');greedy=mode!='stochastic'
    for d in (0,1):
        ix=np.flatnonzero(z['scout']==d);n=len(ix);label=[path.name,d]
        s=senders[d];r=receivers[1-d];inv=torch.tensor(z['inventory'][ix],dtype=torch.float32)
        sr=rng_for(cfg['seed'],cfg['partition'],stream,step,d);rr=rng_for(cfg['seed'],cfg['partition'],stream+1,step,d)
        h=s.observe(scene_visual(z['positions'][ix],z['photo_ids'][ix],banks[d]),memory_mode='reset' if mode=='erase_memory' else 'retain')
        local=torch.cat((h,torch.zeros(n,2),inv/2),dim=1);state=s.send_context(local)
        tokens=[];send_lps=[];send_ents=[]
        for j in range(2):
            token,lp,ent=choose(s.send_out(state),sr,greedy)
            tokens.append(token);send_lps.append(lp);send_ents.append(ent)
            if j==0:state=s.send_recur(s.send_embedding(token),state)
        message=torch.stack(tokens,dim=1).numpy();delivered=message.copy()
        if mode=='blank':delivered[:]=0
        elif mode=='shuffle':
            shuffle=rng_for(cfg['seed'],cfg['partition'],stream+2,step,d)
            for g in (0,1):
                subset=np.flatnonzero(z['goals'][ix,0]==g);delivered[subset]=message[shuffle.permutation(subset)]
        audit.check(np.array_equal(message,z['sent'][ix]),'sender_message_replay',label)
        audit.check(np.array_equal(delivered,z['delivered'][ix]),'delivery_intervention_replay',label)
        actions=[];places=[];rl=[];re=[];rv=[]
        for q in (0,1):
            goal=torch.tensor(np.eye(2,dtype=np.float32)[z['goals'][ix,q]])
            logits,value=r.receive(torch.tensor(delivered),goal,inv,torch.tensor(z['history'][ix]),torch.tensor(z['menu'][ix,q]))
            action,lp,en=choose(logits,rr,greedy)
            actions.append(action.numpy());places.append(z['menu'][ix,q,action.numpy()]);rl.append(lp);re.append(en);rv.append(value)
        action=np.stack(actions,axis=1);place=np.stack(places,axis=1)
        success=(place==np.take_along_axis(z['positions'][ix],z['goals'][ix],axis=1)).astype(np.float32)
        reward=.25*success.sum(1)+.5*success.prod(1)
        audit.check(np.array_equal(action,z['action'][ix]) and np.array_equal(place,z['place'][ix]),'receiver_policy_action_replay',label)
        audit.check(np.array_equal(success,z['successes'][ix]) and np.array_equal(reward,z['reward'][ix]),'two_goal_outcome_replay',label)
        target=torch.tensor(reward-1)
        terms[d]['sender']=(sum(send_lps),sum(send_ents),s.send_value(local).squeeze(-1),target)
        terms[1-d]['receiver']=(sum(rl),sum(re),torch.stack(rv).mean(0),target)
        audit.counts['training_worlds_replayed' if training else 'evaluation_worlds_replayed']+=n
    return terms


def agents_from(seed,prepared,path):
    agents=remake_agents(seed,prepared,7,2,'identity')
    for a,state in zip(agents,torch.load(path,weights_only=True)):a.load_state_dict(state);a.requires_grad_(False)
    return agents


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'results/anchoring_001')
    args=parser.parse_args();batch=args.root.resolve();inv=read(batch/'invocation.json');settings=inv['args'];audit=Audit()
    folders=[batch/f's{s}_p{p}_{arm}' for s in settings['seeds'] for p in settings['partitions'] for arm in inv['arms']]
    assert all((f/'result.json').exists() for f in folders),'All planned runs must finish first'
    torch.set_num_threads(1);bank=ImageBank();sources={};trace_files={}
    for folder in folders:
        cfg=read(folder/'config.json');seed=cfg['seed'];p=cfg['partition']
        prepared=torch.load(cfg['prepared_source']['path'],weights_only=True)
        old=agents_from(seed,prepared,folder/'reference.pt');ob=projected_banks(old,bank)
        lines=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        for step in (0,cfg['release_after'],cfg['entropy_off_after']):
            current=agents_from(seed,prepared,folder/f'checkpoint_{step:04d}.pt');cb=projected_banks(current,bank)
            anchored=cfg['arm']=='anchor' or (cfg['arm']=='release' and step<cfg['release_after'])
            routes={'old_s':(current,old if anchored else current,cb,11),
                    'old_r':(old if anchored else current,current,ob if anchored else cb,21),
                    'new':(current,current,cb,31)}
            terms={}
            for tag,(s,r,b,stream) in routes.items():
                path=folder/f'train_{step+1:04d}_{tag}.npz'
                terms[tag]=replay(path,s,r,b,cfg,step,stream,'stochastic',audit);trace_files[str(path)]=sha(path)
            for who,agent in enumerate(lines[step]['agents']):
                computed=[]
                for part in agent['parts']:
                    lp,en,value,target=terms[part['query']][who][part['role']]
                    policy=-(lp*(target-value)).mean();vl=.5*F.mse_loss(value,target)
                    total=policy+vl-lines[step]['entropy_weight']*en.mean()
                    actual=dict(policy_loss=float(policy),value_loss=float(vl),entropy=float(en.mean()),total=float(total),target_mean=float(target.mean()))
                    audit.check(len(target)==part['n'] and all(abs(actual[k]-part[k])<2e-6 for k in actual),
                        'routed_role_loss_from_replayed_policy',[folder.name,step+1,who,part['query'],part['role']])
                    computed.append(total)
                audit.check(abs(float(torch.stack(computed).mean())-agent['loss'])<2e-6,'four_role_loss_replay',[folder.name,step+1,who])
        current=agents_from(seed,prepared,folder/'final.pt');cb=projected_banks(current,bank)
        for variant in ('current','current_to_old','old_to_current'):
            s=old if variant=='old_to_current' else current;r=old if variant=='current_to_old' else current
            b=ob if variant=='old_to_current' else cb
            modes=('normal','shuffle','blank','stochastic','erase_memory') if variant=='current' else ('normal',)
            for mode in modes:
                path=folder/f'final_{variant}_{mode}.npz';replay(path,s,r,b,cfg,0,91,mode,audit);trace_files[str(path)]=sha(path)
        for file in ('config.json','reference.pt','initial.pt','final.pt'):
            path=folder/file;sources[str(path)]=sha(path)
    result=dict(passed=not audit.failures,counts=dict(audit.counts),failures=audit.failures,runs=len(folders),
        source_hashes=sources,trace_hashes=trace_files,script_sha256=sha(__file__),
        scope='Independent partner routing and forward policy replay, sampled three training steps plus all seven final evaluations; also role losses from the corresponding pre-update checkpoint. No full optimizer replay.')
    (batch/'audit_query_replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','trace_hashes')},ensure_ascii=False,indent=2))
    assert result['passed']


if __name__=='__main__':main()
