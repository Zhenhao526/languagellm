"""Independent experience-transfer statistics and bounded execution replay.

Does not import v23 world, metrics or run_experience. Fixed key-update sample:
formal33101/p1 (development99523/p1), private person0 in old/all and all three
social arms, updates1/2101 when present. Every endpoint960-world policy is
replayed. Full training is not repeated; all external fixtures are reconstructed.
"""
from __future__ import annotations
import argparse,copy,itertools,json,hashlib,math,shutil,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.22'))
from analyze_probe import Checks,fingerprint,average,probability,groups as prior_groups
sys.path.insert(0,str(PROJECT/'redesign_v0.21'))
from audit_execution import enumerate_sender,enumerate_receiver,independent_direction,arrays_sha,groups as role_groups
sys.path.insert(0,str(PROJECT/'redesign_v0.20'))
from temporal_model import observe_sequence,_sequence_states
sys.path.insert(0,str(PROJECT/'redesign_v0.8'))
from camp import ImageBank,remake_agents
MAPS=np.asarray(list(itertools.permutations(range(6),2)),np.int64)
PRIVATE=('memory','slot_phi');SEND=('send_context','send_embedding','send_recur','send_out');RECV=('receive_embedding','actor')
SCOPES=('old','all');ARMS=('old_old','all_old','all_all');CHUNK=256

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def group_maps(p):
    g=prior_groups(p);g['new12']=np.concatenate((g['added'],g['sealed']));return g
def subset(w,idx):return {k:v[idx] for k,v in w.items()}
def seed_value(seed,p,d,kind):return int(np.random.SeedSequence([23023,seed,p,d,kind]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))
def table(bank,split):
    pools=[np.sort(bank.pools[split,k]) for k in (0,1)]
    if split=='test':pools=[a[:4] for a in pools]
    photos=np.asarray(list(itertools.product(*pools)),np.int64);ids=np.repeat(np.arange(30),len(photos))
    return dict(map_id=np.tile(ids,2),positions=np.tile(MAPS[ids],(2,1)),photo_ids=np.tile(photos,(60,1)),shown=np.repeat(np.arange(2),len(ids)))
def render(projected,w):
    n=len(w['map_id']);rows=torch.arange(n);frames=[]
    for full in (True,False):
        pixels=projected.new_zeros(n,6,64);exists=projected.new_zeros(n,6)
        if full:
            for k in (0,1):
                loc=torch.from_numpy(w['positions'][:,k]);photo=torch.from_numpy(w['photo_ids'][:,k]);pixels[rows,loc]=projected[photo];exists[rows,loc]=1.
        else:
            k=w['shown'];loc=torch.from_numpy(w['positions'][np.arange(n),k]);photo=torch.from_numpy(w['photo_ids'][np.arange(n),k]);pixels[rows,loc]=projected[photo];exists[rows,loc]=1.
        frames.append(torch.cat((pixels.flatten(1),exists),-1))
    bits=projected.new_ones(n,2);bits[:,1]=0.
    return torch.stack(frames,1),bits
def fixture(seed,p,d,step,scope,phase,w):
    phase_id=0 if phase=='private' else 1;n=256 if phase=='private' else 128
    rng=np.random.default_rng(np.random.SeedSequence([23023,seed,p,d,phase_id,step,1]))
    uniform=rng.random((n,3));pool=group_maps(p)['old'] if scope=='old' else np.arange(30)
    maps=pool[(uniform[:,0]*len(pool)).astype(np.int64)]
    count=len(w['map_id'])//60;side=math.isqrt(count)
    if side*side!=count:raise ValueError('photo support must be square')
    base=maps*count+(uniform[:,1]*side).astype(np.int64)*side+(uniform[:,2]*side).astype(np.int64)
    indices=np.concatenate((base,base+len(w['map_id'])//2))
    draws=np.random.default_rng(np.random.SeedSequence([23023,seed,p,d,phase_id,step,2])).random((2*n,1 if phase=='private' else 4)).astype(np.float32)
    return dict(indices=indices,uniforms=draws,goals=np.tile(rng.integers(2,size=n),2))


def metrics(raw,p,private):
    pos=raw['positions'];n=len(pos);extra={}
    if private:
        a=raw['logits'].argmax(-1);pr=probability(raw['logits'])
        q=pr[np.arange(n),0,pos[:,0]]*pr[np.arange(n),1,pos[:,1]]
    else:
        code=raw['tokens']@np.asarray([7,1],np.int64);decoder=raw['receiver_logits'].argmax(-1);a=decoder[code]
        rp=probability(raw['receiver_logits']);sp=probability(raw['sender_log_probs'])
        q=np.asarray([np.dot(sp[i],rp[:,0,f]*rp[:,1,w]) for i,(f,w) in enumerate(pos)])
        frequencies=np.bincount(code,minlength=49).astype(np.float64)/n
        extra['blank_J']=(decoder[0]==pos).all(1).astype(float)
        extra['shuffle_J']=np.asarray([np.dot(frequencies,(decoder==x).all(1)) for x in pos])
    good=a==pos;out={}
    for group,maps in group_maps(p).items():
        rows=np.isin(raw['map_id'],maps);out[group]={}
        for label,selected in [('pooled',rows),('food_only',rows&(raw['shown']==0)),('water_only',rows&(raw['shown']==1))]:
            ix=np.flatnonzero(selected);count=len(ix);c=good[ix]
            m=dict(n=count,J=float(np.count_nonzero(c.all(1))/count),food=float(np.count_nonzero(c[:,0])/count),water=float(np.count_nonzero(c[:,1])/count),Q=float(np.sum(q[ix])/count))
            m.update({key:float(np.sum(v[ix])/count) for key,v in extra.items()});out[group][label]=m
    return out


def make_private(seed,p,d,prepared):
    a=remake_agents(seed,prepared,7,2,'identity')[d]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_value(seed,p,d,0));head=nn.Sequential(nn.Linear(96,96),nn.Tanh(),nn.Linear(96,12))
    for key,param in a.named_parameters():param.requires_grad_(key.split('.')[0] in PRIVATE)
    return a,head
def reset_social(a,seed,p,d):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_value(seed,p,d,1))
        for key in SEND+RECV+('send_value','receive_value'):
            for module in getattr(a,key).modules():
                if hasattr(module,'reset_parameters'):module.reset_parameters()
    a.requires_grad_(False)
    for values in role_groups(a).values():
        for value in values:value.requires_grad_(True)
    return a
def private_params(a,head):return [x for x in a.parameters() if x.requires_grad]+list(head.parameters())
def nonprivate(state):return {k:v for k,v in state.items() if k.split('.')[0] not in PRIVATE}
def nonsocial(state):return {k:v for k,v in state.items() if k.split('.')[0] not in SEND+RECV}


def logs_check(folder,cfg,phase,scope,tables,check):
    rows=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    check.exact(len(rows),cfg['updates'],'log count');identities=[]
    for step,row in enumerate(rows):
        check.exact(row['update'],step+1,'zero-based stream/one-based logged update')
        if phase=='private':data=fixture(cfg['seed'],cfg['partition'],cfg['direction'],step,scope,phase,tables['train'])
        else:
            data={f'd{d}__{k}':v for d in (0,1) for k,v in fixture(cfg['seed'],cfg['partition'],d,step,scope,phase,tables['train']).items()}
        check.exact(arrays_sha(data),row['world_sha256'],'all external world/uniform/goal identity')
        if phase=='private':check.check(np.isfinite(row['gradient_norm']),'finite private norm')
        else:check.check(all(np.isfinite([v['loss'],*v['norms'].values()]).all() for v in row['people']),'finite social norms/losses')
        identities.append(row['world_sha256'])
    check.add(phase+'_updates_stream_checked',len(rows));return rows,identities


def private_update(folder,a,head,projected,cfg,w,step,row,check):
    blob=load(folder/f'checkpoint_{step:04d}.pt');a.load_state_dict(blob['agent']);head.load_state_dict(blob['head'])
    params=private_params(a,head);opt=torch.optim.Adam(params,lr=.0007);opt.load_state_dict(load(folder/f'optimizer_{step:04d}.pt'))
    data=fixture(cfg['seed'],cfg['partition'],cfg['direction'],step,cfg['scope'],'private',w);world=subset(w,data['indices']);frames,bits=render(projected,world)
    h,h0,h1=_sequence_states(a,frames,bits,'full');full=head(h).reshape(-1,2,6);goal=data['goals'];selected=full[torch.arange(len(h)),torch.from_numpy(goal)]
    lp=F.log_softmax(selected,-1);prob=lp.exp();action=(prob.detach().cumsum(-1)<torch.from_numpy(data['uniforms'])).sum(-1).clamp(max=5)
    target=world['positions'][np.arange(len(h)),goal];reward=(action.numpy()==target).astype(np.float32)
    policy=-(lp.gather(1,action[:,None]).squeeze(1)*torch.from_numpy(reward-.5)).mean();ent=-(prob*lp).sum(-1).mean();weight=.02 if step<2100 else 0.
    loss=policy-weight*ent;ar=lambda x:x.detach().numpy()
    tr=dict(h0=ar(h0),raw_h1=ar(h1),hfinal=ar(h),full_logits=ar(full),goalselected_logits=ar(selected),selected_probabilities=ar(prob),
        action_uniform=data['uniforms'],action=action.numpy(),reward=reward,selected_target=target)
    check.exact({**{'world__'+k:v for k,v in data.items()},**{'trace__'+k:v for k,v in tr.items()}},npz(folder/f'train_{step+1:04d}.npz'),'private critical full trace')
    check.exact(arrays_sha(tr),row['trace_sha256'],'private critical trace hash')
    comp=dict(loss=float(loss.detach()),policy_loss=float(policy.detach()),entropy=float(ent.detach()),entropy_coefficient=weight,mean_reward=float(reward.mean()),arm='full',rows=len(h))
    check.exact(comp,row['components'],'private independent loss components')
    opt.zero_grad(set_to_none=True);loss.backward();check.check(all(v.grad is not None and torch.isfinite(v.grad).all() for v in params),'private finite gradients')
    norm=float(torch.nn.utils.clip_grad_norm_(params,2.));check.exact(norm,row['gradient_norm'],'private gradient clip norm');opt.step()
    check.exact(dict(agent=a.state_dict(),head=head.state_dict()),load(folder/f'after_{step+1:04d}.pt'),'private after parameters exact')
    check.exact(opt.state_dict(),load(folder/f'after_{step+1:04d}_optimizer.pt'),'private after Adam exact');check.add('private_key_updates_replayed')


def social_update(folder,agents,cache,tables,cfg,step,row,check):
    before=load(folder/f'checkpoint_{step:04d}.pt');os=load(folder/f'optimizer_{step:04d}.pt');opts=[]
    for a,state,s in zip(agents,before,os):
        a.load_state_dict(state);opt=torch.optim.Adam([p for group in role_groups(a).values() for p in group],lr=.0007);opt.load_state_dict(s);opts.append(opt)
    losses=[{},{}];fs={};ts={};scope=cfg['arm'].split('_')[1]
    for d in (0,1):
        f=fixture(cfg['seed'],cfg['partition'],d,step,scope,'social',tables['train']);idx=f['indices']
        sl,rl,tr=independent_direction(agents[d],agents[1-d],cache['train',d][idx],f['uniforms'],tables['train']['positions'][idx],.02 if step<2100 else 0.)
        losses[d]['sender']=sl;losses[1-d]['receiver']=rl;fs.update({f'd{d}__{k}':v for k,v in f.items()});ts.update({f'd{d}__{k}':v for k,v in tr.items()})
    check.exact({**{'world__'+k:v for k,v in fs.items()},**{'trace__'+k:v for k,v in ts.items()}},npz(folder/f'train_{step+1:04d}.npz'),'social critical full trace')
    check.exact(arrays_sha(ts),row['trace_sha256'],'social critical trace hash')
    for d,(a,opt) in enumerate(zip(agents,opts)):
        opt.zero_grad(set_to_none=True);loss=(losses[d]['sender']+losses[d]['receiver'])/2;loss.backward()
        norms={role:float(torch.nn.utils.clip_grad_norm_(params,2.)) for role,params in role_groups(a).items()}
        check.exact(dict(loss=float(loss.detach()),norms=norms),row['people'][d],'social role loss/clip')
        check.check(all(v.grad is None for v in a.parameters() if not v.requires_grad),'social frozen gradient absence')
    for opt in opts:opt.step()
    check.exact([a.state_dict() for a in agents],load(folder/f'after_{step+1:04d}.pt'),'social after parameters exact')
    check.exact([o.state_dict() for o in opts],load(folder/f'after_{step+1:04d}_optimizer.pt'),'social after Adam exact');check.add('social_key_pair_updates_replayed')


def replay_h(a,projected,w):
    values=[]
    with torch.no_grad():
        for lo in range(0,len(w['map_id']),CHUNK):
            f,b=render(projected,subset(w,slice(lo,lo+CHUNK)));values.append(observe_sequence(a,f,b,'full').numpy())
    return np.concatenate(values)


def private_case(seed,p,d,scope,prepared,bank,tables,out,sample,check):
    folder=out/'private'/f's{seed}_p{p}_d{d}_{scope}';cfg=read(folder/'config.json');result=read(folder/'result.json');curve=read(folder/'curve.json')
    check.exact((cfg['seed'],cfg['partition'],cfg['direction'],cfg['scope']),(seed,p,d,scope),'private identity')
    check.exact(cfg['prepared_sha256'],sha(out/f'prepared_{seed}.pt'),'prepared input hash');check.exact(result['status'],'complete','private complete')
    a,head=make_private(seed,p,d,prepared);initial=load(folder/'initial.pt');check.exact(dict(agent=a.state_dict(),head=head.state_dict()),initial,'independent private initialization')
    check.exact(cfg['initial_head_seed'],seed_value(seed,p,d,0),'private init namespace');check.exact(sum(v.numel() for v in private_params(a,head)),155598,'private parameter count')
    check.exact(fingerprint(nonprivate(initial['agent'])),cfg['frozen_sha256'],'private frozen initial');check.exact(load(folder/'initial_optimizer.pt')['state'],{},'private fresh Adam')
    logs,streams=logs_check(folder,cfg,'private',scope,tables,check);points=[]
    for row in curve:
        step=row['update'];blob=load(folder/f'checkpoint_{step:04d}.pt');check.exact(nonprivate(blob['agent']),nonprivate(initial['agent']),'all private frozen checkpoints')
        raw=npz(folder/f'evaluation_{step:04d}.npz');check.exact({k:raw[k] for k in tables['test']},tables['test'],'all private evaluation worlds')
        check.check(raw['logits'].shape==(960,2,6) and np.isfinite(raw['logits']).all() and raw['h'].shape==(960,96),'private raw layout')
        score=metrics(raw,p,True);check.close(score,row['scores'],'all private raw metrics');points.append(dict(update=step,scores=score));check.add('private_evaluation_tables')
    check.exact([x['update'] for x in points],cfg['checkpoints'],'private checkpoint inventory');check.close(points[-1]['scores'],result['scores'],'private endpoint metrics')
    final=load(folder/'final.pt');check.exact(final,load(folder/f'checkpoint_{cfg["updates"]:04d}.pt'),'private final checkpoint')
    a.load_state_dict(final['agent']);head.load_state_dict(final['head'])
    with torch.no_grad():projected=a.project(bank.features).detach()
    h=replay_h(a,projected,tables['test']);raw=npz(folder/f'evaluation_{cfg["updates"]:04d}.npz');check.exact(h,raw['h'],'private complete endpoint h')
    with torch.no_grad():logits=np.concatenate([head(torch.from_numpy(h[lo:lo+CHUNK])).reshape(-1,2,6).numpy() for lo in range(0,len(h),CHUNK)])
    check.exact(logits,raw['logits'],'private complete endpoint logits');check.add('private_endpoint_worlds_replayed',len(h))
    if sample and d==0:
        for step in (0,2100):
            if step<cfg['updates']:private_update(folder,a,head,projected,cfg,tables['train'],step,logs[step],check)
    check.add('private_fits')
    return dict(seed=seed,partition=p,direction=d,condition='private_'+scope,raw_folder=str(folder),curve=points,scores=points[-1]['scores']),initial,final,streams


def social_cache(seed,p,scope,prepared,private_states,bank,tables,out,check):
    agents=remake_agents(seed,prepared,7,2,'identity');cache={};folder=out/'cache'/f's{seed}_p{p}_{scope}';manifest=read(folder/'manifest.json')
    check.exact(manifest['private_head_used'],False,'private head excluded');check.exact(manifest['chunk'],CHUNK,'cache chunk')
    for name,h in manifest['files'].items():check.exact(sha(folder/name),h,'cache bytes')
    for d,a in enumerate(agents):
        a.load_state_dict(private_states[scope,d]['agent']);reset_social(a,seed,p,d)
        with torch.no_grad():projected=a.project(bank.features).detach()
        for split,w in tables.items():
            h=np.load(folder/f'{split}_d{d}.npy');cache[split,d]=torch.from_numpy(h)
            check.check(h.shape==(len(w['map_id']),96) and h.dtype==np.float32 and np.isfinite(h).all(),'all cache layout')
            starts=list(range(0,len(h),CHUNK)) if split=='test' else sorted({0,((len(h)-1)//CHUNK)*CHUNK})
            for lo in starts:
                with torch.no_grad():
                    f,b=render(projected,subset(w,slice(lo,lo+CHUNK)));got=observe_sequence(a,f,b,'full').numpy()
                check.exact(got,h[lo:lo+CHUNK],'cache original chunk h');check.add('cache_'+split+'_worlds_replayed',len(got))
    return cache,agents


def social_case(seed,p,arm,prepared,private_states,cache,bank,tables,out,sample,check):
    folder=out/'social'/f's{seed}_p{p}_{arm}';cfg=read(folder/'config.json');result=read(folder/'result.json');curve=read(folder/'curve.json')
    private_scope,scope=arm.split('_');agents=remake_agents(seed,prepared,7,2,'identity')
    for d,a in enumerate(agents):a.load_state_dict(private_states[private_scope,d]['agent']);reset_social(a,seed,p,d)
    initial=load(folder/'initial.pt');check.exact([a.state_dict() for a in agents],initial,'independent social reset and source frontend')
    check.exact([fingerprint(nonsocial(x)) for x in initial],cfg['frozen_hashes'],'social frozen fingerprints')
    for path,h in cfg['source_private'].items():check.exact(sha(path),h,'social source private checkpoint')
    for opt in load(folder/'initial_optimizer.pt'):check.exact(opt['state'],{},'social fresh Adam')
    logs,streams=logs_check(folder,cfg,'social',scope,tables,check);points=[]
    for row in curve:
        step=row['update'];states=load(folder/f'checkpoint_{step:04d}.pt');scores=[]
        for d,state in enumerate(states):check.exact(nonsocial(state),nonsocial(initial[d]),'all social frozen checkpoint tensors')
        for d in (0,1):
            raw=npz(folder/f'protocol_{step:04d}_d{d}.npz');check.exact({k:raw[k] for k in tables['test']},tables['test'],'all social evaluation worlds')
            lp=raw['sender_log_probs'];check.check(lp.shape==(960,49) and np.isfinite(lp).all() and np.max(np.abs(np.exp(lp.astype(np.float64)).sum(1)-1))<1e-6,'all raw probabilities')
            scores.append(metrics(raw,p,False));check.add('social_evaluation_tables')
        check.close(scores,row['scores'],'all social raw metrics');points.append(dict(update=step,scores=average(scores)))
    check.exact([x['update'] for x in points],cfg['checkpoints'],'social checkpoint inventory');check.close(average(result['scores']),points[-1]['scores'],'social endpoint metrics')
    final=load(folder/'final.pt');check.exact(final,load(folder/f'checkpoint_{cfg["updates"]:04d}.pt'),'social final checkpoint')
    for a,state in zip(agents,final):a.load_state_dict(state)
    for d in (0,1):
        raw=npz(folder/f'protocol_{cfg["updates"]:04d}_d{d}.npz')
        with torch.no_grad():check.exact(enumerate_receiver(agents[1-d]).numpy(),raw['receiver_logits'],'full49 endpoint receiver table')
        for lo in range(0,960,CHUNK):
            with torch.no_grad():lp,tok=enumerate_sender(agents[d],cache['test',d][lo:lo+CHUNK])
            check.exact(lp.numpy(),raw['sender_log_probs'][lo:lo+CHUNK],'complete endpoint sender probability');check.exact(tok.numpy(),raw['tokens'][lo:lo+CHUNK],'complete sequential greedy sender')
            check.add('social_endpoint_sender_worlds_replayed',len(lp))
    if sample:
        for step in (0,2100):
            if step<cfg['updates']:social_update(folder,agents,cache,tables,cfg,step,logs[step],check)
    check.add('social_pairs')
    return dict(seed=seed,partition=p,condition=arm,raw_folder=str(folder),curve=points,scores=points[-1]['scores']),initial,streams


def integrate(points):
    time=[x['update'] for x in points];total=time[-1]-time[0]
    def recurse(values):
        if isinstance(values[0],dict):return {k:recurse([v[k] for v in values]) for k in values[0]}
        return math.fsum((time[i+1]-time[i])*(values[i+1]+values[i])/2 for i in range(len(time)-1))/total
    return recurse([r['scores'] for r in points])
def summarize(rows,seeds,times):
    conditions=('private_old','private_all')+ARMS;source=[]
    for seed,condition in itertools.product(seeds,conditions):
        rr=[r for r in rows if r['seed']==seed and r['condition']==condition]
        curve=[dict(update=t,scores=average([r['curve'][i]['scores'] for r in rr])) for i,t in enumerate(times)]
        source.append(dict(seed=seed,condition=condition,curve=curve,scores=curve[-1]['scores'],auc=integrate(curve)))
    aggregate={}
    for condition in conditions:
        rr=[r for r in source if r['condition']==condition]
        curve=[dict(update=t,scores=average([r['curve'][i]['scores'] for r in rr])) for i,t in enumerate(times)]
        aggregate[condition]=dict(curve=curve,scores=curve[-1]['scores'],auc=integrate(curve))
    primary=[]
    for seed in seeds:
        rr={r['condition']:r for r in source if r['seed']==seed}
        value=lambda c,key:rr[c][key]['new12']['pooled']['J']
        primary.append(dict(seed=seed,A=value('old_old','scores'),B=value('all_old','scores'),C=value('all_all','scores'),
            difference=value('all_old','scores')-value('old_old','scores'),auc_difference=value('all_old','auc')-value('old_old','auc')))
    capability=[]
    for seed in seeds:
        s=next(r for r in source if r['seed']==seed and r['condition']=='private_all')['scores']['new12']
        mask_J={m:s[m]['J'] for m in ('food_only','water_only')};capability.append(dict(seed=seed,mask_J=mask_J,passed=all(v>=.8 for v in mask_J.values())))
    return dict(rows=rows,seed_rows=source,aggregate=aggregate,primary=primary,primary_mean=math.fsum(r['difference'] for r in primary)/len(primary),
        capability=dict(passed=all(r['passed'] for r in capability),source_rows=capability,threshold=.8,filter_applied=False,social_matrix_proceeds_regardless=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();out=args.out.resolve()
    check=Checks();started=time.monotonic();torch.set_num_threads(1)
    if (out/'audit_execution.json').exists():
        history=out/'analysis_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f');history.mkdir(parents=True)
        for name in ('audit_execution.json','analysis.json','comparison.json','analyze_experience_source.py'):
            if (out/name).exists():shutil.copy2(out/name,history/name)
    shutil.copy2(__file__,out/'analyze_experience_source.py')
    deps={str(PROJECT/p):sha(PROJECT/p) for p in ('redesign_v0.22/analyze_probe.py','redesign_v0.21/audit_execution.py','redesign_v0.20/temporal_model.py')}
    audit=dict(passed=False,analysis_source_sha256=sha(__file__),replay_dependency_sha256=deps,production_metrics_called=False,
        coverage=['All saved evaluation tables independently recomputed; full endpoint960-world private/social policies replayed',
            'All external training fixtures/hash identities; all checkpoint frozen tensors; independent paired initialization',
            'Test h cache all rows; train h cache original first/last chunks only',
            'Key updates only source33101/p1 (dev99523/p1), private person0 old/all and social all3arms, updates1/2101 when present',
            'Initial resource preparation not retrained; its source recipe/hash and actual inherited project tensors checked',
            'No full training-trajectory replay; frozen temporal atomic operations and earlier independent social loss helpers reused'])
    try:
        complete=read(out/'training_complete.json');inv=read(out/'invocation.json');check.exact(complete['status'],'complete','whole-batch gate')
        for kind in ('source_hashes','input_hashes'):
            check.exact(complete[kind],inv[kind],'completion source binding')
            for p,h in inv[kind].items():check.exact(sha(p),h,'source/input bytes')
        for p,h in inv['source_hashes'].items():check.exact(sha(out/'frozen_sources'/Path(p).relative_to(PROJECT)),h,'frozen source copy')
        for p,h in complete['files'].items():check.exact(sha(out/p),h,'completed output bytes')
        check.exact(inv['threads'],1,'thread contract');check.exact(inv['torch_version'],str(torch.__version__),'Torch version');check.exact(inv['numpy_version'],np.__version__,'NumPy version')
        if inv['formal']:check.exact((inv['seeds'],inv['partitions'],inv['updates']),([33101,33102,33103,33104],[1,2,3],2400),'formal scope')
        expected_times=sorted({0,inv['updates'],*[t for t in [0,100,600,1200,2100,2400] if t<inv['updates']]})
        bank=ImageBank();tables={s:table(bank,s) for s in ('train','test')}
        for split,w in tables.items():check.exact(w,npz(out/f'{split}_worlds.npz'),'complete independent '+split+' support')
        rows=[];project_hashes=[]
        for seed in inv['seeds']:
            prepared=load(out/f'prepared_{seed}.pt');pr=read(out/f'preparation_{seed}.json');check.exact((pr['updates_per_person'],pr['batch'],pr['source_filter']),(200,64,False),'resource preparation recipe')
            temp=remake_agents(seed,prepared,7,2,'identity')
            for a in temp:project_hashes.append(fingerprint(a.project.state_dict()))
            for p in inv['partitions']:
                sample=(seed==(33101 if inv['formal'] else 99523) and p==1);private_states={};private_initial={};social_initial={};social_streams={}
                for d,scope in itertools.product((0,1),SCOPES):
                    row,initial,final,_=private_case(seed,p,d,scope,prepared,bank,tables,out,sample,check);rows.append(row);private_states[scope,d]=final;private_initial[scope,d]=initial
                    check.exact([r['update'] for r in row['curve']],expected_times,'fixed private times')
                for d in (0,1):check.exact(private_initial['old',d],private_initial['all',d],'paired old/all private initialization')
                for scope in SCOPES:
                    cache,_=social_cache(seed,p,scope,prepared,private_states,bank,tables,out,check)
                    for arm in (('old_old',) if scope=='old' else ('all_old','all_all')):
                        row,initial,stream=social_case(seed,p,arm,prepared,private_states,cache,bank,tables,out,sample,check)
                        rows.append(row);social_initial[arm]=initial;social_streams[arm]=stream;check.exact([r['update'] for r in row['curve']],expected_times,'fixed social times')
                check.exact(social_streams['old_old'],social_streams['all_old'],'A/B exact social worlds')
                for d in (0,1):
                    comm=lambda s:{k:v for k,v in s.items() if k.split('.')[0] in SEND+RECV+('send_value','receive_value')}
                    for arm in ('all_old','all_all'):check.exact(comm(social_initial['old_old'][d]),comm(social_initial[arm][d]),'three-arm communication initialization')
                check.exact(social_initial['all_old'],social_initial['all_all'],'B/C exact full social initialization')
                print(json.dumps(dict(audited_seed=seed,partition=p,checks=check.count)),flush=True)
        check.check(len(project_hashes)==len(set(project_hashes)),'distinct actual prepared project fingerprints')
        check.exact(check.scope_counts['private_fits'],complete['private_fits'],'private fit count');check.exact(check.scope_counts['social_pairs'],complete['social_runs'],'social fit count')
        check.exact(check.scope_counts['private_updates_stream_checked'],complete['private_updates'],'all private stream count');check.exact(check.scope_counts['social_updates_stream_checked'],complete['pair_updates'],'all social stream count')
        analysis=dict(status='complete',formal=inv['formal'],seeds=inv['seeds'],partitions=inv['partitions'],times=expected_times,updates=inv['updates'],
            analysis_source_sha256=sha(__file__),replay_dependency_sha256=deps,training_complete_sha256=sha(out/'training_complete.json'),**summarize(rows,inv['seeds'],expected_times),
            source_project_sha256=project_hashes,boundaries=['PrimaryB−A: private experience support differs at fixed budget; not a pure knowledge or memory intervention',
                'new12 has private exposure in B and both private/social exposure in C; not whole-agent zero-shot',
                'Capability check is source-average and never filters individuals; failure is retained',
                'All source results reported, no significance/equivalence claim;49 codes may encode all30 maps holistically'])
        check.close(analysis['capability'],read(out/'private_applicability.json'),'independent fixed eight-condition applicability')
        write(out/'analysis.json',analysis)
        write(out/'comparison.json',dict(passed=True,comparisons=check.comparisons,max_absolute_error=check.max_error,analysis_sha256=sha(out/'analysis.json'),
            production_metrics_called=False,scope='Every saved private/social curve and endpoint score from raw; applicability independently recomputed. Source aggregation independently implemented; production has no aggregate file.'))
        audit.update(passed=True,formal=inv['formal'],checks=check.count,counts=check.scope_counts,failures=[],source_project_sha256=project_hashes,
            analysis_sha256=sha(out/'analysis.json'),training_complete_sha256=sha(out/'training_complete.json'),seconds=time.monotonic()-started)
        write(out/'audit_execution.json',audit)
        print(json.dumps(dict(passed=True,checks=check.count,comparisons=check.comparisons,max_error=check.max_error,primary_mean=analysis['primary_mean'],counts=check.scope_counts,seconds=time.monotonic()-started)),flush=True)
    except Exception as exc:
        audit.update(passed=False,checks=check.count,counts=check.scope_counts,failures=[str(exc)],traceback=traceback.format_exc(),seconds=time.monotonic()-started)
        write(out/'analysis_failure.json',audit);raise


if __name__=='__main__':main()
