"""Independent fixed-scope analysis and bounded execution replay for v24.

The v23 independent metric implementation is reused and hash-bound. No v24
production metric, sampling, probability-enumeration or update helper is used.
"""
from __future__ import annotations
import argparse,copy,hashlib,itertools,json,math,shutil,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.23'))
import analyze_experience as prior
Checks=prior.Checks
CHUNK=256
ARMS=('mean','attention')

def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {key:z[key] for key in z.files}


def summarize(rows,seeds,times):
    """Mean directions within a pair, partitions within a source, then sources."""
    source=[]
    for seed,condition in itertools.product(seeds,ARMS):
        selected=[r for r in rows if r['seed']==seed and r['condition']==condition]
        curve=[dict(update=t,scores=prior.average([r['curve'][i]['scores'] for r in selected])) for i,t in enumerate(times)]
        source.append(dict(seed=seed,condition=condition,curve=curve,scores=curve[-1]['scores'],auc=prior.integrate(curve)))
    aggregate={}
    for condition in ARMS:
        selected=[r for r in source if r['condition']==condition]
        curve=[dict(update=t,scores=prior.average([r['curve'][i]['scores'] for r in selected])) for i,t in enumerate(times)]
        aggregate[condition]=dict(curve=curve,scores=curve[-1]['scores'],auc=prior.integrate(curve))
    primary=[]
    for seed in seeds:
        by_arm={r['condition']:r for r in source if r['seed']==seed}
        get=lambda arm,kind:by_arm[arm][kind]['new12']['pooled']['J']
        primary.append(dict(seed=seed,mean=get('mean','scores'),attention=get('attention','scores'),
            difference=get('attention','scores')-get('mean','scores'),
            auc_difference=get('attention','auc')-get('mean','auc')))
    return dict(rows=rows,seed_rows=source,aggregate=aggregate,primary=primary,
        primary_mean=math.fsum(r['difference'] for r in primary)/len(primary))


def check_attention(raw,check):
    a=raw['effective_attention']
    check.check(a.shape==(960,2,2) and a.dtype==np.float32 and np.isfinite(a).all(),'effective attention shape/dtype')
    check.check(np.all((a>=0)&(a<=1)) and np.max(np.abs(a.astype(np.float64).sum(-1)-1))<1e-6,'effective attention simplex')


class IndependentSender(nn.Module):
    """Same declared atomic architecture; production FocusSender is not called."""
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.encoder=nn.Linear(8,96);self.embedding=nn.Embedding(8,16)
        self.recur=nn.LSTMCell(16,96);self.bilinear=nn.Linear(96,96,bias=False)
        self.fusion=nn.Linear(192,96);self.out=nn.Linear(96,7)


def roles(agent):
    return dict(sender=list(agent.focus_sender.parameters()),receiver=[v for name in ('receive_embedding','actor') for v in getattr(agent,name).parameters()])


def independent_initialize(agent,mode,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed));agent.focus_sender=IndependentSender(mode)
    agent.requires_grad_(False)
    for group in roles(agent).values():
        for value in group:value.requires_grad_(True)
    for value in agent.parameters():value.grad=None
    return agent


def sender_step(a,key,token,state):
    m=a.focus_sender
    h,c=m.recur(m.embedding(token.detach()),state)
    scores=(m.bilinear(h)[:,None,:]*key).sum(-1)
    weights=F.softmax(scores,-1);context=(weights[:,:,None]*key).sum(1)
    logits=m.out(torch.tanh(m.fusion(torch.cat((h,context),-1))))
    effective=weights.expand(-1,2)*.5 if m.mode=='mean' else weights
    return (h,c),logits,effective


def sender_start(a,x):
    key=F.gelu(a.focus_sender.encoder(x))
    if a.focus_sender.mode=='mean':key=key.mean(1,keepdim=True)
    zero=x.new_zeros(len(x),96)
    state,logits,weights=sender_step(a,key,torch.full((len(x),),7,dtype=torch.int64),(zero,zero))
    return key,state,logits,weights


def enumerate_sender(a,x):
    key,state,first,w0=sender_start(a,x)
    second=torch.stack([F.log_softmax(sender_step(a,key,torch.full((len(x),),t,dtype=torch.int64),state)[1],-1) for t in range(7)],1)
    lp=(F.log_softmax(first,-1)[:,:,None]+second).reshape(len(x),49)
    t0=first.argmax(-1).detach();_,last,w1=sender_step(a,key,t0,state)
    return lp,torch.stack((t0,last.argmax(-1).detach()),1),torch.stack((w0,w1),1)


def receive(a,messages):
    embedded=a.receive_embedding(messages.detach()).flatten(1)
    return a.actor(torch.cat((embedded,embedded.new_zeros(len(messages),20)),-1)).reshape(len(messages),2,6)


def independent_direction(sender,receiver,x,uniforms,positions,weight):
    def draw(logits,u):
        lp=F.log_softmax(logits,-1);prob=lp.exp()
        action=(prob.detach().cumsum(-1)<u).sum(-1).clamp(max=logits.shape[-1]-1)
        selected=lp.gather(1,action[:,None]).squeeze(1);entropy=-(prob*lp).sum(-1)
        return action.detach(),selected,entropy,prob
    key,state,first_logits,w0=sender_start(sender,x);u=torch.from_numpy(uniforms)
    first,l0,e0,p0=draw(first_logits,u[:,0:1])
    _,second_logits,w1=sender_step(sender,key,first,state)
    second,l1,e1,p1=draw(second_logits,u[:,1:2]);messages=torch.stack((first,second),1).detach()
    action_logits=receive(receiver,messages)
    food,lf,ef,pf=draw(action_logits[:,0],u[:,2:3]);water,lw,ew,pw=draw(action_logits[:,1],u[:,3:4])
    actions=torch.stack((food,water),1);success=(actions.numpy()==positions).astype(np.float32)
    reward=(.25*success.sum(1)+.5*success.prod(1)).astype(np.float32);advantage=torch.from_numpy(reward-np.float32(.5))
    slp=l0+l1;rlp=lf+lw;se=e0+e1;re=ef+ew
    sl=-(slp*advantage).mean()-float(weight)*se.mean();rl=-(rlp*advantage).mean()-float(weight)*re.mean()
    ar=lambda value:value.detach().numpy().copy()
    trace=dict(concepts=ar(x),uniforms=uniforms.copy(),messages=ar(messages),first_logits=ar(first_logits),second_logits=ar(second_logits),
        token_probabilities=ar(torch.stack((p0,p1),1)),effective_attention=ar(torch.stack((w0,w1),1)),action_logits=ar(action_logits),
        action_probabilities=ar(torch.stack((pf,pw),1)),actions=ar(actions),positions=positions.copy(),success=success,reward=reward,
        advantage=ar(advantage),sender_logp=ar(slp),receiver_logp=ar(rlp),sender_entropy=ar(se),receiver_entropy=ar(re),
        sender_loss=float(sl.detach()),receiver_loss=float(rl.detach()),entropy_weight=float(weight),baseline=.5)
    return sl,rl,trace


def focus_seed(seed,p,d):
    return int(np.random.SeedSequence([24024,seed,p,d,0]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))


def restore(seed,p,arm,source,check):
    prepared=load(source/f'prepared_{seed}.pt')
    agents=prior.remake_agents(seed,prepared,7,2,'identity')
    for d,a in enumerate(agents):
        original=load(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt')
        a.load_state_dict(original['agent']);prior.reset_social(a,seed,p,d)
        independent_initialize(a,arm,focus_seed(seed,p,d))
        check.exact({k:sum(v.numel() for v in vs) for k,vs in roles(a).items()},dict(sender=73191,receiver=6364),'declared role parameter counts')
        for name,param in a.named_parameters():
            check.exact(param.requires_grad,name.split('.')[0] in ('focus_sender','receive_embedding','actor'),'role trainability')
    check.check(not(set(map(id,agents[0].parameters()))&set(map(id,agents[1].parameters()))),'separate parameter objects')
    return agents


def frozen(state):
    return {k:v for k,v in state.items() if k.split('.')[0] not in ('focus_sender','receive_embedding','actor')}


def cache_check(seed,p,source,out,tables,bank,check):
    folder=out/'cache'/f's{seed}_p{p}_all';manifest=read(folder/'manifest.json')
    check.exact((manifest['seed'],manifest['partition'],manifest['chunk'],manifest['private_head_used'],manifest['gold_inputs']),(seed,p,256,True,False),'concept cache contract')
    for name,h in manifest['files'].items():check.exact(sha(folder/name),h,'concept cache output bytes')
    for name,h in manifest['input_hashes'].items():check.exact(sha(name),h,'concept cache input bytes')
    cache={};capabilities=[]
    for d in (0,1):
        blob=load(source/'private'/f's{seed}_p{p}_d{d}_all'/'final.pt')
        a,head=prior.make_private(seed,p,d,load(source/f'prepared_{seed}.pt'))
        a.load_state_dict(blob['agent']);head.load_state_dict(blob['head']);a.requires_grad_(False);head.requires_grad_(False)
        with torch.no_grad():projected=a.project(bank.features).detach()
        for split,w in tables.items():
            x=np.load(folder/f'{split}_d{d}.npy');h=np.load(source/'cache'/f's{seed}_p{p}_all'/f'{split}_d{d}.npy')
            check.check(x.shape==(len(w['map_id']),2,8) and x.dtype==np.float32 and np.isfinite(x).all(),'all concept cache layout')
            check.exact(x[:,:,6:],np.broadcast_to(np.eye(2,dtype=np.float32),(len(x),2,2)),'constant role identity only')
            check.check(np.all((x[:,:,:6]>=0)&(x[:,:,:6]<=1)) and np.max(np.abs(x[:,:,:6].astype(np.float64).sum(-1)-1))<1e-6,'private prediction probabilities')
            cache[split,d]=torch.from_numpy(x)
            starts=list(range(0,len(x),CHUNK)) if split=='test' else sorted({0,((len(x)-1)//CHUNK)*CHUNK})
            for lo in starts:
                world=prior.subset(w,slice(lo,lo+CHUNK))
                with torch.no_grad():
                    frames,bits=prior.render(projected,world);hidden=prior.observe_sequence(a,frames,bits,'full')
                    check.exact(hidden.numpy(),h[lo:lo+CHUNK],'source frontend original chunk')
                    prediction=head(torch.from_numpy(h[lo:lo+CHUNK])).reshape(-1,2,6).softmax(-1)
                    concept=torch.cat((prediction,torch.eye(2,dtype=torch.float32)[None].expand(len(prediction),-1,-1)),-1)
                check.exact(concept.numpy(),x[lo:lo+CHUNK],'private-head concept original chunk');check.add('cache_'+split+'_worlds_replayed',len(prediction))
            if split=='test':
                correct=(x[:,:,:6].argmax(-1)==w['positions']).all(1)
                score={group:{label:float(correct[np.isin(w['map_id'],maps)&(np.ones(len(x),bool) if label=='pooled' else w['shown']==int(label=='water_only'))].mean()) for label in ('pooled','food_only','water_only')} for group,maps in prior.group_maps(p).items()}
                capabilities.append(dict(seed=seed,partition=p,direction=d,scores=score))
        check.add('source_private_heads')
    return cache,capabilities


def replay_update(folder,agents,cache,w,cfg,step,row,check):
    states=load(folder/f'checkpoint_{step:04d}.pt');optimizers=load(folder/f'optimizer_{step:04d}.pt');opts=[]
    for agent,state,s in zip(agents,states,optimizers):
        agent.load_state_dict(state);opt=torch.optim.Adam([v for vs in roles(agent).values() for v in vs],lr=.0007);opt.load_state_dict(s);opts.append(opt)
    losses=[{},{}];worlds={};traces={}
    for d in (0,1):
        f=prior.fixture(cfg['seed'],cfg['partition'],d,step,'old','social',w);idx=f['indices']
        sl,rl,tr=independent_direction(agents[d],agents[1-d],cache['train',d][idx],f['uniforms'],w['positions'][idx],.02 if step<2100 else 0.)
        losses[d]['sender']=sl;losses[1-d]['receiver']=rl
        worlds.update({f'd{d}__{k}':v for k,v in f.items()});traces.update({f'd{d}__{k}':np.asarray(v) for k,v in tr.items()})
    expected={**{'world__'+k:v for k,v in worlds.items()},**{'trace__'+k:v for k,v in traces.items()}}
    check.exact(expected,npz(folder/f'train_{step+1:04d}.npz'),'sampled key update full trace')
    check.exact(prior.arrays_sha(traces),row['trace_sha256'],'sampled trace hash')
    for d,(agent,opt) in enumerate(zip(agents,opts)):
        opt.zero_grad(set_to_none=True);loss=(losses[d]['sender']+losses[d]['receiver'])/2;loss.backward()
        check.check(all(v.grad is not None and torch.isfinite(v.grad).all() for vs in roles(agent).values() for v in vs),'finite role gradients')
        if cfg['arm']=='mean':check.check(torch.count_nonzero(agent.focus_sender.bilinear.weight.grad)==0,'mean single-key bilinear gradient zero')
        norms={k:float(torch.nn.utils.clip_grad_norm_(v,2.)) for k,v in roles(agent).items()}
        check.exact(dict(loss=float(loss.detach()),norms=norms),row['people'][d],'sampled independent role loss/clip')
        check.check(all(v.grad is None for v in agent.parameters() if not v.requires_grad),'no frozen gradient')
    for opt in opts:opt.step()
    check.exact([a.state_dict() for a in agents],load(folder/f'after_{step+1:04d}.pt'),'sampled after parameters exact')
    check.exact([o.state_dict() for o in opts],load(folder/f'after_{step+1:04d}_optimizer.pt'),'sampled after Adam exact')
    check.add('key_pair_updates_replayed')


def case(seed,p,arm,source,out,cache,tables,sample,check):
    folder=out/'social'/f's{seed}_p{p}_{arm}';cfg=read(folder/'config.json');result=read(folder/'result.json');curve=read(folder/'curve.json')
    check.exact((cfg['seed'],cfg['partition'],cfg['arm'],cfg['batch_per_direction']),(seed,p,arm,256),'case identity/batch')
    check.exact(Path(cfg['source']).resolve(),source,'case source directory');check.exact(result['status'],'complete','case completed')
    for path,h in cfg['source_private'].items():check.exact(sha(path),h,'case private source bytes')
    agents=restore(seed,p,arm,source,check);initial=load(folder/'initial.pt')
    check.exact([a.state_dict() for a in agents],initial,'independent paired initialization')
    check.exact([prior.fingerprint(frozen(s)) for s in initial],cfg['frozen_hashes'],'initial frozen fingerprints')
    for d,r in enumerate(cfg['reset']):
        check.exact((r['focus']['seed'],r['focus']['mode']),(focus_seed(seed,p,d),arm),'new sender initialization namespace')
        check.exact(r['focus']['counts'],dict(sender=73191,receiver=6364),'saved role parameter counts')
    for o in load(folder/'initial_optimizer.pt'):check.exact(o['state'],{},'fresh Adam')
    logs,stream=prior.logs_check(folder,cfg,'social','old',tables,check);points=[]
    for row in curve:
        step=row['update'];states=load(folder/f'checkpoint_{step:04d}.pt');scores=[]
        for d,s in enumerate(states):check.exact(frozen(s),frozen(initial[d]),'all checkpoint frozen tensors')
        for d in (0,1):
            raw=npz(folder/f'protocol_{step:04d}_d{d}.npz')
            check.exact({k:raw[k] for k in tables['test']},tables['test'],'all protocol world tables')
            lp=raw['sender_log_probs'];check.check(lp.shape==(960,49) and np.isfinite(lp).all() and np.max(np.abs(np.exp(lp.astype(np.float64)).sum(1)-1))<1e-6,'all complete-message probabilities')
            check_attention(raw,check)
            if arm=='mean':check.exact(raw['effective_attention'],np.full((960,2,2),.5,np.float32),'mean effective attention constant')
            scores.append(prior.metrics(raw,p,False));check.add('protocol_evaluation_tables')
        check.close(scores,row['scores'],'independent raw protocol statistics');points.append(dict(update=step,scores=prior.average(scores)))
    check.exact([r['update'] for r in points],cfg['checkpoints'],'case checkpoint inventory')
    check.close(prior.average(result['scores']),points[-1]['scores'],'independent endpoint statistics')
    final=load(folder/'final.pt');check.exact(final,load(folder/f'checkpoint_{cfg["updates"]:04d}.pt'),'final checkpoint identity')
    for agent,state in zip(agents,final):agent.load_state_dict(state)
    for d in (0,1):
        raw=npz(folder/f'protocol_{cfg["updates"]:04d}_d{d}.npz')
        with torch.no_grad():check.exact(prior.enumerate_receiver(agents[1-d]).numpy(),raw['receiver_logits'],'all49 endpoint receiver table')
        check.add('endpoint_receiver_tables_replayed')
        for lo in range(0,960,CHUNK):
            with torch.no_grad():lp,tokens,attention=enumerate_sender(agents[d],cache['test',d][lo:lo+CHUNK])
            check.exact(lp.numpy(),raw['sender_log_probs'][lo:lo+CHUNK],'complete endpoint sender distribution')
            check.exact(tokens.numpy(),raw['tokens'][lo:lo+CHUNK],'complete endpoint sequential greedy tokens')
            check.exact(attention.numpy(),raw['effective_attention'][lo:lo+CHUNK],'complete endpoint attention weights')
            check.add('endpoint_sender_worlds_replayed',len(lp))
    if sample:
        for step in (0,2100):
            if step<cfg['updates']:replay_update(folder,agents,cache,tables['train'],cfg,step,logs[step],check)
    check.add('social_pairs')
    return dict(seed=seed,partition=p,condition=arm,raw_folder=str(folder),curve=points,scores=points[-1]['scores']),initial,stream


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();out=args.out.resolve()
    check=Checks();started=time.monotonic();torch.set_num_threads(1)
    if (out/'audit_execution.json').exists():
        history=out/'analysis_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f');history.mkdir(parents=True)
        for name in ('audit_execution.json','analysis.json','comparison.json','analyze_attention_source.py'):
            if (out/name).exists():shutil.copy2(out/name,history/name)
    shutil.copy2(__file__,out/'analyze_attention_source.py')
    dependencies={str(PROJECT/p):sha(PROJECT/p) for p in ('redesign_v0.23/analyze_experience.py','redesign_v0.22/analyze_probe.py','redesign_v0.21/audit_execution.py','redesign_v0.20/temporal_model.py')}
    audit=dict(passed=False,analysis_source_sha256=sha(__file__),analysis_dependency_sha256=dependencies,production_metrics_called=False,
        production_sender_forward_or_loss_called=False,coverage=[
            'Every saved protocol table independently recomputed using the hash-bound v23 independent metrics',
            'Every endpoint960-world sender distribution/sequential-greedy token/attention and all49 receiver rows replayed',
            'All source-private-head test960 predictions and temporal frontends; train concept cache original first/last chunks only',
            'All external training fixtures and hashes, all checkpoint frozen tensors, independent paired initialization',
            'Key updates only33101/p1 (development99523/p1), both arms at1/2101 when present; independent loss/clip/Adam exact',
            'No private retraining or full social training-trajectory replay; earlier frozen temporal atomic operations reused'])
    try:
        complete=read(out/'training_complete.json');inv=read(out/'invocation.json');source=Path(inv['source']).resolve()
        check.exact(complete['status'],'complete','whole-batch completion gate');check.exact(complete['formal'],inv['formal'],'formal flag agreement')
        for field in ('source_hashes','input_hashes'):
            check.exact(complete[field],inv[field],'completion source binding')
            for path,h in inv[field].items():check.exact(sha(path),h,'source/input bytes')
        for path,h in inv['source_hashes'].items():check.exact(sha(out/'frozen_sources'/Path(path).relative_to(PROJECT)),h,'frozen source copy')
        for path,h in complete['files'].items():check.exact(sha(out/path),h,'completed output bytes')
        for name in ('new_dino_inferences','new_private_updates'):check.exact((complete[name],inv[name]),(0,0),'no additional '+name)
        check.exact((inv['threads'],inv['torch_version'],inv['numpy_version']),(1,str(torch.__version__),np.__version__),'runtime contract')
        check.exact(inv['social_arms'],list(ARMS),'fixed two-arm scope')
        if inv['formal']:
            check.exact((inv['seeds'],inv['partitions'],inv['updates']),([33101,33102,33103,33104],[1,2,3],2400),'formal scope')
            gate=inv['preflight'];check.exact(sha(gate['path']),gate['sha256'],'formal preflight receipt hash')
            check.check(read(gate['path'])['passed'],'formal preflight passed')
        else:check.exact((inv['seeds'],inv['partitions'],inv['updates']),([99523],[1],40),'development scope')
        expected_times=sorted({0,inv['updates'],*[t for t in (0,100,600,1200,2100,2400) if t<inv['updates']]})
        bank=prior.ImageBank();tables={split:prior.table(bank,split) for split in ('train','test')}
        for split,w in tables.items():
            check.exact(w,npz(out/f'{split}_worlds.npz'),'complete independent '+split+' world enumeration')
            check.exact(w,npz(source/f'{split}_worlds.npz'),'source world-table reuse')
        rows=[];capabilities=[];seeds_used=[]
        for seed,p in itertools.product(inv['seeds'],inv['partitions']):
            cache,ability=cache_check(seed,p,source,out,tables,bank,check);capabilities.extend(ability)
            initials={};streams={}
            for arm in ARMS:
                sample=seed==(33101 if inv['formal'] else 99523) and p==1
                row,initial,stream=case(seed,p,arm,source,out,cache,tables,sample,check);rows.append(row);initials[arm]=initial;streams[arm]=stream
                check.exact([r['update'] for r in row['curve']],expected_times,'all fixed checkpoint times')
            check.exact(initials['mean'],initials['attention'],'mean/attention all initial tensors identical')
            check.exact(streams['mean'],streams['attention'],'mean/attention world and action-uniform streams identical')
            # Same v23 B fixture stream, not merely the same distribution.
            oldlog=[json.loads(line)['world_sha256'] for line in (source/'social'/f's{seed}_p{p}_all_old'/'training.jsonl').read_text().splitlines()]
            check.exact(streams['mean'],oldlog,'v23 B exact external stream reuse')
            seeds_used.extend(focus_seed(seed,p,d) for d in (0,1))
            print(json.dumps(dict(audited_seed=seed,partition=p,checks=check.count)),flush=True)
        check.exact(len(set(seeds_used)),len(seeds_used),'unique person/source sender initialization identities')
        check.exact(check.scope_counts['social_pairs'],complete['social_runs'],'complete social count')
        check.exact(check.scope_counts['social_updates_stream_checked'],complete['pair_updates'],'complete external update streams')
        check.exact(complete['messages'],complete['pair_updates']*512,'message budget')
        check.exact(complete['actions'],complete['pair_updates']*1024,'action budget')
        check.exact(check.scope_counts['source_private_heads'],len(inv['seeds'])*len(inv['partitions'])*2,'source head count')
        ability_rows=[]
        for seed in inv['seeds']:
            score=prior.average([r['scores'] for r in capabilities if r['seed']==seed])
            ability_rows.append(dict(seed=seed,scores=score))
        analysis=dict(status='complete',formal=inv['formal'],seeds=inv['seeds'],partitions=inv['partitions'],times=expected_times,updates=inv['updates'],
            analysis_source_sha256=sha(__file__),analysis_dependency_sha256=dependencies,training_complete_sha256=sha(out/'training_complete.json'),
            **summarize(rows,inv['seeds'],expected_times),source_private_capability=dict(rows=capabilities,seed_rows=ability_rows,filter_applied=False),
            boundaries=['Four inherited source seeds; partitions and directions are nested measurements, not new independent sources',
                'Attention versus encoded-key mean compares retaining and dynamically reading two keys against mean compression; effective degrees of freedom differ',
                'Inputs are frozen learned private-head probabilities plus supplied goal-role identity; no gold locations or prescribed symbol meanings',
                'New12 is unseen in social training but seen during private preparation; not whole-agent zero-shot',
                'Historical v23 B has a different sender and input readout and is not a concurrent attention-effect control',
                'Attention weights and successful49-code messages do not establish compositional structure or common attention'])
        write(out/'analysis.json',analysis)
        write(out/'comparison.json',dict(passed=True,comparisons=check.comparisons,max_absolute_error=check.max_error,analysis_sha256=sha(out/'analysis.json'),
            production_metrics_called=False,scope='All raw protocol curve/endpoint scores versus saved production scores; independent source aggregation.'))
        audit.update(passed=True,formal=inv['formal'],checks=check.count,counts=check.scope_counts,failures=[],analysis_sha256=sha(out/'analysis.json'),
            training_complete_sha256=sha(out/'training_complete.json'),seconds=time.monotonic()-started)
        write(out/'audit_execution.json',audit)
        print(json.dumps(dict(passed=True,checks=check.count,comparisons=check.comparisons,max_error=check.max_error,primary_mean=analysis['primary_mean'],counts=check.scope_counts,seconds=time.monotonic()-started)),flush=True)
    except Exception as exc:
        audit.update(passed=False,checks=check.count,counts=check.scope_counts,failures=[str(exc)],traceback=traceback.format_exc(),seconds=time.monotonic()-started)
        write(out/'analysis_failure.json',audit);raise


if __name__=='__main__':main()
