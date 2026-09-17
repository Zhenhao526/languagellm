"""Independent analysis and bounded replay of the zero-initialized joint readout.

Only the new joint condition is replayed. The frozen v24 mean/attention rows are
reused as explicitly labelled references, with their analysis and audit hashes.
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
sys.path.insert(0,str(PROJECT/'redesign_v0.24'))
import analyze_attention as prior
experience=prior.prior
Checks=prior.Checks;CHUNK=256

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def summarize(rows,seeds,times):
    arms=('mean','joint','attention');source=[]
    for seed,arm in itertools.product(seeds,arms):
        r=[x for x in rows if x['seed']==seed and x['condition']==arm]
        curve=[dict(update=t,scores=experience.average([x['curve'][i]['scores'] for x in r])) for i,t in enumerate(times)]
        source.append(dict(seed=seed,condition=arm,curve=curve,scores=curve[-1]['scores'],auc=experience.integrate(curve)))
    aggregate={}
    for arm in arms:
        r=[x for x in source if x['condition']==arm]
        curve=[dict(update=t,scores=experience.average([x['curve'][i]['scores'] for x in r])) for i,t in enumerate(times)]
        aggregate[arm]=dict(curve=curve,scores=curve[-1]['scores'],auc=experience.integrate(curve))
    primary=[]
    for seed in seeds:
        r={x['condition']:x for x in source if x['seed']==seed}
        metric=lambda arm,kind,group:r[arm][kind][group]['pooled']['J']
        primary.append(dict(seed=seed,mean=metric('mean','scores','old'),joint=metric('joint','scores','old'),
            difference=metric('joint','scores','old')-metric('mean','scores','old'),
            auc_difference=metric('joint','auc','old')-metric('mean','auc','old'),
            new12_difference=metric('joint','scores','new12')-metric('mean','scores','new12')))
    return dict(rows=rows,seed_rows=source,aggregate=aggregate,primary=primary,
        primary_mean=math.fsum(x['difference'] for x in primary)/len(primary))


class IndependentJoint(nn.Module):
    def __init__(self,reference):
        super().__init__();self.mode='joint'
        self.encoder=copy.deepcopy(reference.encoder);self.embedding=copy.deepcopy(reference.embedding);self.recur=copy.deepcopy(reference.recur)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(25025);self.contrast=nn.Linear(96,96,bias=False);nn.init.zeros_(self.contrast.weight)
        self.fusion=copy.deepcopy(reference.fusion);self.out=copy.deepcopy(reference.out)


def independent_initialize(agent):
    agent.focus_sender=IndependentJoint(agent.focus_sender);agent.requires_grad_(False)
    for vs in prior.roles(agent).values():
        for value in vs:value.requires_grad_(True)
    for value in agent.parameters():value.grad=None
    return agent


def sender_step(a,key,token,state):
    m=a.focus_sender;h,c=m.recur(m.embedding(token.detach()),state)
    average=key.mean(1);difference=(key[:,0]-key[:,1])*.5
    activation=m.fusion(torch.cat((h,average),-1))+m.contrast(difference)
    return (h,c),m.out(torch.tanh(activation))


def sender_start(a,x):
    key=F.gelu(a.focus_sender.encoder(x));zero=x.new_zeros(len(x),96)
    state,first=sender_step(a,key,torch.full((len(x),),7,dtype=torch.int64),(zero,zero))
    return key,state,first


def enumerate_sender(a,x):
    key,state,first=sender_start(a,x)
    second=torch.stack([F.log_softmax(sender_step(a,key,torch.full((len(x),),i,dtype=torch.int64),state)[1],-1) for i in range(7)],1)
    probabilities=(F.log_softmax(first,-1)[:,:,None]+second).reshape(len(x),49)
    first_token=first.argmax(-1).detach();_,last=sender_step(a,key,first_token,state)
    return probabilities,torch.stack((first_token,last.argmax(-1).detach()),1)


def independent_direction(sender,receiver,x,uniforms,positions,weight):
    def draw(logits,u):
        lp=F.log_softmax(logits,-1);p=lp.exp();action=(p.detach().cumsum(-1)<u).sum(-1).clamp(max=logits.shape[-1]-1)
        chosen=lp.gather(1,action[:,None]).squeeze(1);entropy=-(p*lp).sum(-1)
        return action.detach(),chosen,entropy,p
    key,state,fl=sender_start(sender,x);u=torch.from_numpy(uniforms)
    t0,l0,e0,p0=draw(fl,u[:,0:1]);_,second=sender_step(sender,key,t0,state)
    t1,l1,e1,p1=draw(second,u[:,1:2]);messages=torch.stack((t0,t1),1).detach();al=prior.receive(receiver,messages)
    food,lf,ef,pf=draw(al[:,0],u[:,2:3]);water,lw,ew,pw=draw(al[:,1],u[:,3:4])
    actions=torch.stack((food,water),1);success=(actions.numpy()==positions).astype(np.float32)
    reward=(.25*success.sum(1)+.5*success.prod(1)).astype(np.float32);advantage=torch.from_numpy(reward-np.float32(.5))
    slp=l0+l1;rlp=lf+lw;se=e0+e1;re=ef+ew
    sl=-(slp*advantage).mean()-float(weight)*se.mean();rl=-(rlp*advantage).mean()-float(weight)*re.mean()
    ar=lambda value:value.detach().numpy().copy()
    trace=dict(concepts=ar(x),uniforms=uniforms.copy(),messages=ar(messages),first_logits=ar(fl),second_logits=ar(second),
        token_probabilities=ar(torch.stack((p0,p1),1)),action_logits=ar(al),action_probabilities=ar(torch.stack((pf,pw),1)),
        actions=ar(actions),positions=positions.copy(),success=success,reward=reward,advantage=ar(advantage),sender_logp=ar(slp),receiver_logp=ar(rlp),
        sender_entropy=ar(se),receiver_entropy=ar(re),sender_loss=float(sl.detach()),receiver_loss=float(rl.detach()),entropy_weight=float(weight),baseline=.5)
    return sl,rl,trace


def restore(seed,p,reference,check):
    source=Path(read(reference/'invocation.json')['source']).resolve()
    prepared=load(source/f'prepared_{seed}.pt');agents=experience.remake_agents(seed,prepared,7,2,'identity')
    states=load(reference/'social'/f's{seed}_p{p}_mean'/'initial.pt')
    for d,a in enumerate(agents):
        prior.independent_initialize(a,'mean',prior.focus_seed(seed,p,d));a.load_state_dict(states[d])
    mean=copy.deepcopy(agents)
    for d,a in enumerate(agents):
        independent_initialize(a)
        current=a.state_dict()
        check.exact({k:v for k,v in current.items() if k!='focus_sender.contrast.weight'},
            {k:v for k,v in states[d].items() if k!='focus_sender.bilinear.weight'},'all copied shared initial tensors')
        check.exact(current['focus_sender.contrast.weight'],torch.zeros(96,96),'contrast zero initial')
        check.exact({k:sum(v.numel() for v in vs) for k,vs in prior.roles(a).items()},dict(sender=73191,receiver=6364),'role parameter count')
        for name,value in a.named_parameters():check.exact(value.requires_grad,name.split('.')[0] in ('focus_sender','receive_embedding','actor'),'declared trainability')
    check.check(not(set(map(id,agents[0].parameters()))&set(map(id,agents[1].parameters()))),'two private parameter objects')
    return agents,mean


def reused_cache(seed,p,reference,tables,check):
    folder=reference/'cache'/f's{seed}_p{p}_all';cache={}
    for split,d in itertools.product(('train','test'),(0,1)):
        x=np.load(folder/f'{split}_d{d}.npy')
        check.check(x.shape==(len(tables[split]['map_id']),2,8) and x.dtype==np.float32 and np.isfinite(x).all(),'reused concept layout')
        check.exact(x[:,:,6:],np.broadcast_to(np.eye(2,dtype=np.float32),(len(x),2,2)),'reused fixed role bits')
        cache[split,d]=torch.from_numpy(x)
    return cache


def replay_update(folder,agents,cache,w,cfg,step,row,check):
    states=load(folder/f'checkpoint_{step:04d}.pt');saved_opt=load(folder/f'optimizer_{step:04d}.pt');opts=[]
    for a,state,s in zip(agents,states,saved_opt):
        a.load_state_dict(state);opt=torch.optim.Adam([v for vs in prior.roles(a).values() for v in vs],lr=.0007);opt.load_state_dict(s);opts.append(opt)
    losses=[{},{}];worlds={};traces={}
    for d in (0,1):
        fixture=experience.fixture(cfg['seed'],cfg['partition'],d,step,'old','social',w);indices=fixture['indices']
        sl,rl,trace=independent_direction(agents[d],agents[1-d],cache['train',d][indices],fixture['uniforms'],w['positions'][indices],.02 if step<2100 else 0.)
        losses[d]['sender']=sl;losses[1-d]['receiver']=rl
        worlds.update({f'd{d}__{k}':v for k,v in fixture.items()});traces.update({f'd{d}__{k}':np.asarray(v) for k,v in trace.items()})
    check.exact({**{'world__'+k:v for k,v in worlds.items()},**{'trace__'+k:v for k,v in traces.items()}},npz(folder/f'train_{step+1:04d}.npz'),'sampled independent full trace')
    check.exact(experience.arrays_sha(traces),row['trace_sha256'],'sampled full trace hash')
    for d,(a,opt) in enumerate(zip(agents,opts)):
        opt.zero_grad(set_to_none=True);loss=(losses[d]['sender']+losses[d]['receiver'])/2;loss.backward()
        check.check(all(v.grad is not None and torch.isfinite(v.grad).all() for vs in prior.roles(a).values() for v in vs),'finite role gradients')
        contrast_norm=float(a.focus_sender.contrast.weight.grad.norm())
        norms={k:float(torch.nn.utils.clip_grad_norm_(v,2.)) for k,v in prior.roles(a).items()}
        check.exact(dict(loss=float(loss.detach()),norms=norms,contrast_gradient_norm_before_clip=contrast_norm),row['people'][d],'sampled loss/role clip/contrast norm')
        check.check(all(v.grad is None for v in a.parameters() if not v.requires_grad),'frozen gradient absence')
    for opt in opts:opt.step()
    check.exact([a.state_dict() for a in agents],load(folder/f'after_{step+1:04d}.pt'),'sampled after parameters exact')
    check.exact([o.state_dict() for o in opts],load(folder/f'after_{step+1:04d}_optimizer.pt'),'sampled after Adam exact')
    check.add('key_pair_updates_replayed')


def case(seed,p,reference,out,cache,tables,sample,reference_files,check):
    folder=out/'social'/f's{seed}_p{p}_joint';cfg=read(folder/'config.json');result=read(folder/'result.json');curve=read(folder/'curve.json')
    check.exact((cfg['seed'],cfg['partition'],cfg['arm'],cfg['batch_per_direction']),(seed,p,'joint',256),'new case identity/batch')
    check.exact(Path(cfg['reference']).resolve(),reference,'case reference');check.exact(result['status'],'complete','case completed')
    initial_path=reference/'social'/f's{seed}_p{p}_mean'/'initial.pt'
    check.exact(cfg['source_initial'],dict(path=str(initial_path),sha256=sha(initial_path)),'declared actual mean initial source')
    check.exact(cfg['trainable_parameters'],dict(sender=73191,receiver=6364),'declared parameter counts')
    check.exact(cfg['contrast_initialization'],'all zeros','declared zero contrast')
    agents,mean=restore(seed,p,reference,check);initial=load(folder/'initial.pt')
    check.exact([a.state_dict() for a in agents],initial,'independent new initialization')
    check.exact([experience.fingerprint(prior.frozen(s)) for s in initial],cfg['frozen_hashes'],'initial frozen fingerprints')
    for o in load(folder/'initial_optimizer.pt'):check.exact(o['state'],{},'fresh new Adam')
    for d in (0,1):
        path=reference/'social'/f's{seed}_p{p}_mean'/f'protocol_0000_d{d}.npz'
        check.exact(sha(path),reference_files[str(path.relative_to(reference))],'original mean initial protocol bytes')
        raw=npz(path)
        with torch.no_grad():
            r=experience.enumerate_receiver(agents[1-d]);mr=experience.enumerate_receiver(mean[1-d])
        check.exact(r,mr,'initial receiver function same');check.exact(r.numpy(),raw['receiver_logits'],'initial actual mean receiver table')
        for lo in range(0,960,CHUNK):
            with torch.no_grad():
                lp,t=enumerate_sender(agents[d],cache['test',d][lo:lo+CHUNK]);mlp,mt,_=prior.enumerate_sender(mean[d],cache['test',d][lo:lo+CHUNK])
            check.exact(lp,mlp,'initial complete49 function same');check.exact(t,mt,'initial sequential greedy function same')
            check.exact(lp.numpy(),raw['sender_log_probs'][lo:lo+CHUNK],'initial actual mean complete49 probabilities')
            check.exact(t.numpy(),raw['tokens'][lo:lo+CHUNK],'initial actual mean sequential greedy')
            check.add('initial_sender_worlds_equal_mean',len(lp))
        check.add('initial_receiver_tables_equal_mean')
    logs,stream=experience.logs_check(folder,cfg,'social','old',tables,check)
    for row in logs:check.check(all(np.isfinite(v['contrast_gradient_norm_before_clip']) and v['contrast_gradient_norm_before_clip']>=0 for v in row['people']),'all logged contrast gradient norms finite')
    reference_log=reference/'social'/f's{seed}_p{p}_mean'/'training.jsonl'
    check.exact(sha(reference_log),reference_files[str(reference_log.relative_to(reference))],'reference world-stream bytes')
    check.exact(stream,[json.loads(line)['world_sha256'] for line in reference_log.read_text().splitlines()],'complete external world/action stream equal mean')
    points=[]
    for row in curve:
        step=row['update'];states=load(folder/f'checkpoint_{step:04d}.pt');scores=[]
        for d,s in enumerate(states):check.exact(prior.frozen(s),prior.frozen(initial[d]),'every checkpoint frozen tensors')
        for d in (0,1):
            raw=npz(folder/f'protocol_{step:04d}_d{d}.npz')
            check.exact(set(raw),set(tables['test'])|{'sender_log_probs','tokens','receiver_logits'},'new protocol field scope')
            check.exact({k:raw[k] for k in tables['test']},tables['test'],'all protocol world tables')
            lp=raw['sender_log_probs'];check.check(lp.shape==(960,49) and np.isfinite(lp).all() and np.max(np.abs(np.exp(lp.astype(np.float64)).sum(1)-1))<1e-6,'all49 message probability normalization')
            scores.append(experience.metrics(raw,p,False));check.add('protocol_evaluation_tables')
        check.close(scores,row['scores'],'independent all raw scores');points.append(dict(update=step,scores=experience.average(scores)))
    check.exact([r['update'] for r in points],cfg['checkpoints'],'case fixed checkpoint inventory')
    check.close(experience.average(result['scores']),points[-1]['scores'],'independent endpoint scores')
    final=load(folder/'final.pt');check.exact(final,load(folder/f'checkpoint_{cfg["updates"]:04d}.pt'),'final checkpoint same')
    for a,s in zip(agents,final):a.load_state_dict(s)
    for d in (0,1):
        raw=npz(folder/f'protocol_{cfg["updates"]:04d}_d{d}.npz')
        with torch.no_grad():check.exact(experience.enumerate_receiver(agents[1-d]).numpy(),raw['receiver_logits'],'complete endpoint receiver table')
        check.add('endpoint_receiver_tables_replayed')
        for lo in range(0,960,CHUNK):
            with torch.no_grad():lp,tokens=enumerate_sender(agents[d],cache['test',d][lo:lo+CHUNK])
            check.exact(lp.numpy(),raw['sender_log_probs'][lo:lo+CHUNK],'complete endpoint sender49 probabilities')
            check.exact(tokens.numpy(),raw['tokens'][lo:lo+CHUNK],'complete endpoint sequential greedy tokens')
            check.add('endpoint_sender_worlds_replayed',len(lp))
    if sample:
        for step in (0,2100):
            if step<cfg['updates']:replay_update(folder,agents,cache,tables['train'],cfg,step,logs[step],check)
    check.add('new_social_pairs')
    return dict(seed=seed,partition=p,condition='joint',is_reference=False,raw_folder=str(folder),curve=points,scores=points[-1]['scores'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();out=args.out.resolve()
    check=Checks();started=time.monotonic();torch.set_num_threads(1)
    if (out/'audit_execution.json').exists():
        history=out/'analysis_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f');history.mkdir(parents=True)
        for name in ('audit_execution.json','analysis.json','comparison.json','analyze_joint_source.py'):
            if (out/name).exists():shutil.copy2(out/name,history/name)
    shutil.copy2(__file__,out/'analyze_joint_source.py')
    dependencies={str(PROJECT/p):sha(PROJECT/p) for p in ('redesign_v0.24/analyze_attention.py','redesign_v0.23/analyze_experience.py','redesign_v0.22/analyze_probe.py','redesign_v0.21/audit_execution.py','redesign_v0.20/temporal_model.py')}
    audit=dict(passed=False,analysis_source_sha256=sha(__file__),analysis_dependency_sha256=dependencies,production_metrics_called=False,
        production_sender_forward_or_loss_called=False,coverage=[
            'New joint arm only: every saved raw protocol metric independently recomputed',
            'Every initial960-world full49 distribution/greedy function equals actual v24 mean; every new endpoint960-world sender and49-row receiver replayed',
            'All external training streams and source/input/output hashes; all checkpoint frozen tensors and copied shared initialization',
            'Fixed key updates only33101/p1 (development99523/p1) at1/2101 when present: full trace, independent losses/clip/Adam exact',
            'v24 mean/attention outcome rows reused from their hash-bound completed independent analysis; not replayed as new training',
            'Concept cache reused by hash with structural checks only; no new private-head or temporal/DINO inference',
            'No full training-trajectory gradient replay; no donor recombination claim in this audit'])
    try:
        complete=read(out/'training_complete.json');inv=read(out/'invocation.json');reference=Path(inv['reference']).resolve()
        check.exact(complete['status'],'complete','whole-batch completion gate');check.exact(complete['formal'],inv['formal'],'formal flag same')
        for field in ('source_hashes','input_hashes'):
            check.exact(complete[field],inv[field],'completion source binding')
            for p,h in inv[field].items():check.exact(sha(p),h,'source/input bytes')
        for p,h in inv['source_hashes'].items():check.exact(sha(out/'frozen_sources'/Path(p).relative_to(PROJECT)),h,'frozen source copy')
        for p,h in complete['files'].items():check.exact(sha(out/p),h,'completed output bytes')
        for key in ('new_dino_inferences','new_private_updates','new_cache_inferences'):check.exact((complete[key],inv[key]),(0,0),'no additional '+key)
        check.exact((inv['threads'],inv['torch_version'],inv['numpy_version']),(1,str(torch.__version__),np.__version__),'runtime contract')
        check.exact(inv['social_arms'],['joint'],'one new training arm')
        if inv['formal']:
            check.exact((inv['seeds'],inv['partitions'],inv['updates']),([33101,33102,33103,33104],[1,2,3],2400),'formal fixed scope')
            gate=inv['preflight'];check.exact(sha(gate['path']),gate['sha256'],'preflight hash');check.check(read(gate['path'])['passed'],'preflight passed')
        else:check.exact((inv['seeds'],inv['partitions'],inv['updates']),([99523],[1],40),'development fixed scope')
        expected_times=sorted({0,inv['updates'],*[t for t in (0,100,600,1200,2100,2400) if t<inv['updates']]})
        ref_analysis=read(reference/'analysis.json');ref_audit=read(reference/'audit_execution.json');ref_complete=read(reference/'training_complete.json')
        check.check(ref_audit['passed'],'prior independent audit passed')
        check.exact(ref_audit['analysis_sha256'],sha(reference/'analysis.json'),'prior audit binds reused analysis')
        check.exact(ref_audit['training_complete_sha256'],sha(reference/'training_complete.json'),'prior audit binds source completion')
        check.exact(ref_analysis['training_complete_sha256'],sha(reference/'training_complete.json'),'prior analysis binds source completion')
        check.exact((ref_analysis['seeds'],ref_analysis['partitions'],ref_analysis['times']),(inv['seeds'],inv['partitions'],expected_times),'reference paired scope and times')
        bank=experience.ImageBank();tables={split:experience.table(bank,split) for split in ('train','test')}
        for split,w in tables.items():
            check.exact(w,npz(out/f'{split}_worlds.npz'),'independent '+split+' world enumeration')
            check.exact(w,npz(reference/f'{split}_worlds.npz'),'unchanged reference world table')
        rows=[]
        for seed,p in itertools.product(inv['seeds'],inv['partitions']):
            cache=reused_cache(seed,p,reference,tables,check)
            row=case(seed,p,reference,out,cache,tables,seed==(33101 if inv['formal'] else 99523) and p==1,ref_complete['files'],check)
            rows.append(row);check.exact([r['update'] for r in row['curve']],expected_times,'fixed new checkpoint times')
            print(json.dumps(dict(audited_seed=seed,partition=p,checks=check.count)),flush=True)
        prior_rows=[dict(copy.deepcopy(r),is_reference=True) for r in ref_analysis['rows'] if r['condition'] in ('mean','attention')]
        check.exact(len(prior_rows),complete['reused_reference_pairs'],'declared reused reference rows')
        check.exact(check.scope_counts['new_social_pairs'],complete['social_runs'],'complete new social count')
        check.exact(check.scope_counts['social_updates_stream_checked'],complete['pair_updates'],'complete external update count')
        check.exact(complete['messages'],complete['pair_updates']*512,'new message budget');check.exact(complete['actions'],complete['pair_updates']*1024,'new action budget')
        reference_hashes={str(reference/name):sha(reference/name) for name in ('analysis.json','audit_execution.json','training_complete.json')}
        analysis=dict(status='complete',formal=inv['formal'],seeds=inv['seeds'],partitions=inv['partitions'],times=expected_times,updates=inv['updates'],
            analysis_source_sha256=sha(__file__),analysis_dependency_sha256=dependencies,training_complete_sha256=sha(out/'training_complete.json'),
            reference_artifact_sha256=reference_hashes,**summarize(prior_rows+rows,inv['seeds'],expected_times),
            source_private_capability_reused=ref_analysis['source_private_capability'],new_social_pairs=complete['social_runs'],reused_reference_pairs=complete['reused_reference_pairs'],
            boundaries=['Primary old18 J joint-minus-reused-mean at2400; new12 auxiliary; attention only a descriptive historical reference',
                'Four inherited source seeds; three partitions and two directions nested within a source',
                'Zero contrast preserves the initial policy function; it does not preserve gradients or future optimization paths',
                'The new sum/difference path permits distinct learned linear readouts of the two encoded role vectors; no attention mechanism',
                'Same stored parameter count is not matched effective capacity: the prior mean bilinear gradients are zero',
                'Private-head probabilities and role identity reused unchanged; no true locations supplied to sender or receiver',
                'No conclusion about compositional grammar from endpoint performance alone; source images remain the existing finite pool'])
        write(out/'analysis.json',analysis)
        write(out/'comparison.json',dict(passed=True,comparisons=check.comparisons,max_absolute_error=check.max_error,analysis_sha256=sha(out/'analysis.json'),
            production_metrics_called=False,scope='All new-arm raw protocol curve and endpoint statistics. Prior reference rows reused from their bound successful independent analysis.'))
        audit.update(passed=True,formal=inv['formal'],checks=check.count,counts=check.scope_counts,failures=[],analysis_sha256=sha(out/'analysis.json'),
            reference_artifact_sha256=reference_hashes,training_complete_sha256=sha(out/'training_complete.json'),seconds=time.monotonic()-started)
        write(out/'audit_execution.json',audit)
        print(json.dumps(dict(passed=True,checks=check.count,comparisons=check.comparisons,max_error=check.max_error,primary_mean=analysis['primary_mean'],counts=check.scope_counts,seconds=time.monotonic()-started)),flush=True)
    except Exception as exc:
        audit.update(passed=False,checks=check.count,counts=check.scope_counts,failures=[str(exc)],traceback=traceback.format_exc(),seconds=time.monotonic()-started)
        write(out/'analysis_failure.json',audit);raise


if __name__=='__main__':main()
