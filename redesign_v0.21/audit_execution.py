"""Read-only execution audit: full identity tables, bounded model/update replays.

Does not import run_social or call perform_update/direction_loss. Reuses the
CampAgent atomic modules and the frozen temporal observe_sequence operation.
No performance-based selection and no continuation of a trained trajectory.
"""
from __future__ import annotations
import argparse,copy,hashlib,itertools,json,shutil,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.8'))
from camp import ImageBank,remake_agents
sys.path.insert(0,str(PROJECT/'redesign_v0.20'))
from temporal_model import observe_sequence

SEND=('send_context','send_embedding','send_recur','send_out')
RECV=('receive_embedding','actor')
ALL=SEND+RECV+('send_value','receive_value')
MAPS=np.asarray(list(itertools.permutations(range(6),2)),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
CHUNK=256

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def arrays_sha(data):
    h=hashlib.sha256()
    for k in sorted(data):
        a=np.ascontiguousarray(data[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def state_sha(state):return arrays_sha({k:v.detach().numpy() for k,v in state.items()})
def groups(a):return {role:[p for name in names for p in getattr(a,name).parameters()] for role,names in [('sender',SEND),('receiver',RECV)]}
def frozen(state):return {k:v for k,v in state.items() if k.split('.')[0] not in SEND+RECV}
def ss(seed,p,d,purpose,step=0):return np.random.SeedSequence([21021,seed,p,d,purpose,step])
def comm_seed(seed,p,d):return int(ss(seed,p,d,0).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))
def fixture(seed,p,d,step,n):
    return (np.random.default_rng(ss(seed,p,d,1,step)).integers(n,size=256),
            np.random.default_rng(ss(seed,p,d,2,step)).random((256,4)).astype(np.float32))


class Audit:
    def __init__(self):self.checks=0;self.counts={};self.details=[]
    def check(self,ok,label):
        self.checks+=1
        if not bool(ok):raise AssertionError(label)
    def eq(self,a,b,label):
        if isinstance(a,torch.Tensor):
            self.check(isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),label)
        elif isinstance(a,np.ndarray):
            self.check(isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),label)
        elif isinstance(a,dict):
            self.check(isinstance(b,dict) and set(a)==set(b),label+'/keys')
            for k in a:self.eq(a[k],b[k],label+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            self.check(isinstance(b,(list,tuple)) and len(a)==len(b),label+'/length')
            for i,(x,y) in enumerate(zip(a,b)):self.eq(x,y,label+'/'+str(i))
        else:self.check(a==b,label)
    def count(self,name,n=1):self.counts[name]=self.counts.get(name,0)+n


def source(seed,p):
    base=PROJECT/'redesign_v0.20/results'/('smoke_002' if seed==99520 else 'temporal_001')
    return base,[base/f's{seed}_p{p}_d{d}_full/final.pt' for d in (0,1)]


def restore(seed,p,audit):
    base,paths=source(seed,p);prepared=load(base/f'prepared_{seed}.pt')
    agents=remake_agents(seed,prepared,7,2,'identity')
    for d,(a,path) in enumerate(zip(agents,paths)):
        source_state=load(path)['agent'];a.load_state_dict(source_state)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(comm_seed(seed,p,d))
            for name in ALL:
                for module in getattr(a,name).modules():
                    if hasattr(module,'reset_parameters'):module.reset_parameters()
        a.requires_grad_(False)
        for params in groups(a).values():
            for param in params:param.requires_grad_(True)
        for name,param in a.named_parameters():audit.check(param.requires_grad==(name.split('.')[0] in SEND+RECV),'flags/'+name)
        for name,value in a.state_dict().items():
            if name.split('.')[0] not in ALL:audit.eq(value,source_state[name],'inherited frontend/'+name)
    audit.check(not(set(map(id,agents[0].parameters()))&set(map(id,agents[1].parameters()))),'private parameter objects')
    return agents


def tables(bank,p):
    def matching(i):
        pairs={z for a,b in MATCHINGS[i] for z in ((a,b),(b,a))}
        return {i for i,pos in enumerate(MAPS) if tuple(pos) in pairs}
    old=set(range(30))-matching(p-1)-matching(p%3)
    output={}
    for split in ('train','test'):
        events=[]
        for a,b,m in itertools.product(sorted(old),sorted(old) if split=='train' else range(30),(0,1)):
            initial,terminal=MAPS[a],MAPS[b]
            if initial[1-m]==terminal[1-m] and terminal[m] not in initial:
                repeats=1 if split=='train' else (3 if b in old else 2)
                events.extend([(a,b,m)]*repeats)
        event_array=np.asarray(events,np.int64)
        photos=np.asarray(list(itertools.product(*[np.sort(bank.pools[split,k]) if split=='train' else np.sort(bank.pools[split,k])[:4] for k in (0,1)])),np.int64)
        r=np.repeat(event_array,len(photos),axis=0)
        output[split]=dict(source_map=r[:,0],target_map=r[:,1],mover=r[:,2],positions=MAPS[r[:,1]],photo_ids=np.tile(photos,(len(events),1)))
    return output


def frames(projected,w,mode):
    n=len(w['mover']);row=torch.arange(n);ids=w['photo_ids']
    def full(pos):
        pixels=projected.new_zeros(n,6,64);exists=projected.new_zeros(n,6)
        for k in (0,1):
            site=torch.from_numpy(pos[:,k]);pixels[row,site]=projected[torch.from_numpy(ids[:,k])];exists[row,site]=1.
        return torch.cat((pixels.flatten(1),exists),-1)
    first=full(MAPS[w['source_map']])
    if mode=='immediate':second=full(w['positions'])
    else:
        m=w['mover'];site=torch.from_numpy(w['positions'][np.arange(n),m])
        pixels=projected.new_zeros(n,6,64);exists=projected.new_zeros(n,6)
        pixels[row,site]=projected[torch.from_numpy(ids[np.arange(n),m])];exists[row,site]=1.
        second=torch.cat((pixels.flatten(1),exists),-1)
    bits=projected.new_ones(n,2);bits[:,1]=float(mode=='immediate')
    return torch.stack((first,second),1),bits


def blocks(n):return [(lo,min(lo+CHUNK,n)) for lo in sorted({0,((n-1)//CHUNK)*CHUNK})]


def audit_cache(folder,seed,p,bank,audit):
    manifest=read(folder/'cache_manifest.json');expected=tables(bank,p)
    audit.eq(manifest['chunk'],CHUNK,'cache chunk')
    for mapping in ('source_states','source_prepared'):
        for path,h in manifest[mapping].items():audit.eq(sha(path),h,'cache source hash')
    for name,h in manifest['files'].items():audit.eq(sha(folder/name),h,'cache file hash')
    for split,w in expected.items():
        audit.eq(npz(folder/f'{split}_worlds.npz'),w,'independent '+split+' worlds')
        audit.eq(len(w['mover']),manifest[f'{split}_rows'],'cache rows')
        audit.count('cache_world_rows',len(w['mover']))
    agents=restore(seed,p,audit);cached={}
    for d,a in enumerate(agents):
        with torch.no_grad():projected=a.project(bank.features).detach()
        audit.eq(projected.numpy(),np.load(folder/f'projected_d{d}.npy'),'all cached projected features')
        for split,w in expected.items():
            for mode in (('immediate','delayed') if split=='train' else ('immediate','delayed','erase')):
                h=np.load(folder/f'{split}_d{d}_{mode}.npy');cached[split,d,mode]=torch.from_numpy(h)
                audit.check(h.shape==(len(w['mover']),96) and h.dtype==np.float32 and np.isfinite(h).all(),'cached h layout')
                for lo,hi in blocks(len(h)):
                    sub={k:v[lo:hi] for k,v in w.items()}
                    with torch.no_grad():
                        frame,bit=frames(projected,sub,mode)
                        got=observe_sequence(a,frame,bit,'full',erase_history=mode=='erase').numpy()
                    audit.eq(got,h[lo:hi],'original-chunk frontend replay')
                    audit.count('frontend_worlds_replayed',hi-lo)
    return expected,cached,agents


def sender_start(a,h):
    state=a.send_context(torch.cat((h,h.new_zeros(len(h),4)),-1))
    return state,a.send_out(state)
def second_logits(a,state,first):return a.send_out(a.send_recur(a.send_embedding(first.detach()),state))
def rlogits(a,messages):
    emb=a.receive_embedding(messages.detach()).flatten(1)
    return a.actor(torch.cat((emb,emb.new_zeros(len(emb),20)),-1)).reshape(len(emb),2,6)
def enumerate_receiver(a):
    code=torch.arange(49,dtype=torch.int64)
    return rlogits(a,torch.stack((code//7,code%7),1))
def enumerate_sender(a,h):
    state,fl=sender_start(a,h);f=F.log_softmax(fl,-1)
    s=torch.stack([F.log_softmax(second_logits(a,state,torch.full((len(h),),i,dtype=torch.int64)),-1) for i in range(7)],1)
    first=fl.argmax(-1);second=second_logits(a,state,first).argmax(-1)
    return (f[:,:,None]+s).reshape(len(h),49),torch.stack((first,second),1)


def independent_direction(sender,receiver,h,u,pos,weight):
    """Independent constant-baseline loss; no production loss helper called."""
    def draw(logits,uniform):
        lp=F.log_softmax(logits,-1);prob=lp.exp()
        action=(prob.detach().cumsum(-1)<uniform).sum(-1).clamp(max=logits.shape[-1]-1).detach()
        return action,lp.gather(1,action[:,None]).squeeze(1),-(prob*lp).sum(-1),prob
    state,fl=sender_start(sender,h);ut=torch.from_numpy(u)
    t0,l0,e0,p0=draw(fl,ut[:,0:1]);sl=second_logits(sender,state,t0)
    t1,l1,e1,p1=draw(sl,ut[:,1:2]);tokens=torch.stack((t0,t1),1).detach()
    rl=rlogits(receiver,tokens)
    af,lf,ef,pf=draw(rl[:,0],ut[:,2:3]);aw,lw,ew,pw=draw(rl[:,1],ut[:,3:4])
    action=torch.stack((af,aw),1);success=(action.numpy()==pos).astype(np.float32)
    reward=(.25*success.sum(1)+.5*success.prod(1)).astype(np.float32);adv=torch.from_numpy(reward-np.float32(.5))
    log_s=l0+l1;log_r=lf+lw;ent_s=e0+e1;ent_r=ef+ew
    loss_s=-(log_s*adv).mean()-weight*ent_s.mean()
    loss_r=-(log_r*adv).mean()-weight*ent_r.mean()
    ar=lambda x:x.detach().numpy().copy()
    trace=dict(h=ar(h),uniforms=u.copy(),messages=ar(tokens),first_logits=ar(fl),second_logits=ar(sl),
        token_probabilities=ar(torch.stack((p0,p1),1)),action_logits=ar(rl),action_probabilities=ar(torch.stack((pf,pw),1)),
        actions=ar(action),positions=pos.copy(),success=success,reward=reward,advantage=ar(adv),
        sender_logp=ar(log_s),receiver_logp=ar(log_r),sender_entropy=ar(ent_s),receiver_entropy=ar(ent_r),
        sender_loss=float(loss_s.detach()),receiver_loss=float(loss_r.detach()),entropy_weight=float(weight),baseline=.5)
    return loss_s,loss_r,{k:np.asarray(v) for k,v in trace.items()}


def replay_update(folder,agents,cache,w,cfg,step,log,audit):
    before=load(folder/f'checkpoint_{step:04d}.pt');before_opt=load(folder/f'optimizer_{step:04d}.pt')
    opts=[]
    for a,state,os in zip(agents,before,before_opt):
        a.load_state_dict(state)
        opt=torch.optim.Adam([x for g in groups(a).values() for x in g],lr=.0007);opt.load_state_dict(os);opts.append(opt)
    expected=npz(folder/f'train_{step+1:04d}.npz');losses=[{},{}];traces={};world={}
    for d in (0,1):
        indices,u=fixture(cfg['seed'],cfg['partition'],d,step,len(w['mover']))
        world[f'd{d}_indices']=indices;world[f'd{d}_uniforms']=u
        ls,lr,tr=independent_direction(agents[d],agents[1-d],cache['train',d,cfg['mode']][indices],u,w['positions'][indices],.02 if step<2100 else 0.)
        losses[d]['sender']=ls;losses[1-d]['receiver']=lr
        traces.update({f'd{d}__{k}':v for k,v in tr.items()})
    audit.eq({**world,**traces},expected,'saved critical-step trace')
    audit.eq(arrays_sha(world),log['world_sha256'],'key world hash')
    audit.eq(arrays_sha(traces),log['trace_sha256'],'key complete trace hash')
    gradient_hashes=[]
    for who,(a,opt) in enumerate(zip(agents,opts)):
        opt.zero_grad(set_to_none=True);loss=(losses[who]['sender']+losses[who]['receiver'])/2;loss.backward()
        norms={};raw_grad={}
        for role,params in groups(a).items():
            audit.check(all(x.grad is not None and torch.isfinite(x.grad).all() for x in params),'finite role gradients')
            raw_grad.update({f'{role}_{i}':x.grad.detach().numpy() for i,x in enumerate(params)})
        gradient_hashes.append(arrays_sha(raw_grad))
        for role,params in groups(a).items():norms[role]=float(torch.nn.utils.clip_grad_norm_(params,2.))
        audit.eq(dict(loss=float(loss.detach()),norms=norms),log['people'][who],'role loss/clip exact')
        audit.check(all(x.grad is None for x in a.parameters() if not x.requires_grad),'frozen no gradients')
    for opt in opts:opt.step()
    audit.eq([a.state_dict() for a in agents],load(folder/f'after_{step+1:04d}.pt'),'critical-step parameters exact')
    audit.eq([o.state_dict() for o in opts],load(folder/f'after_{step+1:04d}_optimizer.pt'),'critical-step Adam exact')
    audit.count('key_pair_updates_replayed');audit.count('key_person_updates_replayed',2)
    return dict(update=step+1,raw_gradient_sha256=gradient_hashes,scope='independently computed gradients; exact saved trace, loss/norm, resulting parameters and Adam; raw gradients not separately stored by runner')


def audit_run(folder,seed,p,mode,cache,table,initial_agents,audit):
    cfg=read(folder/'config.json');result=read(folder/'result.json')
    for k,v in dict(seed=seed,partition=p,mode=mode,batch_per_direction=256,constant_baseline=.5,learning_rate=.0007,clip_each_role=2.).items():audit.eq(cfg[k],v,'config/'+k)
    audit.eq(result['status'],'complete','run complete')
    times=sorted({0,cfg['updates'],*[t for t in [0,100,300,600,1200,1800,2100,2400] if t<cfg['updates']]})
    audit.eq(cfg['checkpoints'],times,'fixed checkpoints')
    initial=load(folder/'initial.pt');audit.eq(initial,[a.state_dict() for a in initial_agents],'independent communication init')
    audit.eq([state_sha(s) for s in initial],cfg['initial_hashes'],'initial fingerprints')
    audit.eq([state_sha(frozen(s)) for s in initial],cfg['frozen_hashes'],'frozen fingerprints')
    for who in (0,1):audit.eq(cfg['init'][who]['seed'],comm_seed(seed,p,who),'initial RNG identity')
    for path,h in cfg['source_states'].items():audit.eq(sha(path),h,'run source hash')
    logs=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    audit.eq(len(logs),cfg['updates'],'all update logs present')
    world_hashes=[]
    for step,log in enumerate(logs):
        world={}
        for d in (0,1):
            idx,u=fixture(seed,p,d,step,len(table['train']['mover']))
            world[f'd{d}_indices']=idx;world[f'd{d}_uniforms']=u
        audit.eq(log['update'],step+1,'log step')
        audit.eq(log['world_sha256'],arrays_sha(world),'independent full world/uniform stream')
        audit.check(all(np.isfinite([person['loss'],*person['norms'].values()]).all() for person in log['people']),'finite logged losses/norms')
        world_hashes.append(log['world_sha256']);audit.count('pair_updates_stream_checked')
    for step in times:
        states=load(folder/f'checkpoint_{step:04d}.pt')
        for who,state in enumerate(states):audit.eq(frozen(state),frozen(initial[who]),'checkpoint frozen tensors')
        audit.count('pair_checkpoints_checked')
    audit.eq(load(folder/'final.pt'),load(folder/f'checkpoint_{cfg["updates"]:04d}.pt'),'final checkpoint identity')
    audit.eq(load(folder/'final_optimizer.pt'),load(folder/f'optimizer_{cfg["updates"]:04d}.pt'),'final optimizer identity')
    audit.eq(load(folder/'initial_optimizer.pt'),load(folder/'optimizer_0000.pt'),'initial optimizer identity')
    for o in load(folder/'initial_optimizer.pt'):audit.eq(o['state'],{},'fresh Adam state')
    expected_names={f'protocol_{step:04d}_normal_d{d}.npz' for step in times for d in (0,1)}
    expected_names|={f'protocol_{cfg["updates"]:04d}_{label}_d{d}.npz' for label in ('cross_view','erase') for d in (0,1)}
    audit.eq({path.name for path in folder.glob('protocol_*.npz')},expected_names,'evaluation inventory')
    for name in sorted(expected_names):
        _,st,labelpart=name.split('_',2);label,dirpart=labelpart.rsplit('_d',1);d=int(dirpart.split('.')[0]);step=int(st)
        states=load(folder/f'checkpoint_{step:04d}.pt')
        for a,state in zip(initial_agents,states):a.load_state_dict(state)
        raw=npz(folder/name)
        audit.eq({k:raw[k] for k in table['test']},table['test'],'all evaluation worlds')
        lp=raw['sender_log_probs'];tokens=raw['tokens'];rr=raw['receiver_logits'];n=len(table['test']['mover'])
        audit.check(lp.shape==(n,49) and lp.dtype==np.float32 and np.isfinite(lp).all(),'all sender probability layout')
        audit.check(tokens.shape==(n,2) and tokens.dtype==np.int64 and ((tokens>=0)&(tokens<7)).all(),'all message bounds')
        audit.check(np.max(np.abs(np.exp(lp.astype(np.float64)).sum(1)-1))<1e-6,'all sender normalization')
        observation=mode if label=='normal' else ('erase' if label=='erase' else ('delayed' if mode=='immediate' else 'immediate'))
        with torch.no_grad():audit.eq(enumerate_receiver(initial_agents[1-d]).numpy(),rr,'all49 receiver table exact')
        for lo,hi in blocks(n):
            with torch.no_grad():got_lp,got_tokens=enumerate_sender(initial_agents[d],cache['test',d,observation][lo:hi])
            audit.eq(got_lp.numpy(),lp[lo:hi],'original-chunk sender lp exact')
            audit.eq(got_tokens.numpy(),tokens[lo:hi],'original-chunk sequential greedy exact')
            audit.count('evaluation_sender_worlds_replayed',hi-lo)
        audit.count('evaluation_tables_checked');audit.count('evaluation_world_rows_checked',n)
    key_replays=[]
    for step in (0,2100):
        if step<cfg['updates']:key_replays.append(replay_update(folder,initial_agents,cache,table['train'],cfg,step,logs[step],audit))
    audit.count('runs')
    return dict(name=folder.name,updates=cfg['updates'],key_replays=key_replays),world_hashes,initial


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve();started=time.monotonic();audit=Audit();torch.set_num_threads(1)
    output=out/'audit_execution.json';source_copy=out/'audit_execution_source.py'
    if output.exists():
        history=out/'audit_history';history.mkdir(exist_ok=True);tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        shutil.copy2(output,history/f'{tag}_audit_execution.json')
        if source_copy.exists():shutil.copy2(source_copy,history/f'{tag}_audit_execution_source.py')
    shutil.copy2(__file__,source_copy)
    receipt=dict(passed=False,created_utc=datetime.now(timezone.utc).isoformat(),audit_source_sha256=sha(__file__),
        scope=['All declared source/input/output bytes; independent complete legal world tables and full external RNG streams',
               'Independent communication initialization and all checkpoint frozen tensors',
               'Cached h replays on original first/last chunks only; reuses frozen temporal observe_sequence',
               'Each saved evaluation: complete metadata/probability validity, all49 receiver rows, first/last sender chunks',
               'Updates1/2101: independently written loss, gradients, role clip and Adam; no production loss/update helper',
               'No full-trajectory parameter replay; no independent replay of every cached or evaluated h; no performance-based selection'])
    try:
        complete=read(out/'training_complete.json');inv=read(out/'invocation.json')
        audit.eq(complete['status'],'complete','completion gate')
        receipt.update(formal=inv['formal'],source_hashes=inv['source_hashes'],input_hashes=inv['input_hashes'],
            invocation_sha256=sha(out/'invocation.json'),training_complete_sha256=sha(out/'training_complete.json'))
        for kind in ('source_hashes','input_hashes'):
            audit.eq(complete[kind],inv[kind],'complete '+kind)
            for path,h in inv[kind].items():audit.eq(sha(path),h,'declared '+kind)
        for path,h in inv['source_hashes'].items():audit.eq(sha(out/'frozen_sources'/Path(path).relative_to(PROJECT)),h,'archived production source')
        for name,h in complete['files'].items():audit.eq(sha(out/name),h,'complete-file hash')
        expected={f's{seed}_p{p}_{m}' for seed,p,m in itertools.product(inv['seeds'],inv['partitions'],('immediate','delayed'))}
        audit.eq(set(complete['runs']),expected,'complete run inventory');audit.eq(complete['social_runs'],len(expected),'run count')
        audit.eq(inv['threads'],1,'numerical thread contract')
        audit.eq(inv['torch_version'],str(torch.__version__),'Torch version');audit.eq(inv['numpy_version'],np.__version__,'NumPy version')
        if inv['formal']:audit.eq((inv['seeds'],inv['partitions'],inv['updates']),([32101,32102,32103,32104],[1,2,3],2400),'formal fixed scope')
        bank=ImageBank();seed_identities=[]
        for seed,p in itertools.product(inv['seeds'],inv['partitions']):
            cache_path=out/f'cache_s{seed}_p{p}';table,cache,agents=audit_cache(cache_path,seed,p,bank,audit)
            for d in (0,1):seed_identities.append(comm_seed(seed,p,d))
            paired=[]
            for mode in ('immediate','delayed'):
                # Restore exact common initial state; prior evaluation/update
                # replay touched only these disposable in-memory model objects.
                agents=restore(seed,p,audit)
                detail,streams,initial=audit_run(out/f's{seed}_p{p}_{mode}',seed,p,mode,cache,table,agents,audit)
                audit.details.append(detail);paired.append((streams,initial))
                print(json.dumps(dict(audited=detail['name'],checks=audit.checks)),flush=True)
            audit.eq(paired[0][0],paired[1][0],'paired external streams');audit.eq(paired[0][1],paired[1][1],'paired model initialization')
        audit.check(len(set(seed_identities))==len(seed_identities),'unique explicit communication seeds')
        receipt.update(passed=True,failures=[],checks=audit.checks,counts=audit.counts,runs=audit.details)
    except Exception as exc:
        receipt.update(checks=audit.checks,counts=audit.counts,failures=[str(exc)],traceback=traceback.format_exc(),runs=audit.details)
        receipt['seconds']=time.monotonic()-started;write(output,receipt);raise
    receipt['seconds']=time.monotonic()-started;write(output,receipt)
    print(json.dumps(dict(passed=True,checks=audit.checks,counts=audit.counts,seconds=receipt['seconds'],path=str(output))),flush=True)


if __name__=='__main__':main()
