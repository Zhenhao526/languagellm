"""Independent finite-support visibility statistics and complete frozen replay.

No run_probe statistics/aggregate calls. The complete 480-row support is
replayed in original256/224 chunks. Old independent sender enumeration reuses
only CampAgent atomic modules; temporal observation is the frozen v20 helper.
"""
from __future__ import annotations
import argparse,copy,hashlib,itertools,json,math,shutil,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'redesign_v0.8'))
from camp import ImageBank,remake_agents
sys.path.insert(0,str(PROJECT/'redesign_v0.20'))
from temporal_model import observe_sequence
sys.path.insert(0,str(PROJECT/'redesign_v0.21'))
from audit_execution import enumerate_sender,enumerate_receiver

MASKS=('food_only','water_only');SYSTEMS=('private','social_immediate','social_delayed','social_all')
MAPS=np.asarray(list(itertools.permutations(range(6),2)),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
CHUNK=256
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def npz(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def fingerprint(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=np.ascontiguousarray(v.detach().numpy());h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def groups(p):
    def one(i):
        pair={t for a,b in MATCHINGS[i] for t in ((a,b),(b,a))}
        return np.asarray([j for j,t in enumerate(MAPS) if tuple(t) in pair],np.int64)
    added,sealed=one(p-1),one(p%3)
    return dict(old=np.asarray(sorted(set(range(30))-set(added)-set(sealed))),added=added,sealed=sealed,common30=np.arange(30))
def paths(seed,p):
    private=PROJECT/'redesign_v0.20/results'/('smoke_002' if seed==99520 else 'temporal_001')
    social=PROJECT/'redesign_v0.21/results'/('smoke_002' if seed==99520 else 'communication_001')
    return private,social


class Checks:
    def __init__(self):self.count=0;self.comparisons=0;self.max_error=0.;self.scope_counts={}
    def check(self,b,label):
        self.count+=1
        if not bool(b):raise AssertionError(label)
    def exact(self,a,b,label):
        if isinstance(a,torch.Tensor):self.check(isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),label)
        elif isinstance(a,np.ndarray):self.check(isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),label)
        elif isinstance(a,dict):
            self.check(isinstance(b,dict) and set(a)==set(b),label+'/keys')
            for k in a:self.exact(a[k],b[k],label+'/'+str(k))
        elif isinstance(a,(tuple,list)):
            self.check(len(a)==len(b),label+'/length')
            for i,(x,y) in enumerate(zip(a,b)):self.exact(x,y,label+'/'+str(i))
        else:self.check(a==b,label)
    def close(self,a,b,label):
        if isinstance(a,dict):
            self.check(isinstance(b,dict) and set(a)==set(b),label+'/keys')
            for k in a:self.close(a[k],b[k],label+'/'+str(k))
        elif isinstance(a,list):
            self.check(isinstance(b,list) and len(a)==len(b),label+'/length')
            for i,(x,y) in enumerate(zip(a,b)):self.close(x,y,label+'/'+str(i))
        elif isinstance(a,(float,int)) and not isinstance(a,bool):
            self.comparisons+=1;error=abs(a-b);self.max_error=max(self.max_error,error)
            self.check(error<=1e-12,label+f' ({error})')
        else:self.exact(a,b,label)
    def add(self,k,n=1):self.scope_counts[k]=self.scope_counts.get(k,0)+n


def world_table(bank):
    photos=np.asarray(list(itertools.product(np.sort(bank.pools['test',0])[:4],np.sort(bank.pools['test',1])[:4])),np.int64)
    ids=np.repeat(np.arange(30,dtype=np.int64),len(photos))
    return dict(map_id=ids,positions=MAPS[ids],photo_ids=np.tile(photos,(30,1)))


def render(projected,w,kind):
    n=len(w['map_id']);rows=torch.arange(n)
    frames=[]
    for which in ((0,1),(kind,)):
        visual=projected.new_zeros(n,6,64);exists=projected.new_zeros(n,6)
        for k in which:
            loc=torch.from_numpy(w['positions'][:,k]);photo=torch.from_numpy(w['photo_ids'][:,k])
            visual[rows,loc]=projected[photo];exists[rows,loc]=1.
        frames.append(torch.cat((visual.flatten(1),exists),-1))
    bits=projected.new_zeros(n,2);bits[:,0]=1.
    return torch.stack(frames,1),bits


def probability(logits):
    x=np.asarray(logits,np.float64)
    shifted=x-x.max(axis=-1,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(axis=-1,keepdims=True))
    return np.exp(logp)


def scores(raw,p,private):
    """Direct per-world counts; no production metric functions imported."""
    pos=raw['positions'];n=len(pos);good={};action={};joint_prob={};code={}
    for mask in MASKS:
        if private:
            logits=raw[mask+'_logits'];action[mask]=np.argmax(logits,axis=2)
            q=probability(logits);joint_prob[mask]=q[np.arange(n),0,pos[:,0]]*q[np.arange(n),1,pos[:,1]]
        else:
            tokens=raw[mask+'_tokens'];code[mask]=tokens@np.asarray([7,1],np.int64)
            r=raw['receiver_logits'];action[mask]=np.argmax(r,axis=2)[code[mask]]
            rp=probability(r);sp=probability(raw[mask+'_sender_log_probs'])
            # A single message shared across two independent target actions.
            joint_prob[mask]=np.asarray([np.dot(sp[i],rp[:,0,int(f)]*rp[:,1,int(w)]) for i,(f,w) in enumerate(pos)])
        good[mask]=action[mask]==pos
    result={}
    for group,maps in groups(p).items():
        ix=np.flatnonzero(np.isin(raw['map_id'],maps));count=len(ix);entry={}
        for mask in MASKS:
            correct=good[mask][ix]
            entry[mask]=dict(J=float(np.count_nonzero(correct.all(1))/count),food=float(np.count_nonzero(correct[:,0])/count),
                water=float(np.count_nonzero(correct[:,1])/count),Q=float(np.sum(joint_prob[mask][ix],dtype=np.float64)/count))
        f,w=entry[MASKS[0]],entry[MASKS[1]];fe=f['food']-w['food'];we=w['water']-f['water']
        entry.update(D_visible=(fe+we)/2,food_visibility_effect=fe,water_visibility_effect=we,
            visible_accuracy=(f['food']+w['water'])/2,hidden_accuracy=(w['food']+f['water'])/2,
            category_gap_food_only=f['food']-f['water'],category_gap_water_only=w['food']-w['water'],
            stable_correct_J=float(np.count_nonzero(good[MASKS[0]][ix].all(1)&good[MASKS[1]][ix].all(1))/count),
            action_change_rate=float(np.count_nonzero(np.any(action[MASKS[0]][ix]!=action[MASKS[1]][ix],axis=1))/count),
            message_change_rate=None if private else float(np.count_nonzero(code[MASKS[0]][ix]!=code[MASKS[1]][ix])/count),n=count)
        result[group]=entry
    return result


def average(values):
    first=values[0]
    if isinstance(first,dict):return {k:average([v[k] for v in values]) for k in first}
    if first is None:
        if not all(v is None for v in values):raise ValueError('inconsistent missing metric')
        return None
    return math.fsum(values)/len(values)


def summarize(rows,seeds):
    source=[]
    for seed,system in itertools.product(seeds,SYSTEMS):
        wanted=('social_immediate','social_delayed') if system=='social_all' else (system,)
        source.append(dict(seed=seed,system=system,scores=average([r['scores'] for r in rows if r['seed']==seed and r['system'] in wanted])))
    overall={system:average([r['scores'] for r in source if r['system']==system]) for system in SYSTEMS}
    primary=[dict(seed=r['seed'],D_visible=r['scores']['common30']['D_visible'],
        category_gap_food_only=r['scores']['common30']['category_gap_food_only'],category_gap_water_only=r['scores']['common30']['category_gap_water_only'])
        for r in source if r['system']=='social_all']
    return dict(rows=rows,seed_rows=source,aggregate=overall,primary=primary,primary_mean=overall['social_all']['common30']['D_visible'])


@torch.no_grad()
def private_case(seed,p,d,bank,w,out,check):
    base,_=paths(seed,p);source=base/f's{seed}_p{p}_d{d}_full/final.pt';blob=load(source)
    a=remake_agents(seed,load(base/f'prepared_{seed}.pt'),7,2,'identity')[d];a.load_state_dict(blob['agent']);a.requires_grad_(False)
    # Exact original architecture. Initialization is discarded by strict load.
    with torch.random.fork_rng(devices=[]):head=nn.Sequential(nn.Linear(96,96),nn.Tanh(),nn.Linear(96,12))
    head.load_state_dict(blob['head']);head.requires_grad_(False)
    before=(fingerprint(a.state_dict()),fingerprint(head.state_dict()));projected=a.project(bank.features).detach()
    name=f'private_s{seed}_p{p}_d{d}.npz';raw=npz(out/name)
    check.exact({k:raw[k] for k in w},w,'private complete world table')
    for kind,mask in enumerate(MASKS):
        for lo in (0,256):
            sub={k:v[lo:lo+256] for k,v in w.items()};f,b=render(projected,sub,kind)
            h=observe_sequence(a,f,b,'full');logits=head(h).reshape(-1,2,6)
            check.exact(h.numpy(),raw[mask+'_h'][lo:lo+256],'private full h replay')
            check.exact(logits.numpy(),raw[mask+'_logits'][lo:lo+256],'private full logit replay')
            check.add('private_world_forwards_replayed',len(h))
    check.exact(before,(fingerprint(a.state_dict()),fingerprint(head.state_dict())),'private unchanged after replay')
    check.check(all(not q.requires_grad and q.grad is None for q in list(a.parameters())+list(head.parameters())),'private frozen flags/no gradients')
    check.add('private_heads')
    return dict(seed=seed,partition=p,direction=d,system='private',raw=name,source=str(source),source_sha256=sha(source),
        frozen_state_sha256=before[0],frozen_head_sha256=before[1],scores=scores(raw,p,True)),raw,blob['agent']


@torch.no_grad()
def social_cases(seed,p,mode,w,out,private_raw,private_state,check):
    base,communal=paths(seed,p);source=communal/f's{seed}_p{p}_{mode}/final.pt';states=load(source)
    agents=remake_agents(seed,load(base/f'prepared_{seed}.pt'),7,2,'identity')
    for d,(a,state) in enumerate(zip(agents,states)):
        a.load_state_dict(state);a.requires_grad_(False)
        for key,value in state.items():
            if key.split('.')[0] in ('project','memory','slot_phi','input_transform'):check.exact(value,private_state[d][key],'social inherited frontend')
    before=[fingerprint(a.state_dict()) for a in agents];result=[]
    for d in (0,1):
        name=f'social_{mode}_s{seed}_p{p}_d{d}.npz';raw=npz(out/name)
        check.exact({k:raw[k] for k in w},w,'social complete world table')
        check.exact(enumerate_receiver(agents[1-d]).numpy(),raw['receiver_logits'],'full49 receiver table')
        for mask in MASKS:
            h=torch.from_numpy(private_raw[d][mask+'_h'])
            for lo in (0,256):
                lp,tokens=enumerate_sender(agents[d],h[lo:lo+256])
                check.exact(lp.numpy(),raw[mask+'_sender_log_probs'][lo:lo+256],'full sender probabilities')
                check.exact(tokens.numpy(),raw[mask+'_tokens'][lo:lo+256],'full sequential greedy messages')
                check.add('social_sender_world_forwards_replayed',len(lp))
            check.check(np.max(np.abs(np.exp(raw[mask+'_sender_log_probs'].astype(np.float64)).sum(1)-1))<1e-6,'sender normalization')
        result.append(dict(seed=seed,partition=p,direction=d,system='social_'+mode,raw=name,source=str(source),source_sha256=sha(source),
            frozen_sender_sha256=before[d],frozen_receiver_sha256=before[1-d],scores=scores(raw,p,False)))
        check.add('social_directions')
    check.exact(before,[fingerprint(a.state_dict()) for a in agents],'social unchanged after complete replay')
    check.check(all(not q.requires_grad and q.grad is None for a in agents for q in a.parameters()),'social frozen flags/no gradients')
    return result


def self_test():
    """Uniform and perfect finite distributions; no model fits or forward calls."""
    ids=np.arange(30,dtype=np.int64);w=dict(map_id=ids,positions=MAPS.copy(),photo_ids=np.zeros((30,2),np.int64))
    raw=dict(w)
    for mask in MASKS:raw[mask+'_logits']=np.zeros((30,2,6),np.float32)
    for p in (1,2,3):
        s=scores(raw,p,True)['common30'];assert s['D_visible']==0 and abs(s['food_only']['Q']-1/36)<1e-14
    # Exactly one currently displayed goal is correct: D_visible=1, J=0.
    for kind,mask in enumerate(MASKS):
        act=MAPS[:,::-1].copy();act[:,kind]=MAPS[:,kind];z=np.full((30,2,6),-20.,np.float32)
        z[np.arange(30)[:,None],np.arange(2)[None,:],act]=20.;raw[mask+'_logits']=z
    s=scores(raw,1,True)['common30'];assert s['D_visible']==1 and s['food_only']['J']==0 and s['water_only']['J']==0
    assert s['food_visibility_effect']==s['water_visibility_effect']==1
    assert s['category_gap_food_only']==1 and s['category_gap_water_only']==-1
    # Same semantics under different synonym messages: message change is not
    # itself a correctness failure. Here all49 codes decode the same location.
    social=dict(w,receiver_logits=np.zeros((49,2,6),np.float32))
    for kind,mask in enumerate(MASKS):
        social[mask+'_sender_log_probs']=np.full((30,49),-math.log(49),np.float32)
        social[mask+'_tokens']=np.full((30,2),kind,np.int64)
    s=scores(social,1,False)['common30'];assert s['message_change_rate']==1 and s['action_change_rate']==0 and s['D_visible']==0
    assert abs(s['food_only']['Q']-1/36)<1e-14
    return dict(passed=True,training=False,checks=['uniform joint probability','balanced visible effects','category reversal','message change distinct from action/accuracy'])


def figures(out,data):
    # NumPy1.26 is already imported; the extra directory supplies matplotlib,
    # not the NumPy2 runtime bundled alongside some older plotting dependencies.
    sys.path.insert(0,str(PROJECT/'redesign_v0.9/.analysis_deps'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(10.6,4.1));seeds=data['seeds'];colors=plt.get_cmap('tab10').colors
    for index,seed in enumerate(seeds):
        sr={r['system']:r['scores']['common30'] for r in data['seed_rows'] if r['seed']==seed}
        axes[0].plot([0,1],[100*sr[s]['D_visible'] for s in ('private','social_all')],'o-',color=colors[index],alpha=.85,label=str(seed))
        axes[1].plot([0,1],[100*sr['social_all'][k] for k in ('food_visibility_effect','water_visibility_effect')],'o-',color=colors[index],alpha=.85)
    axes[0].set_xticks([0,1],['Private head\n(auxiliary)','Social: both training modes\n(primary)'])
    axes[1].set_xticks([0,1],['Food benefit when\nfood is shown','Water benefit when\nwater is shown'])
    axes[0].set_ylabel('Visible-resource effect (percentage points)');axes[1].set_ylabel('Resource-specific effect (percentage points)')
    for ax in axes:
        ax.axhline(0,color='.5',ls='--',lw=.8);ax.grid(axis='y',alpha=.2);ax.tick_params(axis='y',labelleft=True)
    axes[0].set_title('A  Fixed policies: visibility swap');axes[1].set_title('B  Two components of the social effect')
    axes[0].legend(title='Source seed',fontsize=8,title_fontsize=8)
    figure.text(.5,.02,'No movement; no training. Lines are source seeds; partitions, directions and social modes are nested.\nPrivate/social differences are descriptive, not an identified communication mediation effect.',ha='center',fontsize=8)
    figure.tight_layout(rect=(0,.10,1,1));folder=out/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf'):figure.savefig(folder/f'01_visibility_swap.{ext}',dpi=180,bbox_inches='tight')
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    out=args.out.resolve();started=time.monotonic();torch.set_num_threads(1);check=Checks()
    if (out/'audit_execution.json').exists():
        hist=out/'analysis_history'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f');hist.mkdir(parents=True)
        for name in ('audit_execution.json','analysis.json','comparison.json','analyze_probe_source.py'):
            if (out/name).exists():shutil.copy2(out/name,hist/name)
    shutil.copy2(__file__,out/'analyze_probe_source.py')
    deps={str(PROJECT/n):sha(PROJECT/n) for n in ('redesign_v0.21/audit_execution.py','redesign_v0.20/temporal_model.py','redesign_v0.8/camp.py')}
    receipt=dict(passed=False,analysis_source_sha256=sha(__file__),replay_dependency_sha256=deps,new_training=0,
        production_metrics_called=False,full_support_replay=True,raw_parameter_gradients_computed=False,
        scope='All480 rows in two original256/224 chunks per mask; independent rendering/metrics/aggregation; frozen temporal helper and old independent atomic sender/receiver enumeration reused')
    try:
        complete=read(out/'evaluation_complete.json');inv=read(out/'invocation.json')
        check.exact(complete['status'],'complete','whole-batch completion gate')
        receipt.update(formal=inv['formal'],evaluation_complete_sha256=sha(out/'evaluation_complete.json'))
        for kind in ('source_hashes','input_hashes'):
            check.exact(complete[kind],inv[kind],'completion '+kind)
            for path,h in inv[kind].items():check.exact(sha(path),h,'source/input hash')
        for path,h in inv['source_hashes'].items():check.exact(sha(out/'frozen_sources'/Path(path).relative_to(PROJECT)),h,'frozen source copy')
        for name,h in complete['files'].items():check.exact(sha(out/name),h,'complete output hash')
        check.exact(inv['chunk'],CHUNK,'numerical chunk');check.exact(inv['threads'],1,'single thread')
        check.exact(inv['all_parameter_requires_grad'],False,'all parameter flags false')
        check.exact(inv['no_grad'],True,'no_grad contract');check.exact(inv['new_training'],0,'no training')
        check.exact(inv['torch_version'],str(torch.__version__),'Torch version');check.exact(inv['numpy_version'],np.__version__,'NumPy version')
        if inv['formal']:check.exact((inv['seeds'],inv['partitions']),([32101,32102,32103,32104],[1,2,3]),'fixed formal sources')
        test=self_test();bank=ImageBank();w=world_table(bank);check.exact(npz(out/'worlds.npz'),w,'independent complete worlds')
        check.exact(len(w['map_id']),480,'finite support');rows=[];expected_raw=set()
        for seed,p in itertools.product(inv['seeds'],inv['partitions']):
            private_raw={};private_state={}
            for d in (0,1):
                row,private_raw[d],private_state[d]=private_case(seed,p,d,bank,w,out,check);rows.append(row);expected_raw.add(row['raw'])
            for mode in ('immediate','delayed'):
                added=social_cases(seed,p,mode,w,out,private_raw,private_state,check);rows.extend(added);expected_raw.update(r['raw'] for r in added)
            print(json.dumps(dict(audited_seed=seed,partition=p,checks=check.count)),flush=True)
        check.exact({x.name for x in out.glob('*.npz')},expected_raw|{'worlds.npz'},'complete raw inventory')
        order={r['raw']:r for r in rows}
        # Production stores all private rows first, followed by social pairs.
        names=[f'private_s{s}_p{p}_d{d}.npz' for s,p,d in itertools.product(inv['seeds'],inv['partitions'],(0,1))]
        names += [f'social_{m}_s{s}_p{p}_d{d}.npz' for s,p,m,d in itertools.product(inv['seeds'],inv['partitions'],('immediate','delayed'),(0,1))]
        rows=[order[n] for n in names];summary=summarize(rows,inv['seeds'])
        check.close(summary,read(out/'summary.json'),'independent rows/source/aggregate comparison')
        private_old=average([r['scores']['old'] for r in rows if r['system']=='private'])
        old={m:private_old[m]['J'] for m in MASKS}
        applicability=dict(passed=all(v>=.8 for v in old.values()),threshold=.8,support='old',
            private_heads=sum(r['system']=='private' for r in rows),old_J=old,
            effect='Descriptive applicability only; no model filtering, new training or changed primary outcome regardless of result')
        check.close(applicability,read(out/'private_applicability.json'),'private applicability no filter')
        for key in ('private_heads','social_directions','private_world_forwards','social_sender_world_forwards'):
            ckey=key if key in ('private_heads','social_directions') else key+'_replayed'
            check.exact(check.scope_counts[ckey],complete[key],'complete count/'+key)
        data=dict(status='complete',formal=inv['formal'],seeds=inv['seeds'],partitions=inv['partitions'],**summary,
            analysis_source_sha256=sha(__file__),replay_dependency_sha256=deps,evaluation_complete_sha256=sha(out/'evaluation_complete.json'),
            private_applicability=applicability,self_test=test,
            boundaries=['All policies frozen; these are new no-movement events, not a new training task.',
                'Twelve non-old maps also become initial observations; group scores are descriptive and no source is removed.',
                'Positive D_visible need not make both resource-specific effects positive.',
                'Private/social differences do not identify mediation by communication; private head and social policies differ.',
                'Message changes may be correct synonyms; report accuracy and both-mask stability separately.',
                'Four inherited source seeds are the outer unit; two social training modes are averaged within source.'])
        write(out/'analysis.json',data)
        write(out/'comparison.json',dict(passed=True,comparisons=check.comparisons,max_absolute_error=check.max_error,
            analysis_sha256=sha(out/'analysis.json'),production_summary_sha256=sha(out/'summary.json'),production_metrics_called=False))
        receipt.update(passed=True,checks=check.count,counts=check.scope_counts,failures=[],analysis_sha256=sha(out/'analysis.json'))
        receipt['seconds']=time.monotonic()-started;write(out/'audit_execution.json',receipt)
        figures(out,data)
        a=data['aggregate'];draft=['# 无移动可见性互换：独立分析草稿','',
            f"主社会效应 D_visible 为 {100*data['primary_mean']:.6f} 个百分点；两种社会训练模式均进入每来源均值。",'',
            '四来源分别为：'+ '、'.join(f"{r['seed']}: {100*r['D_visible']:+.6f}" for r in data['primary'])+' 个百分点。','',
            f"社会食物可见效应 {100*a['social_all']['common30']['food_visibility_effect']:+.6f}、水可见效应 {100*a['social_all']['common30']['water_visibility_effect']:+.6f} 个百分点；私人 D_visible 为 {100*a['private']['common30']['D_visible']:+.6f}。",'',
            f"全体私人 old 双目标成绩：food-only {100*old['food_only']:.6f}%，water-only {100*old['water_only']:.6f}%；整批适用性检查 {'通过' if applicability['passed'] else '未通过'}，没有筛人或取消社会评价。",'',
            '本轮不训练任何模型。无移动及部分非old初态属于新支持；正效应不自动表示两个资源均反转。私人/社会差不视为通信的因果放大或中介量。保留各来源、各资源、分组及两mask同时正确的结果。']
        (out/'独立分析草稿.md').write_text('\n'.join(draft)+'\n')
        print(json.dumps(dict(passed=True,checks=check.count,comparisons=check.comparisons,max_error=check.max_error,
            primary_mean=data['primary_mean'],counts=check.scope_counts,seconds=time.monotonic()-started)),flush=True)
    except Exception as exc:
        receipt.update(passed=False,checks=check.count,counts=check.scope_counts,failures=[str(exc)],traceback=traceback.format_exc(),seconds=time.monotonic()-started)
        write(out/'analysis_failure.json',receipt);raise


if __name__=='__main__':main()
