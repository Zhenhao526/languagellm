"""Independent read-only audit of the completed 24-run D1 experiment.

Importing this module reads no results and initializes no networks. Explicit
audit forwards saved FINAL parameters only; intermediate monitor records are
checked as saved arrays. No training or optimizer replay occurs.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path
import platform
import re
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SEEDS=(51101,51102,51103,51104)
CONDITIONS=('FI_silent','FI_live','PL_silent','PL_live','LL_silent','LL_live')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
STEPS=(0,100,500,1500,3000,6000)
PAIRS=((0,1),(0,2),(1,2))
RESOURCE_BITS=(3,12,5,10,1,2,4,8)
DEST_BITS=(1,2,3)
TOL=2e-12
OLD_CORE_SHA='5299c99bb92f3e18f8fae969347084c631a66f78e635d2abc4609c476cb9ae60'


def require(test,message):
    if not test: raise AssertionError(message)


def read(path): return json.loads(Path(path).read_text())
def sha(path): return sha256(Path(path).read_bytes()).hexdigest()
def json_bytes(v): return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def compact(v): return json.dumps(v,separators=(',',':'),ensure_ascii=False)


def array_sha(a):
    a=np.ascontiguousarray(a)
    h=sha256((json.dumps(dict(shape=list(a.shape),dtype=a.dtype.str),sort_keys=True,separators=(',',':'))+'\n').encode())
    h.update(a.tobytes());return h.hexdigest()


def close(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' shape/finite')
    require(np.allclose(a,b,atol=TOL,rtol=0),label+' numerical mismatch')
    return float(np.max(np.abs(a-b))) if a.size else 0.


def finite_tree(v):
    if isinstance(v,dict):
        for x in v.values():finite_tree(x)
    elif isinstance(v,list):
        for x in v:finite_tree(x)
    elif isinstance(v,float):require(math.isfinite(v),'Nonfinite JSON number')


def compare(a,b,label=''):
    if isinstance(b,dict):
        require(isinstance(a,dict),'Mapping '+label)
        for k,v in b.items():require(k in a,'Missing '+label+k);compare(a[k],v,label+k+'/')
    elif isinstance(b,list):
        require(isinstance(a,list) and len(a)==len(b),'List '+label)
        for i,v in enumerate(b):compare(a[i],v,label+str(i)+'/')
    elif isinstance(b,float):close(a,b,label)
    else:require(a==b,'Value '+label)


def structural_actions():
    rows=[]
    for i,j in PAIRS:
        for site,d in product(range(4),range(2)):
            a=[0,0,0]
            for who,peer in ((i,j),(j,i)):
                a[who]=1+4*site+2*d+[b for b in range(3) if b!=who].index(peer)
            rows.append(a)
    return np.asarray(rows,dtype=np.int64)


JOINT=structural_actions()
ACCEPT=np.asarray([[(RESOURCE_BITS[n//3]>>m&1) and (DEST_BITS[n%3]>>d&1)
                     for m,d in product(range(4),range(2))] for n in range(24)],dtype=bool).reshape(24,4,2)


def rewards(states):
    n=len(states);out=np.zeros((n,24),dtype=np.float64)
    for t,row in enumerate(JOINT):
        for a,c in enumerate(row):
            if c:
                site,d=(c-1)//4,((c-1)//2)%2
                out[:,t]+=ACCEPT[states[:,a],states[:,3+site],d]/2
    return out


def features(states,information):
    require(information in ('FI','PL','LL'),'Information condition')
    n=len(states);out=np.zeros((n,3,54),dtype=np.float64);rows=np.arange(n)
    for viewer in range(3):
        for who in range(3):
            if who!=viewer and information!='FI':continue
            need=states[:,who];r=need//3;d=need%3
            out[:,viewer,7*who]=1
            for kind in (0,1):out[:,viewer,7*who+1+kind]=(np.asarray(RESOURCE_BITS)[r] & (3<<(2*kind)))!=0
            for length in (0,1):out[:,viewer,7*who+3+length]=(np.asarray(RESOURCE_BITS)[r] & (5<<length))!=0
            for dest in (0,1):out[:,viewer,7*who+5+dest]=(np.asarray(DEST_BITS)[d] & (1<<dest))!=0
        for site in range(4):
            show=np.ones(n,dtype=bool) if information in ('FI','PL') or site==0 else states[:,7+viewer]==site
            use=rows[show];m=states[show,3+site]
            out[use,viewer,21+5*site]=1
            out[use,viewer,22+5*site+m//2]=1
            out[use,viewer,24+5*site+m%2]=1
        for who in range(3):out[rows,viewer,41+3*(states[:,7+who]-1)+who]=1
        out[:,viewer,50+viewer]=1;out[:,viewer,53]=information=='FI'
    return out


def native(states,actions):
    active=actions!=0;raw=np.maximum(actions.astype(np.int64)-1,0)
    site,dest=raw//4,(raw//2)%2
    partner=np.stack([np.asarray([b for b in range(3) if b!=a])[raw[:,a]%2] for a in range(3)],axis=1)
    executed=np.zeros_like(active)
    for i,j in PAIRS:
        match=(active.sum(1)==2)&active[:,i]&active[:,j]&(partner[:,i]==j)&(partner[:,j]==i)&(site[:,i]==site[:,j])&(dest[:,i]==dest[:,j])
        executed[:,i]|=match;executed[:,j]|=match
    m=states[np.arange(len(states))[:,None],3+site]
    satisfied=executed&ACCEPT[states[:,:3],m,dest]
    return satisfied.sum(1)/2,executed,satisfied


def support():
    all_needs=np.asarray(list(product(range(24),repeat=3)),dtype=np.int16)
    states=np.concatenate((all_needs,np.tile([0,1,2,3,1,2,3],(len(all_needs),1))),axis=1)
    r=rewards(states);return [tuple(map(int,n)) for n in all_needs[(r==1).sum(1)==1]]


def orbit(need):
    values=set()
    for k,l,swap,d in product(range(2),repeat=4):
        transformed=[]
        for v in need:
            mask=0
            for m in range(4):
                if RESOURCE_BITS[v//3]>>m&1:
                    a,b=(m//2)^k,(m%2)^l
                    if swap:a,b=b,a
                    mask|=1<<(2*a+b)
            dest=v%3
            transformed.append(3*RESOURCE_BITS.index(mask)+(1-dest if d and dest<2 else dest))
        values.update(permutations(transformed))
    return sorted(values)


def ranked(prefix,value):return sha256((prefix+'|'+compact(value)).encode()).hexdigest()


def independent_specs():
    remaining=set(support());orbits=[]
    require(len(remaining)==5376,'Full support')
    while remaining:
        members=orbit(min(remaining));require(set(members)<=remaining,'Orbit closed/disjoint');remaining.difference_update(members)
        canonical=members[0];orbits.append(dict(canonical=list(canonical),members=[list(n) for n in members],size=len(members),
            ranking_sha256=ranked('action_dependency_need_split_v1',canonical)))
    orbits.sort(key=lambda x:(x['ranking_sha256'],x['canonical']))
    require(len(orbits)==66,'Expected complete66 orbits')
    for rank,row in enumerate(orbits):row.update(rank=rank,split='heldout' if rank<math.ceil(len(orbits)/4) else 'train')
    needs={s:sorted(n for row in orbits if row['split']==s for n in row['members']) for s in ('train','heldout')}
    layouts=sorted([dict(layout=list(p),ranking_sha256=ranked('action_dependency_layout_split_v1',p)) for p in permutations(range(4))],key=lambda x:(x['ranking_sha256'],x['layout']))
    for rank,row in enumerate(layouts):row.update(rank=rank,split='heldout' if rank<6 else 'train')
    layout_by_split={s:sorted(row['layout'] for row in layouts if row['split']==s) for s in ('train','heldout')}
    owners=[list(p) for p in permutations((1,2,3))];specs={}
    for p in PARTS:
        ns='heldout' if p in ('new_needs','new_needs_and_layouts') else 'train'
        ls='heldout' if p in ('new_layouts','new_needs_and_layouts') else 'train'
        locations=layout_by_split[ls];backgrounds=[]
        candidates=[]
        for li,layout in enumerate(locations):
            for oi,owner in enumerate(owners):
                digest=sha256(('action_dependency_monitor_background_v1|'+compact(layout)+'|'+compact(owner)).encode()).hexdigest()
                candidates.append(dict(layout_index=li,owner_index=oi,layout=layout,private_sites=owner,ranking_sha256=digest))
        candidates.sort(key=lambda r:(r['ranking_sha256'],r['layout'],r['private_sites']))
        for c in candidates:
            if all(c['layout_index']!=b['layout_index'] for b in backgrounds):backgrounds.append(c)
            if len(backgrounds)==2:break
        indices=sorted((ni*len(locations)+b['layout_index'])*6+b['owner_index'] for ni in range(len(needs[ns])) for b in backgrounds)
        specs[p]=dict(needs=needs[ns],layouts=locations,private_sites=owners,world_count=len(needs[ns])*len(locations)*6,
            need_split=ns,layout_split=ls,monitor_backgrounds=backgrounds,monitor_indices=indices)
    return specs,orbits,layouts


def packed(spec,indices=None):
    ids=np.arange(spec['world_count'],dtype=np.int64) if indices is None else np.asarray(indices,dtype=np.int64)
    l,o=len(spec['layouts']),len(spec['private_sites'])
    return np.concatenate((np.asarray(spec['needs'],dtype=np.int16)[ids//(l*o)],
        np.asarray(spec['layouts'],dtype=np.int16)[ids//o%l],np.asarray(spec['private_sites'],dtype=np.int16)[ids%o]),axis=1)


def sample(spec,u):
    require(u.ndim==2 and u.shape[1]==3 and np.isfinite(u).all() and ((u>=0)&(u<1)).all(),'World uniforms')
    indices=(u*np.asarray([len(spec['needs']),len(spec['layouts']),len(spec['private_sites'])])).astype(np.int64)
    return (indices[:,0]*len(spec['layouts'])+indices[:,1])*len(spec['private_sites'])+indices[:,2]


def role_bounds(spec):
    needs=np.asarray(spec['needs'],dtype=np.int16)
    state=np.concatenate((needs,np.tile([0,1,2,3,1,2,3],(len(needs),1))),axis=1)
    table=rewards(state);require(((table==1).sum(1)==1).all(),'Unique role truth')
    target=JOINT[(table==1).argmax(1)];pair=(table==1).argmax(1)//8
    counts=[defaultdict(Counter) for _ in range(3)]
    for a in range(3):
        values=np.where(target[:,a]==0,0,1+(target[:,a]-1)%2)
        for need,role in zip(needs[:,a],values):counts[a][int(need)][int(role)]+=1
    return dict(need_tables=len(needs),fixed_pair_counts={str(p):int((pair==i).sum()) for i,p in enumerate(PAIRS)},
        individual_role_bayes=[dict(agent=a,numerator=sum(max(c.values()) for c in row.values()),denominator=len(needs),
            fraction=str(Fraction(sum(max(c.values()) for c in row.values()),len(needs))),counts_by_own_need={str(k):dict(v) for k,v in row.items()}) for a,row in enumerate(counts)],
        interpretation='Joint role success cannot exceed any individual-role Bayes bound; not an achieved policy or exact optimum. PL/LL same bound.')


def historical_helpers():
    receipt_path=ROOT/'research_program/triadic_message_study/results/messages_001/audit_execution_001/verification.json'
    pure_receipt_path=ROOT/'research_program/triadic_learning_baseline/results/learning_001/audit_execution_001/verification.json'
    receipt,pure_receipt=read(receipt_path),read(pure_receipt_path)
    prevpath=ROOT/'research_program/triadic_message_study/audit_execution.py'
    purepath=ROOT/'research_program/triadic_learning_baseline/audit_execution.py'
    require(receipt['status']==pure_receipt['status']=='passed','Historical audit status')
    require(sha(prevpath)==receipt['audit_source_sha256'] and sha(purepath)==pure_receipt['audit_source_sha256'],'Historical independent helper changed')
    from research_program.triadic_message_study import audit_execution as old
    sources={str(p):sha(p) for p in (receipt_path,pure_receipt_path,prevpath,purepath)}
    return old,sources


def forward_final(networks,states,information,live,old):
    x=features(states,information);messages=[];inputs=x;ties=0
    for window in (0,1):
        p=np.stack([old.probabilities(networks[3*a+window],inputs[:,a]) for a in range(3)],axis=1)
        ties+=int(((p==p.max(-1,keepdims=True)).sum(-1)>1).sum())
        messages.append(p.argmax(-1).astype(np.int8))
        if window==0:inputs=np.concatenate((x,old.route(messages[0],live)),axis=-1)
    inputs=np.concatenate((x,old.route(messages[0],live),old.route(messages[1],live)),axis=-1)
    p=np.stack([old.probabilities(networks[3*a+2],inputs[:,a]) for a in range(3)],axis=1)
    return np.stack(messages,axis=1),p,ties


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def evaluate_saved(entry,path,states,indices,information,live,old,networks=None):
    require(Path(entry['path']).resolve()==path.resolve() and sha(path)==entry['data_sha256'],'Evaluation path/SHA')
    require(entry['information']==information and entry['live'] is live and entry['reused_natural'] is False,'Evaluation mode')
    v=load_npz(path);n=len(indices)
    require(set(v)=={'states','state_indices','messages','action_indices','action_probabilities','greedy_reward','executed','satisfied',
        'conditional_exact_expected_reward','conditional_exact_full_success_probability','conditional_exact_execution_probability'},'Evaluation fields')
    require(np.array_equal(v['state_indices'],indices) and v['state_indices'].dtype==np.int64,'Ordered evaluation ids')
    require(np.array_equal(v['states'],states[indices]),'Exact evaluation worlds')
    require(v['states'].dtype==np.int16,'World dtype')
    for k in ('greedy_reward','conditional_exact_expected_reward','conditional_exact_full_success_probability','conditional_exact_execution_probability'):
        require(v[k].dtype==np.float64 and v[k].shape==(n,) and np.isfinite(v[k]).all(),'Scalar outcome shape/finite '+k)
    require(v['messages'].shape==(n,2,3,4) and v['messages'].dtype==np.int8 and ((v['messages']>=0)&(v['messages']<8)).all(),'Saved message domain')
    a,p=v['action_indices'],v['action_probabilities']
    require(a.shape==(n,3) and a.dtype==np.int16 and ((a>=0)&(a<17)).all(),'Saved action domain')
    require(p.shape==(n,3,17) and p.dtype==np.float64 and np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),'Saved action probabilities')
    close(p.sum(-1),np.ones((n,3)),'Policy normalization');require(np.array_equal(a,p.argmax(-1)),'Greedy actions')
    r,executed,satisfied=native(v['states'],a)
    require(np.array_equal(v['greedy_reward'],r) and np.array_equal(v['executed'],executed) and np.array_equal(v['satisfied'],satisfied),'Native physical outcomes')
    for key in ('executed','satisfied'):require(v[key].dtype==bool and v[key].shape==(n,3),'Boolean outcome fields')
    maxp,maxs,ties=0.,0.,0;target_roles=np.empty((n,3),dtype=np.int8)
    for start in range(0,n,1024):
        end=min(start+1024,n);sl=slice(start,end);chunk=v['states'][sl];probs=p[sl]
        truth=rewards(chunk);require(((truth==1).sum(1)==1).all(),'Single full plan')
        target=JOINT[(truth==1).argmax(1)]
        target_roles[sl]=np.where(target==0,-1,(target-1)%2)
        masses=np.ones((len(chunk),24))
        for who in range(3):masses*=probs[:,who,JOINT[:,who]]
        stats={'conditional_exact_expected_reward':(masses*truth).sum(1),
            'conditional_exact_full_success_probability':(masses*(truth==1)).sum(1),'conditional_exact_execution_probability':masses.sum(1)}
        for key,value in stats.items():maxs=max(maxs,close(v[key][sl],value,'Conditional statistic '+key))
        if networks is not None:
            messages,new_p,new_ties=forward_final(networks,chunk,information,live,old);ties+=new_ties
            require(np.array_equal(messages,v['messages'][sl]),'Independent final messages')
            require(np.array_equal(new_p.argmax(-1),a[sl]),'Independent final actions')
            maxp=max(maxp,close(new_p,probs,'Independent final probabilities'))
    actual_roles=np.where(a==0,-1,(a-1)%2)
    actions,counts=np.unique(a,axis=0,return_counts=True)
    metrics=dict(worlds=n,information=information,live=live,reward_mean=float(r.mean()),full_success_rate=float((r==1).mean()),
        role_success_rate=float(np.all(actual_roles==target_roles,axis=1).mean()),physical_execution_rate=float(executed.any(1).mean()),
        expected_reward_given_greedy_messages=float(v['conditional_exact_expected_reward'].mean()),
        full_probability_given_greedy_messages=float(v['conditional_exact_full_success_probability'].mean()),
        raw_joint_action_counts=[dict(action_indices=x.tolist(),worlds=int(k)) for x,k in zip(actions,counts)],state_indices_sha256=array_sha(indices),reused_natural=False)
    compare(entry,metrics,'evaluation/')
    return v,dict(sha256=sha(path),worlds=n,max_probability_error=maxp,max_conditional_error=maxs,
        independent_forward_worlds=n if networks is not None else 0,independent_network_samples=9*n if networks is not None else 0,
        independent_final_sender_ties=ties if networks is not None else None,metrics=metrics)


def content_summary(spec,values):
    """Independent primary statistic: backgrounds, cases, six S/L, three axes.

    This uses no dataset pair builder or official metrics aggregation. Value
    flips are computed on resource sets and the target is remapped to each site.
    """
    needs=[tuple(n) for n in spec['needs']];lookup={n:i for i,n in enumerate(needs)}
    canonical=np.concatenate((np.asarray(needs,dtype=np.int16),np.tile([0,1,2,3,1,2,3],(len(needs),1))),axis=1)
    truth=JOINT[(rewards(canonical)==1).argmax(1)]
    b=len(spec['layouts'])*6;actions=values['action_indices'].reshape(len(needs),b,3)
    backgrounds=np.repeat(np.asarray(spec['layouts']),6,axis=0)
    groups=defaultdict(list);counts=Counter()
    for i,before in enumerate(needs):
        for sender in range(3):
            n=before[sender];resource,dest=divmod(n,3);mask=RESOURCE_BITS[resource]
            changes=[]
            for axis,bit in ((0,2),(1,1)):
                changed_mask=sum(1<<(m^bit) for m in range(4) if mask>>m&1)
                if changed_mask!=mask:changes.append((axis,3*RESOURCE_BITS.index(changed_mask)+dest))
            if dest<2:changes.append((2,3*resource+1-dest))
            for axis,changed in changes:
                after=list(before);after[sender]=changed;after=tuple(after)
                j=lookup.get(after)
                if j is None or i>=j:continue
                for listener in range(3):
                    if listener==sender:continue
                    a0,a1=truth[[i,j],listener]
                    if not (a0 and a1 and a0!=a1):continue
                    correct=[]
                    for action in (a0,a1):
                        material=(action-1)//4;site=(backgrounds==material).argmax(1)
                        correct.append(1+4*site+(action-1)%4)
                    apt=(actions[i,:,listener]==correct[0])&(actions[j,:,listener]==correct[1])
                    groups[axis,sender,listener].append(float(apt.mean()));counts[axis]+=1
    result=[]
    for axis in range(3):
        strata=[]
        for s,l in product(range(3),repeat=2):
            if s==l:continue
            vals=groups[axis,s,l];require(len(vals)>0,'Missing content stratum')
            strata.append(dict(sender=s,listener=l,cases=len(vals),both_endpoints_apt=float(np.mean(vals))))
        result.append(dict(axis=axis,cases=counts[axis],strata=strata,both_endpoints_apt=float(np.mean([x['both_endpoints_apt'] for x in strata]))))
    return dict(axes=result,both_endpoints_apt=float(np.mean([x['both_endpoints_apt'] for x in result])),backgrounds_per_case=b,
        weighting='uniform backgrounds, cases within S/L, six ordered S/L within axis, three axes')


def training_row(row,seed,condition,update,expected):
    finite_tree(row);require(row['seed']==seed and row['condition']==condition and row['update']==update,'Training identity')
    for key,value in expected.items():require(row[key]==value,'Independent training RNG/hash '+key)
    beta=.001*max(0.,1-(update-1)/1000)
    close(row['entropy_coefficient'],beta,'Entropy schedule')
    close(row['mean_F'],row['mean_log_J']+beta*row['mean_actor_entropy'],'Full objective identity')
    close(row['receiver_loss'],-row['mean_F'],'Receiver loss sign')
    require(-TOL<=row['mean_J']<=1+TOL,'Training expected reward range')
    require(row['min_log_J']<=row['mean_log_J']+TOL<=row['max_log_J']+2*TOL<=3*TOL,'Log expected reward range')
    require(0<=row['zero_float_J_states']<=512 and type(row['zero_float_J_states']) is int,'Underflow state count')
    require(-TOL<=row['mean_actor_entropy']<=math.log(17)+TOL,'Actor entropy range')
    close(row['sender_advantage_mean'],0.,'Antisymmetric paired advantage')
    require(row['sender_advantage_abs_mean']>=0 and row['sender_advantage_squared_mean']>=0 and row['sender_advantage_max_abs']>=0,'Advantage moment range')
    require(row['sender_advantage_abs_mean']<=row['sender_advantage_max_abs']+TOL,'Advantage moment consistency')
    require(row['sender_mean_complete_log_score']<=TOL,'Log score sign')
    norm=row['gradient_norm'];require(norm>=0,'Gradient norm')
    close(row['gradient_clip_scale'],min(1.,5./max(norm,1e-300)),'Gradient clipping schedule')
    require(re.fullmatch('[0-9a-f]{64}',row['sampled_messages_sha256']) is not None,'Sampled message digest')


def audit(run):
    run=Path(run).resolve();execution=run/'execution'
    require(read(execution/'status.json')['status']=='completed','Batch not completed; no interim audit')
    results=read(execution/'results.json');finite_tree(results)
    require(results['status']=='completed' and results['completed_run_count']==24,'Complete24 grid')
    require(not (execution/'failure.json').exists(),'Failure artifact present')
    p,v,f=read(run/'plan.json'),read(run/'prepared.json'),read(run/'freeze.json')
    require(sha(run/'plan.json')=='df5036145a1bea258a40d6e0a65b66a09c6c188b311b732c954429b6eb38fcba','Predetermined plan SHA')
    require(sha(run/'plan.json')==f['plan_sha256']==results['plan_sha256'],'Plan chain')
    require(sha(run/'prepared.json')==f['prepared_sha256']==p['prepared_sha256'],'Prepared chain')
    require(read(execution/'started.json')['plan_sha256']==f['plan_sha256'],'Started plan')
    require(p['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Audit numerical runtime differs')
    config=dict(seeds=list(SEEDS),conditions=list(CONDITIONS),partitions=list(PARTS),updates=6000,batch_size=256,
        checkpoints=list(STEPS),features=54,dtype='float64',learning_rate=.001,global_gradient_clip=5.,
        entropy_initial=.001,entropy_zero_after_updates=1000,sender_entropy_coefficient=0,trajectories_per_state=2,
        sender_windows=2,sender_tokens_per_window=4,actions_per_actor=17,worker_count=4,multiprocessing_start_method='spawn',
        dimensions=dict(sender1=[54,64,64,32],sender2=[153,64,64,32],action=[252,64,64,17]),
        primary='6000_both_holdouts_content_both_apt_PL_live_minus_LL_live',world_sampling='three_uniforms_uniform_need_layout_owner')
    compare(p['config'],config,'config/')
    artifacts={str(path):sha(path) for path in (run/'plan.json',run/'prepared.json',run/'freeze.json',execution/'results.json',execution/'status.json',execution/'started.json')}
    for relative,digest in p['sources'].items():
        for path in (ROOT/relative,run/'source_snapshot'/relative):
            require(sha(path)==digest,'Frozen source changed '+str(path));artifacts[str(path)]=digest
    old,historical=historical_helpers();artifacts.update(historical)
    specs,orbits,layouts=independent_specs()
    compare(v,dict(partitions=specs,need_orbits=orbits,layout_ranks=layouts,orbit_count=66,need_counts=dict(train=3888,heldout=1488)),'independent preparation/')
    worlds=sum(s['world_count'] for s in specs.values());monitor_worlds=sum(len(s['monitor_indices']) for s in specs.values())
    budget=dict(runs=24,training_updates=144000,training_world_samples=24*6000*256,message_trajectories=24*6000*256*2,
        categorical_symbol_samples=24*6000*256*2*24,checkpoints=144,natural_monitor_files=576,closed_monitor_files=288,
        natural_final_files=96,closed_final_files=48,natural_final_worlds=24*worlds,closed_final_worlds=12*worlds,
        final_network_samples=36*worlds*9,monitor_worlds=36*6*monitor_worlds)
    require(v['execution_budget']==results['budget']==budget,'Complete budget')
    require([(r['seed'],r['condition']) for r in results['runs']]==list(product(SEEDS,CONDITIONS)),'Canonical complete24 grid')
    states={part:packed(specs[part]) for part in PARTS};array_hashes={}
    for part,s in states.items():
        h=dict(packed_states=array_sha(s),rewards=array_sha(rewards(s)))
        for info in ('FI','PL','LL'):h['x_'+info]=array_sha(features(s,info))
        array_hashes[part]=h
    require(array_hashes==results['array_hashes'],'Independent complete prepared array hashes')
    for seed in SEEDS:
        path=execution/f'seed_{seed}_arrays.json';require(read(path)['array_hashes']==array_hashes,'Worker input arrays');artifacts[str(path)]=sha(path)
    runmap={(r['seed'],r['condition']):r for r in results['runs']}
    stats=Counter();max_errors=defaultdict(float);evaluations={};final_metrics={};checkpoint_files=set();evaluation_files=set();init_by_seed={}
    for seed in SEEDS:
        rngs_at={};final_nets={};init_hashes=[]
        for cond in CONDITIONS:
            row=runmap[seed,cond];directory=execution/f'seed_{seed}_{cond}'
            path=directory/'result.json';require(read(path)==row,'Run result differs from parent aggregate');artifacts[str(path)]=sha(path)
            require(row['updates']==6000,'Run budget')
            path=directory/'training.jsonl';require(sha(path)==row['training_log_sha256'],'Training log chain');artifacts[str(path)]=sha(path)
            path=directory/'monitor.jsonl';ml=[json.loads(line) for line in path.read_text().splitlines()];require(ml==row['monitor'],'Monitor JSONL/result');artifacts[str(path)]=sha(path)
            require([m['update'] for m in ml]==list(STEPS),'Checkpoint grid')
            rngs_at[cond]={}
            for step,m in zip(STEPS,ml):
                path=directory/f'checkpoint_{step:04d}.npz';require(sha(path)==m['checkpoint_sha256'],'Checkpoint SHA');artifacts[str(path)]=sha(path);checkpoint_files.add(path)
                nets,wr,mr=old.checkpoint(path,step,seed);rngs_at[cond][step]=(wr,mr);stats['checkpoints']+=1
                if step==0:
                    digest=old.network_hash(nets);require(digest==row['initial_parameter_sha256'],'Initial parameter digest');init_hashes.append(digest)
                if step==6000:
                    require(old.network_hash(nets)==row['final_parameter_sha256'] and sha(path)==row['final_checkpoint_sha256'],'Final parameter chain');final_nets[cond]=nets
            require(set(directory.glob('checkpoint_*.npz'))=={directory/f'checkpoint_{s:04d}.npz' for s in STEPS},'Unexpected checkpoint file')
        require(len(set(init_hashes))==1,'Six conditions not paired at initialization');init_by_seed[seed]=init_hashes[0]
        wr=np.random.default_rng(np.random.SeedSequence([seed,200]))
        mr={f't{t}_w{w}_a{a}':np.random.default_rng(np.random.SeedSequence([seed,a,t,w,300])) for t,w,a in product(range(2),range(2),range(3))}
        streams=[(execution/f'seed_{seed}_{c}'/'training.jsonl').open() for c in CONDITIONS]
        def check_rng(step):
            current=(wr.bit_generator.state,{k:g.bit_generator.state for k,g in mr.items()})
            for c in CONDITIONS:require(rngs_at[c][step]==current,'Checkpoint RNG state '+c+'/'+str(step))
        try:
            check_rng(0)
            for update in range(1,6001):
                u=wr.random((256,3));ids=sample(specs['train'],u);mu=np.empty((2,256,2,3,4),dtype=np.float64)
                for t,w,a in product(range(2),range(2),range(3)):mu[t,:,w,a]=mr[f't{t}_w{w}_a{a}'].random((256,4))
                expected=dict(world_uniforms_sha256=array_sha(u),batch_indices_sha256=array_sha(ids),batch_states_sha256=array_sha(states['train'][ids]),sample_uniforms_sha256=array_sha(mu))
                for cond,stream in zip(CONDITIONS,streams):
                    line=stream.readline();require(bool(line),'Premature training log end');training_row(json.loads(line),seed,cond,update,expected);stats['training_rows']+=1
                if update in STEPS:check_rng(update)
            require(all(s.read()=='' for s in streams),'Excess training update')
        finally:
            for s in streams:s.close()
        for cond in CONDITIONS:
            row=runmap[seed,cond];directory=execution/f'seed_{seed}_{cond}';info,visibility=cond.split('_');live=visibility=='live';end_monitor={}
            for step,m in zip(STEPS,row['monitor']):
                require(set(m['monitor'])==set(PARTS),'Monitor partition set')
                for part in PARTS:
                    modes=m['monitor'][part];require(set(modes)=={'natural','closed'},'Monitor modes')
                    ids=np.asarray(specs[part]['monitor_indices'],dtype=np.int64)
                    for mode in ('natural','closed'):
                        if mode=='closed' and not live:
                            target=dict(modes['natural'],reused_natural=True);require(modes[mode]==target,'Silent closed alias');stats['silent_closed_references']+=1
                            if step==6000:end_monitor[part,mode]=end_monitor[part,'natural']
                            continue
                        path=directory/f'monitor_{step:04d}_{part}_{mode}.npz'
                        data,receipt=evaluate_saved(modes[mode],path,states[part],ids,info,live and mode=='natural',old)
                        evaluation_files.add(path);evaluations[str(path)]=receipt;artifacts[str(path)]=receipt['sha256'];stats['monitor_files']+=1;stats['monitor_worlds']+=len(ids)
                        max_errors['conditional']=max(max_errors['conditional'],receipt['max_conditional_error'])
                        if step==6000:end_monitor[part,mode]=data
            require(set(row['final'])==set(PARTS),'Final partition set')
            for part in PARTS:
                ids=np.arange(specs[part]['world_count'],dtype=np.int64);modes=row['final'][part];require(set(modes)=={'natural','closed'},'Final modes')
                for mode in ('natural','closed'):
                    key=f'{seed}/{cond}/{part}/{mode}'
                    if mode=='closed' and not live:
                        require(modes[mode]==dict(modes['natural'],reused_natural=True),'Silent final alias');stats['silent_closed_references']+=1
                        final_metrics[key]=final_metrics[f'{seed}/{cond}/{part}/natural'];continue
                    path=directory/f'final_{part}_{mode}.npz'
                    data,receipt=evaluate_saved(modes[mode],path,states[part],ids,info,live and mode=='natural',old,final_nets[cond])
                    evaluation_files.add(path);evaluations[str(path)]=receipt;artifacts[str(path)]=receipt['sha256'];stats['final_files']+=1
                    for name in ('independent_forward_worlds','independent_network_samples','independent_final_sender_ties'):stats[name]+=receipt[name]
                    for k in ('probability','conditional'):max_errors[k]=max(max_errors[k],receipt['max_'+k+'_error'])
                    mon=end_monitor[part,mode];selected=np.asarray(specs[part]['monitor_indices'])
                    for name,value in mon.items():
                        expected=data[name][selected]
                        if value.dtype.kind=='f':close(value,expected,'Final monitor subset '+name)
                        else:require(np.array_equal(value,expected),'Final monitor subset '+name)
                    semantic=content_summary(specs[part],data)
                    if info in ('PL','LL') and (not live or mode=='closed'):require(semantic['both_endpoints_apt']==0.,'Same-observation silent structural zero')
                    final_metrics[key]=dict(world=receipt['metrics'],content=semantic)
                    print(json.dumps(dict(audit_stage='final_checked',seed=seed,condition=cond,partition=part,mode=mode)),flush=True)
            require(set(directory.glob('*.npz'))=={x for x in checkpoint_files|evaluation_files if x.parent==directory},'Unexpected saved NPZ')
        print(json.dumps(dict(audit_stage='seed_complete',seed=seed)),flush=True)
    require(len(set(init_by_seed.values()))==4,'Independent seed initializations coincide')
    compare(dict(stats),dict(training_rows=144000,checkpoints=144,monitor_files=864,final_files=144,
        monitor_worlds=budget['monitor_worlds'],independent_forward_worlds=36*worlds,independent_network_samples=36*worlds*9,
        silent_closed_references=336),'Audit budget/')
    require({p for p in execution.glob('seed_*_*') if p.is_dir()}=={execution/f'seed_{s}_{c}' for s,c in product(SEEDS,CONDITIONS)},'Unexpected run directory')
    paired=[]
    for seed in SEEDS:
        prefix=f'{seed}/';part='new_needs_and_layouts'
        cells={c:final_metrics[f'{seed}/{c}/{part}/natural'] for c in CONDITIONS}
        paired.append(dict(seed=seed,primary_PL_live_minus_LL_live=cells['PL_live']['content']['both_endpoints_apt']-cells['LL_live']['content']['both_endpoints_apt'],
            secondary_full_success_DiD=cells['PL_live']['world']['full_success_rate']-cells['PL_silent']['world']['full_success_rate']-cells['LL_live']['world']['full_success_rate']+cells['LL_silent']['world']['full_success_rate']))
    for path,digest in artifacts.items():require(sha(path)==digest,'Artifact changed during audit '+path)
    return dict(status='passed',audit_source_sha256=sha(__file__),plan_sha256=f['plan_sha256'],runtime=dict(python=platform.python_version(),numpy=np.__version__),
        source_sha256=p['sources'],artifacts_sha256=artifacts,historical_helpers_sha256=historical,scope=dict(stats),max_errors=dict(max_errors),
        evaluations=evaluations,final_metrics=final_metrics,paired_primary=paired,
        primary_mean=float(np.mean([r['primary_PL_live_minus_LL_live'] for r in paired])),secondary_full_success_DiD_mean=float(np.mean([r['secondary_full_success_DiD'] for r in paired])),
        static_role_bounds={s:role_bounds(specs[p]) for s,p in (('train','train'),('heldout','new_needs'))},
        limits=['No optimizer replay or retraining. No intermediate checkpoint forward.',
            'Training loss values are checked for finite values, algebraic consistency and frozen implementation provenance; gradients and sampled tokens are not regenerated from intermediate weights.',
            'All saved monitor outcomes and conditional statistics are recomputed; only complete final records receive an independent nine-network forward.',
            'Four team seeds are the independent training units; worlds, axes, checkpoints and conditions are repeated measurements.',
            'Natural both-endpoint appropriateness is not itself a causal channel intervention or proof of compositionality.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    output=Path(args.out).resolve();require(not output.exists(),'Never overwrite an audit or retry in place');output.mkdir(parents=True)
    start=time.perf_counter()
    try:
        result=audit(args.run);result.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        (output/'verification.json').write_bytes(json_bytes(result))
        text=f"独立执行核验通过。核对 {result['scope']['training_rows']} 条训练记录、144 个检查点、864 个监测文件与144 个完整末点文件。\n\n"
        text+=f"末点独立前向 {result['scope']['independent_forward_worlds']:,} 个世界，共 {result['scope']['independent_network_samples']:,} 个网络样本；概率最大绝对误差 {result['max_errors']['probability']:.3g}，原生条件统计最大误差 {result['max_errors']['conditional']:.3g}。未重放优化器或重新训练。\n\n"
        text+='训练世界及12条消息随机流从种子逐批重建；六条件初值配对、全量末点和监测原生结算、末点监测子集、冻结源码及文件哈希均通过。\n\n'
        text+='主要双留出比较为 PL_live−LL_live 的三轴等权两端适切率，保留全部4个种子；详见 verification.json。静默结构零与该对比不构成两份独立证据。\n\n'
        text+='限制：保存的中间损失仅核有限性、代数关系和冻结实现来源，没有重新计算中间梯度或抽样消息；未对监测检查点做额外前向。结果不能单凭自然适切率证明因果通信或组合性。\n'
        (output/'独立核验.md').write_text(text)
        print(json.dumps(dict(status='passed',out=str(output),scope=result['scope'],max_errors=result['max_errors']),ensure_ascii=False),flush=True)
    except BaseException as error:
        failure=dict(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,audit_source_sha256=sha(__file__),automatic_retry=False)
        (output/'failure.json').write_bytes(json_bytes(failure));raise


if __name__=='__main__':main()
