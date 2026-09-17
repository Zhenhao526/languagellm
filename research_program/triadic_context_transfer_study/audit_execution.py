"""Independent context-transfer audit kernels; no import-time I/O or forward.

The completion/record wrapper is added only after the formal schema is fixed.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
from hashlib import sha256
from pathlib import Path
from collections import Counter
from functools import lru_cache
from itertools import product
from datetime import datetime,timezone
import argparse
import ast
import json
import multiprocessing
import platform
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
CONTEXT_AUDIT_SHA='20f1be8c017a53874346d1e8865620586d56de4128580b1e428c27899a30e96d'
MESSAGE_AUDIT_SHA='f09f30612ccb7e27b9ec7e3b38b9fa217d133f1df1185e2712fd899254151bd3'
PURE_AUDIT_SHA='739175596c4182068bbfbb578439189208cae72812068b2127d0006393be2281'

def require(condition,message):
    if not condition:raise AssertionError(message)

@lru_cache(maxsize=1)
def references():
    pure_path=HERE.parent/'triadic_learning_baseline/audit_execution.py'
    require(sha256(pure_path.read_bytes()).hexdigest()==PURE_AUDIT_SHA,'Historical pure helper changed')
    from research_program.triadic_action_dependency_study import audit_execution as context
    from research_program.triadic_message_study import audit_execution as messages
    require(sha256(Path(context.__file__).read_bytes()).hexdigest()==CONTEXT_AUDIT_SHA,'Context reference changed')
    require(sha256(Path(messages.__file__).read_bytes()).hexdigest()==MESSAGE_AUDIT_SHA,'MLP reference changed')
    return context,messages

def routes(tokens,live,senders=None,replacement=None):
    """Own payload untouched; each other receiver sees sender's replacement."""
    tokens=np.asarray(tokens);n=len(tokens)
    require(tokens.shape==(n,3,4) and tokens.dtype.kind in 'iu' and ((tokens>=0)&(tokens<8)).all(),'Token domain')
    replace=senders is not None or replacement is not None
    if replace:
        require(senders is not None and replacement is not None,'Incomplete replacement')
        senders=np.asarray(senders);replacement=np.asarray(replacement)
        require(senders.shape==(n,) and senders.dtype.kind in 'iu' and ((senders>=0)&(senders<3)).all(),'Sender domain')
        require(replacement.shape==(n,4) and replacement.dtype.kind in 'iu' and ((replacement>=0)&(replacement<8)).all(),'Replacement domain')
    result=np.zeros((n,3,99),dtype=np.float64);rows=np.arange(n)
    for viewer in range(3):
        for sender in range(3):
            if not (live or viewer==sender):continue
            sent=tokens[:,sender].copy()
            if replace and live and viewer!=sender:
                altered=senders==sender;sent[altered]=replacement[altered]
            result[:,viewer,96+sender]=1
            for position in range(4):result[rows,viewer,32*sender+8*position+sent[:,position]]=1
    return result

def causal_inputs(observations,natural_messages,senders,donor_packets,live,generate_second):
    """Pure scheduling with an injected W2 generator, usable without any model.

    All W2 are regenerated synchronously. No donor observation or needs enter x.
    """
    x=np.asarray(observations);m=np.asarray(natural_messages);packets=np.asarray(donor_packets);n=len(x)
    require(x.shape==(n,3,54) and np.isfinite(x).all(),'Original observations')
    require(m.shape==(n,2,3,4) and packets.shape==(n,2,4),'Message dimensions')
    first=routes(m[:,0],live,senders,packets[:,0])
    second_inputs=np.concatenate((x,first),axis=-1)
    second=np.asarray(generate_second(second_inputs))
    second_received=routes(second,live,senders,packets[:,1])
    action_inputs=np.concatenate((x,first,second_received),axis=-1)
    return dict(generated_messages=np.stack((m[:,0],second),axis=1),first_routes=first,second_routes=second_received,
        second_inputs=second_inputs,action_inputs=action_inputs)

def forward_intervention(networks,states,information,natural_messages,senders,donor_packets,live):
    """Exactly six independent module forwards per intervention world."""
    context,old=references();x=context.features(states,information)
    def second(inputs):
        return np.stack([old.probabilities(networks[3*a+1],inputs[:,a]).argmax(-1).astype(np.int8) for a in range(3)],axis=1)
    result=causal_inputs(x,natural_messages,senders,donor_packets,live,second)
    probs=np.stack([old.probabilities(networks[3*a+2],result['action_inputs'][:,a]) for a in range(3)],axis=1)
    result.update(action_probabilities=probs,action_indices=probs.argmax(-1).astype(np.int16),
        independent_forward_worlds=len(states),independent_module_samples=6*len(states))
    return result

def independent_dataset(part_spec):
    """Re-enumerate every content case and all shifts using independent truth."""
    context,_=references();needs=[tuple(n) for n in part_spec['needs']];lookup={n:i for i,n in enumerate(needs)}
    canonical=np.concatenate((np.asarray(needs,dtype=np.int16),np.tile([0,1,2,3,1,2,3],(len(needs),1))),axis=1)
    r=context.rewards(canonical);require(((r==1).sum(1)==1).all(),'Single full-success support')
    truth=context.JOINT[(r==1).argmax(1)]
    selected=[];source_index=0;strata=Counter()
    for i,before in enumerate(needs):
        for s in range(3):
            resource,destination=divmod(before[s],3);mask=context.RESOURCE_BITS[resource];changes=[]
            for axis,bit in ((0,2),(1,1)):
                changed_mask=sum(1<<(m^bit) for m in range(4) if mask>>m&1)
                if changed_mask!=mask:changes.append((axis,3*context.RESOURCE_BITS.index(changed_mask)+destination))
            if destination<2:changes.append((2,3*resource+1-destination))
            for axis,n in changes:
                after=list(before);after[s]=n;j=lookup.get(tuple(after))
                if j is None or i>=j:continue
                for listener in range(3):
                    if listener==s:continue
                    a0,a1=truth[[i,j],listener]
                    if a0 and a1 and a0!=a1:
                        selected.append(dict(source_index=source_index,axis=axis,sender=s,listener=listener,endpoints=(i,j),
                            actions=truth[[i,j]],materials=((int(a0)-1)//4,(int(a1)-1)//4)))
                        strata[axis,s,listener]+=1
                    source_index+=1
    layouts=np.asarray(part_spec['layouts'],dtype=np.int16);owners=np.asarray(part_spec['private_sites'],dtype=np.int16)
    c,l,o=len(selected),len(layouts),len(owners);b=l*o;n=c*b;k=l-1
    ci=np.repeat(np.arange(c,dtype=np.int32),b);li=np.tile(np.repeat(np.arange(l,dtype=np.int16),o),c);oi=np.tile(np.arange(o,dtype=np.int8),c*l)
    ends=np.asarray([row['endpoints'] for row in selected],dtype=np.int32)[ci]
    ca=np.asarray([row['actions'] for row in selected],dtype=np.int8)[ci]
    target=np.asarray([row['materials'] for row in selected],dtype=np.int8)[ci]
    rankings=sorted(range(l),key=lambda i:(sha256(('triadic_context_transfer_layout_rank_v1|'+json.dumps(layouts[i].tolist(),separators=(',',':'))).encode()).hexdigest(),layouts[i].tolist()))
    shifts=np.empty((k,l),dtype=np.int16)
    for delta in range(1,l):
        for rank,index in enumerate(rankings):shifts[delta-1,index]=rankings[(rank+delta)%l]
    positions=np.argsort(layouts,axis=1).astype(np.int8)
    def correct(canonical,layout_ids):
        target_shape=canonical.shape;expanded=np.broadcast_to(canonical,target_shape)
        out=np.zeros(target_shape,dtype=np.int8)
        for agent in range(3):
            a=expanded[...,agent];material=np.maximum(a-1,0)//4
            pos=positions[layout_ids];site=np.take_along_axis(pos,material,axis=-1)
            out[...,agent]=np.where(a==0,0,1+4*site+(a-1)%4)
        return out
    actual=correct(ca,li)
    dl=shifts[:,li];donor_ca=correct(np.broadcast_to(ca,(k,n,2,3)),dl)
    recv_sites=np.take_along_axis(positions[li],target,axis=-1)
    donor_sites=np.take_along_axis(positions[dl],np.broadcast_to(target,(k,n,2)),axis=-1)
    change=donor_sites!=recv_sites[None];eligible=change.all(-1);count=eligible.sum(0).astype(np.int16)
    require((count>0).all(),'Empty eligible recipient case')
    axis=np.asarray([row['axis'] for row in selected],dtype=np.int8)[ci]
    sender=np.asarray([row['sender'] for row in selected],dtype=np.int8)[ci]
    listener=np.asarray([row['listener'] for row in selected],dtype=np.int8)[ci]
    weights=np.asarray([1/(6*strata[row['axis'],row['sender'],row['listener']]*b) for row in selected])[ci]
    own_site=owners[oi,listener]
    view=(layouts[dl,0]==layouts[li,0][None])&(layouts[dl,own_site[None]]==layouts[li,own_site][None])
    arrays=dict(case_index=ci,source_case_index=np.asarray([row['source_index'] for row in selected],dtype=np.int32)[ci],axis=axis,sender=sender,listener=listener,
        endpoint_indices=((ends*l+li[:,None])*o+oi[:,None]).astype(np.int32),correct_actions=actual,target_materials=target,
        recipient_layout_index=li,owner_index=oi,donor_layout_index=shifts,
        donor_endpoint_indices=((ends[None]*l+dl[:,:,None])*o+oi[None,:,None]).astype(np.int32),donor_correct_actions=donor_ca,
        target_site_changed=change,eligible=eligible,eligible_donor_count=count,base_within_axis_weight=weights,view_equal_LL=view)
    return arrays,dict(n_cases=c,n_rows=n,n_donors=k,rank_indices=rankings,
        content_case_axis_counts={str(axis):sum(row['axis']==axis for row in selected) for axis in range(3)})


SEEDS=(51101,51102,51103,51104)
CONDITIONS=('PL_silent','PL_live','LL_silent','LL_live')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
MODES=('remote_same_both','remote_opposite_both')
MAIN_RESULT_SHA='021cef0cbbf18a1532100d3c6058f04520ffa66a83a18aa10b8b9c53e2e6d2b0'
TRANSFER_PLAN_SHA='af051fa46aedb4653210c47587010d0b0e0e6d971a92f81bc61f2a98d96245c5'
TOL=2e-12

def read(path):return json.loads(Path(path).read_text())
def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def dump_new(path,data):
    with Path(path).open('x') as f:json.dump(data,f,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def close(x,y,label):
    x,y=np.asarray(x),np.asarray(y)
    require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),label+' shape/finite')
    error=float(np.max(np.abs(x-y))) if x.size else 0.
    require(error<=TOL,label+' numerical error');return error
def grid(spec):
    yield 'sham_both',-1
    yield 'local_opposite_both',-1
    for k in range(len(spec['donor_endpoint_indices'])):
        for mode in MODES:yield mode,k
def indices(spec,mode,shift,b):
    if mode=='sham_both':require(shift==-1,'Sham shift');return spec['endpoint_indices'][:,b]
    if mode=='local_opposite_both':require(shift==-1,'Local shift');return spec['endpoint_indices'][:,1-b]
    require(mode in MODES and 0<=shift<len(spec['donor_endpoint_indices']),'Remote selector')
    return spec['donor_endpoint_indices'][shift,:,b if mode=='remote_same_both' else 1-b]
def natural_inputs(x,m,live):return np.concatenate((x,routes(m[:,0],live),routes(m[:,1],live)),axis=-1)


def scalar_sums(spec,pool,record,data):
    """Independent per-axis, unnormalized contributions on fixed denominators."""
    mode,k,b=record['mode'],record['shift_index'],record['direction'];n=record['worlds'];rows=np.arange(n);listener=spec['listener']
    rid=spec['endpoint_indices'][:,b];cfid=spec['endpoint_indices'][:,1-b]
    current=spec['correct_actions'][:,b][rows,listener];target=spec['correct_actions'][:,1-b][rows,listener]
    actions=data['action_indices'][rows,listener];p=data['action_probabilities'][rows,listener]
    if k>=0:d0,d1=spec['donor_endpoint_indices'][k,:,b],spec['donor_endpoint_indices'][k,:,1-b]
    else:d0,d1=rid,cfid
    donor0=pool['action_indices'][d0,listener];donor1=pool['action_indices'][d1,listener]
    gate=(target!=donor0)&(target!=donor1);apt=actions==target
    if 'action_input_equals_donor' in data:
        same_input=data['action_input_equals_donor'][rows,listener]
        require(not ((gate&apt)&same_input.any(1)).any(),'C success contradicts complete donor-input identity')
    values=dict(counterfactual_apt=apt,current_apt=actions==current,counterfactual_probability=p[rows,target],current_probability=p[rows,current],
        native_reward=data['greedy_reward'],native_full_success=data['greedy_reward']==1,
        counterfactual_reward=data['counterfactual_reward'],counterfactual_full_success=data['counterfactual_reward']==1,
        listener_action_changed=actions!=pool['action_indices'][rid,listener],natural_current_apt=pool['action_indices'][rid,listener]==current,
        C_gate_mass=gate,C_counterfactual_apt=gate&apt,copy_same_actual_donor=actions==donor0,copy_opposite_actual_donor=actions==donor1)
    if k<0:weights={'control':spec['base_within_axis_weight']/2}
    else:weights=dict(all_other=spec['base_within_axis_weight']/len(spec['donor_endpoint_indices'])/2,
        eligible=spec['base_within_axis_weight']*spec['eligible'][k]/spec['eligible_donor_count']/2)
    result={}
    for name,w in weights.items():
        result[name]={str(axis):dict(weight_mass=float(w[spec['axis']==axis].sum()),**{
            key:float(np.sum(w[spec['axis']==axis]*value[spec['axis']==axis])) for key,value in values.items()}) for axis in range(3)}
    return result


def audit_cell(record,path,pool,x,spec,networks,live):
    context,_=references();mode,k,b=record['mode'],record['shift_index'],record['direction'];n=len(spec['sender'])
    rid=spec['endpoint_indices'][:,b];cfid=spec['endpoint_indices'][:,1-b];di=indices(spec,mode,k,b);sender=spec['sender']
    require(record['worlds']==n and record['is_silent_alias'] is (not live),'Record mode/domain')
    for field,v in [('recipient_indices_sha256',rid),('counterfactual_recipient_indices_sha256',cfid),('donor_indices_sha256',di)]:require(record[field]==context.array_sha(v),'Dataset binding '+field)
    require(record['new_forward_worlds']==(n if live else 0) and record['new_network_samples']==(6*n if live else 0),'Record forward budget')
    digest=None;max_error=0.;identity_error=0.;signatures=[]
    if live:
        require(Path(record['path'])==path and sha(path)==record['data_sha256'],'Actual NPZ source')
        digest=sha(path);data=load_npz(path)
        keys={'dataset_rows','recipient_indices','counterfactual_recipient_indices','donor_indices','donor_packets','messages','action_indices','action_probabilities',
            'action_input_equals_donor','greedy_reward','executed','satisfied','counterfactual_reward','counterfactual_satisfied'}
        require(set(data)==keys,'NPZ fields')
        for key,value in [('dataset_rows',np.arange(n,dtype=np.int64)),('recipient_indices',rid),('counterfactual_recipient_indices',cfid),('donor_indices',di)]:
            require(data[key].dtype==value.dtype and np.array_equal(data[key],value),'NPZ index '+key)
        require(data['messages'].shape==(n,2,3,4) and data['messages'].dtype==np.int8 and ((data['messages']>=0)&(data['messages']<8)).all(),'Message domain')
        require(data['donor_packets'].shape==(n,2,4) and data['donor_packets'].dtype==np.int8,'Packet shape')
        require(data['action_indices'].shape==(n,3) and data['action_indices'].dtype==np.int16 and ((data['action_indices']>=0)&(data['action_indices']<17)).all(),'Action domain')
        require(data['action_probabilities'].shape==(n,3,17) and data['action_probabilities'].dtype==np.float64 and np.isfinite(data['action_probabilities']).all(),'Probability shape')
        require((data['action_probabilities']>=0).all() and (data['action_probabilities']<=1).all(),'Probability range')
        close(data['action_probabilities'].sum(-1),np.ones((n,3)),'Normalization')
        require(np.array_equal(data['action_indices'],data['action_probabilities'].argmax(-1)),'Full17 first argmax')
        require(data['action_input_equals_donor'].shape==(n,3,2) and data['action_input_equals_donor'].dtype==bool,'Donor-input identity dimensions')
    else:
        require(record['path'] is None and record['data_sha256'] is None and not path.exists(),'Silent file must not exist')
        data={key:pool[key][rid] for key in ('messages','action_indices','action_probabilities','greedy_reward','executed','satisfied')}
    require(len(record['batches'])==(n+1023)//1024,'Batch count')
    for index,start in enumerate(range(0,n,1024)):
        stop=min(start+1024,n);sl=slice(start,stop);r=rid[sl];d=di[sl];s=sender[sl];row=np.arange(stop-start)
        m=pool['messages'][r];packets=pool['messages'][d,:,s,:]
        if live:
            trace=forward_intervention(networks,pool['states'][r],record['condition'].split('_')[0],m,s,packets,True)
            require(np.array_equal(trace['generated_messages'],data['messages'][sl]),'Independent generated messages')
            require(np.array_equal(trace['action_indices'],data['action_indices'][sl]),'Independent action choices')
            max_error=max(max_error,close(trace['action_probabilities'],data['action_probabilities'][sl],'Independent probability'))
            require(np.array_equal(data['donor_packets'][sl],packets),'Focal donor-only packet')
            require(np.array_equal(data['messages'][sl][row,1,s],m[row,1,s]),'Sender self W2 preserved')
            for endpoint in (0,1):
                did=spec['donor_endpoint_indices'][k,sl,endpoint] if k>=0 else spec['endpoint_indices'][sl,endpoint]
                expected=natural_inputs(x[did],pool['messages'][did],True)
                require(np.array_equal(data['action_input_equals_donor'][sl,:,endpoint],np.all(trace['action_inputs']==expected,axis=-1)),'Actual complete donor-input equality')
        else:
            trace=dict(first_routes=routes(m[:,0],False,s,packets[:,0]),second_routes=routes(m[:,1],False,s,packets[:,1]))
            trace['action_inputs']=np.concatenate((x[r],trace['first_routes'],trace['second_routes']),axis=-1)
            require(np.array_equal(trace['action_inputs'],natural_inputs(x[r],m,False)),'Silent route changed original input')
        batch=record['batches'][index];require(batch['start']==start and batch['stop']==stop,'Batch row coverage')
        for key in ('first_routes','second_routes','action_inputs'):require(batch[key+'_sha256']==context.array_sha(trace[key]),'Actual '+key+' hash')
        require(batch['donor_packets_sha256']==context.array_sha(packets),'Packet hash')
        if live and mode=='sham_both':
            require(np.array_equal(trace['action_inputs'],natural_inputs(x[r],m,True)),'Sham full input')
            identity_error=max(identity_error,close(data['action_probabilities'][sl],pool['action_probabilities'][r],'Sham natural probability'))
        foreign=np.arange(3)[None,:]!=s[:,None]
        if live and mode=='local_opposite_both':
            expected=natural_inputs(x[d],pool['messages'][d],True)
            require(np.array_equal(trace['action_inputs'][foreign],expected[foreign]),'Local non-sender donor identity')
            identity_error=max(identity_error,close(data['action_probabilities'][sl][foreign],pool['action_probabilities'][d][foreign],'Local non-sender probability'))
        if live and k>=0:
            signatures.append(dict(start=start,stop=stop,input=context.array_sha(trace['action_inputs'][foreign]),
                probability=context.array_sha(data['action_probabilities'][sl][foreign]),action=context.array_sha(data['action_indices'][sl][foreign]),
                generated_W2=context.array_sha(data['messages'][sl,1][foreign])))
    close(record['max_identity_error'],identity_error,'Saved identity error')
    native,executed,satisfied=context.native(pool['states'][rid],data['action_indices'])
    cf,cfexecuted,cfsatisfied=context.native(pool['states'][cfid],data['action_indices'])
    require(np.array_equal(executed,cfexecuted),'Need-independent physical execution')
    for key,value in [('greedy_reward',native),('executed',executed),('satisfied',satisfied)]:require(np.array_equal(data[key],value),'Native '+key)
    if live:
        for key,value in [('counterfactual_reward',cf),('counterfactual_satisfied',cfsatisfied)]:require(np.array_equal(data[key],value),'Counterfactual '+key)
        for key in ('greedy_reward','counterfactual_reward'):require(data[key].dtype==np.float64,'Reward dtype')
        for key in ('executed','satisfied','counterfactual_satisfied'):require(data[key].dtype==bool,'Outcome dtype')
    else:data.update(counterfactual_reward=cf,counterfactual_satisfied=cfsatisfied)
    require(np.array_equal(cf==1,np.all(data['action_indices']==spec['correct_actions'][:,1-b],axis=1)),'Counterfactual unique full truth')
    return dict(mode=mode,shift_index=k,direction=b,worlds=n,path=str(path) if live else None,data_sha256=digest,
        independent_forward_worlds=n if live else 0,independent_module_samples=6*n if live else 0,max_probability_error=max_error,
        endpoint_swap_signatures=signatures,scalars=scalar_sums(spec,pool,record,data))


def aggregate(records):
    """Add fixed-denominator contributions; never normalize by observed gates."""
    result={}
    for part in PARTS:
        layers={}
        for support in ('eligible','all_other'):
            sums={mode:{str(axis):Counter() for axis in range(3)} for mode in MODES}
            for row in records:
                if row['partition']!=part or row['shift_index']<0:continue
                for axis,values in row['scalars'][support].items():
                    for key,value in values.items():sums[row['mode']][axis][key]+=value
            for mode in MODES:
                for axis in range(3):close(sums[mode][str(axis)]['weight_mass'],1.,'Complete axis/donor/direction weight')
            contrast={axis:{key:sums['remote_opposite_both'][axis][key]-sums['remote_same_both'][axis][key]
                for key in sums['remote_same_both'][axis]} for axis in sums['remote_same_both']}
            # Raw symmetric weights preserve the endpoint-exchange marginal identity.
            for axis in sums['remote_same_both']:
                close(sums['remote_opposite_both'][axis]['counterfactual_apt'],sums['remote_same_both'][axis]['current_apt'],'Endpoint-exchange weighted apt')
                close(sums['remote_opposite_both'][axis]['current_apt'],sums['remote_same_both'][axis]['counterfactual_apt'],'Endpoint-exchange reverse apt')
                close(sums['remote_opposite_both'][axis]['counterfactual_probability'],sums['remote_same_both'][axis]['current_probability'],'Endpoint-exchange weighted probability')
            layers[support]=dict(arms=sums,contrast_by_axis=contrast,macro_contrast={key:float(np.mean([contrast[str(a)][key] for a in range(3)])) for key in contrast['0']})
        result[part]=layers
    return result


def audit_worker(payload):
    seed,run,output=payload;run=Path(run);output=Path(output);plan=read(run/'plan.json');execution=run/'execution'
    context,old=references();original=Path(plan['source_directory']);data_directory=Path(plan['dataset_directory'])
    prior=read(original/'audit_execution_001/verification.json');main=read(execution/'results.json')
    policies=[p for p in plan['policies'] if p['seed']==seed]
    require([p['condition'] for p in policies]==list(CONDITIONS),'Worker policy grid')
    selected=[r for r in main['records'] if r['seed']==seed]
    lookup={(r['condition'],r['partition'],r['mode'],r['shift_index'],r['direction']):r for r in selected}
    require(len(lookup)==len(selected)==768,'Unique worker768 records')
    task_specs=read(original/'prepared.json')['partitions'];receipts=[];artifacts={};seen_npz=set();loads=0;feature_hashes={};condition_summaries={}
    for policy in policies:
        cond=policy['condition'];info,visibility=cond.split('_');live=visibility=='live';directory=execution/f'seed_{seed}_{cond}'
        nets=None
        if live:
            require(sha(policy['checkpoint'])==prior['artifacts_sha256'][policy['checkpoint']],'Previously audited weights')
            nets,_,_=old.checkpoint(policy['checkpoint'],6000,seed);loads+=1
        policy_rows=[]
        for part in PARTS:
            spec=load_npz(data_directory/(part+'.npz'));pool_path=Path(policy['endpoints'][part])
            require(sha(pool_path)==prior['artifacts_sha256'][str(pool_path)],'Previously audited natural endpoint changed')
            pool=load_npz(pool_path);states=context.packed(task_specs[part]);require(np.array_equal(pool['states'],states),'Natural source states/order')
            require(np.array_equal(pool['state_indices'],np.arange(len(states))),'Natural source full ids')
            x=context.features(states,info);feature_hashes.setdefault(part,{})[info]=context.array_sha(x)
            require(main['feature_hashes'][part][info]==context.array_sha(x),'Independent new feature hash')
            swap={}
            for mode,k in grid(spec):
                for b in (0,1):
                    key=(cond,part,mode,k,b);require(key in lookup,'Missing record');record=lookup[key]
                    require(record['natural_source']==str(pool_path),'Wrong policy source')
                    name=f'{part}_{mode}_shift{k:02d}_direction{b}';path=directory/(name+'.npz');jp=directory/(name+'.json')
                    require(read(jp)==record,'Individual/aggregate record differs');artifacts[str(jp)]=sha(jp)
                    checked=audit_cell(record,path,pool,x,spec,nets,live);checked.update(seed=seed,condition=cond,partition=part)
                    if live:seen_npz.add(path);artifacts[str(path)]=checked['data_sha256']
                    if live and k>=0:swap[mode,k,b]=checked['endpoint_swap_signatures']
                    policy_rows.append(checked);receipts.append(checked)
                if mode=='remote_opposite_both' and live:
                    for b in (0,1):require(swap['remote_same_both',k,b]==swap['remote_opposite_both',k,1-b],'Every non-sender endpoint-swap input/output hash')
                    # Compared signatures need not be retained in every worker response.
                    for mm,bb in product(MODES,(0,1)):del swap[mm,k,bb]
                    print(json.dumps(dict(audit_stage='shift_complete',seed=seed,condition=cond,partition=part,shift=k)),flush=True)
            del x,pool,states,spec
        require(set(directory.glob('*.npz'))=={p for p in seen_npz if p.parent==directory},'Extra/missing actual NPZ')
        condition_summaries[cond]=aggregate(policy_rows)
        if not live:
            for part in PARTS:
                for layer in ('eligible','all_other'):
                    for key in ('counterfactual_apt','counterfactual_probability','C_counterfactual_apt'):
                        close(condition_summaries[cond][part][layer]['macro_contrast'][key],0.,'Silent structural zero')
    require(loads==2 and len(receipts)==768 and len(seen_npz)==384,'Worker exact budget')
    worker_source=read(execution/f'seed_{seed}_result.json')
    require(worker_source['records']==selected and worker_source['parameter_loads']==2 and worker_source['feature_hashes']==feature_hashes,'Worker source aggregate')
    require(read(execution/f'seed_{seed}_features.json')['hashes']==feature_hashes,'Worker feature file')
    artifacts[str(execution/f'seed_{seed}_result.json')]=sha(execution/f'seed_{seed}_result.json')
    artifacts[str(execution/f'seed_{seed}_features.json')]=sha(execution/f'seed_{seed}_features.json')
    for path,digest in artifacts.items():require(sha(path)==digest,'Source changed during worker audit')
    result=dict(seed=seed,status='passed',parameter_loads=loads,records=receipts,condition_summaries=condition_summaries,
        artifacts_sha256=artifacts,independent_forward_worlds=sum(r['independent_forward_worlds'] for r in receipts),
        independent_module_samples=sum(r['independent_module_samples'] for r in receipts),max_probability_error=max(r['max_probability_error'] for r in receipts))
    dump_new(output/f'seed_{seed}.json',result)
    return dict(seed=seed,path=str(output/f'seed_{seed}.json'),sha256=sha(output/f'seed_{seed}.json'),
        independent_forward_worlds=result['independent_forward_worlds'],independent_module_samples=result['independent_module_samples'],max_probability_error=result['max_probability_error'])


def audit(run,output):
    run=Path(run).resolve();output=Path(output).resolve();execution=run/'execution'
    require(read(execution/'status.json')['status']=='completed','Main not completed; no audit forward')
    require(not (execution/'failure.json').exists(),'Main has failure artifact')
    plan=read(run/'plan.json');main=read(execution/'results.json')
    require(sha(run/'plan.json')==TRANSFER_PLAN_SHA,'Predetermined transfer plan SHA')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Numerical runtime differs')
    require(main['status']=='completed' and main['plan_sha256']==sha(run/'plan.json')==read(run/'freeze.json')['plan_sha256'],'Completed source chain')
    require(read(execution/'started.json')['plan_sha256']==sha(run/'plan.json'),'Started plan chain')
    artifacts={str(p):sha(p) for p in (run/'plan.json',run/'freeze.json',execution/'results.json',execution/'status.json',execution/'started.json')}
    root=HERE.parents[1]
    for path,digest in plan['sources_sha256'].items():
        require(sha(path)==digest,'Frozen source changed');snapshot=run/'source_snapshot'/Path(path).relative_to(root)
        require(sha(snapshot)==digest,'Source snapshot changed');artifacts[path]=digest;artifacts[str(snapshot)]=digest
    for path,digest in plan['inputs_sha256'].items():require(sha(path)==digest,'Input source changed');artifacts[path]=digest
    original=Path(plan['source_directory']);require(sha(original/'execution/results.json')==MAIN_RESULT_SHA,'Wrong prior main result')
    prior=read(original/'audit_execution_001/verification.json');require(prior['status']=='passed' and prior['audit_source_sha256']==CONTEXT_AUDIT_SHA,'Prior independent proof')
    require(sha(original/'audit_execution_001/verification.json')=='7c0eab86edff602ec6def4e54e92cacf3403cb495e39b195594b0185caeb250a','Prior proof receipt anchor')
    require([(p['seed'],p['condition']) for p in plan['policies']]==list(product(SEEDS,CONDITIONS)),'All16 policies')
    context,_=references();task_specs,_,_=context.independent_specs();dataset_dir=Path(plan['dataset_directory']);manifest=read(dataset_dir/'manifest.json')
    n_budget=file_budget=0;dataset_receipts={};expected_grid=[]
    for part in PARTS:
        spec=load_npz(dataset_dir/(part+'.npz'));expected,meta=independent_dataset(task_specs[part])
        require(set(spec)==set(expected),'Dataset fields')
        for key in spec:
            if spec[key].dtype.kind=='f':close(spec[key],expected[key],'Independent dataset '+key)
            else:require(spec[key].dtype==expected[key].dtype and np.array_equal(spec[key],expected[key]),'Independent dataset '+key)
        require(sha(dataset_dir/(part+'.npz'))==manifest['partitions'][part]['array_sha256'],'Dataset manifest bytes')
        require(sha(dataset_dir/(part+'.json'))==manifest['partitions'][part]['metadata_sha256'],'Dataset metadata bytes')
        dataset_receipts[part]=dict(meta,arrays_sha256={key:context.array_sha(value) for key,value in spec.items()})
        n_budget+=8*4*(meta['n_donors']+1)*meta['n_rows'];file_budget+=8*4*(meta['n_donors']+1)
        del spec,expected
    require(n_budget==184135680 and file_budget==1536,'Independent full budget')
    for seed,cond in product(SEEDS,CONDITIONS):
        for part in PARTS:
            k=dataset_receipts[part]['n_donors']
            modes=[('sham_both',-1),('local_opposite_both',-1)]+[(mode,shift) for shift in range(k) for mode in MODES]
            expected_grid.extend((seed,cond,part,mode,shift,b) for mode,shift in modes for b in (0,1))
    require([(r['seed'],r['condition'],r['partition'],r['mode'],r['shift_index'],r['direction']) for r in main['records']]==expected_grid,'Canonical complete3072 records')
    require(len(main['natural_references'])==64,'64 natural references')
    refs={(r['seed'],r['condition'],r['partition']):r for r in main['natural_references']}
    require(set(refs)==set(product(SEEDS,CONDITIONS,PARTS)),'Natural reference grid')
    for policy in plan['policies']:
        for part,path in policy['endpoints'].items():
            r=refs[policy['seed'],policy['condition'],part]
            require(r['path']==path and r['data_sha256']==prior['artifacts_sha256'][path] and r['worlds']==task_specs[part]['world_count'] and r['new_forward_worlds']==0,'Natural reference provenance')
    config=plan['config']
    for key,value in dict(actual_intervention_files=1536,silent_alias_records=1536,reused_natural_files=64,new_intervention_worlds=n_budget,new_network_samples=n_budget*6,
        reused_natural_worlds=12386304,checkpoint=6000,training_updates=0,new_initializations=0,new_natural_forward_worlds=0,worker_count=4).items():require(config[key]==value,'Plan budget '+key)
    preflight=read(HERE/'independent_preflight_001.json');tree=ast.parse(Path(__file__).read_text())
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in preflight['kernel_ast_sha256']:
            require(sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==preflight['kernel_ast_sha256'][node.name],'Tested kernel AST changed')
    with multiprocessing.get_context('spawn').Pool(4) as pool:workers=pool.map(audit_worker,[(s,str(run),str(output)) for s in SEEDS])
    require(sum(w['independent_forward_worlds'] for w in workers)==n_budget and sum(w['independent_module_samples'] for w in workers)==n_budget*6,'Actual audit forward budget')
    for key,value in dict(actual_intervention_files=1536,silent_aliases=1536,new_intervention_worlds=n_budget,new_network_samples=n_budget*6,parameter_loads=8,training_updates=0).items():require(main[key]==value,'Main budget '+key)
    all_values={w['seed']:read(w['path'])['condition_summaries'] for w in workers};paired={}
    for part in PARTS:
        paired[part]={}
        for layer in ('eligible','all_other'):
            rows=[]
            for seed in SEEDS:
                cells={c:all_values[seed][c][part][layer]['macro_contrast'] for c in CONDITIONS}
                rows.append(dict(seed=seed,cells=cells,PL_minus_LL=cells['PL_live']['counterfactual_apt']-cells['LL_live']['counterfactual_apt'],
                    C_PL_minus_LL=cells['PL_live']['C_counterfactual_apt']-cells['LL_live']['C_counterfactual_apt']))
            paired[part][layer]=dict(seed_rows=rows,equal_seed_mean=float(np.mean([r['PL_minus_LL'] for r in rows])),C_equal_seed_mean=float(np.mean([r['C_PL_minus_LL'] for r in rows])))
    for path,digest in artifacts.items():require(sha(path)==digest,'Artifact changed during complete audit')
    return dict(status='passed',audit_source_sha256=sha(__file__),preflight_sha256=sha(HERE/'independent_preflight_001.json'),plan_sha256=sha(run/'plan.json'),
        artifacts_sha256=artifacts,dataset_checks=dataset_receipts,workers=workers,paired_comparisons=paired,primary=paired['new_needs_and_layouts']['eligible'],
        scope=dict(actual_intervention_npz=1536,silent_alias_records=1536,reused_natural_references=64,independent_forward_worlds=n_budget,
            independent_module_samples=n_budget*6,parameter_loads=8,new_natural_forwards=0,training_updates=0,optimizer_replays=0),
        max_probability_error=max(w['max_probability_error'] for w in workers),
        limits=['Frozen-policy developmental diagnostic; four paired training seeds.',
            'The old complete audit anchors upstream natural messages and donor outputs; no additional natural-policy forward here.',
            'Silent uses exact invisible routing and saved natural outputs, not another observed neural result.',
            'Both endpoint-swap is checked row-wise; C gates are not assumed symmetric across directions.',
            'Independent scalar aggregation covers raw/C appropriateness, probabilities, native/counterfactual scores and selected action diagnostics; it does not claim semantic components or compositionality.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    output=Path(args.out).resolve();require(not output.exists(),'Do not overwrite or resume audit');output.mkdir(parents=True);start=time.perf_counter()
    dump_new(output/'started.json',dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),audit_source_sha256=sha(__file__),automatic_retry=False))
    try:
        result=audit(args.run,output);result.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        dump_new(output/'verification.json',result)
        (output/'独立核验.md').write_text(f"完整独立核验通过：1536实际干预NPZ、1536静默引用、64自然来源。独立后继前向{result['scope']['independent_forward_worlds']:,}行／{result['scope']['independent_module_samples']:,}模块样本；概率最大误差{result['max_probability_error']:.3g}。无新训练或优化器重放。\n\n所有实际双窗路由、完整行动输入、消息、17动作概率、原生和反事实结算及来源SHA均核对；sham、local、silent和remote端点交换按各自适用范围验证。主要eligible权重重新构造，4个seed全部保留。详见verification.json与四份seed收据。\n\n上游自然输出由旧完整独立审计及字节SHA锚定，本次没有新增自然政策前向。C门保留原分母，未将它假定为两方向对称。整包作用不等于组合性。\n")
        print(json.dumps(dict(status='passed',scope=result['scope'],max_probability_error=result['max_probability_error']),ensure_ascii=False),flush=True)
    except BaseException as error:
        dump_new(output/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,
            audit_source_sha256=sha(__file__),automatic_retry=False));raise

if __name__=='__main__':main()
