"""Independent directed W1 audit. No new production numerical imports.

Pinned old independent observations, checkpoints, and native forward are reused.
Mirror cases, delivered W1 routes, six-head replay, proposal selectivity and unique
primary are computed here. Frozen before formal intervention; only completed output may be audited.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_k]='1'
import argparse,hashlib,json,platform,re,time,traceback
from collections import Counter,defaultdict
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import numpy as np
from research_program.triadic_need_response_study import audit_response as response

prior=response.prior;env=prior.env;message=prior.message;old=prior.old
require,read,sha,array_sha,json_bytes,close,compare,finite_tree=prior.require,prior.read,prior.sha,prior.array_sha,prior.json_bytes,prior.close,prior.compare,prior.finite_tree
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SEEDS=(59101,59102,59103,59104);PARTS=env.PARTS;TIMES=(0,6000)
RESPONSE_SHA='589ca516ff962f56f3bda0e23d253bd48a02b2e513d03d84d310eae1d51527ac'
SOURCE_RESULT_SHA='4d80f8395a94a1bedbbbd27a4b287d011db88dbe9cf97ecef4b1e9540fe19f0e'
SOURCE_AUDIT_SHA='3bb1d4894094ec75fcb4ccc3c5187e188bea03c1c834260aa43dd38f6e7084f2'
SOURCE_PLAN_SHA='b5126ca468e16a41db4ee9e96c77f0a7e465c3c2123589d2f423911fc5bf5555'


def references():
    refs=response.references();p=Path(response.__file__).resolve();require(sha(p)==RESPONSE_SHA,'Pinned independent need cases');refs[str(p)]=RESPONSE_SHA;return refs


def build_cases(spec):
    oldcases=response.build_cases(spec);needs=[tuple(n) for n in spec['needs']];lookup={n:i for i,n in enumerate(needs)}
    edges=oldcases['edge_need_indices'];actors=oldcases['changed_person'];axes=oldcases['axis_index'];targets=oldcases['target_pairs']
    edge_index={(u,v,who,axis):k for k,((u,v),who,axis) in enumerate(zip(edges,actors,axes))};seen=set();groups=[]
    for k,((u,v),who,axis) in enumerate(zip(edges,actors,axes)):
        if k in seen:continue
        recipients=[j for j in range(3) if j!=who];mirror=[]
        for index in (u,v):
            n=list(needs[index]);i,j=recipients;n[i],n[j]=n[j],n[i];mirror.append(lookup[tuple(n)])
        other=edge_index[mirror[0],mirror[1],who,axis];require(other!=k and other not in seen,'Disjoint nonfixed mirror')
        require(tuple(needs[u][j] for j in recipients)<tuple(needs[mirror[0]][j] for j in recipients),'Canonical recipient context')
        seen.update((k,other));truth=[targets[k],targets[other]]
        require(truth[0]==truth[1][::-1] and truth[0][0]!=truth[0][1],'Truth context reversal')
        compatible=[]
        for c in range(2):
            compatible.append([next(d for d in range(2) if j in prior.PAIRS[truth[c][d]]) for j in recipients])
        groups.append(dict(indices=[[u,v],mirror],sender=who,axis=axis,recipients=recipients,truth=truth,compatible=compatible))
    require(len(seen)==len(edges),'Complete old-edge coverage')
    groups.sort(key=lambda g:(g['sender'],g['axis'],needs[g['indices'][0][0]][g['sender']],needs[g['indices'][0][1]][g['sender']],*(needs[g['indices'][0][0]][j] for j in g['recipients'])))
    G=len(groups);L=len(spec['layouts']);O=len(spec['private_sites']);B=L*O
    require(L%2==0,'Even layout support')
    strata=[]
    for who,axis in product(range(3),repeat=2):
        ids=[i for i,g in enumerate(groups) if g['sender']==who and g['axis']==axis]
        strata.append(dict(sender=who,axis=response.AXES[axis],axis_index=axis,group_indices=ids,group_count=len(ids),group_background_count=len(ids)*B,cross_cells=len(ids)*B*8,sham_cells=len(ids)*B*4,empty=not ids))
    return dict(schema='mirrored_need_W1_packet_cases_v1',spec_sha256=oldcases['spec_sha256'],state_order=oldcases['state_order'],
        original_need_edges=len(edges),group_background_count=G*B,cross_cells=G*B*8,sham_cells=G*B*4,
        background_layout_owner_indices=[list(x) for x in product(range(L),range(O))],donor_layout_indices=[(li+L//2)%L for li in range(L)],
        strata=strata,empty_strata=[dict(sender=r['sender'],axis=r['axis']) for r in strata if r['empty']],need_target_pairs=oldcases['need_target_pairs'],
        actor_inputs=False,policy_dependent_selection=False,partition=spec['partition'],needs=spec['needs'],layouts=spec['layouts'],private_sites=spec['private_sites'],world_count=spec['world_count'],
        group_count=G,n_backgrounds=B,group_need_indices=[g['indices'] for g in groups],sender=[g['sender'] for g in groups],axis_index=[g['axis'] for g in groups],
        recipients=[g['recipients'] for g in groups],compatible_d=[g['compatible'] for g in groups],truth_pair_indices=[g['truth'] for g in groups],
        donor_background_indices=[((li+L//2)%L)*O+oi for li,oi in product(range(L),range(O))])


def expand(cases,mode,start=0,stop=None):
    require(mode in ('cross','sham'),'Intervention mode');G=cases['group_count'];B=cases['n_backgrounds'];width=8 if mode=='cross' else 4
    n=G*B*width;stop=n if stop is None else stop;require(0<=start<=stop<=n,'Canonical intervention slice')
    ids=np.arange(start,stop,dtype=np.int64);group=ids//(B*width);bg=ids//width%B;local=ids%width
    if mode=='cross':context=local//4;own=local//2%2;packet=local%2
    else:context=local//2;own=local%2;packet=own
    ni=np.asarray(cases['group_need_indices'],np.int64)
    receiver=ni[group,context,own]*B+bg
    donor=ni[group,0,packet]*B+np.asarray(cases['donor_background_indices'])[bg] if mode=='cross' else receiver
    return dict(group_index=group,background_index=bg,context_index=context,self_endpoint=own,packet_endpoint=packet,
        sender=np.asarray(cases['sender'],np.int64)[group],recipient_agents=np.asarray(cases['recipients'],np.int64)[group],
        receiver_state_indices=receiver,donor_state_indices=donor,
        donor_background_index=np.asarray(cases['donor_background_indices'],np.int64)[bg] if mode=='cross' else bg,
        axis_index=np.asarray(cases['axis_index'],np.int64)[group],compatible_d=np.asarray(cases['compatible_d'],np.int64)[group,context],
        receiver_target_pair=np.asarray(cases['truth_pair_indices'],np.int64)[group,context,own])


def first_routes(native_messages,senders,donor_w1):
    """Pure route reconstruction: all original routes, outward sender patch only."""
    n=len(native_messages);rows=np.arange(n);senders=np.asarray(senders,np.int64);cue=np.asarray(donor_w1,np.int8)
    require(native_messages.shape==(n,2,3,4) and cue.shape==(n,4),'Authenticated trace/cue shapes')
    route1=message.route(native_messages[:,0],True)
    for viewer in range(3):
        use=np.flatnonzero(senders!=viewer)
        for offset in range(32):route1[use,viewer,32*senders[use]+offset]=0.
        for pos in range(4):route1[use,viewer,32*senders[use]+8*pos+cue[use,pos]]=1.
    require(np.all(route1[:,:,96:]==1),'Visibility remains unchanged')
    native_routes=message.route(native_messages[:,0],True)
    require(np.array_equal(route1[rows,senders],native_routes[rows,senders]),'Sender self/inbound route retained')
    return route1


def action_distribution(net,local):
    h=np.tanh(local@net['W1']+net['b1']);h=np.tanh(h@net['W2']+net['b2']);z=h@net['W3']+net['b3']
    require(np.isfinite(z).all(),'Independent action logits finite')
    z-=z.max(-1,keepdims=True);mass=np.exp(z);normalizer=mass.sum(-1,keepdims=True)
    return mass/normalizer,z-np.log(normalizer)


def replay(networks,states,native_messages,senders,donor_w1):
    """Six heads; patch outward W1 only; second window is freshly generated."""
    x=env.features(states,'PL');n=len(states);rows=np.arange(n);senders=np.asarray(senders,np.int64);cue=np.asarray(donor_w1,np.int8)
    route1=first_routes(native_messages,senders,cue)
    second_inputs=np.concatenate((x,route1),axis=-1)
    second_prob=np.stack([message.probabilities(networks[3*a+1],second_inputs[:,a]) for a in range(3)],axis=1)
    second=second_prob.argmax(-1).astype(np.int8);route2=message.route(second,True)
    action_inputs=np.concatenate((x,route1,route2),axis=-1);prob=[];logs=[]
    for actor in range(3):
        pp,lp=action_distribution(networks[3*actor+2],action_inputs[:,actor]);prob.append(pp);logs.append(lp)
    p=np.stack(prob,axis=1);lp=np.stack(logs,axis=1)
    generated=np.stack((native_messages[:,0],second),axis=1)
    delivered=np.broadcast_to(generated[:,:,None,:,:],(n,2,3,3,4)).copy()
    for viewer in range(3):
        use=np.flatnonzero(senders!=viewer);delivered[use,0,viewer,senders[use]]=cue[use]
    return dict(generated_messages=generated,delivered_tokens=delivered,delivery_visibility=np.ones((n,2,3,3),bool),
        action_probabilities=p,action_log_probabilities=lp,action_indices=p.argmax(-1).astype(np.int16),
        first_routes=route1,second_routes=route2,independent_module_samples=6*n,
        sender2_ties=int(((second_prob==second_prob.max(-1,keepdims=True)).sum(-1)>1).sum()))


def recipient_probabilities(probabilities,senders,recipients):
    n=len(probabilities);output=np.zeros((n,2));senders=np.asarray(senders);recipients=np.asarray(recipients);rows=np.arange(n)
    for actor in range(3):
        for slot in range(2):
            use=rows[recipients[:,slot]==actor]
            for target in range(3):
                if actor==target:continue
                ix=use[senders[use]==target];other=[j for j in range(3) if j!=actor];offset=other.index(target)
                choices=np.arange(1+offset,17,2)
                mask=np.zeros(17,bool);mask[choices]=True
                output[ix,slot]=(probabilities[ix,actor]*mask).sum(1)
    return output


def score_arrays(cases,p):
    G=cases['group_count'];B=cases['n_backgrounds'];p=np.asarray(p)
    require(p.shape==(G*B*8,2) and np.isfinite(p).all() and np.all((p>=0)&(p<=1+2e-12)),'Recipient proposal probabilities')
    q=p.reshape(G,B,2,2,2,2);average=q.mean(axis=3) # G,B,context,packet,recipient
    compatible=np.asarray(cases['compatible_d']);good=np.empty((G,B,2,2));bad=np.empty_like(good);delta=np.empty_like(good)
    for g,c,j in product(range(G),range(2),range(2)):
        d=compatible[g,c,j];good[g,:,c,j]=average[g,:,c,d,j];bad[g,:,c,j]=average[g,:,c,1-d,j]
        delta[g,:,c,j]=(q[g,:,c,:,d,j]-q[g,:,c,:,1-d,j]).mean(axis=1)
    return dict(delta=delta,compatible_probability=good,incompatible_probability=bad,L=delta.min(axis=(2,3)),D=delta.mean(axis=(2,3)))


def score(cases,p):
    arrays=score_arrays(cases,p);L=arrays['L'];D=arrays['D'];delta=arrays['delta'];B=cases['n_backgrounds'];G=cases['group_count']
    metrics=dict(L=L,D=D,mean_compatible_probability=arrays['compatible_probability'].mean(axis=(2,3)),
        mean_incompatible_probability=arrays['incompatible_probability'].mean(axis=(2,3)),strict_positive_L_fraction=(L>0).astype(float))
    strata=[]
    for actor,axis in product(range(3),repeat=2):
        mask=(np.asarray(cases['sender'])==actor)&(np.asarray(cases['axis_index'])==axis);count=int(mask.sum());require(count>0,'All nine nonempty strata')
        row=dict(sender=actor,axis=response.AXES[axis],axis_index=axis,group_count=count,group_background_count=count*B,empty=False,
            positive_L_group_backgrounds=int((L[mask]>0).sum()),zero_L_group_backgrounds=int((L[mask]==0).sum()),negative_L_group_backgrounds=int((L[mask]<0).sum()),
            mean_four_deltas=delta[mask].mean(axis=1).mean(axis=0).tolist())
        row.update({key:float(v[mask].mean(axis=1).mean()) for key,v in metrics.items()});strata.append(row)
    result={key:float(np.mean([s[key] for s in strata])) for key in metrics}
    result.update(schema='mirrored_need_W1_selectivity_v1',partition=cases['partition'],group_count=G,n_backgrounds=B,group_background_count=G*B,cross_cells=G*B*8,
        complete_nine_strata=True,empty_strata=[],strata=strata,raw_group_background_counts=dict(positive_L=int((L>0).sum()),zero_L=int((L==0).sum()),negative_L=int((L<0).sum())))
    return result,arrays


def primary(values):
    require(set(values)==set(product(SEEDS,TIMES)),'Eight complete double-holdout score cells')
    rows=[]
    for seed in SEEDS:
        first=values[seed,0];final=values[seed,6000]
        require(all(np.isfinite(v[k]) and -1<=v[k]<=1 for v in (first,final) for k in ('L','D')),'Finite selectivity')
        rows.append(dict(seed=seed,L_initial=first['L'],L_final=final['L'],L_final_minus_initial=final['L']-first['L'],D_initial=first['D'],D_final=final['D']))
    return dict(name='mean_four_seed_final_complete_double_holdout_L',partition=PARTS[-1],independent_societies=4,by_seed=rows,
        mean_L_final=float(np.mean([v['L_final'] for v in rows])),auxiliary=dict(mean_L_initial=float(np.mean([v['L_initial'] for v in rows])),
        mean_L_final_minus_initial=float(np.mean([v['L_final_minus_initial'] for v in rows])),mean_D_final=float(np.mean([v['D_final'] for v in rows])),mean_D_initial=float(np.mean([v['D_initial'] for v in rows]))))


def budget(specs,cases):
    W=sum(s['world_count'] for s in specs.values());C=sum(c['group_count']*c['n_backgrounds']*8 for c in cases.values());S=C//2
    require((W,C,S)==(774144,1009152,504576),'Fixed support budget')
    return dict(policy_states=8,trained_policies=4,training_updates=0,checkpoint_files_read=8,initial_natural_bank_files=16,reused_final_natural_bank_files=16,
        new_natural_worlds=4*W,reused_final_natural_worlds=4*W,new_natural_module_samples=4*W*9,cross_files=32,sham_files=32,group_metric_files=32,
        cross_rows=8*C,sham_rows=8*S,intervention_rows=8*(C+S),intervention_module_samples=8*(C+S)*6,
        total_new_module_samples=4*W*9+8*(C+S)*6,generated_data_files=112,formal_random_draws=0)


def npz_record(entry,path):
    path=Path(path);require(Path(entry['path']).resolve()==path.resolve() and sha(path)==entry['sha256'],'NPZ identity/hash')
    value=env.load_npz(path)
    if 'arrays' in entry:require(entry['arrays']=={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in value.items()},'NPZ array manifest')
    return value


def natural_bank(entry,path,states,networks,initial):
    value=npz_record(entry,path);n=len(states);ids=np.arange(n,dtype=np.int64)
    require(np.array_equal(value['states'],states) and value['states'].dtype==np.int16,'Natural complete states')
    require(np.array_equal(value['state_indices'],ids) and value['state_indices'].dtype==np.int64,'Natural complete state order')
    m=value['messages'];p=value['action_probabilities'];act=value['action_indices']
    require(m.dtype==np.int8 and m.shape==(n,2,3,4) and np.all((m>=0)&(m<8)),'Natural messages')
    require(p.dtype==np.float64 and p.shape==(n,3,17) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Natural probabilities')
    close(p.sum(-1),np.ones((n,3)),'Natural policy normalization')
    require(act.dtype==np.int16 and np.array_equal(act,p.argmax(-1)),'Natural action identity')
    require(entry['worlds']==n and entry['reused'] is (not initial) and entry['new_module_samples']==(9*n if initial else 0),'Native source/new forward distinction')
    maximum=0.;ties=0
    if initial:
        settled=prior.settle(states,act,'reciprocal')
        require(set(value)==set(settled)|{'states','state_indices','messages','action_probabilities','action_indices'},'New initial bank exact schema')
        for k,v in settled.items():require(value[k].dtype==v.dtype and np.array_equal(value[k],v),'Initial native settlement '+k)
        for begin in range(0,n,1024):
            sl=slice(begin,begin+1024);mm,pp,_,tt=old.forward_final(networks,states[sl],'PL',True);ties+=tt
            require(np.array_equal(mm,m[sl]) and np.array_equal(pp.argmax(-1),act[sl]),'Full initial native discrete replay')
            maximum=max(maximum,close(pp,p[sl],'Full initial native probability replay'))
        compare(entry['physical_summary'],prior.summarize(states,act,'reciprocal'),'Initial native summary/')
    return {k:value[k] for k in ('states','state_indices','messages','action_probabilities','action_indices')},dict(worlds=n,independent_module_samples=9*n if initial else 0,
        max_probability_error=maximum,sender_ties=ties,initial_native_freshly_replayed=initial,reused_final_bound_to_previous_audit=not initial)


def context_check(cases,bank):
    group=np.asarray(cases['group_need_indices']);actors=np.asarray(cases['sender']);B=cases['n_backgrounds'];count=0
    for b in range(B):
        for r in range(2):
            left=bank['messages'][group[:,0,r]*B+b,0,actors];right=bank['messages'][group[:,1,r]*B+b,0,actors]
            require(np.array_equal(left,right),'Native sender W1 must not depend on other private needs');count+=len(group)
    return dict(compared_sender_packets=count,all_equal=True,neural_forward_samples=0)


def intervention_record(entry,path,cases,bank,networks,mode):
    value=npz_record(entry,path);meta=expand(cases,mode);n=len(meta['sender']);native=bank['messages'];recipient=np.empty((n,2));errors=defaultdict(float);ties=0
    require(entry['mode']==mode and entry['rows']==n and entry['new_module_samples']==6*n,'Intervention fixed grid/budget')
    for k,v in meta.items():require(value[k].dtype.kind in 'iu' and np.array_equal(value[k],v),'Independent case metadata '+k)
    receiver=meta['receiver_state_indices'];donor=meta['donor_state_indices'];sender=meta['sender'];recipients=meta['recipient_agents']
    states=bank['states'][receiver];p=value['action_probabilities'];act=value['action_indices']
    require(value['receiver_states'].dtype==np.int16 and np.array_equal(value['receiver_states'],states),'Receiver observations source')
    require(p.dtype==np.float64 and p.shape==(n,3,17) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Saved intervened probability domain')
    require(act.dtype==np.int16 and np.array_equal(act,p.argmax(-1)),'Intervened greedy actions')
    require(value['overwrite_windows'].dtype==bool and np.array_equal(value['overwrite_windows'],[True,False]),'Only W1 overwritten')
    for key,shape,dtype in (('generated_messages',(n,2,3,4),np.int8),('delivered_tokens',(n,2,3,3,4),np.int8),('donor_packets',(n,2,4),np.int8),('delivery_visibility',(n,2,3,3),bool),('recipient_partner_probs',(n,2),np.float64)):
        require(value[key].shape==shape and value[key].dtype==dtype,'Intervention array shape/dtype '+key)
    for begin in range(0,n,1024):
        sl=slice(begin,begin+1024);ri=receiver[sl];di=donor[sl];a=sender[sl];packets=native[di,:,a]
        require(np.array_equal(value['donor_packets'][sl],packets),'Bound natural donor packets')
        replayed=replay(networks,states[sl],native[ri],a,packets[:,0]);ties+=replayed['sender2_ties']
        for key in ('generated_messages','delivered_tokens','delivery_visibility','action_indices'):
            require(np.array_equal(value[key][sl],replayed[key]),'Independent W1-only replay '+key)
        errors['probability']=max(errors['probability'],close(p[sl],replayed['action_probabilities'],'Independent six-head replay'))
        # Saved probabilities already matched a fresh independent replay; sum the
        #17 entries in their saved order for exact cancellation/sign bookkeeping.
        rp=recipient_probabilities(p[sl],a,recipients[sl]);recipient[sl]=rp
        errors['recipient']=max(errors['recipient'],close(value['recipient_partner_probs'][sl],rp,'Independent recipient action event'))
        if mode=='sham':
            require(np.array_equal(value['generated_messages'][sl],native[ri]),'Native sham message identity')
            require(np.array_equal(act[sl],bank['action_indices'][ri]),'Native sham action identity')
            error=float(np.max(np.abs(p[sl]-bank['action_probabilities'][ri])));require(error<=1e-13,'Native sham probability identity');errors['sham']=max(errors['sham'],error)
    settled=prior.settle(states,act,'reciprocal')
    for k,v in settled.items():require(value[k].dtype==v.dtype and np.array_equal(value[k],v),'Intervened physical settlement '+k)
    require(set(value)==set(meta)|set(settled)|{'receiver_states','generated_messages','delivered_tokens','delivery_visibility','donor_packets','action_probabilities','action_indices','recipient_partner_probs','overwrite_windows'},'Complete intervention NPZ schema')
    compare(entry['physical_summary'],prior.summarize(states,act,'reciprocal'),'Intervened physical summary/')
    if mode=='sham':compare(entry['native_replay'],dict(all_messages_equal=True,all_actions_equal=True,max_probability_absolute_error=errors['sham'],checked_rows=n),'Full sham check/')
    return recipient,dict(rows=n,independent_module_samples=n*6,max_errors=dict(errors),sender2_ties=ties)


def audit(run,expected_plan_sha):
    run=Path(run).resolve();execution=run/'execution';refs=references();artifacts=dict(refs)
    require(re.fullmatch('[0-9a-f]{64}',expected_plan_sha) is not None,'Explicit frozen plan hash')
    def bind(path,digest=None):
        path=Path(path).resolve();actual=sha(path)
        if digest is not None:require(actual==digest,'Changed artifact '+str(path))
        artifacts[str(path)]=actual;return actual
    for path in (run/'plan.json',run/'prepared.json',run/'freeze.json',execution/'started.json',execution/'status.json',execution/'results.json',HERE/'audit_freeze_001.json'):bind(path)
    plan=read(run/'plan.json');prepared=read(run/'prepared.json');result=read(execution/'results.json');status=read(execution/'status.json');finite_tree(result)
    require(status['status']==result['status']=='completed' and not (execution/'failure.json').exists(),'No incomplete-run audit')
    require(status['results_sha256']==sha(execution/'results.json'),'Completed result hash')
    require(sha(run/'plan.json')==expected_plan_sha==read(run/'freeze.json')['plan_sha256']==result['plan_sha256']==read(execution/'started.json')['plan_sha256'],'Plan chain')
    require(sha(run/'prepared.json')==plan['prepared_sha256']==read(run/'freeze.json')['prepared_sha256'],'Prepared chain')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Exact numerical runtime')
    compare(plan['config'],dict(schema='directed_W1_context_reversal_v1',seeds=list(SEEDS),checkpoints=list(TIMES),partitions=list(PARTS),condition='PL_live',execution_rule='reciprocal',
        overwrite_windows=[True,False],primary='mean_four_seed_final_complete_double_holdout_L',training_updates=0,worker_count=4,chunk_size=1024,automatic_followon_experiment=False),'Frozen core plan/')
    af=read(HERE/'audit_freeze_001.json');require(af['status']=='frozen_before_formal_intervention' and af['plan_sha256']==expected_plan_sha,'Independent source frozen before model intervention')
    require(af['at']<read(execution/'started.json')['at'],'Pre-execution audit freeze time')
    require(set(af['source_sha256'])=={str(HERE/'audit_execution.py'),str(HERE/'test_audit.py'),str(HERE/'audit_preflight_001.json')},'Audit source coverage')
    for p,d in af['source_sha256'].items():bind(p,d)
    for p,d in plan['source_sha256'].items():bind(p,d);bind(run/'source_snapshot'/Path(p).relative_to(ROOT),d)
    for p,d in plan['inputs_sha256'].items():bind(p,d)
    original=ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json';bind(original,response.ORIGINAL_SHA)
    specs=read(original)['partitions'];independent,_,_=env.independent_specs()
    require(prepared['partitions']==specs and prepared['original_prepared_sha256']==response.ORIGINAL_SHA and prepared['independent_societies']==4,'Anchored static metadata')
    for part in PARTS:compare(specs[part],independent[part],'Independent partition support/')
    cc={p:build_cases(specs[p]) for p in PARTS};compare(prepared['cases'],cc,'Independent mirror support/')
    expected_budget=budget(specs,cc);require(prepared['budget']==result['budget']==expected_budget,'Complete no-training budget')
    source=ROOT/'research_program/triadic_private_partner_study/results/private_001';bind(source/'plan.json',SOURCE_PLAN_SHA);bind(source/'execution/results.json',SOURCE_RESULT_SHA);bind(source/'audit_execution_001/verification.json',SOURCE_AUDIT_SHA)
    source_result=read(source/'execution/results.json');source_audit=read(source/'audit_execution_001/verification.json')
    require(source_result['status']=='completed' and source_audit['status']=='passed','Completed independently audited source')
    oldruns={(r['seed'],r['condition']):r for r in source_result['runs']};expected_meta=[]
    for seed,t in product(SEEDS,TIMES):
        oldrow=oldruns[seed,'PL_live'];directory=source/'execution'/f'seed_{seed}_PL_live';checkpoint=directory/f'checkpoint_{t:04d}.npz';digest=next(m['checkpoint_sha256'] for m in oldrow['monitor'] if m['update']==t)
        require(source_audit['artifacts_sha256'][str(checkpoint)]==digest,'Old checkpoint audit binding')
        meta=dict(seed=seed,checkpoint=t,checkpoint_path=str(checkpoint),checkpoint_sha256=digest,parameter_sha256=oldrow['initial_parameter_sha256'] if t==0 else oldrow['final_parameter_sha256'],natural_banks={})
        if t==6000:
            for part in PARTS:
                bank=oldrow['final'][part]['natural'];p=Path(bank['path']);require(source_audit['artifacts_sha256'][str(p)]==bank['data_sha256'],'Old complete final bank audit binding')
                meta['natural_banks'][part]=dict(path=str(p),sha256=bank['data_sha256'],worlds=bank['worlds'])
        expected_meta.append(meta)
    require(plan['policy_states']==expected_meta,'Exact preexisting checkpoint/final-bank manifest')
    require([(r['seed'],r['checkpoint']) for r in result['policy_states']]==list(product(SEEDS,TIMES)),'Canonical eight completed policy states')
    states={p:env.packed(specs[p]) for p in PARTS};hashes={}
    for p,s in states.items():hashes[p]=dict(packed_states=array_sha(s),rewards=array_sha(env.rewards(s)),x_PL=array_sha(env.features(s,'PL')))
    require(result['array_hashes']==hashes,'Independent complete PL world arrays')
    for seed in SEEDS:
        p=execution/f'seed_{seed}'/'array_hashes.json';bind(p);require(read(p)==hashes,'Worker arrays')
        p=execution/f'seed_{seed}'/'status.json';bind(p);require(read(p)['status']=='completed','Completed worker')
    counts=Counter();errors=defaultdict(float);receipts={};primary_cells={};paths=set();parameter_hashes={}
    for meta,row in zip(expected_meta,result['policy_states']):
        seed=meta['seed'];t=meta['checkpoint'];directory=execution/f'seed_{seed}'/f'checkpoint_{t:04d}'
        path=directory/'result.json';bind(path);require(read(path)==row,'Per-policy result');require(set(row['parts'])==set(PARTS),'Complete four policy partitions')
        require(row['checkpoint_sha256']==meta['checkpoint_sha256'] and row['parameter_sha256']==meta['parameter_sha256'],'Policy identity')
        bind(meta['checkpoint_path'],meta['checkpoint_sha256']);networks,_,_=message.checkpoint(meta['checkpoint_path'],t,seed)
        require(message.network_hash(networks)==meta['parameter_sha256'],'Actual independent loaded parameters');parameter_hashes[seed,t]=message.network_hash(networks);counts['checkpoint_files']+=1
        for part in PARTS:
            r=row['parts'][part];pdir=directory/part;p=pdir/'result.json';bind(p);require(read(p)==r,'Per-part result identity')
            bank_entry=r['natural_bank'];bank_path=pdir/'natural_bank.npz' if t==0 else Path(meta['natural_banks'][part]['path'])
            if t==6000:require(bank_entry['sha256']==meta['natural_banks'][part]['sha256'],'Reused bank hash')
            bind(bank_path,bank_entry['sha256']);bank,receipt=natural_bank(bank_entry,bank_path,states[part],networks,t==0);receipts[str(bank_path)]=receipt
            counts['initial_native_banks' if t==0 else 'reused_final_banks']+=1;counts['initial_native_module_samples']+=receipt['independent_module_samples'];errors['native_probability']=max(errors['native_probability'],receipt['max_probability_error'])
            if t==0:paths.add(bank_path)
            compared=context_check(cc[part],bank);compare(r['context_packet_check'],compared,'Complete native W1 context equality/');counts['context_packet_comparisons']+=compared['compared_sender_packets']
            for mode in ('cross','sham'):
                p=pdir/(mode+'.npz');bind(p,r[mode]['sha256']);paths.add(p);probabilities,receipt=intervention_record(r[mode],p,cc[part],bank,networks,mode);receipts[str(p)]=receipt
                counts[mode+'_files']+=1;counts[mode+'_rows']+=receipt['rows'];counts['intervention_module_samples']+=receipt['independent_module_samples']
                for k,v in receipt['max_errors'].items():errors[k]=max(errors[k],v)
                if mode=='cross':
                    calculated,details=score(cc[part],probabilities);compare(r['metrics'],calculated,'Independent L/D/strata/')
                    mp=pdir/'cross_metrics.npz';entry=r['metrics']['per_group_background_file'];bind(mp,entry['sha256']);paths.add(mp);saved=npz_record(entry,mp)
                    require(set(saved)==set(details),'All group/background metric arrays')
                    for k,v in details.items():close(saved[k],v,'Independent full numeric metric array '+k)
                    counts['metric_files']+=1
                    if part==PARTS[-1]:primary_cells[seed,t]=calculated
            require(set(pdir.glob('*.npz'))=={p for p in paths if p.parent==pdir},'No extra per-part NPZ')
            print(json.dumps(dict(stage='directed_partition_audited',seed=seed,checkpoint=t,part=part)),flush=True)
        require(message.network_hash(networks)==meta['parameter_sha256'],'Audit leaves source parameters unchanged')
    require(len(set(parameter_hashes[s,0] for s in SEEDS))==4,'Four initialization blocks')
    expected_scope=dict(checkpoint_files=8,initial_native_banks=16,reused_final_banks=16,initial_native_module_samples=27869184,
        context_packet_comparisons=2018304,cross_files=32,sham_files=32,cross_rows=8073216,sham_rows=4036608,intervention_module_samples=72658944,metric_files=32)
    require(dict(counts)==expected_scope,'Complete independent actual scope')
    measured=dict(new_natural_module_samples=counts['initial_native_module_samples'],intervention_module_samples=counts['intervention_module_samples'],total_new_module_samples=counts['initial_native_module_samples']+counts['intervention_module_samples'])
    require(result['measured_module_samples']==measured and measured['total_new_module_samples']==100528128,'Exact full forward accounting')
    calculated=primary(primary_cells);compare(result['primary'],calculated,'Unique final L primary/')
    require(len(paths)==112 and set(execution.rglob('*.npz'))==paths,'All112 generated data files exactly covered')
    for p,d in artifacts.items():require(sha(p)==d,'Artifact changed during audit '+p)
    return dict(status='passed',audit_source_sha256=sha(__file__),plan_sha256=expected_plan_sha,scope=dict(counts),measured_module_samples=measured,max_errors=dict(errors),
        artifacts_sha256=artifacts,historical_helpers_sha256=refs,evaluations=receipts,primary=calculated,
        limits=['No training, gradient or optimizer replay. All16 initial native banks and64 intervention/sham files freshly forwarded with independent features/routes.',
            'The16 final native banks are reused through the completed previous independent audit and exact hashes, not counted as new forwards.',
            'All mirror indices, recipient proposal events, complete group/background arrays and nine-stratum L/D are independently recomputed.',
            'Primary is final L on complete double holdout. Initial L and final-minus-initial are auxiliary, not information-creation measures.',
            'Selectivity is stronger than task necessity and concerns the outward first-window pathway only; no linguistic semantics or composition claim.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--plan-sha',required=True);a=p.parse_args()
    out=Path(a.out).resolve();require(not out.exists(),'Never overwrite audit');out.mkdir(parents=True);start=time.perf_counter()
    (out/'started.json').write_bytes(json_bytes(dict(at=datetime.now(timezone.utc).isoformat(),run=a.run,expected_plan_sha256=a.plan_sha,audit_source_sha256=sha(__file__))))
    try:
        v=audit(a.run,a.plan_sha);v.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        (out/'verification.json').write_bytes(json_bytes(v));print(json.dumps({k:v[k] for k in ('status','scope','max_errors','elapsed_seconds')}))
    except BaseException as e:
        (out/'failure.json').write_bytes(json_bytes(dict(status='failed',error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,audit_source_sha256=sha(__file__),automatic_retry=False)));raise

if __name__=='__main__':main()
