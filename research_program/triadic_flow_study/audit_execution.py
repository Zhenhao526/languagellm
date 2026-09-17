"""Independent same-background I/O audit; no new production numerics imported.

Pinned historical independent helpers supply static worlds, checkpoint parsing,
and MLP arithmetic. Flow cases, W1 routing, exact reciprocal event masses,
semantic action marginals, paired contrasts, and the unique primary are rebuilt.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[_key]='1'
import argparse,json,platform,re,time,traceback
from collections import Counter,defaultdict
from datetime import datetime,timezone
from itertools import product
from pathlib import Path
import numpy as np
from research_program.triadic_directed_message_study import audit_execution as directed

prior=directed.prior;env=directed.env;message=directed.message;old=directed.old
require,read,sha,array_sha,json_bytes,close,compare,finite_tree=(directed.require,directed.read,directed.sha,directed.array_sha,directed.json_bytes,directed.close,directed.compare,directed.finite_tree)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SEEDS=directed.SEEDS;PARTS=env.PARTS
DIRECTED_AUDIT_SHA='31eb4e45c09054afe7dc4c41155ee6d62aca30d5b38eb5cc2636bad5c9f8deb6'
SOURCE_PLAN_SHA=directed.SOURCE_PLAN_SHA;SOURCE_RESULT_SHA=directed.SOURCE_RESULT_SHA;SOURCE_AUDIT_SHA=directed.SOURCE_AUDIT_SHA
CONTRASTS=('S00','S01','S10','S11','I_given_O0','I_given_O1','O_given_I0','O_given_I1','interaction')


def references():
    refs=directed.references();p=Path(directed.__file__).resolve();require(sha(p)==DIRECTED_AUDIT_SHA,'Pinned directed independent helper');refs[str(p)]=DIRECTED_AUDIT_SHA;return refs


def build_cases(spec):
    previous=directed.build_cases(spec)
    keys=('partition','spec_sha256','state_order','needs','layouts','private_sites','world_count','n_backgrounds','background_layout_owner_indices','group_count','group_background_count','original_need_edges','group_need_indices','sender','axis_index','recipients','compatible_d','truth_pair_indices','need_target_pairs','empty_strata')
    out={k:previous[k] for k in keys};rows=[]
    for s in previous['strata']:
        r={k:s[k] for k in ('sender','axis','axis_index','group_indices','group_count','group_background_count','empty')};gb=r['group_background_count']
        r.update(flow_cells=16*gb,arm_cells=4*gb,sham_cells=4*gb);rows.append(r)
    gb=out['group_background_count'];out.update(schema='mirrored_same_background_W1_flow_cases_v1',strata=rows,flow_cells=16*gb,arm_cells=4*gb,sham_cells=4*gb,nonsham_cells=12*gb,
        flow_order='group,background,context,self_endpoint,incoming,outgoing',actor_inputs=False,policy_dependent_selection=False)
    return out


def expand(cases,start=0,stop=None):
    G=cases['group_count'];B=cases['n_backgrounds'];n=G*B*16;stop=n if stop is None else stop
    require(0<=start<=stop<=n,'Complete canonical flow slice');index=np.arange(start,stop,dtype=np.int64)
    g=index//(B*16);b=index//16%B;c=index//8%2;r=index//4%2;incoming=index//2%2;outgoing=index%2
    ni=np.asarray(cases['group_need_indices'],np.int64);truth=np.asarray(cases['truth_pair_indices'],np.int8)
    receiver=ni[g,c,r]*B+b;di=ni[g,1-c,r]*B+b;do=ni[g,c,1-r]*B+b;s=np.asarray(cases['sender'],np.int8)[g]
    return dict(group_index=g,background_index=b,context_index=c,self_endpoint=r,incoming=incoming,outgoing=outgoing,focus_actor=s,sender=s.copy(),
        recipient_agents=np.asarray(cases['recipients'],np.int8)[g],axis_index=np.asarray(cases['axis_index'],np.int8)[g],receiver_state_indices=receiver,
        incoming_donor_state_indices=di,outgoing_donor_state_indices=do,effective_incoming_donor_state_indices=np.where(incoming,di,receiver),effective_outgoing_donor_state_indices=np.where(outgoing,do,receiver),
        receiver_target_pair=truth[g,c,r],incoming_donor_target_pair=truth[g,1-c,r],outgoing_donor_target_pair=truth[g,c,1-r],compatible_d=np.asarray(cases['compatible_d'],np.int8)[g,c],sham=(incoming==0)&(outgoing==0))


def first_routes(native_messages,focal,incoming_w1,outgoing_w1,incoming,outgoing):
    native=np.asarray(native_messages);n=len(native);focal=np.asarray(focal);rows=np.arange(n)
    require(native.shape==(n,2,3,4) and incoming_w1.shape==(n,3,4) and outgoing_w1.shape==(n,4),'Native/candidate W1 shape')
    require(np.all((focal>=0)&(focal<3)) and np.isin(incoming,(0,1)).all() and np.isin(outgoing,(0,1)).all(),'Factorial switches')
    delivered=np.broadcast_to(native[:,0,None,:,:],(n,3,3,4)).copy()
    for actor in range(3):
        use=rows[(focal!=actor)&np.asarray(incoming,dtype=bool)];delivered[use,focal[use],actor]=incoming_w1[use,actor]
        use=rows[(focal!=actor)&np.asarray(outgoing,dtype=bool)];delivered[use,actor,focal[use]]=outgoing_w1[use]
    route=np.zeros((n,3,99),np.float64);route[:,:,96:]=1.
    for viewer,actor,position in product(range(3),range(3),range(4)):
        route[rows,viewer,32*actor+8*position+delivered[:,viewer,actor,position]]=1.
    for actor in range(3):require(np.array_equal(delivered[:,actor,actor],native[:,0,actor]),'Every self W1 retained')
    return route,delivered


action_distribution=directed.action_distribution


def replay(networks,states,native_messages,focal,incoming_w1,outgoing_w1,incoming,outgoing):
    n=len(states);x=env.features(states,'PL');route1,delivered1=first_routes(native_messages,focal,incoming_w1,outgoing_w1,incoming,outgoing)
    x2=np.concatenate((x,route1),axis=-1);second_prob=np.stack([message.probabilities(networks[3*a+1],x2[:,a]) for a in range(3)],axis=1)
    second=second_prob.argmax(-1).astype(np.int8);route2=message.route(second,True);xa=np.concatenate((x,route1,route2),axis=-1)
    p=np.stack([action_distribution(networks[3*a+2],xa[:,a])[0] for a in range(3)],axis=1)
    generated=np.stack((native_messages[:,0],second),axis=1);delivered=np.broadcast_to(generated[:,:,None,:,:],(n,2,3,3,4)).copy();delivered[:,0]=delivered1
    return dict(generated_messages=generated,delivered_tokens=delivered,delivery_visibility=np.ones((n,2,3,3),bool),action_probabilities=p,action_indices=p.argmax(-1).astype(np.int16),independent_module_samples=n*6,
        sender2_ties=int(((second_prob==second_prob.max(-1,keepdims=True)).sum(-1)>1).sum()))


def exact_responses(states,p):
    """24 disjoint reciprocal execution events; ignored actor integrates to1."""
    n=len(states);require(p.shape==(n,3,17) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Action distribution domain');close(p.sum(-1),np.ones((n,3)),'Actions normalized')
    native=env.rewards(states);require(np.all((native==1).sum(1)==1),'Unique original-world full event');target=(native==1).argmax(1)
    event=np.empty((n,24));rows=np.arange(n)
    for k,(i,j) in enumerate(prior.PAIRS):
        for site,dest in product(range(4),range(2)):
            ix=k*8+site*2+dest;ai=1+site*4+dest*2+[a for a in range(3) if a!=i].index(j);aj=1+site*4+dest*2+[a for a in range(3) if a!=j].index(i)
            event[:,ix]=p[:,i,ai]*p[:,j,aj]
    pairs=event.reshape(n,3,8).sum(-1);execution=np.column_stack((1-pairs.sum(-1),pairs));require(np.all(execution>=-2e-12),'Mutually exclusive execution events')
    return dict(full_success_probability=event[rows,target],expected_reward=(native*event).sum(1),execution=execution),target


def action_responses(states,p,focal):
    """Absolute probabilities including waiting; actor slots [focus,u,v]."""
    n=len(states);rows=np.arange(n);focal=np.asarray(focal);actor_order=np.array([[0,1,2],[1,0,2],[2,0,1]],np.int8)[focal]
    out={k:np.zeros((n,3,width)) for k,width in (('partner',4),('site',5),('material',5),('kind',3),('length',3),('destination',3))}
    for slot in range(3):
        actor=actor_order[:,slot];mass=p[rows,actor]
        for val in out.values():val[:,slot,0]=mass[:,0]
        for choice in range(1,17):
            site=(choice-1)//4;dest=(choice-1)//2%2;ordinal=(choice-1)%2
            peer=np.array([[1,2],[0,2],[0,1]],np.int8)[actor,ordinal];rel=(actor_order==peer[:,None]).argmax(1)
            material=states[:,3+site];w=mass[:,choice]
            out['partner'][rows,slot,1+rel]+=w;out['site'][:,slot,1+site]+=w;out['material'][rows,slot,1+material]+=w
            out['kind'][rows,slot,1+material//2]+=w;out['length'][rows,slot,1+material%2]+=w;out['destination'][:,slot,1+dest]+=w
    for key,v in out.items():close(v.sum(-1),np.ones((n,3)),'Marginals preserve waiting '+key)
    for slot in range(3):require(np.all(out['partner'][:,slot,slot+1]==0),'Self partner probability zero')
    return out


def score(cases,values):
    raw=np.asarray(values,dtype=np.float64);G=cases['group_count'];B=cases['n_backgrounds'];tail=raw.shape[1:]
    require(raw.shape[0]==G*B*16 and np.isfinite(raw).all(),'Finite full factorial response')
    v=raw.reshape((G,B,2,2,2,2)+tail);arms={f'S{i}{o}':v[:,:,:,:,i,o] for i,o in product(range(2),repeat=2)}
    pairs=dict(I_given_O0=arms['S10']-arms['S00'],I_given_O1=arms['S11']-arms['S01'],O_given_I0=arms['S01']-arms['S00'],O_given_I1=arms['S11']-arms['S10'])
    pairs['interaction']=pairs['I_given_O1']-pairs['I_given_O0'];arrays={k:x.mean(axis=(2,3)) for k,x in {**arms,**pairs}.items()}
    def num(x):return float(x) if np.ndim(x)==0 else np.asarray(x).tolist()
    strata=[]
    for who,axis in product(range(3),repeat=2):
        ids=np.flatnonzero((np.asarray(cases['sender'])==who)&(np.asarray(cases['axis_index'])==axis));require(len(ids)>0,'All nine strata nonempty')
        s=dict(sender=who,axis=directed.response.AXES[axis],axis_index=axis,group_count=len(ids),group_background_count=len(ids)*B,empty=False)
        s.update({k:num(x[ids].mean(axis=1).mean(axis=0)) for k,x in arrays.items()});strata.append(s)
    result={k:num(np.mean([s[k] for s in strata],axis=0)) for k in arrays}
    result.update(schema='same_background_flow_response_metrics_v1',partition=cases['partition'],response_shape=list(tail),group_count=G,n_backgrounds=B,group_background_count=G*B,flow_cells=G*B*16,arm_cells=G*B*4,complete_nine_strata=True,empty_strata=[],strata=strata)
    return result,arrays


def changed_symbols(native,delivered,focal):
    n=len(native);rows=np.arange(n);delta=delivered[:,0]!=native[:,0,None,:,:]
    count=np.zeros((n,2),np.int16)
    for actor in range(3):
        use=rows[focal!=actor]
        count[use,0]+=delta[use,focal[use],actor].sum(-1).astype(np.int16)
        count[use,1]+=delta[use,actor,focal[use]].sum(-1).astype(np.int16)
    require(np.all((count>=0)&(count<=8)),'Actual changed delivery symbols')
    return count


def invariances(cases,bank):
    """Unchanged own observations imply unchanged native W1, all base cells."""
    ni=np.asarray(cases['group_need_indices']);G,B=cases['group_count'],cases['n_backgrounds'];focal=np.asarray(cases['sender']);actors=np.asarray(cases['recipients']);count_i=count_o=0
    for b,c,r in product(range(B),range(2),range(2)):
        receiver=ni[:,c,r]*B+b;di=ni[:,1-c,r]*B+b;do=ni[:,c,1-r]*B+b
        require(np.array_equal(bank['messages'][receiver,0,focal],bank['messages'][di,0,focal]),'Incoming donor leaves focal native W1 unchanged');count_i+=G
        for slot in range(2):
            a=actors[:,slot];require(np.array_equal(bank['messages'][receiver,0,a],bank['messages'][do,0,a]),'Outgoing donor leaves recipients native W1 unchanged');count_o+=G
    return dict(compared_sender_packets=count_i//2,compared_other_packets=count_o//2,all_equal=True,neural_forward_samples=0)


def response_bundle(states,p,focal):
    exact,_=exact_responses(states,p)
    return dict(S=exact['full_success_probability'],expected_reward=exact['expected_reward'],execution=exact['execution'],**action_responses(states,p,focal))


def budget(specs,cases):
    W=sum(s['world_count'] for s in specs.values());N=sum(c['flow_cells'] for c in cases.values());S=N//4
    require((W,N,S)==(774144,2018304,504576),'Fixed full-support flow budget')
    return dict(policy_states=4,trained_policies=4,training_updates=0,checkpoint_files_read=4,new_natural_bank_files=0,reused_final_natural_bank_files=16,new_natural_worlds=0,reused_final_natural_worlds=4*W,new_natural_module_samples=0,
        flow_files=16,group_metric_files=16,flow_rows=4*N,sham_rows=4*S,intervention_rows=4*N,intervention_module_samples=4*N*6,total_new_module_samples=4*N*6,generated_data_files=32,formal_random_draws=0)


def primary(values):
    require(set(values)==set(SEEDS),'Exactly four final independent societies')
    rows=[]
    for seed in SEEDS:
        v=values[seed];require(np.isfinite(v['I_given_O0']) and -1<=v['I_given_O0']<=1,'Finite signed primary')
        rows.append(dict(seed=seed,S10_minus_S00=v['I_given_O0'],**{k:v[k] for k in ('S00','S01','S10','S11')}))
    return dict(name='mean_four_seed_complete_double_holdout_exact_S10_minus_S00',partition=PARTS[-1],response='S',contrast='I_given_O0',independent_societies=4,by_seed=rows,
        mean_S10_minus_S00=float(np.mean([r['S10_minus_S00'] for r in rows])))


def flow_record(entry,path,cases,bank,networks):
    data=directed.npz_record(entry,path);meta=expand(cases);n=len(meta['sender']);receiver=meta['receiver_state_indices'];focal=meta['focus_actor'];states=bank['states'][receiver]
    require(entry['rows']==n and entry['new_module_samples']==6*n,'Full16-cell forward budget')
    for k,v in meta.items():require(data[k].dtype.kind in 'biu' and np.array_equal(data[k],v),'Independent complete case metadata '+k)
    require(np.array_equal(data['receiver_states'],states) and data['receiver_states'].dtype==np.int16,'Unchanged original receiver states')
    p=data['action_probabilities'];act=data['action_indices']
    require(p.dtype==np.float64 and p.shape==(n,3,17) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Raw probabilities');close(p.sum(-1),np.ones((n,3)),'Probability normalization')
    require(act.dtype==np.int16 and np.array_equal(act,p.argmax(-1)),'Raw greedy action choice')
    for key,shape,dtype in (('generated_messages',(n,2,3,4),np.int8),('delivered_tokens',(n,2,3,3,4),np.int8),('delivery_visibility',(n,2,3,3),bool),('incoming_donor_W1',(n,3,4),np.int8),('outgoing_donor_W1',(n,4),np.int8),('response_actor_agents',(n,3),np.int8),('changed_W1_symbols',(n,2),np.int8),('exact_full_success_probability',(n,),np.float64),('expected_native_reward',(n,),np.float64),('execution_probabilities',(n,4),np.float64)):
        require(data[key].shape==shape and data[key].dtype==dtype,'Exact stored shape/dtype '+key)
    order=np.array([[0,1,2],[1,0,2],[2,0,1]],np.int8)[focal];require(np.array_equal(data['response_actor_agents'],order),'Actor response slots')
    errors=defaultdict(float);ties=checked=0
    for start in range(0,n,1024):
        sl=slice(start,start+1024);ri=receiver[sl];who=focal[sl];native=bank['messages'][ri]
        incoming=bank['messages'][meta['incoming_donor_state_indices'][sl],0];outgoing=bank['messages'][meta['outgoing_donor_state_indices'][sl],0,who]
        require(np.array_equal(data['incoming_donor_W1'][sl],incoming) and np.array_equal(data['outgoing_donor_W1'][sl],outgoing),'Candidate packet identities')
        fresh=replay(networks,states[sl],native,who,incoming,outgoing,meta['incoming'][sl],meta['outgoing'][sl]);ties+=fresh['sender2_ties']
        for key in ('generated_messages','delivered_tokens','delivery_visibility','action_indices'):require(np.array_equal(data[key][sl],fresh[key]),'Independent six-head replay '+key)
        errors['probability']=max(errors['probability'],close(p[sl],fresh['action_probabilities'],'Fresh six-head full probability'))
        changed=changed_symbols(native,fresh['delivered_tokens'],who);require(np.array_equal(data['changed_W1_symbols'][sl],changed),'Actual changed W1 symbols')
        require(np.all(changed[meta['incoming'][sl]==0,0]==0) and np.all(changed[meta['outgoing'][sl]==0,1]==0),'No change on unselected edge set')
        mask=meta['sham'][sl]
        if mask.any():
            require(np.array_equal(data['generated_messages'][sl][mask],native[mask]),'00 native messages identity')
            require(np.array_equal(act[sl][mask],bank['action_indices'][ri[mask]]),'00 native actions identity')
            err=float(np.max(np.abs(p[sl][mask]-bank['action_probabilities'][ri[mask]])));require(err<=1e-13,'00 native probability identity');errors['sham']=max(errors['sham'],err);checked+=int(mask.sum())
    settled=prior.settle(states,act,'reciprocal')
    for k,v in settled.items():require(data[k].dtype==v.dtype and np.array_equal(data[k],v),'Physical reciprocal settlement '+k)
    expected=set(meta)|set(settled)|{'receiver_states','generated_messages','delivered_tokens','delivery_visibility','incoming_donor_W1','outgoing_donor_W1','action_probabilities','action_indices','exact_full_success_probability','expected_native_reward','execution_probabilities','changed_W1_symbols','response_actor_agents'}
    require(set(data)==expected,'Exact flow NPZ field coverage')
    compare(entry['physical_summary'],prior.summarize(states,act,'reciprocal'),'Independent physical summary/')
    require(checked==cases['sham_cells'],'Complete00 subset coverage')
    compare(entry['native_replay'],dict(all_messages_equal=True,all_actions_equal=True,max_probability_absolute_error=errors['sham'],checked_rows=checked),'Native00 receipt/')
    values=response_bundle(states,p,focal);values['changed_W1_symbols']=data['changed_W1_symbols']
    for saved,computed in (('exact_full_success_probability','S'),('expected_native_reward','expected_reward'),('execution_probabilities','execution')):
        errors[saved]=close(data[saved],values[computed],'Independent exact original-world event '+saved)
    return values,dict(rows=n,sham_rows=checked,independent_module_samples=n*6,max_errors=dict(errors),sender2_ties=ties)


def audit(run,expected_plan_sha):
    run=Path(run).resolve();execution=run/'execution';refs=references();artifacts=dict(refs)
    require(re.fullmatch('[0-9a-f]{64}',expected_plan_sha) is not None,'Explicit frozen plan hash')
    def bind(path,digest=None):
        path=Path(path).resolve();actual=sha(path)
        if digest is not None:require(actual==digest,'Changed artifact '+str(path))
        artifacts[str(path)]=actual;return actual
    for path in (run/'plan.json',run/'prepared.json',run/'freeze.json',execution/'started.json',execution/'status.json',execution/'results.json',HERE/'audit_freeze_001.json'):bind(path)
    plan=read(run/'plan.json');prepared=read(run/'prepared.json');result=read(execution/'results.json');status=read(execution/'status.json');finite_tree(result)
    require(status['status']==result['status']=='completed' and not (execution/'failure.json').exists(),'No incomplete audit')
    require(status['results_sha256']==sha(execution/'results.json'),'Completed result hash')
    require(sha(run/'plan.json')==expected_plan_sha==read(run/'freeze.json')['plan_sha256']==result['plan_sha256']==read(execution/'started.json')['plan_sha256'],'Plan chain')
    require(sha(run/'prepared.json')==plan['prepared_sha256']==read(run/'freeze.json')['prepared_sha256'],'Prepared chain')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Exact numerical runtime')
    compare(plan['config'],dict(schema='same_background_W1_flow_v1',seeds=list(SEEDS),checkpoints=[6000],partitions=list(PARTS),condition='PL_live',execution_rule='reciprocal',
        primary='mean_four_seed_complete_double_holdout_exact_S10_minus_S00',primary_response='S',primary_contrast='I_given_O0',case_order='group/background/context/self_endpoint/incoming/outgoing',
        training_updates=0,worker_count=4,chunk_size=1024,automatic_followon_experiment=False,
        responses=dict(actor_order=['focal','u','v'],u_v='other identities ascending',partner=['wait','focal','u','v'],site=['wait',0,1,2,3],material=['wait',0,1,2,3],kind=['wait',0,1],length=['wait',0,1],destination=['wait',0,1],execution=['none','AB','AC','BC'],changed_W1_symbols=['incoming','outgoing'])),'Frozen core config/')
    af=read(HERE/'audit_freeze_001.json');require(af['status']=='frozen_before_formal_intervention' and af['plan_sha256']==expected_plan_sha,'Preformal independent source freeze')
    require(af['at']<read(execution/'started.json')['at'],'Audit freeze precedes formal forward')
    require(set(af['source_sha256'])=={str(HERE/'audit_execution.py'),str(HERE/'test_audit.py'),str(HERE/'audit_preflight_001.json')},'Audit source coverage')
    for p,d in af['source_sha256'].items():bind(p,d)
    for p,d in plan['source_sha256'].items():bind(p,d);bind(run/'source_snapshot'/Path(p).relative_to(ROOT),d)
    for p,d in plan['inputs_sha256'].items():bind(p,d)
    original=ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json';bind(original,directed.response.ORIGINAL_SHA)
    specs=read(original)['partitions'];independent,_,_=env.independent_specs()
    require(prepared['partitions']==specs and prepared['original_prepared_sha256']==directed.response.ORIGINAL_SHA and prepared['independent_societies']==4,'Anchored static metadata')
    for part in PARTS:compare(specs[part],independent[part],'Independent full support/')
    cc={p:build_cases(specs[p]) for p in PARTS};compare(prepared['cases'],cc,'Independent flow group metadata/')
    require(prepared['budget']==result['budget']==budget(specs,cc),'Exact no-training budget')
    source=ROOT/'research_program/triadic_private_partner_study/results/private_001';bind(source/'plan.json',SOURCE_PLAN_SHA);bind(source/'execution/results.json',SOURCE_RESULT_SHA);bind(source/'audit_execution_001/verification.json',SOURCE_AUDIT_SHA)
    oldresult=read(source/'execution/results.json');oldaudit=read(source/'audit_execution_001/verification.json');require(oldresult['status']=='completed' and oldaudit['status']=='passed','Completed audited old source')
    oldruns={(r['seed'],r['condition']):r for r in oldresult['runs']};expected_meta=[]
    for seed in SEEDS:
        oldrow=oldruns[seed,'PL_live'];directory=source/'execution'/f'seed_{seed}_PL_live';checkpoint=directory/'checkpoint_6000.npz';digest=oldrow['final_checkpoint_sha256']
        require(oldaudit['artifacts_sha256'][str(checkpoint)]==digest,'Old checkpoint audit binding')
        meta=dict(seed=seed,checkpoint=6000,checkpoint_path=str(checkpoint),checkpoint_sha256=digest,parameter_sha256=oldrow['final_parameter_sha256'],natural_banks={})
        for part in PARTS:
            b=oldrow['final'][part]['natural'];p=directory/f'final_{part}_natural.npz';require(Path(b['path'])==p and oldaudit['artifacts_sha256'][str(p)]==b['data_sha256'],'Old natural full-bank audit binding')
            meta['natural_banks'][part]=dict(path=str(p),sha256=b['data_sha256'],worlds=b['worlds'])
        expected_meta.append(meta)
    require(plan['policy_states']==expected_meta,'Four exact final policies/banks')
    require([(r['seed'],r['checkpoint']) for r in result['policy_states']]==[(s,6000) for s in SEEDS],'Canonical completed four policy grid')
    states={p:env.packed(specs[p]) for p in PARTS};hashes={p:dict(packed_states=array_sha(s),rewards=array_sha(env.rewards(s)),x_PL=array_sha(env.features(s,'PL'))) for p,s in states.items()}
    require(result['array_hashes']==hashes,'Independent full world/PL arrays')
    counts=Counter();errors=defaultdict(float);receipts={};primary_cells={};paths=set()
    for meta,row in zip(expected_meta,result['policy_states']):
        seed=meta['seed'];directory=execution/f'seed_{seed}'
        bind(directory/'array_hashes.json');require(read(directory/'array_hashes.json')==hashes,'Worker array hashes')
        bind(directory/'status.json');require(read(directory/'status.json')['status']=='completed','Complete worker')
        bind(directory/'result.json');require(read(directory/'result.json')==row,'Worker result identity');require(set(row['parts'])==set(PARTS),'All four partitions')
        require(row['checkpoint_sha256']==meta['checkpoint_sha256'] and row['parameter_sha256']==meta['parameter_sha256'],'Exact policy identity')
        bind(meta['checkpoint_path'],meta['checkpoint_sha256']);networks,_,_=message.checkpoint(meta['checkpoint_path'],6000,seed)
        require(message.network_hash(networks)==meta['parameter_sha256'],'Independent loaded parameter identity');counts['checkpoint_files']+=1
        for part in PARTS:
            rec=row['parts'][part];pdir=directory/part;bind(pdir/'result.json');require(read(pdir/'result.json')==rec,'Partition result identity')
            bank_entry=rec['natural_bank'];bp=Path(meta['natural_banks'][part]['path']);require(bank_entry['sha256']==meta['natural_banks'][part]['sha256'],'Reused native source hash');bind(bp,bank_entry['sha256'])
            bank,receipt=directed.natural_bank(bank_entry,bp,states[part],networks,False);receipts[str(bp)]=receipt;counts['reused_final_banks']+=1
            inv=invariances(cc[part],bank);compare(rec['context_packet_check'],inv,'Own-information W1 invariance/');counts['context_sender_packets']+=inv['compared_sender_packets'];counts['context_other_packets']+=inv['compared_other_packets']
            fp=pdir/'flow.npz';bind(fp,rec['flow']['sha256']);paths.add(fp);values,receipt=flow_record(rec['flow'],fp,cc[part],bank,networks);receipts[str(fp)]=receipt
            counts['flow_files']+=1;counts['flow_rows']+=receipt['rows'];counts['sham_rows']+=receipt['sham_rows'];counts['intervention_module_samples']+=receipt['independent_module_samples']
            for key,value in receipt['max_errors'].items():errors[key]=max(errors[key],value)
            mp=pdir/'flow_metrics.npz';bind(mp,rec['metric_file']['sha256']);paths.add(mp);saved=directed.npz_record(rec['metric_file'],mp);expected_keys=set();require(set(rec['metrics'])==set(values),'Complete ten response families')
            for response,v in values.items():
                summary,details=score(cc[part],v);record=rec['metrics'][response];compare(record,summary,'Independent all responses '+response+'/')
                require(record['per_group_background_key_prefix']==response+'__' and record['per_group_background_file']==dict(path=str(mp),sha256=sha(mp)),'Full metric detail links')
                for key,value in details.items():
                    k=response+'__'+key;expected_keys.add(k);errors['metrics']=max(errors['metrics'],close(saved[k],value,'Full group/background response '+k))
                if part==PARTS[-1] and response=='S':primary_cells[seed]=summary
            require(set(saved)==expected_keys and len(saved)==90,'All90 scalar/vector group-background arrays');counts['metric_files']+=1;counts['response_summaries']+=len(values)
            require(set(pdir.glob('*.npz'))=={fp,mp},'No extra per-part data file')
            del values,saved,bank
            print(json.dumps(dict(stage='flow_partition_audited',seed=seed,part=part)),flush=True)
        require(message.network_hash(networks)==meta['parameter_sha256'],'Audit never modifies source weights')
    expected=dict(checkpoint_files=4,reused_final_banks=16,context_sender_packets=1009152,context_other_packets=2018304,flow_files=16,flow_rows=8073216,sham_rows=2018304,intervention_module_samples=48439296,metric_files=16,response_summaries=160)
    require(dict(counts)==expected,'Complete independent actual audit scope');measured=dict(new_natural_module_samples=0,intervention_module_samples=counts['intervention_module_samples'],total_new_module_samples=counts['intervention_module_samples'])
    require(result['measured_module_samples']==measured,'Exact new forward accounting')
    calculated=primary(primary_cells);compare(result['primary'],calculated,'Unique exactS signed incoming primary/')
    require(len(paths)==32 and set(execution.rglob('*.npz'))==paths,'All32 generated files covered')
    for p,d in artifacts.items():require(sha(p)==d,'Artifact changed during audit '+p)
    return dict(status='passed',audit_source_sha256=sha(__file__),plan_sha256=expected_plan_sha,scope=dict(counts),measured_module_samples=measured,max_errors=dict(errors),artifacts_sha256=artifacts,historical_helpers_sha256=refs,evaluations=receipts,primary=calculated,
        limits=['No training, random sampling, optimizer, or gradient replay. All16 complete intervention files freshly replay six downstream modules per row.',
        'The16 natural final banks and four final checkpoints are SHA-bound to the previous completed independent audit; no new native forward is counted.',
        'Cases, W1 edge routing, exact original-world reciprocal event probabilities, waiting-inclusive semantic marginals, all90 group/background arrays per partition, and nine-stratum paired contrasts are independently recomputed.',
        'Primary is the complete double-holdout S10 minus S00 averaged over four societies. Exact probability conditions on greedy generated messages; it does not integrate message sampling.',
        'The result localizes a specified functional message pathway and does not establish demand semantics, information centrality, linguistic structure, or language formation.'])


def freeze(run,expected_plan_sha):
    run=Path(run).resolve();path=HERE/'audit_freeze_001.json'
    require(not path.exists(),'Never overwrite independent audit freeze')
    require(not (run/'execution').exists(),'Freeze only before formal intervention')
    require(sha(run/'plan.json')==expected_plan_sha==read(run/'freeze.json')['plan_sha256'],'Exact prepared plan before audit freeze')
    plan=read(run/'plan.json');require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Prepared numerical runtime')
    require(sha(run/'prepared.json')==plan['prepared_sha256'],'Prepared static metadata hash')
    receipt=read(HERE/'audit_preflight_001.json');require(receipt['status']=='passed' and receipt['real_neural_forward_samples']==0 and receipt['formal_output_files_read']==0,'Passed pure preflight')
    require(receipt['source_sha256'][str(HERE/'audit_execution.py')]==sha(__file__) and receipt['source_sha256'][str(HERE/'test_audit.py')]==sha(HERE/'test_audit.py'),'Preflight current source identity')
    for p,d in plan['source_sha256'].items():require(sha(p)==d,'Main source changed before audit freeze')
    value=dict(status='frozen_before_formal_intervention',at=datetime.now(timezone.utc).isoformat(),plan_sha256=expected_plan_sha,
        source_sha256={str(p):sha(p) for p in (HERE/'audit_execution.py',HERE/'test_audit.py',HERE/'audit_preflight_001.json')},historical_helpers_sha256=references(),
        formal_output_files_read=0,real_neural_forward_samples=0,expected_independent_module_samples=48439296)
    path.write_bytes(json_bytes(value));return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--out');parser.add_argument('--plan-sha',required=True);parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:
        require(args.out is None,'Freeze creates fixed receipt, not an audit output');print(json.dumps(freeze(args.run,args.plan_sha)));return
    require(args.out is not None,'Audit output required');out=Path(args.out).resolve();require(not out.exists(),'Never overwrite an audit');out.mkdir(parents=True);start=time.perf_counter()
    (out/'started.json').write_bytes(json_bytes(dict(at=datetime.now(timezone.utc).isoformat(),run=args.run,expected_plan_sha256=args.plan_sha,audit_source_sha256=sha(__file__))))
    try:
        value=audit(args.run,args.plan_sha);value.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        (out/'verification.json').write_bytes(json_bytes(value));print(json.dumps({k:value[k] for k in ('status','scope','max_errors','elapsed_seconds')}))
    except BaseException as error:
        (out/'failure.json').write_bytes(json_bytes(dict(status='failed',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,audit_source_sha256=sha(__file__),automatic_retry=False)));raise

if __name__=='__main__':main()
