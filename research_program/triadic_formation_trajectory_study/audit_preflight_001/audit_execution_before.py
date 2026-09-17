"""Independent saved-checkpoint audit. No model work or result reads on import.

Only the explicitly invoked completed-batch audit performs neural forwards.
Old endpoint anchors are compared as saved bytes/arrays, never regenerated.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k]='1'
from pathlib import Path
from hashlib import sha256
from functools import lru_cache
from fractions import Fraction
import json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=HERE.parent/'triadic_position_reuse_study/results/position_001'
STEPS=(0,100,500,1500,3000,6000)
SEEDS=(51101,51102,51103,51104)
CONDITIONS=('PL_silent','PL_live','LL_silent','LL_live')
AXES=('kind','length','destination')
PART='new_needs_and_layouts'
POSITION_AUDIT_SHA='54da7fbf694ceb3499d846aa3a489a88a4f928944029f225fb835c2cb5035717'
POSITION_VERIFICATION_SHA='b5f0b8f2400c8c9208d89d0232d920667956d67a2939d5341dd9954ea70f1e6d'
POSITION_RESULT_SHA='c9f5ce31c8329db29c7afb2ca124dba006a257beb65d781c76b612ce2914391c'
CONTEXT_VERIFICATION_SHA='7c0eab86edff602ec6def4e54e92cacf3403cb495e39b195594b0185caeb250a'
TOL=2e-12
BATCH=1024

def require(ok,message):
    if not ok:raise AssertionError(message)

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False);f.write('\n')
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

@lru_cache(maxsize=1)
def references():
    path=HERE.parent/'triadic_position_reuse_study/audit_execution.py'
    require(sha(path)==POSITION_AUDIT_SHA,'Pinned independent position auditor changed')
    from research_program.triadic_position_reuse_study import audit_execution as position
    context,old=position.references()
    return position,context,old

def close(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' shape/finite')
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    require(error<=TOL,label+' numerical mismatch: '+str(error))
    return error

def arrays_equal(expected,actual,label):
    require(set(expected)==set(actual),label+' complete array fields')
    maximum=0.
    for key,want in expected.items():
        got=actual[key];require(got.dtype==want.dtype and got.shape==want.shape,label+'/'+key+' dtype/shape')
        if key=='action_probabilities':maximum=max(maximum,close(want,got,label+'/'+key))
        else:require(np.array_equal(got,want),label+'/'+key+' exact values')
    return maximum

def numeric_tree(expected,actual,counts,path=''):
    """Compare only explicitly independently derived fields, preserving scope."""
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(expected)<=set(actual),'Missing summary fields '+path)
        for k,v in expected.items():numeric_tree(v,actual[k],counts,path+'/'+str(k))
    elif isinstance(expected,(list,tuple)):
        require(len(expected)==len(actual),'Summary length '+path)
        for i,v in enumerate(expected):numeric_tree(v,actual[i],counts,path+'/'+str(i))
    elif isinstance(expected,(int,float,np.integer,np.floating)):
        e=close(expected,actual,path);counts['scalars']+=1;counts['max_error']=max(counts['max_error'],e)
    else:require(expected==actual,'Summary label '+path)

def exact_time_weights():
    result=[]
    for i,t in enumerate(STEPS):
        span=(t-STEPS[i-1] if i else 0)+(STEPS[i+1]-t if i+1<len(STEPS) else 0)
        result.append(Fraction(span,2*(STEPS[-1]-STEPS[0])))
    require(sum(result)==1,'Time weights sum')
    return result

def area(values):
    v=np.asarray(values,dtype=np.float64)
    require(v.shape==(6,) and np.isfinite(v).all(),'Six finite fixed-time values')
    return float(sum(float(w)*float(x) for w,x in zip(exact_time_weights(),v)))

def static_pool(validation,held_states,train_count):
    ids=np.unique(np.r_[validation['endpoint_indices'].ravel(),validation['donor_endpoint_indices'].ravel()]).astype(np.int32)
    require(np.all(ids>=0) and np.all(ids<len(held_states)),'Pool IDs valid')
    return dict(validation_pool_endpoint_indices=ids,validation_pool_states=held_states[ids],
        recipient_pool_rows=np.searchsorted(ids,validation['endpoint_indices']).astype(np.int32),
        donor_pool_rows=np.searchsorted(ids,validation['donor_endpoint_indices']).astype(np.int32),
        train_world_indices=np.arange(train_count,dtype=np.int32))

def natural_trace(networks,x,live,*,actions=True,probability=None):
    """Independent nine-module scheduler; callback enables weight-free tests."""
    p,_,old=references();x=np.asarray(x);n=len(x)
    require(x.shape==(n,3,54) and n>0 and np.isfinite(x).all(),'Natural input')
    require(len(networks)==9 and isinstance(live,(bool,np.bool_)),'Natural module/visibility')
    predict=old.probabilities if probability is None else probability
    first=np.stack([predict(networks[3*a],x[:,a]).argmax(-1).astype(np.int8) for a in range(3)],axis=1)
    r1=p.route(first,live);x2=np.concatenate((x,r1),axis=-1)
    second=np.stack([predict(networks[3*a+1],x2[:,a]).argmax(-1).astype(np.int8) for a in range(3)],axis=1)
    r2=p.route(second,live)
    out=dict(messages=np.stack((first,second),axis=1),first_routes=r1,second_routes=r2,independent_module_samples=6*n)
    if actions:
        inputs=np.concatenate((x,r1,r2),axis=-1)
        probs=np.stack([predict(networks[3*a+2],inputs[:,a]) for a in range(3)],axis=1)
        out.update(action_inputs=inputs,action_probabilities=probs,action_indices=probs.argmax(-1).astype(np.int16),independent_module_samples=9*n)
    return out

def whole_route(tokens,live,senders,packets):
    p,_,_=references();tokens=np.asarray(tokens);n=len(tokens);s=p.vec(senders,n,3,'sender')
    donor=np.asarray(packets);require(donor.shape==(n,4) and donor.dtype.kind in 'iu' and ((donor>=0)&(donor<8)).all(),'Whole packet')
    out=p.route(tokens,live)
    if live:
        rows=np.arange(n)
        for listener in range(3):
            take=np.flatnonzero(s!=listener)
            for slot in range(4):
                start=32*s[take]+8*slot
                out[take[:,None],listener,start[:,None]+np.arange(8)[None]]=0.
                out[take,listener,start+donor[take,slot]]=1.
    return out

def whole_trace(networks,x,m,s,donor,live,*,probability=None):
    """Receiver-time nets/history only; a donor is an integer packet, not nets."""
    p,_,old=references();x=np.asarray(x);m=np.asarray(m);n=len(x);s=p.vec(s,n,3,'sender')
    require(m.shape==(n,2,3,4) and m.dtype.kind in 'iu' and ((m>=0)&(m<8)).all(),'Whole receiver natural')
    donor=np.asarray(donor);require(donor.shape==(n,2,4),'Whole donor shape')
    r1=whole_route(m[:,0],live,s,donor[:,0]);messages=m.copy()
    predict=old.probabilities if probability is None else probability
    if live:
        x2=np.concatenate((x,r1),axis=-1)
        messages[:,1]=np.stack([predict(networks[3*a+1],x2[:,a]).argmax(-1).astype(np.int8) for a in range(3)],axis=1)
        require(np.array_equal(messages[np.arange(n),1,s],m[np.arange(n),1,s]),'Focal own W2 must stay receiver-natural')
    r2=whole_route(messages[:,1],live,s,donor[:,1]);inputs=np.concatenate((x,r1,r2),axis=-1)
    out=dict(messages=messages,patched_outward_packets=donor.copy(),first_routes=r1,second_routes=r2,action_inputs=inputs,independent_module_samples=6*n if live else 0)
    if live:
        probs=np.stack([predict(networks[3*a+2],inputs[:,a]) for a in range(3)],axis=1)
        out.update(action_probabilities=probs,action_indices=probs.argmax(-1).astype(np.int16))
    return out

def indices(spec,arm,b):
    r=spec['endpoint_indices'][:,b]
    d=r if arm=='sham' else spec['donor_endpoint_indices'][:,b if arm=='same' else 1-b]
    return r,spec['endpoint_indices'][:,1-b],d

def settle_arrays(pool,spec,b,trace,donor_packets,d,*,position=False):
    _,context,_=references();r=spec['endpoint_indices'][:,b];cf=spec['endpoint_indices'][:,1-b]
    keys=('messages','patched_outward_packets','action_indices','action_probabilities')+ (('replacement_symbols',) if position else ())
    data={k:trace[k] for k in keys}
    reward,ex,sat=context.native(pool['states'][r],data['action_indices']);cfr,_,cfs=context.native(pool['states'][cf],data['action_indices'])
    data.update(dataset_rows=np.arange(len(r)),recipient_indices=r,counterfactual_recipient_indices=cf,donor_indices=d,donor_packets=donor_packets,
        greedy_reward=reward,executed=ex,satisfied=sat,counterfactual_reward=cfr,counterfactual_satisfied=cfs)
    return data

def pair_values(values):return {k:np.stack((values[0][k],values[1][k])) for k in values[0]}

def point_summary(spec,profile,final_profile,positions,natural,shams):
    p,_,_=references()
    def selected(selection):
        value=p.policy_summary(spec,selection,positions,{},{});value.pop('matched_references');value.pop('shams')
        return value
    current=selected(profile);retro=selected(final_profile)
    retro.pop('all_positions')
    rates=profile['response_rates'];scores=profile['scores'];choice=profile['selected_positions']
    resp=rates[np.arange(3)[:,None],np.arange(3)[None,:],choice]
    score=scores[np.arange(3)[:,None],np.arange(3)[None,:],choice]
    desc=dict(selected_response=resp.tolist(),selected_scores=score.tolist(),mean_selected_response=float(resp.mean()),mean_selected_score=float(score.mean()),
        positive_selected_scores=int((score>0).sum()),zero_selected_scores=int((score==0).sum()),negative_selected_scores=int((score<0).sum()))
    return dict(current=current,retrospective_final=retro,discovery=desc,natural=p.aggregate(spec,natural),shams={str(u):p.aggregate(spec,v) for u,v in shams.items()})

def temporal_change(a,b,previous,current):
    require(a.shape==b.shape==(419904,2,3,4),'Complete train order for change')
    diff=a!=b;choice=previous['selected_positions']!=current['selected_positions']
    return dict(train_token_change_rate=float(diff.mean()),by_window_sender_position=diff.mean(0).tolist(),
        sender_whole_packet_change_rate=np.any(diff,axis=(1,3)).mean(0).tolist(),selected_position_changed=choice.tolist(),selected_position_change_rate=float(choice.mean()))

CURVE_KEYS=('diagonal_effect','offdiagonal_effect','selectivity','uniform_position_effect','diagonal_minus_uniform_position')
def trajectory_summary(points,cross):
    require(set(points)==set(STEPS) and set(cross)=={(r,s) for r in STEPS for s in STEPS},'Complete six-time grid')
    curves={k:[points[t]['current'][k] for t in STEPS] for k in CURVE_KEYS}
    retro={k:[points[t]['retrospective_final'][k] for t in STEPS] for k in CURVE_KEYS}
    matrix=np.asarray([[cross[r,s]['contrast']['macro']['target_apt'] for s in STEPS] for r in STEPS])
    return dict(steps=list(STEPS),current_discovery_curves=curves,normalized_areas={k:area(v) for k,v in curves.items()},
        retrospective_final_position_curves=retro,retrospective_normalized_areas={k:area(v) for k,v in retro.items()},
        cross_time_target_effect=matrix.tolist(),cross_time_minus_receiver_same_time=(matrix-np.diag(matrix)[:,None]).tolist(),
        cross_time_axes=dict(rows='receiver checkpoint',columns='donor-message checkpoint'),cross_time_same_time_effect=np.diag(matrix).tolist())


def configured_counts():
    return dict(new_train_message_worlds=33592320,new_train_message_module_samples=201553920,
        new_validation_natural_worlds=44320,new_validation_natural_module_samples=398880,
        new_position_worlds=207360,new_position_module_samples=933120,
        new_cross_time_worlds=161280,new_cross_time_module_samples=967680,
        total_new_module_samples=203853600,parameter_loads=88,train_records=96,natural_records=96,
        position_records=3456,cross_time_records=2304,new_position_npz=1440,new_cross_time_npz=1120,
        reused_position_records=576,reused_cross_time_records=64)


def audit(run,out,expected_plan_sha):
    import platform,time
    from itertools import product
    start=time.perf_counter();p,context,old=references();run=Path(run).resolve();out=Path(out).resolve();execution=run/'execution'
    require(read(execution/'status.json')['status']=='completed' and not (execution/'failure.json').exists(),'Only a fully completed main batch can be audited')
    require(sha(run/'plan.json')==expected_plan_sha==read(run/'freeze.json')['plan_sha256'],'Authorized plan/freeze SHA')
    plan=read(run/'plan.json');main=read(execution/'results.json');frozen=read(run/'frozen_spec.json')
    require(main['status']=='completed' and main['plan_sha256']==expected_plan_sha and main['training_updates']==0,'Completed source run')
    require(read(execution/'started.json')['plan_sha256']==expected_plan_sha,'Execution started from this plan')
    require(plan['status']=='prepared_without_new_forward' and frozen['new_checkpoint_outputs_read'] is False,'Plan predates new forwards')
    require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Runtime version')
    require(main['config']==plan['config']==frozen['config'] and plan['sources_sha256']==frozen['sources_sha256'],'Configuration/source consistency')
    required=dict(steps=list(STEPS),seeds=list(SEEDS),conditions=list(CONDITIONS),batch_size=BATCH,
        positions=list(range(8)),arms=['same','opposite'],shams=[0,4],directions=[0,1],validation_rows=144,
        validation_pool_worlds=554,train_worlds=419904,training_updates=0,new_initializations=0,
        positive_score_filter=False,force_distinct_positions=False,onset_threshold=None,automatic_retry=False)
    required.update(configured_counts())
    for k,v in required.items():require(plan['config'][k]==v,'Fixed budget/selection '+k)
    require(main['totals']==configured_counts(),'Saved complete stage budgets')
    artifacts={};new_npz=set();record_json=set();summary_paths=set();counts=dict(scalars=0,max_error=0.)
    def bind(path,digest=None):
        path=Path(path).resolve();actual=sha(path)
        require(digest is None or actual==digest,'SHA changed '+str(path));artifacts[str(path)]=actual;return path
    for path in (run/'plan.json',run/'freeze.json',run/'frozen_spec.json',execution/'results.json',execution/'status.json',execution/'started.json'):bind(path)
    for path,digest in plan['sources_sha256'].items():
        bind(path,digest);bind(run/'source_snapshot'/Path(path).relative_to(ROOT),digest)
    for group in ('inputs_sha256','prepared_files_sha256'):
        for path,digest in plan[group].items():bind(path,digest)
    bind(PRIOR/'audit_execution_001/verification.json',POSITION_VERIFICATION_SHA);bind(PRIOR/'execution/results.json',POSITION_RESULT_SHA)
    prior_proof=read(PRIOR/'audit_execution_001/verification.json')
    require(prior_proof['status']=='passed' and prior_proof['audit_source_sha256']==POSITION_AUDIT_SHA,'Prior full audit anchor')
    prior_main=read(PRIOR/'execution/results.json');prior_plan=read(PRIOR/'plan.json')
    original_context=HERE.parent/'triadic_action_dependency_study/results/context_001'
    bind(original_context/'audit_execution_001/verification.json',CONTEXT_VERIFICATION_SHA)
    context_proof=read(original_context/'audit_execution_001/verification.json')
    require(context_proof['status']=='passed','Original training audit passed')
    original_runs=read(original_context/'execution/results.json')['runs']
    specs,_,_=context.independent_specs();expected_static,static_proof=p.independent_dataset(specs)
    saved_static={}
    for kind in ('discovery','validation'):
        saved_static[kind]=load_npz(run/'dataset'/(kind+'.npz'))
        arrays_equal(expected_static[kind],saved_static[kind],'Independent static '+kind)
    ds=expected_static['discovery'];full_spec=expected_static['validation']
    train_states=context.packed(specs['train']);held_states=context.packed(specs[PART])
    pool_index=static_pool(full_spec,held_states,len(train_states));arrays_equal(pool_index,load_npz(run/'dataset/pool.npz'),'Independent 554 pool')
    require(len(pool_index['validation_pool_endpoint_indices'])==554 and len(train_states)==419904,'Complete static domains')
    pool_meta=read(run/'dataset/pool.json')
    require(pool_meta['training_spec']==specs['train'] and pool_meta['validation_spec']==specs[PART],'Pool semantic domain')
    ids=pool_index['validation_pool_endpoint_indices'];states=pool_index['validation_pool_states']
    spec=dict(full_spec,endpoint_indices=pool_index['recipient_pool_rows'],donor_endpoint_indices=pool_index['donor_pool_rows'])
    require(np.array_equal(ids[spec['endpoint_indices']],full_spec['endpoint_indices']) and np.array_equal(ids[spec['donor_endpoint_indices']],full_spec['donor_endpoint_indices']),'Exact full/local mapping')
    manifest=read(run/'dataset/manifest.json')
    for path,digest in manifest['source_sha256'].items():bind(path,digest)
    for name,rec in manifest['outputs'].items():require(Path(rec['path'])==run/'dataset'/name,'Static output path');bind(rec['path'],rec['sha256'])
    feature_record=read(execution/'features.json');bind(execution/'features.json')
    # One physical array at a time; both feature domains are cached because all policies use the same states.
    train_x={info:context.features(train_states,info) for info in ('PL','LL')}
    val_x={info:context.features(states,info) for info in ('PL','LL')}
    require(feature_record=={info:dict(train=context.array_sha(train_x[info]),validation=context.array_sha(val_x[info])) for info in ('PL','LL')},'Independent 54-feature arrays')
    policies=list(product(SEEDS,CONDITIONS));require([(r['seed'],r['condition']) for r in plan['policies']]==policies,'Canonical16 policies')
    for label in ('train_records','natural_records'):
        require([(r['seed'],r['condition'],r['update']) for r in main[label]]==[(s,c,t) for s,c in policies for t in STEPS],'Canonical96 '+label)
    want_keys=[('position',s,c,t,t,u,a,b) for s,c in policies for t in STEPS for u,a,b in p.cells()]
    want_keys += [('cross_time_whole',s,c,r,t,-1,a,b) for s,c in policies for r in STEPS for t in STEPS for a,b in product(('same','opposite'),(0,1))]
    def rec_key(r):return (r['kind'],r['seed'],r['condition'],r['receiver_update'],r['donor_update'],r.get('unit',-1),r['arm'],r['direction'])
    lookup={rec_key(r):r for r in main['records']}
    require(len(lookup)==len(main['records'])==5760 and set(lookup)==set(want_keys),'All3456 positions and2304 cross-time records exactly once')
    canonical=[]
    for s,c in policies:
        canonical += [('position',s,c,t,t,u,a,b) for t in STEPS for u,a,b in p.cells()]
        canonical += [('cross_time_whole',s,c,r,t,-1,a,b) for r in STEPS for t in STEPS for a,b in product(('same','opposite'),(0,1))]
    require([rec_key(r) for r in main['records']]==canonical,'Actual canonical execution order')
    tlookup={(r['seed'],r['condition'],r['update']):r for r in main['train_records']};nlookup={(r['seed'],r['condition'],r['update']):r for r in main['natural_records']}
    require([(r['seed'],r['condition']) for r in main['policies']]==policies,'All policy summary records')
    stat=dict(train_worlds=0,train_modules=0,natural_worlds=0,natural_modules=0,position_worlds=0,position_modules=0,cross_worlds=0,cross_modules=0,
        parameter_loads=0,new_position_npz=0,new_cross_npz=0,reused_position_records=0,reused_cross_records=0,silent_position_records=0,silent_cross_records=0,
        maximum_probability_error=0.,endpoint_exchange_checks=0,sham_checks=0)
    reports=[]
    for policy in plan['policies']:
        seed,condition=policy['seed'],policy['condition'];info=condition.split('_')[0];live=condition.endswith('_live')
        directory=execution/f'seed_{seed}_{condition}';prior_policy=next(r for r in prior_plan['policies'] if r['seed']==seed and r['condition']==condition)
        require(policy['prior_policy']==prior_policy,'Exact old policy/reference domain')
        old_pos=[r for r in prior_main['records'] if r['seed']==seed and r['condition']==condition]
        old_whole=[r for r in prior_main['references'] if r['seed']==seed and r['condition']==condition and r['kind']=='whole']
        require(policy['prior_position_records']==old_pos and policy['prior_whole_references']==old_whole,'Whole and position anchors')
        actual_original=next(r for r in original_runs if r['seed']==seed and r['condition']==condition)
        require([c['update'] for c in policy['checkpoints']]==list(STEPS),'All checkpoints ordered')
        nets={}
        for checkpoint in policy['checkpoints']:
            t=checkpoint['update'];path=Path(checkpoint['path']);original_path=original_context/'execution'/f'seed_{seed}_{condition}'/f'checkpoint_{t:04d}.npz'
            require(path==original_path and checkpoint['seed']==seed and checkpoint['condition']==condition,'No checkpoint cohort/time substitution')
            original_meta=next(r for r in actual_original['monitor'] if r['update']==t)
            digest=context_proof['artifacts_sha256'][str(path)];require(digest==checkpoint['sha256']==original_meta['checkpoint_sha256'],'Historical checkpoint SHA')
            bind(path,digest)
            if t!=6000 or live:nets[t],_,_=old.checkpoint(path,t,seed);stat['parameter_loads']+=1
        pools={};messages={};profiles={};codes={};points={};changes=[];cross={}
        stored_final=read(prior_policy['selection'])
        for t in STEPS:
            stepdir=directory/f'checkpoint_{t:04d}';tr=tlookup[seed,condition,t];nr=nlookup[seed,condition,t]
            for rec,label in ((tr,'train_messages'),(nr,'natural_validation')):
                jp=stepdir/(label+'.json');require(read(jp)==rec,'Natural per-record/index agreement');bind(jp);record_json.add(jp)
                require(rec['path']==str(stepdir/(label+'.npz')) and rec['is_reused'] is (t==6000),'Natural path/reuse')
                bind(rec['path'],rec['data_sha256']);new_npz.add(Path(rec['path']))
            tmdata=load_npz(tr['path']);require(set(tmdata)=={'messages'},'Train records contain messages only');tm=tmdata['messages']
            require(tm.shape==(419904,2,3,4) and tm.dtype==np.int8 and ((tm>=0)&(tm<8)).all(),'Complete natural train messages')
            pool=load_npz(nr['path'])
            if t==6000:
                require(tr['source_path']==prior_policy['endpoints']['train'] and nr['source_path']==prior_policy['endpoints'][PART],'6000 natural source identity')
                with np.load(tr['source_path'],allow_pickle=False) as z:require(np.array_equal(tm,z['messages']) and tm.dtype==z['messages'].dtype,'6000 full train copy')
                with np.load(nr['source_path'],allow_pickle=False) as z:
                    expect={k:z[k][ids] for k in ('states','messages','action_indices','action_probabilities','greedy_reward','executed','satisfied')}
                expect['state_indices']=ids.copy();arrays_equal(expect,pool,'6000 exact pooled natural copy')
                require(all(r[k]==0 for r in (tr,nr) for k in ('new_forward_worlds','new_network_samples','neural_forward_calls')),'6000 copies have no forward')
            else:
                for begin in range(0,len(train_states),BATCH):
                    stop=min(begin+BATCH,len(train_states));trace=natural_trace(nets[t],train_x[info][begin:stop],live,actions=False)
                    require(np.array_equal(tm[begin:stop],trace['messages']),'Independent complete train greedy messages')
                    stat['train_worlds']+=stop-begin;stat['train_modules']+=trace['independent_module_samples']
                trace=natural_trace(nets[t],val_x[info],live);reward,ex,sat=context.native(states,trace['action_indices'])
                expect={k:trace[k] for k in ('messages','action_indices','action_probabilities')};expect.update(states=states,state_indices=ids,greedy_reward=reward,executed=ex,satisfied=sat)
                err=arrays_equal(expect,pool,'Independent natural validation');stat['maximum_probability_error']=max(stat['maximum_probability_error'],err)
                stat['natural_worlds']+=len(states);stat['natural_modules']+=trace['independent_module_samples']
                require(nr['routing_sha256']=={k:context.array_sha(trace[k]) for k in ('first_routes','second_routes','action_inputs')},'Validation natural route hashes')
                require(tr['new_forward_worlds']==419904 and tr['new_network_samples']==419904*6 and tr['neural_forward_calls']==6*((419904+BATCH-1)//BATCH),'Sender-only batch/call budget')
                require(nr['new_forward_worlds']==554 and nr['new_network_samples']==554*9 and nr['neural_forward_calls']==9,'Natural validation budget')
            require(tr['actions_generated'] is False and tr['worlds']==419904 and nr['worlds']==554,'Natural domain metadata')
            require(np.array_equal(pool['states'],states) and np.array_equal(pool['state_indices'],ids),'Canonical validation world IDs')
            reward,ex,sat=context.native(states,pool['action_indices'])
            for k,v in (('greedy_reward',reward),('executed',ex),('satisfied',sat)):require(np.array_equal(pool[k],v),'Natural native settlement '+k)
            profile=p.discovery(tm,ds['endpoint_indices'],ds['axis'],ds['sender'],ds['listener'],ds['base_within_axis_weight'])
            stored=read(stepdir/'selection.json');p.verify_selection(profile,stored);bind(stepdir/'selection.json')
            if t==6000:require(stored==stored_final,'Exact6000 discovery anchor')
            code=[np.unique(p.packet_codes(tm[:,:,s,:])) for s in range(3)];saved=load_npz(stepdir/'train_packet_codes.npz')
            arrays_equal({str(s):code[s] for s in range(3)},saved,'Complete train greedy packet sets');bind(stepdir/'train_packet_codes.npz');new_npz.add(stepdir/'train_packet_codes.npz')
            profiles[t]=profile;codes[t]=code;pools[t]=pool;messages[t]=tm
            if t:
                previous=STEPS[STEPS.index(t)-1];change=temporal_change(messages[previous],tm,profiles[previous],profile);change.update(earlier=previous,later=t);changes.append(change)
            print(json.dumps(dict(audit_stage='natural_complete',seed=seed,condition=condition,update=t,train_modules=stat['train_modules'],elapsed_seconds=time.perf_counter()-start)),flush=True)
        # New position records can use the previously frozen independent single-symbol auditor unchanged.
        for t in STEPS:
            stepdir=directory/f'checkpoint_{t:04d}';pool=pools[t];values={};shams={}
            for u,arm,b in p.cells():
                rec=lookup['position',seed,condition,t,t,u,arm,b];name=f'position_unit{u}_{arm}_direction{b}';jp=stepdir/(name+'.json')
                require(read(jp)==rec,'Position row JSON/index equality');bind(jp);record_json.add(jp)
                require(rec['receiver_natural_source']==str(stepdir/'natural_validation.npz') and rec['is_reused'] is (t==6000),'Position receiver time/source')
                require(rec['recipient_full_indices_sha256']==context.array_sha(full_spec['endpoint_indices'][:,b]),'Position global recipient identity')
                if t==6000 and live:
                    anchor=next(r for r in old_pos if (r['unit'],r['arm'],r['direction'])==(u,arm,b))
                    for key,value in anchor.items():
                        if key not in ('new_forward_worlds','new_network_samples','neural_forward_calls'):require(rec[key]==value,'6000 old position metadata '+key)
                    require(rec['new_forward_worlds']==rec['new_network_samples']==rec['neural_forward_calls']==0 and rec['old_actual_forward_worlds']==144,'6000 no regenerated position')
                    require(rec['index_scope']=='prior_full_domain','Old position index scope');bind(rec['path'],rec['data_sha256']);data=load_npz(rec['path'])
                    r,cf,d=indices(full_spec,arm,b)
                    for k,w in (('recipient_indices',r),('counterfactual_recipient_indices',cf),('donor_indices',d)):require(np.array_equal(data[k],w),'6000 global index '+k)
                    _,_,di=indices(spec,arm,b);require(np.array_equal(data['donor_packets'],pool['messages'][di,:,spec['sender'],:]),'6000 source packet')
                    value=p.measure(spec,b,pool,data,codes[t])
                else:
                    require(rec['index_scope']=='local554_pool','New/silent local indices')
                    path=stepdir/(name+'.npz')
                    value,scope=p.audit_cell(rec,path,pool,spec,nets.get(t),info,codes[t])
                    stat['position_worlds']+=scope['independent_worlds'];stat['position_modules']+=scope['independent_modules'];stat['maximum_probability_error']=max(stat['maximum_probability_error'],scope['max_probability_error'])
                    if live:bind(path,rec['data_sha256']);new_npz.add(path);stat['new_position_npz']+=1
                if t==6000:stat['reused_position_records']+=1
                if not live:stat['silent_position_records']+=1
                if arm=='sham':shams[u,b]=value;stat['sham_checks']+=1
                else:values[u,arm,b]=value
            positions={}
            for u in range(8):
                arms={a:pair_values({b:values[u,a,b] for b in (0,1)}) for a in ('same','opposite')};arms['contrast']=p.contrast(arms['same'],arms['opposite']);positions[u]=arms
            natural=pair_values({b:p.measure(spec,b,pool,p.reference_arrays(pool,spec,b,'natural'),codes[t]) for b in (0,1)})
            shams={u:pair_values({b:shams[u,b] for b in (0,1)}) for u in (0,4)}
            point=point_summary(spec,profiles[t],profiles[6000],positions,natural,shams);point['update']=t
            saved=read(stepdir/'summary.json');numeric_tree(point,saved,counts,f'{seed}/{condition}/{t}');bind(stepdir/'summary.json');summary_paths.add(stepdir/'summary.json');points[t]=point
            if not live:close(point['current']['selectivity'],0,'Silent position structural zero')
        for rt,st in product(STEPS,STEPS):
            pool=pools[rt];donor_pool=pools[st];values={};inputs={};actions={};cell_dir=directory/f'cross_receiver{rt:04d}_donor{st:04d}'
            for arm,b in product(('same','opposite'),(0,1)):
                rec=lookup['cross_time_whole',seed,condition,rt,st,-1,arm,b];jp=cell_dir/f'{arm}_direction{b}.json'
                require(read(jp)==rec,'Cross row JSON/index equality');bind(jp);record_json.add(jp)
                reused=rt==st==6000;r,cf,d=indices(spec,arm,b);sender=spec['sender'];packets=donor_pool['messages'][d,:,sender,:]
                require(rec['is_reused'] is reused and rec['is_silent_alias'] is (not live) and rec['outward_patch_visible'] is live and rec['worlds']==144,'Cross reuse/live/worlds')
                require(rec['receiver_natural_source']==str(directory/f'checkpoint_{rt:04d}/natural_validation.npz') and rec['donor_natural_source']==str(directory/f'checkpoint_{st:04d}/natural_validation.npz'),'Both checkpoint source bindings')
                require(rec['donor_packets_sha256']==context.array_sha(packets),'Cross donor packet source SHA')
                if reused:
                    anchor=next(z for z in old_whole if z['arm']==arm and z['direction']==b)
                    require(rec['path']==anchor['path'] and rec['data_sha256']==anchor['data_sha256'] and rec['index_scope']=='prior_reference','6000 whole old reference identity')
                    bind(rec['path'],rec['data_sha256']);data=load_npz(rec['path'])
                    require(np.array_equal(data['patched_outward_packets'],packets),'6000 whole packet anchor')
                    require(rec['new_forward_worlds']==rec['new_network_samples']==rec['neural_forward_calls']==0,'6000 whole no regeneration');stat['reused_cross_records']+=1
                else:
                    x=val_x[info][r];m=pool['messages'][r];trace=whole_trace(nets.get(rt),x,m,sender,packets,live)
                    if not live:trace.update(action_indices=pool['action_indices'][r].copy(),action_probabilities=pool['action_probabilities'][r].copy())
                    expected=settle_arrays(pool,spec,b,trace,packets,d)
                    require(rec['routing_sha256']=={k:context.array_sha(trace[k]) for k in ('first_routes','second_routes','action_inputs')},'Cross complete routing and action-input hashes')
                    require(rec['new_forward_worlds']==(144 if live else 0) and rec['new_network_samples']==(864 if live else 0) and rec['neural_forward_calls']==(6 if live else 0),'Cross new forward budget')
                    require(rec['index_scope']=='local554_pool','Cross local index scope')
                    path=cell_dir/f'{arm}_direction{b}.npz'
                    if live:
                        require(rec['path']==str(path),'Cross actual output path');bind(path,rec['data_sha256']);data=load_npz(path)
                        error=arrays_equal(expected,data,'Cross independent complete output');stat['maximum_probability_error']=max(stat['maximum_probability_error'],error)
                        new_npz.add(path);stat['new_cross_npz']+=1
                    else:require(rec['path'] is None and rec['data_sha256'] is None and not path.exists(),'Silent cross no NPZ');data=expected
                    inputs[arm,b]=trace['action_inputs'];stat['cross_worlds']+=144 if live else 0;stat['cross_modules']+=trace['independent_module_samples']
                values[arm,b]=p.measure(spec,b,pool,data,codes[rt]);actions[arm,b]=data['action_indices']
                if not live:stat['silent_cross_records']+=1
            rows=np.arange(144)
            for b in (0,1):
                for listener in range(3):
                    take=spec['sender']!=listener
                    require(np.array_equal(actions['opposite',b][take,listener],actions['same',1-b][take,listener]),'Cross endpoint-exchange action identity')
                    if not (rt==st==6000):require(np.array_equal(inputs['opposite',b][take,listener],inputs['same',1-b][take,listener]),'Cross endpoint-exchange complete input identity')
                    stat['endpoint_exchange_checks']+=1
            arms={a:pair_values({b:values[a,b] for b in (0,1)}) for a in ('same','opposite')};arms['contrast']=p.contrast(arms['same'],arms['opposite'])
            summary={a:p.aggregate(spec,v) for a,v in arms.items()};cross[rt,st]=summary
            numeric_tree(summary,read(cell_dir/'summary.json'),counts,f'{seed}/{condition}/cross{rt},{st}');bind(cell_dir/'summary.json');summary_paths.add(cell_dir/'summary.json')
        expected=trajectory_summary(points,cross);expected.update(seed=seed,condition=condition,temporal_changes=changes,discovery_at_checkpoints={str(t):points[t]['discovery'] for t in STEPS})
        sp=directory/'summary.json';rec=next(z for z in main['policies'] if z['seed']==seed and z['condition']==condition)
        require(rec['path']==str(sp),'Policy summary path');bind(sp,rec['sha256']);summary_paths.add(sp);saved=read(sp)
        numeric_tree(expected,saved,counts,f'{seed}/{condition}/trajectory');numeric_tree(expected['normalized_areas'],rec['normalized_areas'],counts,'Index area')
        require(saved['point_summaries']=={str(t):dict(path=str(directory/f'checkpoint_{t:04d}/summary.json'),sha256=sha(directory/f'checkpoint_{t:04d}/summary.json')) for t in STEPS},'Checkpoint summary hashes')
        require(saved['cross_summaries']==[dict(receiver_update=r,donor_update=s,path=str(directory/f'cross_receiver{r:04d}_donor{s:04d}/summary.json'),sha256=sha(directory/f'cross_receiver{r:04d}_donor{s:04d}/summary.json')) for r in STEPS for s in STEPS],'Cross summary complete hashes')
        report=dict(seed=seed,condition=condition,trajectory=expected,selected_positions={str(t):profiles[t]['selected_positions'].tolist() for t in STEPS},
            exact_discovery_checked_at=list(STEPS),train_packet_set_sizes={str(t):[len(z) for z in codes[t]] for t in STEPS})
        write(out/f'seed_{seed}_{condition}.json',report);reports.append(report)
        del nets,pools,messages,codes
        print(json.dumps(dict(audit_stage='policy_complete',seed=seed,condition=condition,totals=stat,elapsed_seconds=time.perf_counter()-start)),flush=True)
    expected_stat=dict(train_worlds=33592320,train_modules=201553920,natural_worlds=44320,natural_modules=398880,
        position_worlds=207360,position_modules=933120,cross_worlds=161280,cross_modules=967680,parameter_loads=88,
        new_position_npz=1440,new_cross_npz=1120,reused_position_records=576,reused_cross_records=64,silent_position_records=1728,silent_cross_records=1152,
        endpoint_exchange_checks=16*36*6,sham_checks=16*6*4)
    for key,want in expected_stat.items():require(stat[key]==want,'Independent actual scope '+key)
    require(set(execution.rglob('*.npz'))==new_npz and len(new_npz)==2848,'All materialized new NPZ exactly, no extra output')
    require(len(record_json)==5952 and len(summary_paths)==688,'All row/checkpoint/cross/policy summary files')
    paired=[]
    for seed in SEEDS:
        a=next(r for r in reports if r['seed']==seed and r['condition']=='PL_live')['trajectory']['normalized_areas']['selectivity']
        b=next(r for r in reports if r['seed']==seed and r['condition']=='LL_live')['trajectory']['normalized_areas']['selectivity']
        paired.append(dict(seed=seed,PL=a,LL=b,difference=a-b))
    primary=dict(paired_seeds=paired,mean_difference=sum(z['difference'] for z in paired)/4);numeric_tree(primary,main['primary'],counts,'Primary')
    for path,digest in artifacts.items():require(sha(path)==digest,'Source/output changed during audit '+path)
    result=dict(status='passed',plan_sha256=expected_plan_sha,audit_source_sha256=sha(__file__),runtime=dict(python=platform.python_version(),numpy=np.__version__),
        independent_scope=stat,new_neural_module_samples=sum(stat[k] for k in ('train_modules','natural_modules','position_modules','cross_modules')),
        new_training_updates=0,optimizer_replays=0,reused6000_neural_forwards=0,primary=primary,
        numerical_comparison=counts,normalized_trapezoid_weights=[dict(numerator=w.numerator,denominator=w.denominator) for w in exact_time_weights()],
        materialized_npz_files=len(new_npz),record_json_files=len(record_json),summary_json_files=len(summary_paths),artifacts_sha256=artifacts,
        checkpoint_replay_boundary='All newly generated train/validation/intervention outputs independently forwarded;6000 saved anchors are SHA/array checked with the prior full independent audit, not regenerated.',
        scalar_scope='All independently reconstructed numeric leaves of96 checkpoint current/retrospective/all-position/natural/sham summaries,576 cross-pair summaries,16 trajectories/changes and primary. Producer prose is not interpreted as evidence.',
        elapsed_seconds=time.perf_counter()-start)
    write(out/'verification.json',result)
    (out/'独立核验.md').write_text('完整独立核验通过。新增模块样本 '+str(result['new_neural_module_samples'])+'，参数加载 '+str(stat['parameter_loads'])+'；无训练或优化器重放。6000旧锚按原SHA与逐数组引用核对，不重复前向。\n\n独立概率最大误差 '+str(stat['maximum_probability_error'])+'；独立摘要数字 '+str(counts['scalars'])+' 项，最大绝对差 '+str(counts['max_error'])+'。具体覆盖、文件SHA和四种子主量见 verification.json。\n')
    return result


if __name__=='__main__':
    import argparse,time,traceback
    from datetime import datetime,timezone
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--out',required=True);parser.add_argument('--plan-sha',required=True)
    args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    write(out/'started.json',dict(at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),run=str(Path(args.run).resolve()),plan_sha256=args.plan_sha,audit_source_sha256=sha(__file__)))
    try:
        result=audit(args.run,out,args.plan_sha);write(out/'terminal.json',dict(status='completed',at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,verification_sha256=sha(out/'verification.json')))
        print(json.dumps({k:result[k] for k in ('status','new_neural_module_samples','independent_scope','numerical_comparison','primary','elapsed_seconds')},ensure_ascii=False),flush=True)
    except BaseException as error:
        write(out/'failure.json',dict(status='failed',at=datetime.now(timezone.utc).isoformat(),error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started));raise
