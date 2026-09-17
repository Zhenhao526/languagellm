"""Independent, read-only verification of the fixed role-decoder experiment.

No current runner, dataset or learner is imported. The completed initial sender
transcripts are replayed before the 42 complete endpoint decoder evaluations.
Monitor arrays and training random streams are checked as records; optimizer,
gradient, training-input and intermediate neural computations are NOT replayed.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
from itertools import zip_longest
import json
import math
from pathlib import Path
import platform
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PLAN_SHA='9cf2995a75acb9cd5d69d36897019299bb9123095b684922345b3ae41ddc9f41'
SOURCE_PLAN_SHA='99b9663dfae02868692d53707ad5b7db40b7f2ee31c951edf2fd68f05859f33f'
SOURCE_RESULT_SHA='0d0fd3bbea91eae728384b0bffc4e040f250193004465451305eb56d451615e2'
SOURCE_AUDIT_SHA='b270398c609132623d83b01fb6e7fe65077688af013a0b36aa83e501436cbf97'
HELPER_SHA={
 'triadic_action_dependency_study/audit_execution.py':'20f1be8c017a53874346d1e8865620586d56de4128580b1e428c27899a30e96d',
 'triadic_message_study/audit_execution.py':'f09f30612ccb7e27b9ec7e3b38b9fa217d133f1df1185e2712fd899254151bd3',
 'triadic_learning_baseline/audit_execution.py':'739175596c4182068bbfbb578439189208cae72812068b2127d0006393be2281'}
PROBE_SEEDS=(55101,55102,55103)
OLD_SEEDS=(53101,53102,53103,53104)
PAYOFFS=('a50','a10')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
STEPS=(0,100,500,1500,3000,6000)
VIEWS=('Own','FI','Initial','Final')
DIMS=(252,64,64,3)
SHAPES={k:s for layer,(left,right) in enumerate(zip(DIMS,DIMS[1:]),1)
        for k,s in ((f'W{layer}',(left,right)),(f'b{layer}',(right,)))}
OPTIMIZER=dict(learning_rate=.001,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,global_gradient_clip=5.)
TOL=2e-12


def require(ok,message):
    if not ok:raise AssertionError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def json_bytes(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def json_sha(value):return sha256(json_bytes(value)).hexdigest()


def sha(path):
    h=sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def array_sha(value):
    a=np.ascontiguousarray(value);h=sha256(json_bytes(dict(shape=list(a.shape),dtype=a.dtype.str)))
    h.update(a.tobytes());return h.hexdigest()


def close(actual,expected,label):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' shape/finite')
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    require(np.allclose(a,b,atol=TOL,rtol=0),label+f' numerical difference {error}')
    return error


def compare(actual,expected,label=''):
    if isinstance(expected,dict):
        require(isinstance(actual,dict),label+' mapping')
        for k,v in expected.items():
            require(k in actual,label+' missing '+k);compare(actual[k],v,label+k+'/')
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),label+' list')
        for i,v in enumerate(expected):compare(actual[i],v,label+str(i)+'/')
    elif isinstance(expected,float):close(actual,expected,label)
    else:require(actual==expected,label+' value')


def finite_tree(value):
    if isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,list):
        for v in value:finite_tree(v)
    elif isinstance(value,float):require(math.isfinite(value),'Nonfinite JSON value')


def helpers():
    for name,digest in HELPER_SHA.items():require(sha(ROOT/'research_program'/name)==digest,'Historical independent helper changed: '+name)
    from research_program.triadic_action_dependency_study import audit_execution as env
    from research_program.triadic_message_study import audit_execution as sender
    return env,sender


def role_truth(states):
    """Research labels from the unique physical solution, independent of learner.

    Enumerate the eight material/destination combinations for each partner pair.
    Layouts are complete permutations, so this also counts physical site plans.
    Role 1/2 means first/second OTHER identity, not an absolute actor identity.
    """
    states=np.asarray(states)
    require(states.ndim==2 and states.shape[1]==10 and states.dtype.kind in 'iu','Truth states')
    needs=states[:,:3]
    require(np.all((needs>=0)&(needs<24)),'Truth need domain')
    resource=np.asarray((3,12,5,10,1,2,4,8))[needs//3]
    dest=np.asarray((1,2,3))[needs%3]
    accept=((resource[:,:,None,None]>>np.arange(4)[None,None,:,None])&1).astype(bool)
    accept=accept&(((dest[:,:,None,None]>>np.arange(2)[None,None,None,:])&1).astype(bool))
    counts=np.stack([(accept[:,i]&accept[:,j]).sum((1,2)) for i,j in ((0,1),(0,2),(1,2))],axis=1)
    require(np.all(counts.sum(1)==1),'Truth requires exactly one full physical plan')
    labels=np.zeros((len(states),3),dtype=np.int8)
    for column,(i,j) in enumerate(((0,1),(0,2),(1,2))):
        mask=counts[:,column]==1
        labels[mask,i]=1+[a for a in range(3) if a!=i].index(j)
        labels[mask,j]=1+[a for a in range(3) if a!=j].index(i)
    return labels


def route(messages):
    """Legal all-live window: 96 payload columns followed by 3 visibility bits."""
    m=np.asarray(messages)
    require(m.ndim==3 and m.shape[1:]==(3,4) and m.dtype.kind in 'iu' and np.all((m>=0)&(m<8)),'Route message domain')
    # Construct one public transcript, then copy it to each actor. Own messages
    # are included here; Own/FI controls instead leave all 198 columns at zero.
    payload=np.eye(8,dtype=np.float64)[m].reshape(len(m),96)
    public=np.concatenate((payload,np.ones((len(m),3))),axis=1)
    return np.repeat(public[:,None,:],3,axis=1)


def inputs(states,view,messages,env):
    require(view in VIEWS,'Input view')
    x=np.zeros((len(states),3,252),dtype=np.float64)
    x[:,:,:54]=env.features(states,'FI' if view=='FI' else 'PL')
    if view in ('Initial','Final'):
        require(messages is not None and messages.shape==(len(states),2,3,4),'Transcript input shape')
        x[:,:,54:153]=route(messages[:,0]);x[:,:,153:252]=route(messages[:,1])
    else:require(not x[:,:,54:].any(),'Control contains a message column')
    return x


def probe_forward(networks,x):
    require(len(networks)==3 and x.ndim==3 and x.shape[1:]==(3,252),'Probe forward shapes')
    logits=[]
    for a,net in enumerate(networks):
        h=np.tanh(x[:,a]@net['W1']+net['b1'])
        h=np.tanh(h@net['W2']+net['b2'])
        logits.append(h@net['W3']+net['b3'])
    logits=np.stack(logits,axis=1);require(np.isfinite(logits).all(),'Probe nonfinite logits')
    shifted=logits-logits.max(-1,keepdims=True);e=np.exp(shifted);z=e.sum(-1,keepdims=True)
    return e/z,shifted-np.log(z)


def cross_entropy(log_probabilities,labels):
    require(log_probabilities.shape==(len(labels),3,3),'CE dimensions')
    require(labels.shape==(len(labels),3) and labels.dtype.kind in 'iu' and np.all((labels>=0)&(labels<3)),'CE labels')
    return -np.take_along_axis(log_probabilities,labels[:,:,None],axis=-1)[:,:,0].mean(1)


def parameter_hash(networks):
    return json_sha({f'agent{a}_{k}':array_sha(net[k]) for a,net in enumerate(networks) for k in SHAPES})


def probe_checkpoint(path,step,seed,expected_rng):
    keys={'schema','dimensions','update','world_rng_json','optimizer_config_json'};nets=[]
    with np.load(path,allow_pickle=False) as z:
        require(z['schema'].shape==() and z['schema'].dtype.kind=='U' and str(z['schema'].item())=='triadic_role_decoder_checkpoint_v1','Probe checkpoint schema')
        require(z['dimensions'].dtype==np.int64 and np.array_equal(z['dimensions'],DIMS),'Probe checkpoint dimensions')
        require(z['update'].shape==() and z['update'].dtype==np.int64 and int(z['update'])==step,'Probe checkpoint update')
        for a in range(3):
            rng=np.random.default_rng(np.random.SeedSequence([seed,a,730])) if step==0 else None
            net={}
            for layer,(left,right) in enumerate(zip(DIMS,DIMS[1:]),1):
                initial=rng.normal(0,math.sqrt(2/(left+right)),(left,right)) if rng is not None else None
                for prefix,shape in (('W',(left,right)),('b',(right,))):
                    k=f'{prefix}{layer}';name=f'agent{a}_{k}';keys.add(name);v=z[name]
                    require(v.shape==shape and v.dtype==np.float64 and np.isfinite(v).all(),'Probe parameter '+name)
                    if step==0:require(np.array_equal(v,initial if prefix=='W' else np.zeros(right)),'Probe initial parameter stream')
                    net[k]=v.copy()
                    for moment in ('m','v'):
                        name=f'adam_agent{a}_{moment}_{k}';keys.add(name);v=z[name]
                        require(v.shape==shape and v.dtype==np.float64 and np.isfinite(v).all(),'Probe Adam '+name)
                        if moment=='v':require(np.all(v>=0),'Negative Adam variance')
                        if step==0:require(not v.any(),'Initial Adam state is nonzero')
            nets.append(net)
        require(set(z.files)==keys,'Unexpected probe checkpoint fields')
        for key in ('world_rng_json','optimizer_config_json'):require(z[key].shape==() and z[key].dtype.kind=='U','Checkpoint JSON type')
        require(json.loads(str(z['world_rng_json'].item()))==expected_rng,'Independent checkpoint world RNG')
        require(json.loads(str(z['optimizer_config_json'].item()))==OPTIMIZER,'Checkpoint optimizer configuration')
    return nets


def evaluation_summary(p,labels,indices):
    require(p.shape==(len(labels),3,3) and p.dtype==np.float64 and np.isfinite(p).all() and np.all((p>0)&(p<=1)),'Saved probability domain')
    close(p.sum(-1),np.ones((len(labels),3)),'Saved probability normalization')
    pred=p.argmax(-1).astype(np.int8);correct=pred==labels;joint=correct.all(1)
    nll=cross_entropy(np.log(p),labels);chosen=np.take_along_axis(p,labels[:,:,None],axis=-1)[:,:,0]
    strata={}
    for wait,pair in ((2,'AB'),(1,'AC'),(0,'BC')):
        mask=labels[:,wait]==0;require(mask.any(),'Evaluation lacks true pair '+pair)
        strata[pair]=dict(worlds=int(mask.sum()),joint_accuracy=float(joint[mask].mean()))
    unique,counts=np.unique(pred,axis=0,return_counts=True)
    result=dict(worlds=len(labels),joint_accuracy=float(joint.mean()),indiv_accuracy=float(correct.mean()),
        cross_entropy=float(nll.mean()),expected_joint_correct_probability=float(np.prod(chosen,axis=1).mean()),
        per_actor_accuracy=correct.mean(0).tolist(),true_pair_strata=strata,
        predicted_joint_role_counts=[dict(labels=v.tolist(),worlds=int(c)) for v,c in zip(unique,counts)],
        state_indices_sha256=array_sha(indices))
    return result,pred,nll,joint


def identity(probe_seed,view,old_seed=None,payoff=None):
    name=f'probe_{probe_seed}_{view}'
    if old_seed is not None:name+=f'_protocol_{old_seed}'
    if payoff is not None:name+='_'+payoff
    return name


def expected_jobs():
    rows=[]
    for probe in PROBE_SEEDS:
        for view in ('Own','FI'):rows.append(dict(probe_seed=probe,view=view,old_seed=None,payoff=None))
        for seed in OLD_SEEDS:
            rows.append(dict(probe_seed=probe,view='Initial',old_seed=seed,payoff=None))
            for payoff in PAYOFFS:rows.append(dict(probe_seed=probe,view='Final',old_seed=seed,payoff=payoff))
    for row in rows:row['id']=identity(**row)
    return rows


def logical_and_primary(runs):
    by={r['job']['id']:r for r in runs}
    require(len(runs)==len(by)==42 and set(by)=={j['id'] for j in expected_jobs()},'Actual decoder identities')
    logical=[];blocks=[]
    for seed in OLD_SEEDS:
        cells=[]
        for payoff in PAYOFFS:
            probes=[]
            for probe in PROBE_SEEDS:
                scores={}
                for view in VIEWS:
                    actual=identity(probe,view,seed if view in ('Initial','Final') else None,payoff if view=='Final' else None)
                    logical.append(dict(old_seed=seed,payoff=payoff,probe_seed=probe,view=view,actual_job_id=actual))
                    scores[view]=by[actual]['final']['new_needs_and_layouts']['joint_accuracy']
                probes.append(dict(probe_seed=probe,accuracies=scores,
                    final_minus_initial=scores['Final']-scores['Initial'],final_minus_own=scores['Final']-scores['Own'],
                    final_minus_bound=scores['Final']-23/62,initial_minus_bound=scores['Initial']-23/62))
            cells.append(dict(payoff=payoff,probes=probes,mean_final_minus_initial=float(np.mean([p['final_minus_initial'] for p in probes]))))
        blocks.append(dict(old_seed=seed,payoffs=cells,mean_final_minus_initial=float(np.mean([c['mean_final_minus_initial'] for c in cells]))))
    counts=Counter(r['actual_job_id'] for r in logical)
    require(len(logical)==96 and len(counts)==42 and sum(v-1 for v in counts.values())==54,'Logical aliases')
    require(Counter(counts.values())==Counter({1:24,2:12,8:6}),'Own/FI/Initial alias multiplicities')
    return logical,dict(metric='joint_role_accuracy_final_minus_initial',partition='new_needs_and_layouts',
        protocol_seed_blocks=blocks,mean_difference=float(np.mean([b['mean_final_minus_initial'] for b in blocks])))


def expected_budget(parts):
    worlds=sum(s['world_count'] for s in parts.values());monitor=sum(len(s['monitor_indices']) for s in parts.values())
    require(worlds==774144 and monitor==21504,'World/monitor support')
    return dict(actual_decoder_runs=42,logical_decoder_runs=96,aliased_decoder_runs=54,updates=252000,
        training_world_samples=64512000,training_forward_module_samples=193536000,checkpoints=252,
        actual_monitor_files=1008,actual_final_files=168,aliased_evaluation_records=1512,
        actual_monitor_worlds=42*6*monitor,actual_final_worlds=42*worlds,
        decoder_evaluation_module_samples=(42*6*monitor+42*worlds)*3,
        initial_transcript_files=16,initial_sender_worlds=4*worlds,initial_sender_module_samples=4*worlds*6,
        old_parameter_loads=4,reused_final_transcript_files=32,reused_final_transcript_worlds=8*worlds)


def feature_hash(states,information,env):
    h=sha256(json_bytes(dict(shape=[len(states),3,54],dtype=np.dtype(np.float64).str)))
    for start in range(0,len(states),1024):h.update(np.ascontiguousarray(env.features(states[start:start+1024],information)).tobytes())
    return h.hexdigest()


def audit(run_directory,audit_directory):
    start=time.perf_counter();run=Path(run_directory).resolve();out=Path(audit_directory).resolve()
    require(not out.exists(),'Audit destination exists');out.mkdir(parents=True)
    artifacts={};evaluations={};counts=Counter();max_probability_error=0.;max_ce_error=0.
    def bind(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None:require(digest==expected,'Artifact SHA: '+str(path))
        artifacts[str(path)]=digest;return digest
    try:
        env,sender=helpers()
        for name,digest in HELPER_SHA.items():bind(ROOT/'research_program'/name,digest)
        bind(__file__);bind(HERE/'tests/test_audit.py');bind(HERE/'audit_preflight_001.json')
        freeze=read(HERE/'audit_freeze_001.json');bind(HERE/'audit_freeze_001.json')
        require(freeze['plan_sha256']==PLAN_SHA and freeze['status']=='frozen_before_formal_audit','Audit source freeze')
        require(set(freeze['source_sha256'])=={str(Path(__file__).resolve()),str(HERE/'tests/test_audit.py'),str(HERE/'audit_preflight_001.json')},'Audit freeze source coverage')
        for path,digest in freeze['source_sha256'].items():bind(path,digest)
        plan=read(run/'plan.json');prepared=read(run/'prepared.json');source=prepared['source'];execution=run/'execution'
        bind(run/'plan.json',PLAN_SHA);bind(run/'freeze.json');bind(run/'prepared.json',plan['prepared_sha256'])
        require(read(run/'freeze.json')['plan_sha256']==PLAN_SHA,'Plan freeze')
        require(prepared['schema']=='frozen_transcript_role_decoder_v1','Prepared schema')
        require(plan['runtime']==dict(python=platform.python_version(),numpy=np.__version__),'Audit/runtime mismatch')
        result=read(execution/'results.json');bind(execution/'results.json');bind(execution/'status.json');bind(execution/'started.json')
        require(result['status']==read(execution/'status.json')['status']=='completed' and not (execution/'failure.json').exists(),'Incomplete/failed decoder run')
        require(result['plan_sha256']==read(execution/'started.json')['plan_sha256']==PLAN_SHA,'Result/started plan identity')
        finite_tree(result)
        compare(plan['config'],dict(probe_seeds=list(PROBE_SEEDS),protocol_seeds=list(OLD_SEEDS),payoffs=list(PAYOFFS),
            updates=6000,batch_size=256,checkpoints=list(STEPS),dimensions=list(DIMS),heads=3,
            objective='mean_supervised_role_cross_entropy_over_worlds_and_actors; no entropy',
            primary='double_holdout_joint_role_accuracy_final_minus_initial; mean_probe_then_payoff_then_protocol_seed',
            auxiliary_absolute_bound=23/62,views=list(VIEWS),checkpoint_selection='fixed_6000_no_early_stopping',
            world_sampling='original_three_uniform_need_layout_owner',learning_rate=.001,global_gradient_clip=5.,dtype='float64'))
        require(plan['inputs_sha256']==source['source_sha256'],'Frozen source manifest differs')
        for path,digest in plan['inputs_sha256'].items():bind(path,digest)
        for path,digest in plan['sources'].items():
            bind(path,digest);bind(run/'source_snapshot'/Path(path).relative_to(ROOT),digest)
        oldrun=Path(source['source_run']);bind(oldrun/'plan.json',SOURCE_PLAN_SHA)
        bind(oldrun/'execution/results.json',SOURCE_RESULT_SHA);bind(oldrun/'audit_execution_001/verification.json',SOURCE_AUDIT_SHA)
        oldplan=read(oldrun/'plan.json');bind(oldrun/'prepared.json',oldplan['prepared_sha256'])
        oldparts=read(oldrun/'prepared.json')['partitions'];parts=source['partitions']
        require(parts==oldparts,'Original full partition metadata changed')
        independent,_,_=env.independent_specs()
        require(set(parts)==set(PARTS),'Partition names')
        # Original metadata has additional descriptive fields: compare only the
        # independently reconstructed core against each whole anchored record.
        for part in PARTS:compare(parts[part],independent[part],'independent partition '+part+'/')
        compare(prepared['budget'],expected_budget(parts),'Prepared budget/')
        compare(result['budget'],expected_budget(parts),'Result budget/')
        require(prepared['jobs']==expected_jobs() and [r['job'] for r in result['runs']]==expected_jobs(),'Run order/identities')
        require(source['schema']=='triadic_role_decoder_sources_v1','Source schema')
        compare(source['counts'],dict(policies=8,independent_source_seeds=4,initial_protocols=4,final_npz=32,
            monitor0_npz=32,checkpoint0_files_bound=8,total_worlds_per_protocol=774144,
            parameter_arrays_loaded=0,neural_forward_calls=0,training_updates=0),'Preparation source counts/')
        policies={(p['seed'],p['payoff']):p for p in source['policies']}
        require(len(source['policies'])==len(policies)==8 and set(policies)=={(s,p) for s in OLD_SEEDS for p in PAYOFFS},'Source policy identities')
        initial={r['seed']:r for r in source['initial_sources']}
        require(len(source['initial_sources'])==len(initial)==4 and set(initial)==set(OLD_SEEDS),'Initial source identities')
        oldresult=read(oldrun/'execution/results.json');oldby={(r['seed'],r['payoff'],r['condition']):r for r in oldresult['runs']}
        oldaudit=read(oldrun/'audit_execution_001/verification.json')
        require(oldaudit['status']=='passed','Prior independent audit did not pass')
        for (seed,payoff),policy in policies.items():
            require(policy['condition']=='PL_live','Wrong sender condition')
            oldrecord=oldby[seed,payoff,'PL_live'];cp=policy['checkpoint0']
            bind(cp['path'],cp['sha256']);require(cp['sha256']==oldrecord['monitor'][0]['checkpoint_sha256'],'Old checkpoint identity')
            require(policy['initial_parameter_sha256']==oldrecord['initial_parameter_sha256'],'Old initial parameter identity')
            for part in PARTS:
                for name,entry in (('final',oldrecord['final'][part]['natural']),('monitor0',oldrecord['monitor'][0]['monitor'][part]['natural'])):
                    value=policy[name][part]
                    require(value['path']==entry['path'] and value['sha256']==entry['data_sha256'],'Old transcript identity')
                    require(value['worlds']==entry['worlds'] and value['state_indices_sha256']==entry['state_indices_sha256'],'Old transcript coverage')
                    require(value['sha256']==oldaudit['artifacts_sha256'][value['path']],'Old transcript not bound by audit')
        equalities=[]
        for seed in OLD_SEEDS:
            a,b=policies[seed,'a50'],policies[seed,'a10']
            require(a['checkpoint0']['sha256']==b['checkpoint0']['sha256'] and a['initial_parameter_sha256']==b['initial_parameter_sha256'],'Payoff initial equality')
            compare(initial[seed],dict(seed=seed,payoff='a50',checkpoint0=a['checkpoint0'],monitor0=a['monitor0'],paired_monitor0=b['monitor0']))
            equalities.append(dict(seed=seed,a50=a['checkpoint0'],a10=b['checkpoint0'],byte_identical=True,
                parameter_hash=a['initial_parameter_sha256']))
        require(source['checkpoint0_equalities']==equalities,'Four initial source aliases')
        states={p:env.packed(parts[p]) for p in PARTS};labels={p:role_truth(s) for p,s in states.items()}
        for part in PARTS:
            hashes=dict(packed_states=array_sha(states[part]),labels=array_sha(labels[part]),
                x_PL=feature_hash(states[part],'PL',env),x_FI=feature_hash(states[part],'FI',env))
            require(result['array_hashes'][part]==hashes,'Independent array hashes '+part)
            for key in ('packed_states','x_PL','x_FI'):require(hashes[key]==oldresult['array_hashes'][part][key],'Original information arrays '+part+'/'+key)
        for seed in PROBE_SEEDS:
            path=execution/f'probe_{seed}_arrays.json';bind(path);require(read(path)==result['array_hashes'],'Worker array identity')

        # Phase 1: all four initialization senders, all four full partitions.
        initial_records=result['initial_transcripts'];initial_by={(r['seed'],r['partition']):r for r in initial_records}
        require(len(initial_records)==len(initial_by)==16 and set(initial_by)=={(s,p) for s in OLD_SEEDS for p in PARTS},'Initial transcript coverage')
        path=execution/'initial_transcripts/results.json';bind(path)
        compare(read(path),dict(status='completed',records=initial_records,module_samples=18579456,old_parameter_loads=4),'Initial transcript receipt/')
        initial_nets={}
        for seed in OLD_SEEDS:
            cp=initial[seed]['checkpoint0'];nets,_,_=sender.checkpoint(cp['path'],0,seed)
            require(sender.network_hash(nets)==policies[seed,'a50']['initial_parameter_sha256'],'Old independently initialized parameter hash')
            initial_nets[seed]=nets;counts['old_checkpoint_loads']+=1
        for seed in OLD_SEEDS:
            for part in PARTS:
                row=initial_by[seed,part];path=execution/'initial_transcripts'/f'protocol_{seed}_{part}.npz'
                require(Path(row['path']).resolve()==path.resolve(),'Initial transcript path');bind(path,row['sha256'])
                require(row['worlds']==len(states[part]) and row['module_samples']==len(states[part])*6,'Initial transcript counts')
                require(row['initial_checkpoint_sha256']==initial[seed]['checkpoint0']['sha256'] and row['monitor_anchor_payoffs']==list(PAYOFFS),'Initial source anchor')
                with np.load(path,allow_pickle=False) as z:
                    require(set(z.files)=={'states','state_indices','messages'},'Initial transcript NPZ schema')
                    ids=z['state_indices'];saved_states=z['states'];messages=z['messages']
                require(ids.dtype==np.int64 and np.array_equal(ids,np.arange(len(states[part]))),'Initial transcript full ids')
                require(saved_states.dtype==np.int16 and np.array_equal(saved_states,states[part]),'Initial transcript full states')
                require(messages.dtype==np.int8 and messages.shape==(len(ids),2,3,4) and np.all((messages>=0)&(messages<8)),'Initial message domain')
                for start0 in range(0,len(ids),1024):
                    sl=slice(start0,start0+1024);x=env.features(states[part][sl],'PL');transcript=[];xx=x
                    for window in (0,1):
                        pp=np.stack([sender.probabilities(initial_nets[seed][3*a+window],xx[:,a]) for a in range(3)],axis=1)
                        mm=pp.argmax(-1).astype(np.int8);transcript.append(mm)
                        if window==0:xx=np.concatenate((x,route(mm)),axis=-1)
                    require(np.array_equal(np.stack(transcript,axis=1),messages[sl]),'Independent initial sender replay')
                    counts['initial_sender_module_samples']+=len(x)*6
                monitor=np.asarray(parts[part]['monitor_indices'],dtype=np.int64)
                for payoff in PAYOFFS:
                    anchor=policies[seed,payoff]['monitor0'][part]
                    with np.load(anchor['path'],allow_pickle=False) as z:
                        require(np.array_equal(z['state_indices'],monitor) and np.array_equal(z['states'],states[part][monitor]),'Old initial monitor worlds')
                        require(np.array_equal(z['messages'],messages[monitor]),'Old initial monitor transcript equality')
                counts['initial_transcript_files']+=1
            print(json.dumps(dict(stage='initial_replay_complete',old_seed=seed)),flush=True)
        del initial_nets

        # Phase 2: rebuild each seed's shared random world stream, validating all
        # 14 logged uses. No network forward or gradient calculation occurs here.
        rng_states={};runs=result['runs']
        for probe in PROBE_SEEDS:
            subset=[r for r in runs if r['job']['probe_seed']==probe]
            require(len(subset)==14 and len({r['initial_parameter_sha256'] for r in subset})==1,'Shared probe initialization')
            rng=np.random.default_rng(np.random.SeedSequence([probe,800]));rng_states[probe]={0:rng.bit_generator.state}
            with ExitStack() as stack:
                streams=[]
                for record in subset:
                    path=execution/record['job']['id']/'training.jsonl';bind(path,record['training_log_sha256'])
                    streams.append(stack.enter_context(path.open()))
                seen=0
                for step,lines in enumerate(zip_longest(*streams),1):
                    require(step<=6000 and all(line is not None for line in lines),'Training row coverage')
                    u=rng.random((256,3));ids=env.sample(parts['train'],u)
                    expected=dict(world_uniforms_sha256=array_sha(u),batch_indices_sha256=array_sha(ids),
                        batch_states_sha256=array_sha(states['train'][ids]),labels_sha256=array_sha(labels['train'][ids]))
                    for record,line in zip(subset,lines):
                        row=json.loads(line);finite_tree(row)
                        require(set(row)=={'loss','indiv_accuracy','joint_accuracy','update','job_id','world_uniforms_sha256',
                            'batch_indices_sha256','batch_states_sha256','labels_sha256','input_features_sha256','gradient_norm','gradient_clip_scale'},'Training log schema')
                        compare(row,dict(update=step,job_id=record['job']['id'],**expected),'Independent training stream/')
                        require(row['loss']>=0 and 0<=row['indiv_accuracy']<=1 and 0<=row['joint_accuracy']<=1,'Training logged metrics domain')
                        require(row['gradient_norm']>=0 and 0<row['gradient_clip_scale']<=1,'Training gradient log domain')
                        require(isinstance(row['input_features_sha256'],str) and len(row['input_features_sha256'])==64 and all(c in '0123456789abcdef' for c in row['input_features_sha256']),'Training input hash syntax only')
                        counts['training_rows_world_and_labels_rebuilt']+=1
                    if step in STEPS:rng_states[probe][step]=rng.bit_generator.state
                    seen=step
                require(seen==6000,'Training updates')
            print(json.dumps(dict(stage='training_stream_verified',probe_seed=probe)),flush=True)

        # Phase 3: all 252 checkpoints/1008 saved monitors, and full independent
        # three-head endpoint replay for each of the 42 physical decoder jobs.
        verified_runs=[]
        for record in runs:
            job=record['job'];directory=execution/job['id'];path=directory/'result.json';bind(path)
            require(read(path)==record,'Per-job/aggregate result mismatch')
            monitor_path=directory/'monitor.jsonl';bind(monitor_path)
            monitor_rows=[json.loads(line) for line in monitor_path.read_text().splitlines()]
            require(monitor_rows==record['monitor'] and [r['update'] for r in monitor_rows]==list(STEPS),'Monitor record coverage')
            final_networks=None
            for row in monitor_rows:
                step=row['update'];path=directory/f'checkpoint_{step:04d}.npz';bind(path,row['checkpoint_sha256'])
                nets=probe_checkpoint(path,step,job['probe_seed'],rng_states[job['probe_seed']][step]);counts['probe_checkpoints']+=1
                if step==0:require(parameter_hash(nets)==record['initial_parameter_sha256'],'Probe initial parameter hash')
                if step==6000:
                    require(parameter_hash(nets)==record['final_parameter_sha256'],'Probe final parameter hash');final_networks=nets
                require(set(row['evaluations'])==set(PARTS),'Monitor partitions')
            verified_final={}
            for part in PARTS:
                messages=None
                if job['view']=='Initial':
                    path=initial_by[job['old_seed'],part]['path']
                elif job['view']=='Final':path=policies[job['old_seed'],job['payoff']]['final'][part]['path']
                if job['view'] in ('Initial','Final'):
                    with np.load(path,allow_pickle=False) as z:
                        require(z['state_indices'].dtype==np.int64 and np.array_equal(z['state_indices'],np.arange(len(states[part]))),'Source transcript full ids')
                        require(z['states'].dtype==np.int16 and np.array_equal(z['states'],states[part]),'Source transcript full worlds')
                        messages=z['messages']
                    require(messages.dtype==np.int8 and messages.shape==(len(states[part]),2,3,4) and np.all((messages>=0)&(messages<8)),'Source transcript message domain')
                for is_final,row in [(False,r) for r in monitor_rows]+[(True,None)]:
                    if is_final:
                        entry=record['final'][part];ids=np.arange(len(states[part]),dtype=np.int64);path=directory/f'final_{part}.npz'
                    else:
                        entry=row['evaluations'][part];ids=np.asarray(parts[part]['monitor_indices'],dtype=np.int64);path=directory/f"monitor_{row['update']:04d}_{part}.npz"
                    require(Path(entry['path']).resolve()==path.resolve(),'Evaluation path');bind(path,entry['sha256'])
                    with np.load(path,allow_pickle=False) as z:
                        require(set(z.files)=={'states','state_indices','labels','probabilities','predictions','negative_log_likelihood','joint_correct'},'Evaluation NPZ schema')
                        value={k:z[k] for k in z.files}
                    require(value['state_indices'].dtype==np.int64 and np.array_equal(value['state_indices'],ids),'Evaluation ordered ids')
                    require(value['states'].dtype==np.int16 and np.array_equal(value['states'],states[part][ids]),'Evaluation packed worlds')
                    target=labels[part][ids]
                    require(value['labels'].dtype==np.int8 and np.array_equal(value['labels'],target),'Independent role truth')
                    stats,pred,nll,joint=evaluation_summary(value['probabilities'],target,ids)
                    require(value['predictions'].dtype==np.int8 and np.array_equal(value['predictions'],pred),'Saved greedy roles')
                    require(value['joint_correct'].dtype==bool and np.array_equal(value['joint_correct'],joint),'Saved joint correctness')
                    require(value['negative_log_likelihood'].dtype==np.float64,'Saved CE dtype')
                    max_ce_error=max(max_ce_error,close(value['negative_log_likelihood'],nll,'CE from saved probabilities'))
                    compare(entry,stats,'Evaluation statistics/')
                    err=0.
                    if is_final:
                        for start0 in range(0,len(ids),1024):
                            sl=slice(start0,start0+1024);batch=ids[sl]
                            x=inputs(states[part][batch],job['view'],None if messages is None else messages[batch],env)
                            p,lp=probe_forward(final_networks,x)
                            err=max(err,close(value['probabilities'][sl],p,'Independent complete endpoint probability'))
                            max_ce_error=max(max_ce_error,close(value['negative_log_likelihood'][sl],cross_entropy(lp,target[sl]),'Independent complete endpoint CE'))
                            require(np.array_equal(value['predictions'][sl],p.argmax(-1)),'Independent endpoint greedy roles')
                            counts['decoder_final_module_samples']+=len(batch)*3
                        max_probability_error=max(max_probability_error,err);verified_final[part]=stats
                        counts['final_files']+=1;counts['final_worlds']+=len(ids)
                    else:counts['monitor_files']+=1;counts['monitor_worlds']+=len(ids)
                    evaluations[str(path)]=dict(sha256=entry['sha256'],worlds=len(ids),neural_replayed=is_final,max_probability_error=err,**{k:stats[k] for k in ('joint_accuracy','cross_entropy')})
            verified_runs.append(dict(job=job,final=verified_final))
            print(json.dumps(dict(stage='decoder_audited',job=job['id'])),flush=True)
        logical,primary=logical_and_primary(verified_runs)
        require(result['logical_results']==logical,'Logical reuse mapping');compare(result['primary'],primary,'Primary nesting/')
        expected_counts=dict(old_checkpoint_loads=4,initial_sender_module_samples=18579456,initial_transcript_files=16,
            training_rows_world_and_labels_rebuilt=252000,probe_checkpoints=252,decoder_final_module_samples=97542144,
            final_files=168,final_worlds=32514048,monitor_files=1008,monitor_worlds=5419008)
        require(dict(counts)==expected_counts,'Independent audit coverage counts')
        verification=dict(status='passed',at=datetime.now(timezone.utc).isoformat(),plan_sha256=PLAN_SHA,
            audit_source_sha256=sha(__file__),elapsed_seconds=time.perf_counter()-start,
            runtime=dict(python=platform.python_version(),numpy=np.__version__),counts=dict(counts),
            primary=primary,logical_runs=96,actual_runs=42,aliases=54,
            max_probability_error=max_probability_error,max_cross_entropy_error=max_ce_error,
            evidence_scope=dict(initial_sender_full_domain_replayed=True,endpoint_decoder_full_domain_replayed=True,
                all_saved_monitor_probabilities_labels_ce_statistics_checked=True,
                all_training_world_uniforms_indices_states_labels_rebuilt=True,
                checkpoint_world_rng_states_checked=True,all_checkpoint_parameter_and_adam_schemas_checked=True,
                training_input_features_regenerated=False,training_gradients_replayed=False,optimizer_updates_replayed=False,
                intermediate_decoder_forward_replayed=False,old_action_heads_forwarded=False,
                inference='Fixed-budget decoder extractability change, not direct information quantity or new language formation.'),
            evaluations=evaluations,artifacts_sha256=artifacts)
        (out/'verification.json').write_bytes(json_bytes(verification))
        return verification
    except BaseException as error:
        (out/'failure.json').write_bytes(json_bytes(dict(status='failed',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,help='Completed decoder run directory')
    parser.add_argument('--audit-out',required=True,help='New audit output directory')
    args=parser.parse_args();v=audit(args.out,args.audit_out)
    print(json.dumps({k:v[k] for k in ('status','elapsed_seconds','counts','max_probability_error','max_cross_entropy_error')},ensure_ascii=False))
