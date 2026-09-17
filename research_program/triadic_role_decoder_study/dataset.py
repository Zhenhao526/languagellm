"""Frozen payoff-study transcripts and official observations for role decoders.

Import and source_manifest perform no network construction/load/forward. Targets
are researcher-side labels only. Own/FI decoder inputs contain no message routes.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

import numpy as np

from research_program.triadic_action_dependency_study import dataset as original_dataset
from research_program.triadic_action_dependency_study import environment as env
from research_program.triadic_message_study import runner as core

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'triadic_partial_payoff_study/results/payoff_001'
PLAN_SHA='99b9663dfae02868692d53707ad5b7db40b7f2ee31c951edf2fd68f05859f33f'
RESULT_SHA='0d0fd3bbea91eae728384b0bffc4e040f250193004465451305eb56d451615e2'
AUDIT_SHA='b270398c609132623d83b01fb6e7fe65077688af013a0b36aa83e501436cbf97'
SEEDS=(53101,53102,53103,53104)
PAYOFFS=('a50','a10')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
VIEWS=('Own','FI','Transcript')
_SAVED_MESSAGE_HASHES={}


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_manifest():
    """Bind metadata,64 saved evaluations,8 checkpoint files, and old sources.

NPZ bytes are hashed, never opened as arrays here. The four initial protocols
choose a50 only after equality of each seed's two checkpoint0 byte hashes.
    """
    inputs={}
    def bind(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        if expected is not None:require(digest==expected,'Changed source: '+str(path))
        inputs[str(path)]=digest
        return dict(path=str(path),sha256=digest)
    plan_path=SOURCE/'plan.json';result_path=SOURCE/'execution/results.json'
    audit_path=SOURCE/'audit_execution_001/verification.json'
    bind(plan_path,PLAN_SHA);bind(result_path,RESULT_SHA);bind(audit_path,AUDIT_SHA)
    plan,result,audit=read(plan_path),read(result_path),read(audit_path)
    freeze_path=SOURCE/'freeze.json';prepared_path=SOURCE/'prepared.json';status_path=SOURCE/'execution/status.json'
    bind(freeze_path);bind(prepared_path,plan['prepared_sha256']);bind(status_path)
    require(read(freeze_path)['plan_sha256']==PLAN_SHA,'Source freeze mismatch')
    require(result['status']=='completed' and read(status_path)['status']=='completed','Source execution incomplete')
    require(not (SOURCE/'execution/failure.json').exists(),'Source execution has a failure marker')
    require(audit['status']=='passed' and audit['plan_sha256']==result['plan_sha256']==PLAN_SHA,'Source audit mismatch')
    prepared=read(prepared_path);parts=prepared['partitions']
    require(set(parts)==set(PARTS) and sum(p['world_count'] for p in parts.values())==774144,'Source world support changed')
    require(sum(len(p['monitor_indices']) for p in parts.values())==21504,'Source monitoring support changed')
    old_sources={}
    repository=HERE.parents[1]
    for path,digest in plan['source_sha256'].items():
        bind(path,digest)
        require(audit['artifacts_sha256'].get(path)==digest,'Old source missing from independent audit')
        snapshot=SOURCE/'source_snapshot'/Path(path).relative_to(repository)
        bind(snapshot,digest)
        if Path(path).suffix=='.py':old_sources[path]=digest
    for path,digest in plan['inputs_sha256'].items():bind(path,digest)
    for module in (original_dataset,env,core,core.base):
        require(str(Path(module.__file__).resolve()) in old_sources,'Imported dependency is not frozen')
    runs={(r['seed'],r['payoff'],r['condition']):r for r in result['runs']}
    require(len(runs)==24,'Source run identities are not unique')
    policies=[]
    for seed in SEEDS:
        for payoff in PAYOFFS:
            run=runs[seed,payoff,'PL_live']
            require(run['updates']==6000 and run['monitor'][0]['update']==0,'Checkpoint endpoint changed')
            directory=SOURCE/'execution'/f'seed_{seed}_{payoff}_PL_live'
            run_path=directory/'result.json';bind(run_path)
            require(read(run_path)==run,'Per-policy result differs from source aggregate')
            checkpoint=bind(directory/'checkpoint_0000.npz',run['monitor'][0]['checkpoint_sha256'])
            require(audit['artifacts_sha256'].get(checkpoint['path'])==checkpoint['sha256'],'Checkpoint absent from audit')
            policy=dict(seed=seed,payoff=payoff,condition='PL_live',checkpoint0=checkpoint,
                        initial_parameter_sha256=run['initial_parameter_sha256'],final={},monitor0={})
            for part in PARTS:
                for key,record in (('final',run['final'][part]['natural']),
                                   ('monitor0',run['monitor'][0]['monitor'][part]['natural'])):
                    expected=parts[part]['world_count'] if key=='final' else len(parts[part]['monitor_indices'])
                    require(record['information']=='PL' and record['live'] and not record['reused_natural'] and record['worlds']==expected,
                            'Source transcript identity mismatch')
                    artifact=bind(record['path'],record['data_sha256'])
                    require(audit['artifacts_sha256'].get(artifact['path'])==artifact['sha256'] and
                            audit['evaluations'][artifact['path']]['sha256']==artifact['sha256'],
                            'Source transcript absent from independent audit')
                    artifact.update(worlds=expected,state_indices_sha256=record['state_indices_sha256'])
                    policy[key][part]=artifact
                    _SAVED_MESSAGE_HASHES[artifact['path']]=artifact['sha256']
            policies.append(policy)
    by={(p['seed'],p['payoff']):p for p in policies}
    initial_sources=[];equalities=[]
    for seed in SEEDS:
        a,b=by[seed,'a50'],by[seed,'a10']
        require(a['checkpoint0']['sha256']==b['checkpoint0']['sha256'] and
                a['initial_parameter_sha256']==b['initial_parameter_sha256'],'Initial payoff arms differ')
        initial_sources.append(dict(seed=seed,payoff='a50',checkpoint0=deepcopy(a['checkpoint0']),
                                    monitor0=deepcopy(a['monitor0']),paired_monitor0=deepcopy(b['monitor0'])))
        equalities.append(dict(seed=seed,a50=deepcopy(a['checkpoint0']),a10=deepcopy(b['checkpoint0']),
                               byte_identical=True,parameter_hash=a['initial_parameter_sha256']))
    return dict(schema='triadic_role_decoder_sources_v1',source_run=str(SOURCE.resolve()),
                source_sha256=inputs,source_hashes=old_sources,partitions=deepcopy(parts),policies=policies,
                initial_sources=initial_sources,checkpoint0_equalities=equalities,
                counts=dict(policies=8,independent_source_seeds=4,initial_protocols=4,final_npz=32,monitor0_npz=32,
                            checkpoint0_files_bound=8,total_worlds_per_protocol=774144,parameter_arrays_loaded=0,
                            neural_forward_calls=0,training_updates=0),
                initial_equivalence_scope='Whole checkpoint0 bytes match across payoffs. Monitor NPZ bytes need not match because utility fields differ.')


def _integers(value,shape,low,high,label):
    a=np.asarray(value)
    require(a.shape==shape and a.dtype.kind in 'iu' and np.all((a>=low)&(a<=high)),label+' shape/dtype/range')
    return a


def make_arrays(spec):
    """Vectorize official54 columns; no full-domain Python State list or reward.

Labels:0 wait;1/2 choose the other agents in ascending identity order. They are
separate from observations and are never passed into initial_messages.
    """
    n=spec['world_count'];nn,nl,no=len(spec['needs']),len(spec['layouts']),len(spec['private_sites'])
    require(type(n) is int and n>0 and n==nn*nl*no,'World product mismatch')
    needs=_integers(spec['needs'],(nn,3),0,23,'Needs')
    layouts=_integers(spec['layouts'],(nl,4),0,3,'Layouts')
    owners=_integers(spec['private_sites'],(no,3),1,3,'Owners')
    require(np.all(np.sort(layouts,axis=1)==np.arange(4)) and np.all(np.sort(owners,axis=1)==np.arange(1,4)),
            'Layouts/owners must be permutations')
    labels_by_need=np.zeros((nn,3),dtype=np.int8)
    for index,need in enumerate(needs):
        plans=env.full_success_plans(tuple(map(int,need)))
        require(len(plans)==1,'Role label needs one unique full-success plan')
        first,second,_,_=plans[0]
        for actor,partner in ((first,second),(second,first)):
            labels_by_need[index,actor]=1+tuple(a for a in range(3) if a!=actor).index(partner)
    packed=original_dataset.pack_states(spec)
    x_FI=np.zeros((n,3,54),dtype=np.float64)
    need_table=np.zeros((24,7),dtype=np.float64);need_table[:,0]=1
    for need in range(24):
        view=env.need_view(need)
        for field,options,offset in (('kinds',('wood','fiber'),1),('lengths',('short','long'),3),('destinations',('L','R'),5)):
            for value in view[field]:need_table[need,offset+options.index(value)]=1
    x_FI[:,:,:21]=need_table[packed[:,:3]].reshape(n,1,21)
    material_table=np.zeros((4,5),dtype=np.float64);material_table[:,0]=1
    for material,(kind,length) in enumerate(env.MATERIALS):
        material_table[material,1+('wood','fiber').index(kind)]=1
        material_table[material,3+('short','long').index(length)]=1
    x_FI[:,:,21:41]=material_table[packed[:,3:7]].reshape(n,1,20)
    owner_table=np.zeros((n,3,3),dtype=np.float64)
    owner_table[np.arange(n)[:,None],packed[:,7:10]-1,np.arange(3)[None,:]]=1
    x_FI[:,:,41:50]=owner_table.reshape(n,1,9)
    x_FI[:,np.arange(3),50+np.arange(3)]=1;x_FI[:,:,53]=1
    x_PL=x_FI.copy();x_PL[:,:,53]=0
    for actor in range(3):
        for other in range(3):
            if actor!=other:x_PL[:,actor,7*other:7*(other+1)]=0
    labels=np.repeat(labels_by_need,nl*no,axis=0)
    return dict(packed_states=packed,x_PL=x_PL,x_FI=x_FI,labels=labels)


def _message_shape(messages,n):
    m=np.asarray(messages)
    require(m.shape==(n,2,3,4) and m.dtype==np.int8,'Messages must be int8 [domain,2,3,4]')
    return m


def build_inputs(arrays,indices,view,messages=None):
    """Build [B,3,252]; indices may repeat for sampled training batches.

Transcript messages must cover the full arrays domain, in its exact row order;
this function applies indices to both observations and messages. Own/FI ignore
the supplied messages entirely, including self-generated messages and flags.
    """
    require(view in VIEWS,'Unknown decoder view')
    x=np.asarray(arrays['x_FI' if view=='FI' else 'x_PL'])
    require(x.ndim==3 and x.shape[1:]==(3,54) and len(x)>0 and x.dtype==np.float64,'Official feature shape/dtype')
    ids=np.asarray(indices)
    require(ids.ndim==1 and len(ids)>0 and ids.dtype.kind in 'iu' and np.all((ids>=0)&(ids<len(x))),'Invalid input indices')
    observed=x[ids]
    require(np.isfinite(observed).all(),'Nonfinite selected observations')
    result=np.zeros((len(ids),3,252),dtype=np.float64);result[:,:,:54]=observed
    if view=='Transcript':
        selected=_message_shape(messages,len(x))[ids]
        require(np.all((selected>=0)&(selected<8)),'Invalid selected message symbols')
        result[:,:,54:153]=core.routed_window(selected[:,0],True)
        result[:,:,153:252]=core.routed_window(selected[:,1],True)
    return result


def initial_messages(networks,x_PL):
    """Six sender-module calls only, synchronous and greedy, for one batch."""
    x=np.asarray(x_PL)
    require(x.ndim==3 and x.shape[1:]==(3,54) and len(x)>0 and x.dtype==np.float64 and np.isfinite(x).all(),
            'Initial sender observations must be finite float64 [B,3,54]')
    require(len(networks)==9,'Nine saved modules required; action modules will not be called')
    inputs=x;messages=[]
    for window in range(2):
        logits=[]
        for actor in range(3):
            z,_=core.base.actor_forward(networks[3*actor+window],inputs[:,actor])
            z=np.asarray(z)
            require(z.shape==(len(x),32) and np.isfinite(z).all(),'Invalid sender logits')
            logits.append(z.reshape(len(x),4,8))
        probabilities,_=core.base.policy_distribution(np.stack(logits,axis=1))
        tokens=core.categorical_tokens(probabilities,None)
        messages.append(tokens)
        if window==0:inputs=np.concatenate((x,core.routed_window(tokens,True)),axis=-1)
    return dict(messages=np.stack(messages,axis=1),neural_forward_calls=6,neural_forward_samples=6*len(x),
                first_window_forward_samples=3*len(x),second_window_forward_samples=3*len(x),
                action_forward_calls=0,action_forward_samples=0)


def load_final_messages(path,expected_states,expected_sha256=None):
    """Load only messages and row identity arrays; never parameters or actions.

The optional digest must come from the caller's frozen source manifest. Without
it the path is resolved against source_manifest (cached path digests after its
first explicit call). Full final files must use indices0..N-1 without reordering.
    """
    path=Path(path).resolve();states=np.asarray(expected_states)
    require(states.ndim==2 and states.shape[1:]==(10,) and len(states)>0 and states.dtype.kind in 'iu',
            'Expected states must be nonempty integer [N,10]')
    if expected_sha256 is None:
        if str(path) not in _SAVED_MESSAGE_HASHES:source_manifest()
        require(str(path) in _SAVED_MESSAGE_HASHES,'Unbound saved transcript path')
        expected_sha256=_SAVED_MESSAGE_HASHES[str(path)]
    require(isinstance(expected_sha256,str) and len(expected_sha256)==64 and sha(path)==expected_sha256,'Saved transcript hash mismatch')
    with np.load(path,allow_pickle=False) as saved:
        require({'states','state_indices','messages'}<=set(saved.files),'Missing transcript identity arrays')
        indices=saved['state_indices']
        require(indices.dtype.kind in 'iu' and np.array_equal(indices,np.arange(len(states))),'Final transcript indices mismatch')
        actual=saved['states']
        require(actual.dtype.kind in 'iu' and np.array_equal(actual,states),'Final transcript states mismatch')
        messages=_message_shape(saved['messages'],len(states)).copy()
    require(np.all((messages>=0)&(messages<8)),'Saved transcript symbols out of range')
    return messages
