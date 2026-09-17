"""Three independent supervised role probes on fixed receiver-input features.

Each actor predicts its own role: 0 waits; 1/2 select the other two actors in
ascending identity order. The loss is cross-entropy averaged over B*3 labels.
This module has no environment reward, message learning, training loop, early
stopping or checkpoint selection. The runner fixes 6000 updates of batch 256.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'

from copy import deepcopy
import json
import math
import numbers
from pathlib import Path
import numpy as np
from research_program.triadic_learning_baseline import runner as base

DIMENSIONS=(252,64,64,3)
PARAMETER_SHAPES={key:shape for layer,(left,right) in enumerate(zip(DIMENSIONS,DIMENSIONS[1:]),1)
                  for key,shape in ((f'W{layer}',(left,right)),(f'b{layer}',(right,)))}
PARAMETER_KEYS=tuple(PARAMETER_SHAPES)
UPDATES=6000
BATCH_SIZE=256
SCHEMA='triadic_role_decoder_checkpoint_v1'
BASE_SHA='1f3cae631bc15e8f2004e27187370fcc8f9448ac6f99495e867c6ac40e2e0d6a'
OPTIMIZER_CONFIG=dict(learning_rate=.001,adam_beta1=.9,adam_beta2=.999,adam_epsilon=1e-8,global_gradient_clip=5.)
require,finite,sha,array_sha,json_hash=base.require,base.finite,base.sha,base.array_sha,base.json_hash


def _networks(networks):
    require(isinstance(networks,(tuple,list)) and len(networks)==3,'Exactly three role networks required')
    for network in networks:
        require(isinstance(network,dict) and set(network)==set(PARAMETER_KEYS),'Role network parameter schema')
        for key,shape in PARAMETER_SHAPES.items():
            value=network[key]
            require(isinstance(value,np.ndarray) and value.shape==shape and value.dtype==np.float64,
                    'Role parameter shape/dtype '+key)
            finite(value,'Role parameter '+key)
    for right in range(3):
        for left in range(right):
            require(all(not np.shares_memory(a,b) for a in networks[left].values() for b in networks[right].values()),
                    'Role networks cannot share parameter memory')


def _observations(x):
    raw=np.asarray(x)
    require(raw.ndim==3 and raw.shape[1:]==(3,252) and len(raw)>0 and raw.dtype.kind in 'fiub',
            'Observations must be nonempty real numeric [B,3,252]')
    value=np.asarray(raw,dtype=np.float64)
    finite(value,'Role probe observations')
    return value


def _labels(labels,batch_size):
    labels=np.asarray(labels)
    require(labels.shape==(batch_size,3) and labels.dtype.kind in 'iu'
            and ((labels>=0)&(labels<3)).all(),'Role labels must be integer [B,3] in 0..2')
    return labels


def make_networks(seed):
    require(isinstance(seed,numbers.Integral) and not isinstance(seed,(bool,np.bool_)) and seed>=0,
            'Probe seed must be a nonnegative integer')
    networks=[]
    for actor in range(3):
        rng=np.random.default_rng(np.random.SeedSequence([int(seed),actor,730]))
        network={}
        for layer,(left,right) in enumerate(zip(DIMENSIONS,DIMENSIONS[1:]),1):
            network[f'W{layer}']=rng.normal(0,math.sqrt(2/(left+right)),(left,right))
            network[f'b{layer}']=np.zeros(right,dtype=np.float64)
        networks.append(network)
    _networks(networks)
    return networks


def prediction_terms(networks,x):
    """One call per head; return probabilities and stable log probabilities."""
    _networks(networks);x=_observations(x)
    logits=np.stack([base.actor_forward(networks[actor],x[:,actor])[0] for actor in range(3)],axis=1)
    probability,log_probability=base.policy_distribution(logits)
    require(probability.shape==(len(x),3,3),'Role probability shape')
    return probability,log_probability


def probabilities(networks,x):
    return prediction_terms(networks,x)[0]


def training_gradients(networks,x,labels):
    """Return CE gradients only; labels never enter any network forward input."""
    _networks(networks);x=_observations(x);labels=_labels(labels,len(x))
    traces=[base.actor_forward(networks[actor],x[:,actor]) for actor in range(3)]
    logits=np.stack([trace[0] for trace in traces],axis=1)
    probability,log_probability=base.policy_distribution(logits)
    selected=np.take_along_axis(log_probability,labels[...,None],axis=-1)[...,0]
    derivative=(probability-np.eye(3,dtype=np.float64)[labels])/(len(x)*3)
    gradients=[base.actor_backward(networks[actor],traces[actor][1],derivative[:,actor]) for actor in range(3)]
    prediction=probability.argmax(-1)
    row=dict(loss=-float(selected.mean()),indiv_accuracy=float((prediction==labels).mean()),
             joint_accuracy=float(np.all(prediction==labels,axis=1).mean()))
    require(math.isfinite(row['loss']) and row['loss']>=0,'Role CE loss finite/nonnegative')
    for gradient in gradients:
        for key,value in gradient.items():finite(value,'Role gradient '+key)
    return gradients,row


def parameter_hash(networks):
    _networks(networks)
    return json_hash({f'agent{actor}_{key}':array_sha(networks[actor][key])
                     for actor in range(3) for key in PARAMETER_KEYS})


def _optimizer(optimizer,update):
    require(isinstance(update,numbers.Integral) and not isinstance(update,(bool,np.bool_)) and 0<=update<=UPDATES,
            'Role checkpoint update must be integer 0..6000')
    require(isinstance(optimizer,(tuple,list)) and len(optimizer)==3,'Three role optimizer states required')
    for state in optimizer:
        require(isinstance(state,dict) and set(state)=={'m','v'},'Role optimizer moment schema')
        for moment in ('m','v'):
            require(isinstance(state[moment],dict) and set(state[moment])==set(PARAMETER_KEYS),'Role optimizer parameter schema')
            for key,shape in PARAMETER_SHAPES.items():
                value=state[moment][key]
                require(isinstance(value,np.ndarray) and value.shape==shape and value.dtype==np.float64,
                        'Role optimizer shape/dtype '+key)
                finite(value,'Role optimizer '+moment+'/'+key)
                require(moment!='v' or (value>=0).all(),'Role Adam second moment must be nonnegative')
                require(update!=0 or not value.any(),'Initial role optimizer moments must be zero')


def _rng_from_state(state):
    require(isinstance(state,dict) and set(state)=={'bit_generator','state','has_uint32','uinteger'}
            and state['bit_generator']=='PCG64','Role world RNG must use PCG64 schema')
    require(isinstance(state['state'],dict) and set(state['state'])=={'state','inc'},'Role PCG64 internal schema')
    for key in ('state','inc'):
        value=state['state'][key]
        require(type(value) is int and 0<=value<2**128,'Role PCG64 internal integer range')
    require(state['state']['inc']%2==1,'Role PCG64 increment must be odd')
    require(type(state['has_uint32']) is int and state['has_uint32'] in (0,1),'Role PCG64 uint32 cache flag')
    require(type(state['uinteger']) is int and 0<=state['uinteger']<2**32,'Role PCG64 uint32 cache value')
    engine=np.random.PCG64(0);engine.state=deepcopy(state)
    require(engine.state==state,'Role PCG64 state roundtrip')
    return np.random.Generator(engine)


def save_checkpoint(path,networks,optimizer,update,world_rng):
    """Save all three independent probes, Adam moments, update and world RNG."""
    _networks(networks);_optimizer(optimizer,update)
    require(isinstance(world_rng,np.random.Generator) and type(world_rng.bit_generator) is np.random.PCG64,
            'Role world RNG must be a PCG64 Generator')
    state=deepcopy(world_rng.bit_generator.state);_rng_from_state(state)
    require(all(base.CONFIG[k]==value for k,value in OPTIMIZER_CONFIG.items()),'Frozen base Adam configuration changed')
    path=Path(path);require(not path.exists(),'Role checkpoint exists')
    payload={f'agent{actor}_{key}':networks[actor][key] for actor in range(3) for key in PARAMETER_KEYS}
    payload.update({f'adam_agent{actor}_{moment}_{key}':optimizer[actor][moment][key]
                    for actor in range(3) for moment in ('m','v') for key in PARAMETER_KEYS})
    payload.update(schema=np.array(SCHEMA),dimensions=np.asarray(DIMENSIONS,dtype=np.int64),
        update=np.array(update,dtype=np.int64),world_rng_json=np.array(json.dumps(state,sort_keys=True)),
        optimizer_config_json=np.array(json.dumps(OPTIMIZER_CONFIG,sort_keys=True)))
    with path.open('xb') as output:np.savez_compressed(output,**payload)
    return sha(path)


def load_checkpoint(path):
    """Return (networks, optimizer, update, PCG64 Generator) after full validation."""
    expected={f'agent{actor}_{key}' for actor in range(3) for key in PARAMETER_KEYS}
    expected|={f'adam_agent{actor}_{moment}_{key}' for actor in range(3) for moment in ('m','v') for key in PARAMETER_KEYS}
    expected|={'schema','dimensions','update','world_rng_json','optimizer_config_json'}
    with np.load(path,allow_pickle=False) as saved:
        require(set(saved.files)==expected,'Role checkpoint exact key schema')
        for key in ('schema','world_rng_json','optimizer_config_json'):
            require(saved[key].shape==() and saved[key].dtype.kind=='U','Role checkpoint metadata string '+key)
        require(str(saved['schema'].item())==SCHEMA,'Role checkpoint schema version')
        require(saved['dimensions'].dtype==np.int64 and np.array_equal(saved['dimensions'],DIMENSIONS),'Role checkpoint dimensions')
        require(saved['update'].shape==() and saved['update'].dtype==np.int64,'Role checkpoint update dtype')
        update=int(saved['update'])
        config=json.loads(str(saved['optimizer_config_json'].item()))
        require(config==OPTIMIZER_CONFIG and all(base.CONFIG[k]==v for k,v in OPTIMIZER_CONFIG.items()),'Role checkpoint optimizer configuration')
        networks=[{key:saved[f'agent{actor}_{key}'].copy() for key in PARAMETER_KEYS} for actor in range(3)]
        optimizer=[{moment:{key:saved[f'adam_agent{actor}_{moment}_{key}'].copy() for key in PARAMETER_KEYS}
                    for moment in ('m','v')} for actor in range(3)]
        rng_state=json.loads(str(saved['world_rng_json'].item()))
    _networks(networks);_optimizer(optimizer,update)
    return networks,optimizer,update,_rng_from_state(rng_state)
