"""Independent incoming/outgoing W1 edge routing, followed by natural W2.

No model-loading, training, world construction, settlement, or metric code.
The caller authenticates each native receiver trace against the fixed policy
and PL observations. This module reuses W1 and always recomputes W2/actions.
"""
from pathlib import Path
import hashlib
import numpy as np
from research_program.triadic_message_study import runner as core

CORE_SHA256='5299c99bb92f3e18f8fae969347084c631a66f78e635d2abc4609c476cb9ae60'
BASE_SHA256='1f3cae631bc15e8f2004e27187370fcc8f9448ac6f99495e867c6ac40e2e0d6a'


def require(ok,message):
    if not ok:raise ValueError(message)


def frozen_sources():
    output={}
    for path,expected in ((Path(core.__file__),CORE_SHA256),(Path(core.base.__file__),BASE_SHA256)):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest==expected,f'Frozen forward source changed: {path}')
        output[str(path.resolve())]=digest
    return output


def _tokens(value,shape,name):
    raw=np.asarray(value)
    require(raw.shape==shape and raw.dtype.kind in 'iu' and np.all((raw>=0)&(raw<8)),
        f'{name} must contain integer symbols 0..7 with shape {shape}')
    return raw.astype(np.int8,copy=False)


def _focal(value,n):
    raw=np.asarray(value)
    require(raw.shape==(n,) and raw.dtype.kind in 'iu' and np.all((raw>=0)&(raw<3)),
        'focal must contain one integer actor 0..2 per row')
    return raw.astype(np.int64,copy=False)


def _flags(value,n,name):
    raw=np.asarray(value)
    require(raw.shape==(n,) and raw.dtype.kind in 'biu' and np.all((raw==0)|(raw==1)),
        f'{name} must be a B-vector of booleans or binary integers')
    return raw.astype(bool,copy=False)


def _observations(value):
    x=np.asarray(value,dtype=np.float64)
    require(x.ndim==3 and x.shape[1:]==(3,54) and len(x)>0 and np.isfinite(x).all(),
        'receiver_observations must be finite nonempty B by 3 by 54')
    require(np.all((x==0)|(x==1)) and np.all(x[:,:,53]==0),'Frozen PL features must be binary with FI flag zero')
    for actor in range(3):
        require(np.all(x[:,actor,7*actor]==1),'Own private need must be visible')
        for other in range(3):
            if actor!=other:require(np.all(x[:,actor,7*other:7*other+7]==0),'Other private needs must be masked')
    require(np.all(x[:,:,[21,26,31,36]]==1),'All four PL material sites must be visible')
    return x


def route_first(native_tokens,focal,incoming_donor_W1,outgoing_donor_W1,flags_I,flags_O):
    """B×viewer×source×token delivery with two disjoint directed edge sets.

I: other sources -> focal viewer. O: focal source -> other viewers.
No self edge belongs to either set; edges between the two others stay native.
"""
    raw=np.asarray(native_tokens)
    require(raw.ndim==3 and raw.shape[1:]==(3,4) and len(raw)>0,'Native W1 must be B by 3 by 4')
    n=len(raw);tokens=_tokens(raw,(n,3,4),'native_tokens');who=_focal(focal,n)
    incoming=_tokens(incoming_donor_W1,(n,3,4),'incoming_donor_W1')
    outgoing=_tokens(outgoing_donor_W1,(n,4),'outgoing_donor_W1')
    incoming_on=_flags(flags_I,n,'flags_I');outgoing_on=_flags(flags_O,n,'flags_O')
    delivered=np.broadcast_to(tokens[:,None,:,:],(n,3,3,4)).copy()
    # These assignments use source and viewer explicitly; no whole-sender
    # substitution helper is reused from an earlier intervention study.
    for source in range(3):
        rows=np.flatnonzero(incoming_on & (who!=source))
        delivered[rows,who[rows],source]=incoming[rows,source]
    for viewer in range(3):
        rows=np.flatnonzero(outgoing_on & (who!=viewer))
        delivered[rows,viewer,who[rows]]=outgoing[rows]
    visibility=np.ones((n,3,3),bool)
    routes=np.concatenate((np.eye(8,dtype=np.float64)[delivered].reshape(n,3,96),visibility.astype(np.float64)),axis=-1)
    return dict(delivered_tokens=delivered,visibility=visibility,routes=routes)


def intervene(networks,receiver_observations,native_messages,focal,incoming_donor_W1,outgoing_donor_W1,flags_I,flags_O):
    """All four I/O conditions, including full sham, cost6B module samples.

Generated W1 is the authenticated native W1. W2 is newly generated from the
patched W1 inputs and naturally broadcast, including each actor's own newly
generated W2. No saved native W2 or donor W2 is substituted into that window.
"""
    x=_observations(receiver_observations);n=len(x)
    require(len(networks)==9,'Exactly nine fixed policy heads required')
    native=_tokens(native_messages,(n,2,3,4),'native_messages');who=_focal(focal,n)
    incoming=_tokens(incoming_donor_W1,(n,3,4),'incoming_donor_W1')
    outgoing=_tokens(outgoing_donor_W1,(n,4),'outgoing_donor_W1')
    incoming_on=_flags(flags_I,n,'flags_I');outgoing_on=_flags(flags_O,n,'flags_O')
    first=route_first(native[:,0],who,incoming,outgoing,incoming_on,outgoing_on)
    second_inputs=np.concatenate((x,first['routes']),axis=-1)
    sender2_logits=np.stack([core.base.actor_forward(networks[3*actor+1],second_inputs[:,actor])[0]
        .reshape(n,4,8) for actor in range(3)],axis=1)
    sender2_probabilities,sender2_log_probabilities=core.base.policy_distribution(sender2_logits)
    second_tokens=core.categorical_tokens(sender2_probabilities)
    second_routes=core.routed_window(second_tokens,True)
    delivered_second=np.broadcast_to(second_tokens[:,None,:,:],(n,3,3,4)).copy()
    action_inputs=np.concatenate((x,first['routes'],second_routes),axis=-1)
    action_logits=np.stack([core.base.actor_forward(networks[3*actor+2],action_inputs[:,actor])[0]
        for actor in range(3)],axis=1)
    probabilities,log_probabilities=core.base.policy_distribution(action_logits)
    return dict(generated_messages=np.stack((native[:,0],second_tokens),axis=1),
        delivered_tokens=np.stack((first['delivered_tokens'],delivered_second),axis=1),
        delivery_visibility=np.ones((n,2,3,3),bool),
        first_routes=first['routes'],second_routes=second_routes,second_inputs=second_inputs,action_inputs=action_inputs,
        sender2_logits=sender2_logits,sender2_probabilities=sender2_probabilities,sender2_log_probabilities=sender2_log_probabilities,
        action_logits=action_logits,action_probabilities=probabilities,action_log_probabilities=log_probabilities,
        action_indices=np.argmax(probabilities,axis=-1).astype(np.int16),
        focal=who.copy(),flags_I=incoming_on.copy(),flags_O=outgoing_on.copy(),
        incoming_donor_W1=incoming.copy(),outgoing_donor_W1=outgoing.copy(),
        neural_forward_calls=6,neural_forward_samples=6*n,reused_native_first_window=True,reused_natural_actions=False)
