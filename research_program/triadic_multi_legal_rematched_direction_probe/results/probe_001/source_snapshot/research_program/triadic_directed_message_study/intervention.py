"""Replace one sender's two outward packets without changing receiver views.

Only the three W2 sender heads and three action heads are recomputed. The caller
must authenticate receiver_messages as the native greedy trace for these exact
receiver observations and parameters; this function does not replay W1 or read
checkpoints. Every route remains visible. Settlement and scoring are external.
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
    """Source-only verification for preparation; no parameter/result reads."""
    output={}
    for path,expected in ((Path(core.__file__),CORE_SHA256),(Path(core.base.__file__),BASE_SHA256)):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest==expected,f'Frozen forward source changed: {path}')
        output[str(path.resolve())]=digest
    return output


def _tokens(value,shape,name):
    raw=np.asarray(value)
    require(raw.shape==shape and raw.dtype.kind in 'iu' and np.all((raw>=0)&(raw<8)),
        f'{name} must be integer symbols 0..7 with shape {shape}')
    return raw.astype(np.int8,copy=False)


def _persons(value,n):
    raw=np.asarray(value)
    require(raw.shape==(n,) and raw.dtype.kind in 'iu' and np.all((raw>=0)&(raw<3)),
        'changed_person must contain B integer actor indices 0..2')
    return raw.astype(np.int64,copy=False)


def _observations(value):
    x=np.asarray(value,dtype=np.float64)
    require(x.ndim==3 and x.shape[1:]==(3,54) and len(x)>0 and np.isfinite(x).all(),
        'receiver_observations must be nonempty finite B by 3 by 54')
    require(np.all((x==0)|(x==1)),'Frozen PL observations contain only binary features')
    require(np.all(x[:,:,53]==0),'FI flag must be absent from PL receiver observations')
    for actor in range(3):
        require(np.all(x[:,actor,7*actor]==1),'Own need must be visible')
        for other in range(3):
            if other!=actor:require(np.all(x[:,actor,7*other:7*other+7]==0),'Other private needs must be masked')
    require(np.all(x[:,:,[21,26,31,36]]==1),'PL receiver must retain the complete public layout')
    return x


def replace_outward(generated_tokens,changed_person,donor_window):
    """One synchronous window: B, viewer, source, token; self stays generated."""
    raw=np.asarray(generated_tokens)
    require(raw.ndim==3 and raw.shape[1:]==(3,4) and len(raw)>0,'Generated window must be B by 3 by 4')
    n=len(raw);tokens=_tokens(raw,(n,3,4),'generated_tokens')
    persons=_persons(changed_person,n);donor=_tokens(donor_window,(n,4),'donor_window')
    delivered=np.broadcast_to(tokens[:,None,:,:],(n,3,3,4)).copy()
    for viewer in range(3):
        rows=np.flatnonzero(persons!=viewer)
        delivered[rows,viewer,persons[rows]]=donor[rows]
    visible=np.ones((n,3,3),dtype=bool)
    routes=np.concatenate((np.eye(8,dtype=np.float64)[delivered].reshape(n,3,96),
        visible.astype(np.float64)),axis=-1)
    return dict(delivered_tokens=delivered,visibility=visible,routes=routes)


def intervene(networks,receiver_observations,receiver_messages,changed_person,donor_packets,*,overwrite_windows=(True,True)):
    """Greedy two-window outward replacement; six module calls for B rows.

Inputs: observations[B,3,54], native receiver_messages[B,2,3,4],
changed_person[B], donor_packets[B,2,4]. Inputs are never mutated. The donor
provides symbols only; receiver observations, actor identities and visibility
never come from the donor. All W2 heads finish before any W2 routing or action.
overwrite_windows is a two-boolean tuple with at least one True. An unpatched
W2 delivers the recomputed W2 symbols, not the previously saved native W2.

generated_messages[B,2,source,4] and
delivered_tokens[B,2,viewer,source,4] are different records. No correctness
target, world identity, reward, optimizer, or policy-loading operation enters.
"""
    x=_observations(receiver_observations);n=len(x)
    require(len(networks)==9,'Expected nine independent receiver-policy heads')
    require(isinstance(overwrite_windows,tuple) and len(overwrite_windows)==2 and
        all(isinstance(value,(bool,np.bool_)) for value in overwrite_windows) and any(overwrite_windows),
        'overwrite_windows must be a two-boolean tuple with at least one True')
    native=_tokens(receiver_messages,(n,2,3,4),'receiver_messages')
    persons=_persons(changed_person,n);donor=_tokens(donor_packets,(n,2,4),'donor_packets')
    rows=np.arange(n)
    first=replace_outward(native[:,0],persons,donor[:,0] if overwrite_windows[0] else native[rows,0,persons])
    second_inputs=np.concatenate((x,first['routes']),axis=-1)
    sender2_logits=np.stack([core.base.actor_forward(networks[3*actor+1],second_inputs[:,actor])[0]
        .reshape(n,4,8) for actor in range(3)],axis=1)
    sender2_probabilities,sender2_log_probabilities=core.base.policy_distribution(sender2_logits)
    second_tokens=core.categorical_tokens(sender2_probabilities)
    second=replace_outward(second_tokens,persons,donor[:,1] if overwrite_windows[1] else second_tokens[rows,persons])
    action_inputs=np.concatenate((x,first['routes'],second['routes']),axis=-1)
    action_logits=np.stack([core.base.actor_forward(networks[3*actor+2],action_inputs[:,actor])[0]
        for actor in range(3)],axis=1)
    probabilities,log_probabilities=core.base.policy_distribution(action_logits)
    return dict(generated_messages=np.stack((native[:,0],second_tokens),axis=1),
        delivered_tokens=np.stack((first['delivered_tokens'],second['delivered_tokens']),axis=1),
        delivery_visibility=np.stack((first['visibility'],second['visibility']),axis=1),
        first_routes=first['routes'],second_routes=second['routes'],
        second_inputs=second_inputs,action_inputs=action_inputs,
        sender2_logits=sender2_logits,sender2_probabilities=sender2_probabilities,
        sender2_log_probabilities=sender2_log_probabilities,
        action_logits=action_logits,action_probabilities=probabilities,action_log_probabilities=log_probabilities,
        action_indices=np.argmax(probabilities,axis=-1).astype(np.int16),
        changed_person=persons.copy(),donor_packets=donor.copy(),
        overwrite_windows=tuple(bool(value) for value in overwrite_windows),
        neural_forward_calls=6,neural_forward_samples=6*n,
        reused_native_first_window=True,reused_natural_actions=False)
