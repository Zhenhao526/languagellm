"""Fast invariants for the tabular signaling control."""
from __future__ import annotations
import numpy as np
from .. import design, policy, environment


def test_pairing_and_shapes():
    assert len(design.CONDITIONS)==32
    a=design.episode_stream(68101,'scarce','persistent',32,update=4)
    b=design.episode_stream(68101,'abundant','persistent',32,update=4)
    for k in ('site_type','goal','context','message_uniforms','action_uniforms','sender'):
        np.testing.assert_array_equal(a[k],b[k])
    assert np.all(a['capacity']==2) and np.all(b['capacity']==design.HORIZON)
    assert a['goal'].shape==(32,2,design.HORIZON)


def test_policy_and_protocol():
    ep=design.episode_stream(68101,'scarce','persistent',64,update=1)
    p=policy.make_policy(68101,'persistent','recurrent')
    tr=environment.rollout(p,ep,'PI','live',sample=True)
    assert np.all(tr['messages'][:,1:,0]==design.NULL_MESSAGE)
    assert np.all(tr['actions'][:,0,1]==0) and np.all(tr['actions'][:,:,0]==0)
    assert tr['action_probs'][1].shape==(64,3)
    oracle=environment.oracle_team_return(ep)
    assert np.all(oracle>=-1e-12) and np.all(oracle<=1+1e-12)
    assert np.all(oracle+1e-12>=tr['team_return'])
    closed=environment.rollout(p,ep,'PI','live',message_mode='closed',sample=False)
    assert np.all(closed['messages']==design.NULL_MESSAGE)


def test_memory_lookup_and_hash():
    s=policy.make_policy(68101,'persistent','stateless')
    r=policy.make_policy(68101,'persistent','recurrent')
    np.testing.assert_array_equal(s['sender_logits'],r['sender_logits'])
    np.testing.assert_array_equal(s['worker_logits'],r['worker_logits'])
    assert policy.parameter_hash(s)!=policy.parameter_hash(r)


def main():
    for fn in (test_pairing_and_shapes,test_policy_and_protocol,test_memory_lookup_and_hash): fn()
    print('tabular_signaling_study tests passed')


if __name__=='__main__': main()
