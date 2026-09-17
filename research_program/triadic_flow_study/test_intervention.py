"""Small fake-forward checks only; no real policy, checkpoint, or neural call."""
from itertools import product
import unittest
from unittest.mock import patch
import numpy as np
from . import intervention as c

COST=dict(fake_forward_calls=0,fake_module_samples=0,real_forward_calls=0,optimizer_updates=0,
    formal_checkpoint_loads=0,formal_result_reads=0)


def observations(n):
    x=np.zeros((n,3,54))
    for actor in range(3):
        x[:,actor,7*actor:7*actor+7]=[1,1,0,1,0,1,0]
        x[:,actor,50+actor]=1;x[:,actor,[41,45,49]]=1
        for site in range(4):
            x[:,actor,21+5*site]=1;x[:,actor,22+5*site+site//2]=1;x[:,actor,24+5*site+site%2]=1
    return x


def fake_forward(net,x):
    COST['fake_forward_calls']+=1;COST['fake_module_samples']+=len(x)
    value=np.rint(x @ (1+np.arange(x.shape[1])%7)).astype(int)
    if net%3==2:
        logits=np.full((len(x),17),-4.);logits[np.arange(len(x)),(value+net)%17]=4.
    else:
        logits=np.full((len(x),4,8),-4.)
        for token in range(4):logits[np.arange(len(x)),token,(value+net+token)%8]=4.
        logits=logits.reshape(len(x),32)
    return logits,None


class InterventionTests(unittest.TestCase):
    def test_pinned_forward_sources(self):
        self.assertEqual(len(c.frozen_sources()),2)

    def test_all_focal_roles_and_four_flag_cells_edge_by_edge(self):
        grid=np.array(list(product(range(3),range(2),range(2))),int);n=len(grid)
        who,flags_I,flags_O=grid.T
        native=np.arange(n*3*4).reshape(n,3,4)%8;incoming=(native+1)%8;outgoing=(native[np.arange(n),who]+3)%8
        originals=[v.copy() for v in (native,incoming,outgoing,who,flags_I,flags_O)]
        got=c.route_first(native,who,incoming,outgoing,flags_I,flags_O)
        for row,(focal,on_I,on_O) in enumerate(grid):
            for viewer in range(3):
                for source in range(3):
                    expected=native[row,source]
                    if on_I and viewer==focal and source!=focal:expected=incoming[row,source]
                    if on_O and source==focal and viewer!=focal:expected=outgoing[row]
                    np.testing.assert_array_equal(got['delivered_tokens'][row,viewer,source],expected)
                    np.testing.assert_array_equal(got['routes'][row,viewer,source*32:(source+1)*32],np.eye(8)[expected].reshape(32))
        self.assertTrue(np.all(got['visibility']));self.assertTrue(np.all(got['routes'][:,:,-3:]==1))
        for before,after in zip(originals,(native,incoming,outgoing,who,flags_I,flags_O)):np.testing.assert_array_equal(before,after)

    @patch.object(c.core.base,'actor_forward',side_effect=fake_forward)
    def test_full_sham_exact_native_replay_still_uses_six_calls(self,forward):
        x=observations(3);who=np.arange(3);native=c.core.rollout(tuple(range(9)),x,True)
        forward.reset_mock()
        got=c.intervene(tuple(range(9)),x,native['messages'],who,np.full((3,3,4),7),np.ones((3,4),int),np.zeros(3,bool),np.zeros(3,bool))
        self.assertEqual([call.args[0] for call in forward.call_args_list],[1,4,7,2,5,8])
        self.assertEqual(got['neural_forward_samples'],18);self.assertFalse(got['reused_natural_actions'])
        np.testing.assert_array_equal(got['generated_messages'],native['messages'])
        for key in ('action_inputs','action_logits'):np.testing.assert_array_equal(got[key],native[key])
        p,lp=c.core.base.policy_distribution(native['action_logits'])
        np.testing.assert_array_equal(got['action_probabilities'],p);np.testing.assert_array_equal(got['action_log_probabilities'],lp)
        np.testing.assert_array_equal(got['action_indices'],p.argmax(-1))

    @patch.object(c.core.base,'actor_forward',side_effect=fake_forward)
    def test_mixed_rows_fixed_views_and_recomputed_W2_natural_broadcast(self,forward):
        grid=np.array(list(product(range(3),range(2),range(2))),int);who,flags_I,flags_O=grid.T;n=len(grid)
        x=observations(n);native=c.core.rollout(tuple(range(9)),x,True)
        incoming=(native['messages'][:,0]+1)%8;outgoing=(native['messages'][np.arange(n),0,who]+3)%8
        snapshots=[a.copy() for a in (x,native['messages'],incoming,outgoing)]
        forward.reset_mock();got=c.intervene(tuple(range(9)),x,native['messages'],who,incoming,outgoing,flags_I,flags_O)
        self.assertEqual(forward.call_count,6);self.assertEqual(got['neural_forward_samples'],6*n)
        self.assertEqual([call.args[0] for call in forward.call_args_list],[1,4,7,2,5,8])
        for actor,call in enumerate(forward.call_args_list[:3]):np.testing.assert_array_equal(call.args[1],got['second_inputs'][:,actor])
        for actor,call in enumerate(forward.call_args_list[3:]):np.testing.assert_array_equal(call.args[1],got['action_inputs'][:,actor])
        np.testing.assert_array_equal(got['second_inputs'][:,:,:54],x);np.testing.assert_array_equal(got['action_inputs'][:,:,:54],x)
        for viewer in range(3):np.testing.assert_array_equal(got['delivered_tokens'][:,1,viewer],got['generated_messages'][:,1])
        for row,(focal,on_I,on_O) in enumerate(grid):
            for actor in range(3):
                np.testing.assert_array_equal(got['delivered_tokens'][row,0,actor,actor],native['messages'][row,0,actor])
            if not on_I:np.testing.assert_array_equal(got['generated_messages'][row,1,focal],native['messages'][row,1,focal])
            if not on_O:
                for actor in range(3):
                    if actor!=focal:np.testing.assert_array_equal(got['generated_messages'][row,1,actor],native['messages'][row,1,actor])
        self.assertTrue(np.all(got['delivery_visibility']));self.assertEqual(got['action_probabilities'].shape,(n,3,17))
        for before,after in zip(snapshots,(x,native['messages'],incoming,outgoing)):np.testing.assert_array_equal(before,after)

    def test_softmax_first_argmax_tie_behavior(self):
        def near_tie(net,x):
            COST['fake_forward_calls']+=1;COST['fake_module_samples']+=len(x)
            logits=np.zeros((len(x),17 if net%3==2 else 32))
            if net%3==2:logits[:,1]=1e-18
            else:logits[:,1::8]=1e-18
            return logits,None
        with patch.object(c.core.base,'actor_forward',side_effect=near_tie):
            got=c.intervene(tuple(range(9)),observations(1),np.zeros((1,2,3,4),int),np.array([1]),
                np.ones((1,3,4),int),np.ones((1,4),int),np.ones(1,bool),np.ones(1,bool))
        np.testing.assert_array_equal(got['generated_messages'][:,1],0);np.testing.assert_array_equal(got['action_indices'],0)

    def test_invalid_inputs_rejected_before_any_forward(self):
        base=[tuple(range(9)),observations(1),np.zeros((1,2,3,4),int),np.array([0]),np.zeros((1,3,4),int),np.zeros((1,4),int),np.array([0]),np.array([1])]
        invalid=[]
        for position,value in [(0,tuple(range(8))),(1,np.full((1,3,54),np.nan)),(2,np.zeros((1,2,3,4))),
            (3,np.array([3])),(3,np.array([0.])),(4,np.full((1,3,4),8)),(5,np.zeros((1,3))),
            (6,np.array([2])),(6,np.array([0.])),(7,np.array([-1])),(7,np.array([[1]]))]:
            args=list(base);args[position]=value;invalid.append(args)
        leaked=base[1].copy();leaked[:,0,7]=1;args=list(base);args[1]=leaked;invalid.append(args)
        with patch.object(c.core.base,'actor_forward') as forward:
            for args in invalid:
                with self.assertRaises(ValueError):c.intervene(*args)
            self.assertEqual(forward.call_count,0)


if __name__=='__main__':unittest.main()
