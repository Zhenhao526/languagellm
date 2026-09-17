"""Mock routing checks and one B=2 random-network native/sham compatibility check."""
import unittest
from unittest.mock import patch
import numpy as np
from . import intervention as c

COST=dict(fake_forward_calls=0,fake_module_samples=0,real_forward_calls=0,real_module_samples=0,
    real_optimizer_updates=0,formal_checkpoint_loads=0,formal_result_reads=0)


def observations(n):
    x=np.zeros((n,3,54))
    for actor in range(3):
        x[:,actor,7*actor:7*actor+7]=[1,1,0,1,0,1,0]
        x[:,actor,50+actor]=1
        for site in range(4):
            x[:,actor,21+5*site]=1
            x[:,actor,22+5*site+site//2]=1
            x[:,actor,24+5*site+site%2]=1
        x[:,actor,[41,45,49]]=1
    return x


def fake_forward(net,x):
    COST['fake_forward_calls']+=1;COST['fake_module_samples']+=len(x)
    value=np.rint(x @ (1+np.arange(x.shape[1])%7)).astype(int)
    if net%3==2:
        logits=np.full((len(x),17),-4.)
        logits[np.arange(len(x)),(value+net)%17]=4.
    else:
        logits=np.full((len(x),4,8),-4.)
        for token in range(4):logits[np.arange(len(x)),token,(value+net+token)%8]=4.
        logits=logits.reshape(len(x),32)
    return logits,None


class InterventionTests(unittest.TestCase):
    def test_frozen_forward_sources(self):
        self.assertEqual(len(c.frozen_sources()),2)

    def test_every_outward_edge_and_no_input_mutation(self):
        tokens=np.arange(3*3*4,dtype=np.int8).reshape(3,3,4)%8
        persons=np.arange(3);donor=(tokens[np.arange(3),persons]+3)%8
        before=tokens.copy();donor_before=donor.copy()
        result=c.replace_outward(tokens,persons,donor)
        self.assertTrue(np.all(result['visibility']))
        self.assertTrue(np.all(result['routes'][:,:,-3:]==1))
        for row in range(3):
            for viewer in range(3):
                for source in range(3):
                    expected=donor[row] if source==persons[row] and viewer!=source else tokens[row,source]
                    np.testing.assert_array_equal(result['delivered_tokens'][row,viewer,source],expected)
                    np.testing.assert_array_equal(result['routes'][row,viewer,32*source:32*(source+1)],np.eye(8)[expected].reshape(32))
        np.testing.assert_array_equal(tokens,before);np.testing.assert_array_equal(donor,donor_before)

    @patch.object(c.core.base,'actor_forward',side_effect=fake_forward)
    def test_mock_sham_replays_core_exactly_and_recomputes_in_synchronous_order(self,forward):
        x=observations(3);persons=np.arange(3);native=c.core.rollout(tuple(range(9)),x,True)
        donor=native['messages'][np.arange(3),:,persons]
        forward.reset_mock();got=c.intervene(tuple(range(9)),x,native['messages'],persons,donor)
        self.assertEqual([call.args[0] for call in forward.call_args_list],[1,4,7,2,5,8])
        self.assertEqual(got['neural_forward_samples'],18)
        np.testing.assert_array_equal(got['generated_messages'],native['messages'])
        for key in ('action_inputs','action_logits'):np.testing.assert_array_equal(got[key],native[key])
        p,lp=c.core.base.policy_distribution(native['action_logits'])
        np.testing.assert_array_equal(got['action_probabilities'],p);np.testing.assert_array_equal(got['action_log_probabilities'],lp)
        np.testing.assert_array_equal(got['action_indices'],p.argmax(-1))

    @patch.object(c.core.base,'actor_forward',side_effect=fake_forward)
    def test_changed_packets_recompute_descendants_but_keep_receiver_views_and_self(self,forward):
        x=observations(3);persons=np.arange(3);native=c.core.rollout(tuple(range(9)),x,True)
        packet=(native['messages'][np.arange(3),:,persons]+1)%8
        x_before=x.copy();m_before=native['messages'].copy();packet_before=packet.copy()
        forward.reset_mock();got=c.intervene(tuple(range(9)),x,native['messages'],persons,packet)
        self.assertEqual(forward.call_count,6)
        np.testing.assert_array_equal(got['second_inputs'][:,:,:54],x)
        np.testing.assert_array_equal(got['action_inputs'][:,:,:54],x)
        for row,person in enumerate(persons):
            np.testing.assert_array_equal(got['generated_messages'][row,:,person],native['messages'][row,:,person])
            for window in range(2):
                for viewer in range(3):
                    for source in range(3):
                        wanted=packet[row,window] if source==person and viewer!=person else got['generated_messages'][row,window,source]
                        np.testing.assert_array_equal(got['delivered_tokens'][row,window,viewer,source],wanted)
        # For this fixture, outward W1 changes at least one downstream W2 packet.
        self.assertTrue(np.any(got['generated_messages'][:,1]!=native['messages'][:,1]))
        self.assertTrue(np.all(got['delivery_visibility']))
        np.testing.assert_array_equal(x,x_before);np.testing.assert_array_equal(native['messages'],m_before)
        np.testing.assert_array_equal(packet,packet_before)

    def test_softmax_rounding_ties_use_probability_argmax(self):
        def near_tie(net,x):
            COST['fake_forward_calls']+=1;COST['fake_module_samples']+=len(x)
            logits=np.zeros((len(x),17 if net%3==2 else 32))
            if net%3==2:logits[:,1]=1e-18
            else:logits[:,1::8]=1e-18
            return logits,None
        with patch.object(c.core.base,'actor_forward',side_effect=near_tie):
            got=c.intervene(tuple(range(9)),observations(1),np.zeros((1,2,3,4),int),np.array([0]),np.ones((1,2,4),int))
        np.testing.assert_array_equal(got['generated_messages'][:,1],0)
        np.testing.assert_array_equal(got['action_indices'],0)

    @patch.object(c.core.base,'actor_forward',side_effect=fake_forward)
    def test_single_window_modes_deliver_recomputed_second_window(self,forward):
        x=observations(3);persons=np.arange(3);native=c.core.rollout(tuple(range(9)),x,True)
        donor=(native['messages'][np.arange(3),:,persons]+1)%8
        for windows in ((True,False),(False,True)):
            forward.reset_mock()
            got=c.intervene(tuple(range(9)),x,native['messages'],persons,donor,overwrite_windows=windows)
            self.assertEqual(forward.call_count,6);self.assertEqual(got['neural_forward_samples'],18)
            self.assertEqual(got['overwrite_windows'],windows)
            if windows==(True,False):
                for viewer in range(3):
                    np.testing.assert_array_equal(got['delivered_tokens'][:,1,viewer],got['generated_messages'][:,1])
                self.assertTrue(np.any(got['generated_messages'][:,1]!=native['messages'][:,1]))
            else:
                for viewer in range(3):
                    np.testing.assert_array_equal(got['delivered_tokens'][:,0,viewer],native['messages'][:,0])
                np.testing.assert_array_equal(got['generated_messages'],native['messages'])

    def test_invalid_inputs_fail_before_forward(self):
        x=observations(1);m=np.zeros((1,2,3,4),int);person=np.array([0]);donor=np.zeros((1,2,4),int)
        invalid=[]
        for broken in (np.full_like(x,np.nan),x[:,:,:53]):invalid.append((tuple(range(9)),broken,m,person,donor))
        leaked=x.copy();leaked[:,0,7]=1;invalid.append((tuple(range(9)),leaked,m,person,donor))
        fi=x.copy();fi[:,:,53]=1;invalid.append((tuple(range(9)),fi,m,person,donor))
        for bad in (np.full_like(donor,8),donor.astype(float),donor[:,:,:3]):invalid.append((tuple(range(9)),x,m,person,bad))
        for bad in (np.array([3]),np.array([-1]),np.array([0.0]),np.array([True])):invalid.append((tuple(range(9)),x,m,bad,donor))
        invalid.append((tuple(range(8)),x,m,person,donor));invalid.append((tuple(range(9)),x,m.astype(float),person,donor))
        with patch.object(c.core.base,'actor_forward') as forward:
            for args in invalid:
                with self.assertRaises(ValueError):c.intervene(*args)
            self.assertEqual(forward.call_count,0)
            for bad in ((False,False),(True,),(1,False),[True,False]):
                with self.assertRaises(ValueError):c.intervene(tuple(range(9)),x,m,person,donor,overwrite_windows=bad)
            self.assertEqual(forward.call_count,0)

    def test_one_real_random_network_native_sham_bitwise_compatibility(self):
        # One untrained synthetic nine-head policy; B=2 => (9+6)*2=30 module samples.
        networks=c.core.make_networks(948317);before=c.core.parameter_hash(networks)
        x=observations(2);persons=np.array([0,2]);original_forward=c.core.base.actor_forward
        def counted_forward(net,inputs):
            COST['real_forward_calls']+=1;COST['real_module_samples']+=len(inputs)
            return original_forward(net,inputs)
        with patch.object(c.core.base,'actor_forward',side_effect=counted_forward) as forward:
            native=c.core.rollout(networks,x,True)
            donor=native['messages'][np.arange(2),:,persons]
            got=c.intervene(networks,x,native['messages'],persons,donor)
            self.assertEqual(forward.call_count,15)
        np.testing.assert_array_equal(got['generated_messages'],native['messages'])
        for key in ('action_inputs','action_logits'):np.testing.assert_array_equal(got[key],native[key])
        for window,key in enumerate(('first_routes','second_routes')):
            np.testing.assert_array_equal(got[key],c.core.routed_window(native['messages'][:,window],True))
        p,lp=c.core.base.policy_distribution(native['action_logits'])
        np.testing.assert_array_equal(got['action_probabilities'],p);np.testing.assert_array_equal(got['action_log_probabilities'],lp)
        np.testing.assert_array_equal(got['action_indices'],p.argmax(-1))
        self.assertEqual(c.core.parameter_hash(networks),before)


if __name__=='__main__':unittest.main()
