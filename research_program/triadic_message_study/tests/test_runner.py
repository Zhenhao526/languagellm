"""Offline interface, causal-routing, reproducibility and failure-boundary tests.

No study-seed actor or optimizer is run. The one random scratch-network check
uses artificial arrays only; it never evaluates task performance.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from research_program.triadic_message_study import runner as r


class MessageProtocolTests(unittest.TestCase):
    def test_categorical_cdf_and_first_index_ties(self):
        p=np.full((1,3,4,8),1/8)
        np.testing.assert_array_equal(r.categorical_tokens(p),0)
        u=np.array([0,.125,.5,np.nextafter(1.,0.)])[None,None,:]
        u=np.broadcast_to(u,(1,3,4))
        np.testing.assert_array_equal(r.categorical_tokens(p,u),np.broadcast_to([0,1,4,7],(1,3,4)))
        with self.assertRaisesRegex(ValueError,'Invalid categorical uniforms'):
            r.categorical_tokens(p,np.ones((1,3,4)))

    def test_silent_self_visible_category_zero_not_missing(self):
        messages=np.array([[[0,1,2,3],[4,5,6,7],[7,6,5,4]]],dtype=np.int8)
        live=r.routed_window(messages,True)
        silent=r.routed_window(messages,False)
        for viewer in range(3):
            for sender in range(3):
                for token in range(4):
                    block=silent[0,viewer,32*sender+8*token:32*sender+8*(token+1)]
                    if sender==viewer:
                        self.assertEqual(block.sum(),1)
                        self.assertEqual(block[messages[0,sender,token]],1)
                    else:np.testing.assert_array_equal(block,0)
            np.testing.assert_array_equal(silent[0,viewer,96:],np.eye(3)[viewer])
            np.testing.assert_array_equal(live[0,viewer,96:],1)
        self.assertEqual(silent[0,0,0],1)
        with self.assertRaisesRegex(ValueError,'Invalid actual message'):
            r.routed_window(np.full((1,3,4),8),True)

    def test_two_synchronous_barriers_and_no_foreign_observation_input(self):
        x=np.arange(2*3*54,dtype=float).reshape(2,3,54)
        nets=[{'agent':a,'module':m} for a in range(3) for m in range(3)]
        calls=[]
        def fake_forward(net,inputs):
            a,m=net['agent'],net['module'];calls.append((a,m,inputs.copy()))
            if m<2:
                z=np.zeros((len(inputs),4,8));z[:,:,(a+2*m)%8]=10
                return z.reshape(len(inputs),32),(inputs,None,None)
            return np.zeros((len(inputs),17)),(inputs,None,None)
        with patch.object(r.base,'actor_forward',side_effect=fake_forward):
            trace=r.rollout(nets,x,True)
        self.assertEqual([(a,m) for a,m,_ in calls],[(a,m) for m in range(3) for a in range(3)])
        for a in range(3):
            np.testing.assert_array_equal(calls[a][2],x[:,a])
            expected=np.concatenate((x[:,a],r.routed_window(trace['messages'][:,0],True)[:,a]),axis=-1)
            np.testing.assert_array_equal(calls[3+a][2],expected)
            expected=np.concatenate((expected,r.routed_window(trace['messages'][:,1],True)[:,a]),axis=-1)
            np.testing.assert_array_equal(calls[6+a][2],expected)
        self.assertEqual(trace['messages'].shape,(2,2,3,4))

    def test_independent_trajectory_streams_matched_across_conditions(self):
        a,b=r.make_message_rngs(91),r.make_message_rngs(91)
        first=r.draw_uniforms(a,7)
        np.testing.assert_array_equal(first,r.draw_uniforms(b,7))
        second=r.draw_uniforms(a,7)
        np.testing.assert_array_equal(second,r.draw_uniforms(b,7))
        self.assertFalse(np.array_equal(first[0],first[1]))
        self.assertFalse(np.array_equal(first[:,:,0],first[:,:,1]))
        self.assertFalse(np.array_equal(first[:,:,:,0],first[:,:,:,1]))
        self.assertFalse(np.array_equal(first,second))
        self.assertEqual(len(a),12)

    def test_official_private_observation_hidden_fact_invariance(self):
        first=r.base.env.State((0,1,2),(0,1,2,3),(1,2,3))
        second=r.base.env.State((0,4,5),(0,1,3,2),(1,2,3))
        def features(full):
            return r.base.encode_observations([{a:r.base.env.observe(s,a,shared_needs=full,full_information=full)
                    for a in r.base.AGENTS} for s in (first,second)])
        private=features(False);full=features(True)
        np.testing.assert_array_equal(private[0,0],private[1,0])
        self.assertFalse(np.array_equal(full[0,0],full[1,0]))
        self.assertEqual(private[0,0,53],0);self.assertEqual(full[0,0,53],1)

    def test_scratch_network_silent_causal_isolation_and_no_mutation(self):
        nets=r.make_networks(19)
        rng=np.random.default_rng(25);x=rng.normal(size=(2,3,54));changed=x.copy();changed[:,1:]+=3
        u=rng.random((2,2,3,4));before=r.parameter_hash(nets)
        original=r.rollout(nets,x,False,u);other=r.rollout(nets,changed,False,u)
        np.testing.assert_array_equal(original['messages'][:,:,0],other['messages'][:,:,0])
        np.testing.assert_array_equal(original['action_logits'][:,0],other['action_logits'][:,0])
        reward=np.zeros((2,24));reward[:,0]=1;reward[:,1]=.5
        gradients,row=r.training_gradients(nets,x,reward,False,r.draw_uniforms(r.make_message_rngs(25),2),10)
        self.assertEqual(len(gradients),9)
        for net,g in zip(nets,gradients):
            for key in net:
                self.assertEqual(g[key].shape,net[key].shape);self.assertTrue(np.isfinite(g[key]).all())
        self.assertEqual(before,r.parameter_hash(nets))
        self.assertAlmostEqual(row['sender_advantage_mean'],0)
        for i in range(9):
            for j in range(i):
                self.assertFalse(np.shares_memory(nets[i]['b2'],nets[j]['b2']))

    def test_equal_returns_cancel_sender_not_receiver(self):
        p=np.full((2,2,2,3,4,8),1/8);lp=np.log(p);m=np.zeros(p.shape[:-1],dtype=np.int8)
        result=r.paired_sender_derivative(p,lp,m,np.full((2,2),-4.))
        np.testing.assert_array_equal(result['derivative'],0)
        logits=np.zeros((4,3,17));reward=np.zeros((4,24));reward[:,0]=1
        _,receiver,_=r.trajectory_objective(logits,reward,6000)
        self.assertGreater(np.linalg.norm(receiver['derivative']),0)

    def test_saved_rollout_contains_real_messages_and_original_settlement(self):
        state=r.base.env.State((0,0,0),(0,1,2,3),(1,2,3))
        obs=r.base.encode_observations([{a:r.base.env.observe(state,a,shared_needs=True,full_information=True) for a in r.base.AGENTS}])
        arrays={'states':[state],'x_FI':obs,'x_PI':obs,'packed_states':np.array([state.needs+state.layout+state.private_sites],dtype=np.int16),
                'rewards':r.base.reward_terms([state])}
        z=np.full((1,3,17),-9.);z[:,:,0]=9 # deliberate all-wait, never a teacher answer
        messages=np.arange(24,dtype=np.int8).reshape(1,2,3,4)%8
        fake={'action_logits':z,'messages':messages,'sender_probabilities':np.full((1,2,3,4,8),1/8)}
        with tempfile.TemporaryDirectory() as d,patch.object(r,'rollout',return_value=fake):
            path=Path(d)/'out.npz';result=r.evaluate([],arrays,'FI_live',save_path=path)
            self.assertEqual(result['greedy_reward_counts'],{'0.0':1,'0.5':0,'1.0':0})
            self.assertEqual(result['greedy_failure_categories']['all_wait'],1)
            self.assertEqual(result['greedy_sender_token_argmax_ties'],24)
            with np.load(path,allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['messages'],messages)
                np.testing.assert_array_equal(saved['state_indices'],[0])
                np.testing.assert_array_equal(saved['action_indices'],0)
            with self.assertRaisesRegex(ValueError,'Evaluation output exists'):
                r.evaluate([],arrays,'FI_live',save_path=path)


class LifecycleTests(unittest.TestCase):
    def test_prepared_retains_old_domain_and_exact_budget(self):
        before=deepcopy(r.base.CONFIG);old=r.base.make_prepared();prepared=r.make_prepared()
        self.assertEqual(r.base.CONFIG,before);self.assertEqual(prepared['partitions'],old['partitions'])
        self.assertEqual(prepared['actions'],old['actions'])
        self.assertEqual([(x['seed'],x['condition']) for x in prepared['runs']],[(s,c) for s in r.SEEDS for c in r.CONDITIONS])
        self.assertEqual(prepared['training_state_samples_total'],24576000)
        self.assertEqual(prepared['sampled_complete_message_trajectories_total'],49152000)
        self.assertEqual(prepared['categorical_message_samples_total'],1179648000)
        self.assertEqual(prepared['weighted_structural_action_contributions'],1179648000)
        self.assertEqual(prepared['offline_reward_table_entries_actual'],13768704)
        self.assertEqual(prepared['complete_final_world_evaluations'],2294784)
        self.assertEqual(prepared['parameter_count_per_agent'],47313)

    def test_prepare_no_network_no_training_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run'
            with (patch.object(r,'make_networks',side_effect=AssertionError('no actor')),
                  patch.object(r,'train_run',side_effect=AssertionError('no training'))):
                r.prepare(p);r.verify(p)
                with self.assertRaisesRegex(ValueError,'overwrite'):r.prepare(p)
            self.assertFalse((p/'execution').exists())

    def test_tamper_and_started_run_rejected_before_worker_spawn(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run';r.prepare(p)
            with patch.object(r.multiprocessing,'get_context',side_effect=AssertionError('no spawn')):
                (p/'execution').mkdir()
                with self.assertRaisesRegex(ValueError,'Never resume'):r.execute(p)
                (p/'execution').rmdir()
                saved=p/'source_snapshot/research_program/triadic_message_study/runner.py'
                saved.write_text(saved.read_text()+'\n# changed\n')
                with self.assertRaisesRegex(ValueError,'snapshot changed'):r.execute(p)
                self.assertFalse((p/'execution').exists())

    def test_failure_stops_only_explicit_worker_handles(self):
        class Fake:
            def __init__(self,name,code):self.name=name;self.exitcode=code;self.pid=100+len(name);self.events=[]
            def is_alive(self):return self.exitcode is None
            def terminate(self):self.events.append('terminate');self.exitcode=-15
            def join(self,timeout=None):self.events.append('join')
            def kill(self):self.events.append('kill');self.exitcode=-9
        bad,running,unrelated=Fake('bad',1),Fake('own',None),Fake('unrelated',None)
        with self.assertRaisesRegex(RuntimeError,'worker failed'):
            r.wait_for_workers([bad,running])
        self.assertEqual(running.events,['terminate','join'])
        self.assertEqual(unrelated.events,[])

    def test_primary_four_pairs_and_all_signs_preserved(self):
        records=[]
        for i,s in enumerate(r.SEEDS):
            values={'FI_silent':10,'FI_live':12,'PI_silent':5,'PI_live':4+i}
            for c in r.CONDITIONS:
                records.append({'seed':s,'condition':c,'final':{'new_needs_and_layouts':
                    {'worlds':8244,'greedy_full_success_rate':values[c]/8244}}})
        result=r.primary_comparison(records)
        self.assertLess(result['seed_pairs'][0]['PI_live_minus_silent'],0)
        self.assertAlmostEqual(result['equal_weight_means']['PI_live_minus_silent'],.5/8244)
        self.assertAlmostEqual(result['equal_weight_means']['difference_in_differences_PI_minus_FI'],-1.5/8244)
        with self.assertRaisesRegex(ValueError,'sixteen'):r.primary_comparison(records[:-1])


if __name__=='__main__':unittest.main()
