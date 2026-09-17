"""Synthetic audit tests; no production utility/runner imports or run results.

The one final-forward arithmetic test uses zero scratch parameters of hidden
width 1 on two artificial worlds: 9 module calls / 18 module samples. No
optimizer, saved formal checkpoint or formal policy is read.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from research_program.triadic_partial_payoff_study import audit_execution as a


def native_table():
    r=np.zeros((2,24));r[:,0]=1;r[:,3:6]=.5
    return r


def training_record(alpha=.1):
    return dict(seed=53101,payoff='a10' if alpha==.1 else 'a50',condition='PL_live',update=500,
        partial_utility=alpha,partial_success_utility=alpha,
        entropy_coefficient=.001*(1-499/1000),mean_actor_entropy=2.,mean_log_expected_utility=-3.,
        min_log_expected_utility=-4.,max_log_expected_utility=-2.,
        mean_F=-3.+.001*(1-499/1000)*2.,receiver_loss=3.-.001*(1-499/1000)*2.,
        mean_expected_utility=.1+alpha*.2,mean_native_expected_reward=.2,
        mean_full_success_probability=.1,mean_partial_success_probability=.2,
        mean_full_success_posterior_mass=.7,mean_partial_success_posterior_mass=.3,
        zero_float_expected_utility_states=0,sender_advantage_mean=0.,sender_advantage_abs_mean=.2,
        sender_advantage_squared_mean=.05,sender_advantage_max_abs=.3,sender_mean_complete_log_score=-25.,
        sampled_messages_sha256='f'*64,gradient_norm=10.,gradient_clip_scale=.5,stream='verified')


class ConditionalTests(unittest.TestCase):
    def test_frozen_helper_sources(self):
        self.assertEqual(len(a.references()),3)

    def test_uniform_exact_utility_native_and_posterior(self):
        p=np.full((2,3,17),1/17);r=native_table()
        for alpha in (.5,.1):
            v=a.conditional_statistics(p,r,alpha)
            np.testing.assert_allclose(v['conditional_exact_expected_utility'],(1+3*alpha)/17**3,rtol=2e-15)
            np.testing.assert_allclose(v['conditional_exact_expected_reward'],2.5/17**3,rtol=2e-15)
            np.testing.assert_allclose(v['conditional_full_posterior_mass'],1/(1+3*alpha),rtol=2e-15)
            np.testing.assert_array_equal(v['posterior'][r==0],0)
            np.testing.assert_allclose(v['posterior'].sum(-1),1,rtol=0,atol=2e-15)

    def test_random_reference_direct_sum_and_reweighting(self):
        random=np.random.default_rng(707).random((2,3,17));p=random/random.sum(-1,keepdims=True)
        r=native_table();results=[]
        for alpha in (.5,.1):
            value=a.conditional_statistics(p,r,alpha);expected=np.zeros(2);numerator=np.zeros(2)
            for index,plan in enumerate(a.env.JOINT):
                mass=p[:,0,plan[0]]*p[:,1,plan[1]]*p[:,2,plan[2]]
                expected+=mass*((r[:,index]==1)+alpha*(r[:,index]==.5))
                numerator+=mass*(r[:,index]==1)
            np.testing.assert_allclose(value['conditional_exact_expected_utility'],expected,rtol=2e-15)
            np.testing.assert_allclose(value['conditional_full_posterior_mass'],numerator/expected,rtol=3e-15)
            results.append(value)
        self.assertTrue((results[1]['conditional_full_posterior_mass']>results[0]['conditional_full_posterior_mass']).all())
        np.testing.assert_array_equal(results[0]['conditional_exact_expected_reward'],results[1]['conditional_exact_expected_reward'])

    def test_joint_underflow_is_identifiable_but_single_probability_loss_fails(self):
        lp=np.full((2,3,17),-500.);lp[:,:,0]=0.
        p=np.exp(lp);r=native_table()
        result=a.conditional_statistics(p,r,.1)
        np.testing.assert_array_equal(result['conditional_exact_expected_utility'],0)
        self.assertTrue(np.isfinite(result['log_expected_utility']).all())
        self.assertTrue((result['conditional_full_posterior_mass']>0).all())
        lost=np.full_like(lp,-1000.);lost[:,:,0]=0.;lost_p=np.exp(lost)
        with self.assertRaisesRegex(AssertionError,'underflow'):
            a.conditional_statistics(lost_p,r,.1)
        # Final replay can retain finite receiver log probabilities despite p=0.
        recovered=a.conditional_statistics(lost_p,r,.1,lost)
        np.testing.assert_array_equal(recovered['conditional_exact_expected_utility'],0)
        np.testing.assert_allclose(recovered['posterior'].sum(-1),1,atol=1e-12)

    def test_invalid_shapes_support_and_probabilities(self):
        p=np.full((2,3,17),1/17);r=native_table()
        for bad in (np.zeros((2,24)),np.ones((2,24)),np.full((2,24),np.nan)):
            with self.assertRaises(AssertionError):a.conditional_statistics(p,bad,.1)
        for bad in (np.full_like(p,-1),np.full_like(p,np.nan),np.full_like(p,1),p[:,:,:16]):
            with self.assertRaises(AssertionError):a.conditional_statistics(bad,r,.1)
        for alpha in (0,-.1,1,True):
            with self.assertRaises(AssertionError):a.conditional_statistics(p,r,alpha)

    def test_partner_ids_include_wait_and_actor_identity(self):
        actions=np.asarray([[0,0,0],[1,1,1],[2,2,2],[16,16,16]],dtype=np.int16)
        np.testing.assert_array_equal(a.role_indices(actions),[[-1,-1,-1],[1,0,0],[2,2,1],[2,2,1]])

    def test_final_forward_zero_scratch_modules(self):
        networks=[]
        for actor in range(3):
            for inputs,outputs in ((54,32),(153,32),(252,17)):
                networks.append(dict(W1=np.zeros((inputs,1)),b1=np.zeros(1),W2=np.zeros((1,1)),b2=np.zeros(1),
                                     W3=np.zeros((1,outputs)),b3=np.zeros(outputs)))
        states=np.asarray([[0,1,4,0,1,2,3,1,2,3],[2,3,4,3,2,1,0,3,1,2]],dtype=np.int16)
        messages,p,lp,ties=a.forward_final(networks,states,'PL',True)
        np.testing.assert_array_equal(messages,np.zeros((2,2,3,4),dtype=np.int8))
        np.testing.assert_array_equal(p,np.full((2,3,17),1/17))
        np.testing.assert_array_equal(lp,np.full((2,3,17),-np.log(17)))
        self.assertEqual(ties,48)


class RecordTests(unittest.TestCase):
    def test_eight_domain_fields_and_five_inherited_metadata_fields(self):
        path=a.ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json'
        self.assertEqual(a.sha(path),a.ORIGINAL_PREPARED_SHA)
        inherited=a.read(path)['partitions'];independent,_,_=a.env.independent_specs()
        self.assertNotEqual(inherited,independent)  # Metadata is absent in the independent constructor.
        proof=a.validate_partitions(inherited,independent,deepcopy(inherited))
        self.assertEqual(len(proof['independently_reconstructed_fields']),8)
        self.assertEqual(len(proof['inherited_fields_checked_against_fixed_source']),5)
        bad=deepcopy(inherited);bad['train']['monitor_scope']='altered metadata'
        with self.assertRaisesRegex(AssertionError,'frozen partition copy'):
            a.validate_partitions(bad,independent,inherited)
        bad=deepcopy(independent);bad['train']['world_count']+=1
        with self.assertRaisesRegex(AssertionError,'Independent domain field'):
            a.validate_partitions(inherited,bad,inherited)

    def test_training_row_independent_algebra_and_failures(self):
        row=training_record()
        a.training_row(row,53101,'a10',.1,'PL_live',500,dict(stream='verified'))
        for key,value in (('mean_expected_utility',.2),('mean_native_expected_reward',.3),
                          ('mean_full_success_posterior_mass',.9),('mean_F',-1),('gradient_clip_scale',1),
                          ('sender_advantage_squared_mean',.001),('zero_float_expected_utility_states',513),
                          ('sampled_messages_sha256','bad'),('stream','changed'),('partial_success_utility',.5)):
            changed=dict(row);changed[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):
                a.training_row(changed,53101,'a10',.1,'PL_live',500,dict(stream='verified'))

    def test_budget_and_nonzero_world_role_did(self):
        specs={part:dict(world_count=count,monitor_indices=range(mon)) for part,count,mon in
            zip(a.PARTS,(419904,160704,139968,53568),(7776,2976,7776,2976))}
        budget=a.budget(specs)
        self.assertEqual(budget['actual_final_files'],128)
        self.assertEqual(budget['actual_monitor_files'],768)
        self.assertEqual(budget['silent_closed_aliases'],448)
        self.assertEqual(budget['training_forward_module_samples'],663552000)
        metrics={}
        for seed in a.SEEDS:
            for payoff,alpha,condition in a.CELLS:
                value={('a50','PL_silent'):.2,('a50','PL_live'):.3,
                       ('a10','PL_silent'):.1,('a10','PL_live'):.5}.get((payoff,condition),.8)
                metrics[f'{seed}/{payoff}/{condition}/new_needs_and_layouts/natural']={
                    'role_success_rate':value,'full_success_rate':value/2,'reward_mean':value*.7,'utility_mean':value*.4}
        result=a.primary(metrics)
        self.assertAlmostEqual(result['mean_difference'],.3)
        for row in result['paired_seeds']:
            self.assertAlmostEqual(row['contrasts']['full_success_rate'],.15)
            self.assertAlmostEqual(row['contrasts']['reward_mean'],.21)

    def fixture(self,directory,alpha=.1):
        needs=a.env.support()[:4]
        states=np.asarray([tuple(n)+(0,1,2,3,1,2,3) for n in needs],dtype=np.int16)
        reward_table=a.env.rewards(states);actions=np.zeros((4,3),dtype=np.int16)
        actions[0]=a.env.JOINT[(reward_table[0]==1).argmax()]
        actions[1]=a.env.JOINT[(reward_table[1]==.5).argmax()]
        actions[3]=[1,1,1]
        p=np.full((4,3,17),.002)
        p[np.arange(4)[:,None],np.arange(3),actions]=.968
        conditional=a.conditional_statistics(p,reward_table,alpha)
        reward,executed,satisfied=a.env.native(states,actions)
        values=dict(states=states,state_indices=np.arange(4,dtype=np.int64),messages=np.zeros((4,2,3,4),dtype=np.int8),
            action_indices=actions,action_probabilities=p,greedy_reward=reward,executed=executed,satisfied=satisfied,
            greedy_utility=np.where(reward==.5,alpha,reward))
        values.update({key:value for key,value in conditional.items() if key.startswith('conditional_')})
        path=Path(directory)/'synthetic.npz';np.savez_compressed(path,**values)
        truth=a.env.JOINT[(reward_table==1).argmax(-1)]
        roles=a.role_indices(actions);correct=a.role_indices(truth)
        acts,counts=np.unique(actions,axis=0,return_counts=True);rp,rc=np.unique(roles,axis=0,return_counts=True)
        entry=dict(path=str(path),data_sha256=a.sha(path),worlds=4,information='PL',live=True,partial_utility=alpha,
            reward_mean=float(reward.mean()),utility_mean=float(values['greedy_utility'].mean()),
            full_success_rate=float((reward==1).mean()),role_success_rate=float(np.all(roles==correct,axis=1).mean()),
            physical_execution_rate=float(executed.any(-1).mean()),
            expected_reward_given_greedy_messages=float(conditional['conditional_exact_expected_reward'].mean()),
            expected_utility_given_greedy_messages=float(conditional['conditional_exact_expected_utility'].mean()),
            full_probability_given_greedy_messages=float(conditional['conditional_exact_full_success_probability'].mean()),
            full_posterior_mass_given_greedy_messages=float(conditional['conditional_full_posterior_mass'].mean()),
            raw_joint_action_counts=[dict(action_indices=x.tolist(),worlds=int(c)) for x,c in zip(acts,counts)],
            raw_joint_role_counts=[dict(partner_indices=x.tolist(),worlds=int(c)) for x,c in zip(rp,rc)],
            state_indices_sha256=a.array_sha(values['state_indices']),reused_natural=False)
        return path,states,values,entry

    def test_saved_schema_native_utility_roles_and_no_monitor_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            path,states,values,entry=self.fixture(directory)
            with patch.object(a,'forward_final',side_effect=AssertionError('Monitor must not forward')):
                checked,receipt=a.evaluate_saved(entry,path,states,np.arange(4),'PL',True,.1)
            self.assertEqual(receipt['independent_forward_worlds'],0)
            self.assertEqual(receipt['metrics']['raw_joint_role_counts'],entry['raw_joint_role_counts'])
            for key in values:np.testing.assert_array_equal(checked[key],values[key])
            for key in ('raw_joint_role_counts','raw_joint_action_counts'):
                changed=deepcopy(entry);changed[key][0]['worlds']+=1
                with self.assertRaises(AssertionError):a.evaluate_saved(changed,path,states,np.arange(4),'PL',True,.1)

    def test_saved_settlement_and_posterior_tampering_rejected(self):
        for field in ('greedy_utility','conditional_full_posterior_mass','state_indices'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                path,states,values,entry=self.fixture(directory)
                values[field][0]+=1
                np.savez_compressed(path,**values);entry['data_sha256']=a.sha(path)
                with self.assertRaises(AssertionError):a.evaluate_saved(entry,path,states,np.arange(4),'PL',True,.1)

    def test_final_replay_counts_and_action_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path,states,values,entry=self.fixture(directory)
            returned=(values['messages'],values['action_probabilities'],np.log(values['action_probabilities']),0)
            with patch.object(a,'forward_final',return_value=returned) as forward:
                _,receipt=a.evaluate_saved(entry,path,states,np.arange(4),'PL',True,.1,networks=['synthetic'])
                forward.assert_called_once()
            self.assertEqual(receipt['independent_forward_worlds'],4)
            self.assertEqual(receipt['independent_network_samples'],36)
            changed=list(returned);changed[0]=changed[0].copy();changed[0][0,0,0,0]=1
            with patch.object(a,'forward_final',return_value=tuple(changed)),self.assertRaises(AssertionError):
                a.evaluate_saved(entry,path,states,np.arange(4),'PL',True,.1,networks=['synthetic'])


if __name__=='__main__':unittest.main()
