"""Pure static/synthetic audit tests, no production reciprocal numerics."""
from itertools import combinations,product
import tempfile
from pathlib import Path
import unittest
import numpy as np
from research_program.triadic_reciprocal_execution_study import audit_execution as a

RESOURCE=({0,1},{2,3},{0,2},{1,3},{0},{1},{2},{3});DEST=({0},{1},{0,1})


def fixture_states():
    return np.array([[12,12,15,0,1,2,3,1,2,3],[12,15,12,0,1,2,3,1,2,3],
                     [15,12,12,0,1,2,3,1,2,3]],dtype=np.int16)


def reference(state,actions,rule):
    decoded=[]
    for actor,choice in enumerate(actions):
        if choice==0:decoded.append(None)
        else:
            value=int(choice)-1
            decoded.append(dict(partner=[i for i in range(3) if i!=actor][value%2],site=value//4,destination=(value//2)%2))
    matches=[]
    for pair,(i,j) in enumerate(combinations(range(3),2)):
        first,second=decoded[i],decoded[j]
        if first is None or second is None:continue
        if rule=='strict' and decoded[3-i-j] is not None:continue
        if first['partner']==j and second['partner']==i and first['site']==second['site'] and first['destination']==second['destination']:
            matches.append((pair,i,j,first))
    assert len(matches)<=1
    executed=[False]*3;satisfied=[False]*3;pair=-1
    if matches:
        pair,i,j,plan=matches[0];material=int(state[3+plan['site']])
        for actor in (i,j):
            executed[actor]=True;need=int(state[actor]);satisfied[actor]=material in RESOURCE[need//3] and plan['destination'] in DEST[need%3]
    return pair,executed,satisfied,sum(satisfied)/2


class AuditTests(unittest.TestCase):
    def test_references_and_fixed_budget(self):
        self.assertEqual(len(a.references()),4)
        specs={p:dict(world_count=n,monitor_indices=range(m)) for p,n,m in zip(a.PARTS,(419904,160704,139968,53568),(7776,2976,7776,2976))}
        b=a.budget(specs)
        self.assertEqual(b['training_updates'],48000);self.assertEqual(b['final_forward_module_samples'],55738368)
        self.assertEqual(b['monitor_forward_module_samples'],9289728);self.assertEqual(b['aliased_evaluations'],0)

    def test_all_joint_proposals_independent_dictionary_settlement(self):
        actions=np.array(list(product(range(17),repeat=3)),dtype=np.int16)
        fixtures=list(fixture_states())+[np.array([0,1,6,3,1,2,0,3,1,2],np.int16)]
        for state in fixtures:
            states=np.repeat(state[None,:],len(actions),axis=0)
            for rule in a.RULES:
                result=a.settle(states,actions,rule);expected=[reference(state,row,rule) for row in actions]
                np.testing.assert_array_equal(result['actual_pair_index'],[r[0] for r in expected])
                np.testing.assert_array_equal(result['executed'],[r[1] for r in expected])
                np.testing.assert_array_equal(result['satisfied'],[r[2] for r in expected])
                np.testing.assert_array_equal(result['greedy_reward'],[r[3] for r in expected])
                self.assertEqual(int(result['executed'].any(1).sum()),24 if rule=='strict' else 408)
                self.assertEqual(int((result['greedy_reward']==1).sum()),1 if rule=='strict' else 17)
                unsigned=a.settle(states.astype(np.uint8),actions.astype(np.uint8),rule)
                for key in result:np.testing.assert_array_equal(result[key],unsigned[key])
                failed=~result['executed'].any(1)
                for key in ('executed_site','executed_material','executed_destination'):
                    self.assertEqual(result[key].dtype,np.int8);self.assertTrue(np.all(result[key][failed]==-1))

    def test_conditional_probabilities_match_full_4913_sum(self):
        states=fixture_states();rng=np.random.default_rng(41);p=rng.uniform(.2,1,(3,3,17));p/=p.sum(-1,keepdims=True)
        native=a.env.rewards(states)
        for rule in a.RULES:
            result=a.conditional_statistics(p,native,rule)
            for world,state in enumerate(states):
                reward=full=execution=0.
                for choices in product(range(17),repeat=3):
                    pair,_,_,r=reference(state,choices,rule);mass=np.prod([p[world,i,choice] for i,choice in enumerate(choices)])
                    reward+=mass*r;full+=mass*(r==1);execution+=mass*(pair>=0)
                self.assertAlmostEqual(result['conditional_exact_expected_reward'][world],reward,places=14)
                self.assertAlmostEqual(result['conditional_exact_full_success_probability'][world],full,places=14)
                self.assertAlmostEqual(result['conditional_exact_execution_probability'][world],execution,places=14)
                self.assertAlmostEqual(result['conditional_full_posterior_mass'][world],full/reward,places=14)
        uniform=np.full((3,3,17),1/17)
        strict=a.conditional_statistics(uniform,native,'strict');reciprocal=a.conditional_statistics(uniform,native,'reciprocal')
        np.testing.assert_allclose(strict['conditional_exact_full_success_probability'],1/17**3)
        np.testing.assert_allclose(reciprocal['conditional_exact_full_success_probability'],1/17**2)
        np.testing.assert_allclose(reciprocal['conditional_exact_expected_reward'],17*strict['conditional_exact_expected_reward'])

    def test_underflow_requires_stable_log_probabilities(self):
        s=fixture_states();lp=np.full((3,3,17),-1000.);lp[:,:,0]=0.;p=np.exp(lp)
        for rule in a.RULES:
            with self.assertRaisesRegex(AssertionError,'underflow'):a.conditional_statistics(p,a.env.rewards(s),rule)
            result=a.conditional_statistics(p,a.env.rewards(s),rule,lp)
            self.assertTrue(np.isfinite(result['log_expected_reward']).all())
            self.assertTrue(np.all(result['conditional_exact_expected_reward']==0))
            np.testing.assert_allclose(result['posterior'].sum(1),1.,atol=1e-12)

    def test_summary_distinguishes_roles_axes_and_ignored_actor(self):
        s=fixture_states()[1:2];actions=np.array([[1,1,1]],np.int16)
        row=a.summarize(s,actions,'reciprocal')
        self.assertEqual(row['full_success_rate'],0);self.assertEqual(row['executed_partner_correct_rate'],0)
        self.assertEqual(row['kind_correct_rate'],1);self.assertEqual(row['length_correct_rate'],1);self.assertEqual(row['destination_correct_rate'],1)
        self.assertEqual(row['ignored_proposal_world_rate'],1);self.assertEqual(row['ignored_proposal_agent_rate'],1/3)
        self.assertEqual(row['ignored_proposal_counts_by_actor'],[0,0,1]);self.assertEqual(row['proposal_role_success_rate'],0)
        self.assertIsNone(row['true_pair_strata']['AB']['reward_mean'])
        strict=a.summarize(s,actions,'strict')
        self.assertEqual(strict['physical_execution_rate'],0);self.assertEqual(strict['unexecuted_proposal_agent_rate'],1)
        self.assertEqual(strict['ignored_proposal_agent_rate'],0)

    def test_primary_four_pairs_and_both_decompositions(self):
        metrics={}
        for seed,delta in zip(a.SEEDS,(-.1,.02,.08,.01)):
            for rule in a.RULES:
                cross={'strict':.20,'reciprocal':.30} if rule=='strict' else {'strict':.15+delta,'reciprocal':.20+delta}
                score=cross[rule]
                metrics[seed,rule]=dict(full_success_rate=score,executed_partner_correct_rate=.4,proposal_role_success_rate=.3,reward_mean=.5,
                    cross_settlement={r:dict(full_success_rate=v) for r,v in cross.items()})
        result=a.primary(metrics);self.assertAlmostEqual(result['mean_difference'],.0025)
        for row,delta in zip(result['paired_seeds'],(-.1,.02,.08,.01)):
            self.assertAlmostEqual(row['contrasts']['full_success_rate'],delta)
            self.assertAlmostEqual(row['decomposition']['strict_policy_mechanical_release'],.1)
            self.assertAlmostEqual(row['decomposition']['common_reciprocal_policy_difference'],delta-.1)
        with self.assertRaises(AssertionError):a.primary({k:v for i,(k,v) in enumerate(metrics.items()) if i})

    def test_saved_synthetic_record_and_cross_settlement_corruption(self):
        s=fixture_states();actions=np.ones((3,3),np.int16);p=np.full((3,3,17),.5/16);p[:,:,1]=.5
        ids=np.arange(3,dtype=np.int64);rule='reciprocal';native=a.env.rewards(s)
        values=dict(states=s,state_indices=ids,messages=np.zeros((3,2,3,4),np.int8),action_indices=actions,action_probabilities=p)
        values.update(a.settle(s,actions,rule));terms=a.conditional_statistics(p,native,rule)
        for key in ('conditional_exact_expected_reward','conditional_exact_full_success_probability','conditional_exact_execution_probability','conditional_full_posterior_mass'):values[key]=terms[key]
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'synthetic.npz';np.savez_compressed(path,**values)
            entry=a.summarize(s,actions,rule);entry.update(path=str(path),data_sha256=a.sha(path),information='FI',live=False,
                expected_reward_given_greedy_messages=float(terms['conditional_exact_expected_reward'].mean()),
                full_probability_given_greedy_messages=float(terms['conditional_exact_full_success_probability'].mean()),
                execution_probability_given_greedy_messages=float(terms['conditional_exact_execution_probability'].mean()),
                full_posterior_mass_given_greedy_messages=float(terms['conditional_full_posterior_mass'].mean()),state_indices_sha256=a.array_sha(ids),
                cross_settlement={r:a.summarize(s,actions,r) for r in a.RULES})
            _,receipt=a.evaluate_saved(entry,path,s,ids,rule)
            self.assertEqual(receipt['independent_network_samples'],0)
            entry['cross_settlement']['strict']['full_success_rate']=1
            with self.assertRaises(AssertionError):a.evaluate_saved(entry,path,s,ids,rule)

    def test_training_record_algebra_and_entropy_schedule(self):
        expected={k:'a'*64 for k in ('world_uniforms_sha256','batch_indices_sha256','batch_states_sha256','sample_uniforms_sha256')}
        for update in (1,1000,1001,6000):
            beta=.001*max(0.,1-(update-1)/1000)
            row=dict(seed=57101,rule='reciprocal',settlement_rule='reciprocal',condition='FI_silent',update=update,
                partial_success_utility=.5,entropy_coefficient=beta,mean_log_expected_utility=-1.3,mean_actor_entropy=2.,
                mean_F=-1.3+2*beta,receiver_loss=1.3-2*beta,mean_expected_utility=.3,mean_native_expected_reward=.3,
                mean_full_success_probability=.2,mean_partial_success_probability=.2,mean_execution_probability=.6,
                mean_full_success_posterior_mass=2/3,mean_partial_success_posterior_mass=1/3,
                min_log_expected_utility=-1.5,max_log_expected_utility=-1.,zero_float_expected_utility_states=0,
                sender_advantage_mean=0.,sender_advantage_abs_mean=1.,sender_advantage_squared_mean=1.,
                sender_advantage_max_abs=1.,sender_mean_complete_log_score=-10.,sampled_messages_sha256='b'*64,
                gradient_norm=10.,gradient_clip_scale=.5,**expected)
            a.training_row(row,57101,'reciprocal',update,expected)
            row['entropy_coefficient']+=.01
            with self.assertRaises(AssertionError):a.training_row(row,57101,'reciprocal',update,expected)


if __name__=='__main__':unittest.main()
