"""Independent rule/encoding/interface fixtures; no learned arrays or forward."""
from itertools import product,permutations
from pathlib import Path
from unittest.mock import patch
import unittest
from copy import deepcopy
import numpy as np
from research_program.triadic_action_dependency_study import audit_execution as a
from research_program.triadic_action_dependency_study import environment as env,dataset,runner


class AuditInterfaceTests(unittest.TestCase):
    def test_all_acceptance_cells_and_full_support(self):
        for need,m,d in product(range(24),range(4),range(2)):
            self.assertEqual(bool(a.ACCEPT[need,m,d]),env.accepts(need,m,d))
            view=env.need_view(need)
            self.assertEqual(bool(a.ACCEPT[need,m,d]),
                ('wood','fiber')[m//2] in view['kinds'] and ('short','long')[m%2] in view['lengths'] and 'LR'[d] in view['destinations'])
        self.assertEqual(tuple(a.support()),env.support())
        self.assertEqual(len(a.support()),5376)

    def test_all_original_actions_and_all_4913_joint_choices(self):
        self.assertTrue(np.array_equal(a.JOINT,dataset.JOINT_ACTIONS))
        self.assertTrue(np.array_equal(a.JOINT,runner.core.base.JOINT_ACTIONS))
        for i,name in enumerate(env.AGENTS):self.assertEqual(env.all_actions(name),runner.core.base.ACTIONS[i])
        choices=np.asarray(list(product(range(17),repeat=3)),dtype=np.int16)
        for needs,layout in (((0,3,14),(0,1,2,3)),((12,13,23),(3,1,0,2)),((8,17,22),(2,0,3,1))):
            state=env.State(needs,layout,(3,1,2));packed=np.tile([*needs,*layout,3,1,2],(len(choices),1)).astype(np.int16)
            reward,executed,satisfied=a.native(packed,choices)
            for k,row in enumerate(choices):
                result=env.settle(state,{who:env.all_actions(who)[int(row[i])] for i,who in enumerate(env.AGENTS)})
                self.assertEqual(reward[k],result['reward'])
                self.assertEqual(executed[k].tolist(),[result['individual_feedback'][who]['executed'] for who in env.AGENTS])
                self.assertEqual(satisfied[k].tolist(),[result['individual_feedback'][who]['own_need_satisfied'] for who in env.AGENTS])

    def test_acceptance_54_columns_and_information_boundaries(self):
        states=[env.State((n,(n+7)%24,(n+11)%24),l,o) for n,l,o in product(range(24),permutations(range(4)),permutations((1,2,3)))]
        packed=np.asarray([[*s.needs,*s.layout,*s.private_sites] for s in states],dtype=np.int16)
        for mode in ('FI','PL','LL'):
            observed=[{who:env.observe(s,who,information=mode) for who in env.AGENTS} for s in states]
            actual=dataset.encode_observations(observed);expected=a.features(packed,mode)
            self.assertTrue(np.array_equal(actual,expected))
            for viewer in range(3):
                if mode!='FI':
                    for other in range(3):
                        if other!=viewer:self.assertFalse(actual[:,viewer,7*other:7*(other+1)].any())
                self.assertTrue(np.all(actual[:,viewer,53]==(mode=='FI')))
            self.assertTrue(np.all(actual[:,:,21:41:5].sum(-1)==(2 if mode=='LL' else 4)))

    def test_no_private_need_leak_after_changing_other_agents(self):
        one=np.asarray([[0,3,14,0,1,2,3,1,2,3]],dtype=np.int16)
        for viewer in range(3):
            variants=np.repeat(one,24*24,axis=0);others=[i for i in range(3) if i!=viewer]
            variants[:,others]=np.asarray(list(product(range(24),repeat=2)))
            for mode in ('PL','LL'):
                observations=[{who:env.observe(env.State(tuple(s[:3]),tuple(s[3:7]),tuple(s[7:])),who,information=mode) for who in env.AGENTS} for s in variants]
                actual=dataset.encode_observations(observations)
                self.assertTrue(np.all(actual[:,viewer]==actual[0,viewer]))

    def test_orbit_split_and_monitor_reconstructed_independently(self):
        expected,orbits,layouts=a.independent_specs();prepared=dataset.make_prepared()
        self.assertEqual(orbits,prepared['need_orbits']);self.assertEqual(layouts,prepared['layout_ranks'])
        for part in a.PARTS:
            a.compare(prepared['partitions'][part],expected[part],part+'/')
            self.assertEqual(len(expected[part]['monitor_indices']),2*len(expected[part]['needs']))
        self.assertEqual(sum(s['world_count'] for s in expected.values()),774144)

    def test_uniform_sampler_centers_and_half_open_boundaries(self):
        spec=dict(needs=[0,1,2],layouts=list(range(4)),private_sites=list(range(6)))
        u=np.asarray([((n+.5)/3,(l+.5)/4,(o+.5)/6) for n,l,o in product(range(3),range(4),range(6))])
        np.testing.assert_array_equal(a.sample(spec,u),np.arange(72))
        np.testing.assert_array_equal(dataset.sample_indices(spec,u),np.arange(72))
        for k,size in enumerate((3,4,6)):
            for cut in range(1,size):
                for value in (np.nextafter(cut/size,0),cut/size,np.nextafter(cut/size,1)):
                    x=np.zeros((1,3));x[0,k]=value
                    np.testing.assert_array_equal(a.sample(spec,x),dataset.sample_indices(spec,x))
        with self.assertRaises(AssertionError):a.sample(spec,np.ones((1,3)))

    def test_new_rewards_feed_frozen_array_learner_without_network_initialization(self):
        self.assertEqual(a.sha(runner.core.__file__),a.OLD_CORE_SHA)
        self.assertEqual(a.sha(runner.core.base.__file__),runner.core.BASE_SHA)
        self.assertEqual(a.sha(runner.core.coordination.__file__),runner.core.COORDINATION_SHA)
        self.assertEqual([list(x) for x in runner.core.DIMENSIONS.values()],[[54,64,64,32],[153,64,64,32],[252,64,64,17]])
        sample=a.support()[::83]
        states=[env.State(n,(3,1,0,2),(2,1,3)) for n in sample]
        packed=np.asarray([[*s.needs,*s.layout,*s.private_sites] for s in states],dtype=np.int16)
        expected=a.rewards(packed)
        np.testing.assert_array_equal(dataset.reward_terms(states),expected)
        with patch.object(runner.core,'make_networks',side_effect=AssertionError('No network init')),patch.object(runner.core.base,'actor_forward',side_effect=AssertionError('No model forward')):
            terms=runner.core.coordination.objective_terms(np.zeros((len(states),3,17)),expected)
        np.testing.assert_allclose(terms['J'],expected.sum(1)/(17**3),atol=1e-15,rtol=0)
        self.assertTrue(np.isfinite(terms['log_J']).all())

    def test_independent_content_macro_uses_actual_sites_and_all_strata(self):
        from research_program.triadic_action_dependency_study import metrics
        spec=deepcopy(dataset.make_prepared()['partitions']['new_needs_and_layouts'])
        states=a.packed(spec);truth=a.JOINT[(a.rewards(states)==1).argmax(1)].astype(np.int16)
        # Deterministic synthetic mistakes, independent of any learned policy.
        actions=truth.copy();actions[(states[:,0]+states[:,1]*3+states[:,4])%5<2]=0
        probs=np.eye(17)[actions]
        saved=dict(states=states,state_indices=np.arange(len(states)),action_indices=actions,action_probabilities=probs)
        expected=metrics.semantic_metrics(spec=spec,**saved)
        result=a.content_summary(spec,saved)
        self.assertAlmostEqual(result['both_endpoints_apt'],expected['content']['macro']['both_endpoints_apt'],places=14)
        self.assertEqual(len(result['axes']),3)
        self.assertTrue(all(len(row['strata'])==6 for row in result['axes']))

    def test_training_log_numeric_contract_and_negative_controls(self):
        expected={k:'a'*64 for k in ('world_uniforms_sha256','batch_indices_sha256','batch_states_sha256','sample_uniforms_sha256')}
        row=dict(expected,seed=51101,condition='LL_live',update=1,entropy_coefficient=.001,mean_J=.2,
            mean_log_J=-2.,min_log_J=-3.,max_log_J=-1.,zero_float_J_states=0,mean_F=-1.998,receiver_loss=1.998,
            mean_actor_entropy=2.,sender_advantage_mean=0.,sender_advantage_abs_mean=.2,sender_advantage_squared_mean=.06,
            sender_advantage_max_abs=.3,sender_mean_complete_log_score=-20.,gradient_norm=10.,gradient_clip_scale=.5,
            sampled_messages_sha256='b'*64)
        a.training_row(row,51101,'LL_live',1,expected)
        for key,value in (('receiver_loss',-1.998),('entropy_coefficient',0.),('gradient_clip_scale',1.),('sender_advantage_mean',1.),('sampled_messages_sha256','invalid')):
            with self.assertRaises(AssertionError):a.training_row(dict(row,**{key:value}),51101,'LL_live',1,expected)


if __name__=='__main__':unittest.main()
