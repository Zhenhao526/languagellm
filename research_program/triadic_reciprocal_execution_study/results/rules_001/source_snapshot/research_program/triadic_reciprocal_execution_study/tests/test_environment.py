"""Exhaustive native-action fixtures; no model construction or calls."""
from itertools import product
import json
import unittest

import numpy as np

from research_program.triadic_reciprocal_execution_study import environment as e
from research_program.triadic_action_dependency_study import dataset as old_data


def independent_reciprocal(state,choices):
    menus=[e.original.all_actions(a) for a in e.original.AGENTS]
    proposals={a:menus[i][int(choices[i])] for i,a in enumerate(e.original.AGENTS)}
    executed=[False]*3;satisfied=[False]*3;roles=[-1]*3;pair_index=-1;site=material=dest=-1
    for pi,(i,j) in enumerate(((0,1),(0,2),(1,2))):
        a,b=e.original.AGENTS[i],e.original.AGENTS[j];left,right=proposals[a],proposals[b]
        if (left.get('kind')=='transport' and right.get('kind')=='transport' and left['partner']==b and right['partner']==a
                and left['site']==right['site'] and left['destination']==right['destination']):
            assert pair_index==-1
            pair_index=pi;site=e.original.SITES.index(left['site']);dest=e.original.DESTINATIONS.index(left['destination']);material=state.layout[site]
            for who,partner in ((i,j),(j,i)):
                executed[who]=True;roles[who]=partner;satisfied[who]=e.original.accepts(state.needs[who],material,dest)
    ignored=[bool(pair_index>=0 and choices[i]!=0 and not executed[i]) for i in range(3)]
    return dict(executed=executed,satisfied=satisfied,greedy_reward=sum(satisfied)/2,executed_roles=roles,
                actual_pair_index=pair_index,ignored_proposal=ignored,executed_site=site,executed_material=material,executed_destination=dest)


class EnvironmentTests(unittest.TestCase):
    def test_all4913_actions_four_states_strict_and_independent_reciprocal(self):
        choices=np.asarray(list(product(range(17),repeat=3)),dtype=np.int16)
        states=[e.original.State((0,1,6),(0,1,2,3),(1,2,3)),
                e.original.State((0,6,1),(2,0,3,1),(3,1,2)),
                e.original.State((1,0,6),(3,2,0,1),(2,3,1)),
                e.original.State((0,4,12),(3,1,0,2),(3,2,1))]
        expected_pairs=[]
        for state in states:
            plans=e.original.full_success_plans(state.needs,state.layout)
            self.assertEqual(len(plans),1);expected_pairs.append(plans[0][:2])
            packed=np.tile(np.array(state.needs+state.layout+state.private_sites,dtype=np.int16),(4913,1))
            strict=e.settle(packed,choices,'strict');reciprocal=e.settle(packed,choices,'reciprocal')
            self.assertEqual(int(strict['executed'].any(1).sum()),24)
            self.assertEqual(int(reciprocal['executed'].any(1).sum()),408)
            self.assertEqual(int((strict['greedy_reward']==1).sum()),1)
            self.assertEqual(int((reciprocal['greedy_reward']==1).sum()),17)
            self.assertFalse(strict['ignored_proposal'].any())
            self.assertEqual(int(reciprocal['ignored_proposal'].any(1).sum()),384)
            menus=[e.original.all_actions(a) for a in e.original.AGENTS]
            for index,actions in enumerate(choices):
                reference=e.original.settle(state,{a:menus[i][int(actions[i])] for i,a in enumerate(e.original.AGENTS)})
                self.assertEqual(strict['greedy_reward'][index],reference['reward'])
                for i,a in enumerate(e.original.AGENTS):
                    self.assertEqual(strict['executed'][index,i],reference['individual_feedback'][a]['executed'])
                    self.assertEqual(strict['satisfied'][index,i],reference['individual_feedback'][a]['own_need_satisfied'])
                expected=independent_reciprocal(state,actions)
                for key,value in expected.items():np.testing.assert_array_equal(reciprocal[key][index],value)
            for reward in (0.5,1.):
                self.assertEqual(int((reciprocal['greedy_reward']==reward).sum()),17*int((strict['greedy_reward']==reward).sum()))
        self.assertEqual(set(expected_pairs),set(e.PAIRS))

    def test_full_execution_may_have_wrong_proposal_roles(self):
        state=e.original.State((0,1,6),(0,1,2,3),(1,2,3))
        packed=np.array([state.needs+state.layout+state.private_sites],dtype=np.int16)
        actions=np.array([[2,1,1]],dtype=np.int16);rewards=old_data.reward_terms([state])
        new=e.settle(packed,actions,'reciprocal');old=e.settle(packed,actions,'strict')
        np.testing.assert_array_equal(new['executed'],[[True,False,True]])
        np.testing.assert_array_equal(new['executed_roles'],[[2,-1,0]])
        np.testing.assert_array_equal(new['proposal_roles'],[[2,0,0]])
        np.testing.assert_array_equal(new['ignored_proposal'],[[False,True,False]])
        self.assertEqual(new['greedy_reward'][0],1);self.assertEqual(old['greedy_reward'][0],0)
        row=e.summarize(packed,actions,'reciprocal',rewards)
        self.assertEqual(row['full_success_rate'],1)
        self.assertEqual(row['executed_partner_correct_rate'],1)
        self.assertEqual(row['proposal_role_success_rate'],0)
        self.assertEqual(row['ignored_proposal_world_rate'],1)
        self.assertEqual(row['ignored_proposal_agent_rate'],1/3)

    def test_summary_preserves_all_denominators_and_content_has_no_pair_gate(self):
        state=e.original.State((0,1,6),(0,1,2,3),(1,2,3))
        actions=np.array([[2,1,1],[1,1,0],[0,0,0],[2,0,5],[4,0,3],[6,0,5]],dtype=np.int16)
        packed=np.tile(np.array(state.needs+state.layout+state.private_sites,dtype=np.int16),(6,1))
        rewards=old_data.reward_terms([state]*6)
        row=e.summarize(packed,actions,'reciprocal',rewards)
        expected={'reward_mean':1/3,'full_success_rate':1/6,'physical_execution_rate':4/6,
                  'executed_partner_correct_rate':3/6,'proposal_role_success_rate':3/6,
                  'kind_correct_rate':4/6,'length_correct_rate':3/6,'destination_correct_rate':3/6,
                  'material_identity_correct_rate':3/6,'ignored_proposal_world_rate':1/6,
                  'ignored_proposal_agent_rate':1/18,'unexecuted_proposal_agent_rate':3/18}
        for key,value in expected.items():self.assertAlmostEqual(row[key],value,msg=key)
        wrong=e.summarize(packed[1:2],actions[1:2],'reciprocal',rewards[1:2])
        self.assertEqual(wrong['executed_partner_correct_rate'],0)
        for key in ('kind_correct_rate','length_correct_rate','destination_correct_rate','material_identity_correct_rate'):
            self.assertEqual(wrong[key],1,key)
        self.assertEqual(row['actual_pair_counts'],dict(none=2,AB=1,AC=3,BC=0))
        for key in ('raw_joint_action_counts','raw_proposal_role_counts','raw_executed_role_counts'):
            self.assertEqual(sum(r['worlds'] for r in row[key]),6)
        self.assertEqual(row['true_pair_strata']['AC']['worlds'],6)
        self.assertIsNone(row['true_pair_strata']['AB']['reward_mean'])
        self.assertEqual(row['true_pair_strata']['BC']['worlds'],0)
        json.dumps(row,allow_nan=False)

    def test_structural_order_truth_and_inputs_unchanged(self):
        np.testing.assert_array_equal(e.STRUCTURAL_ACTIONS,old_data.JOINT_ACTIONS)
        states=[e.original.State(ns,(0,1,2,3)) for ns in ((0,1,6),(0,6,1),(1,0,6))]
        packed=np.array([s.needs+s.layout+s.private_sites for s in states],dtype=np.int16)
        rewards=old_data.reward_terms(states);truth=e.truth_from_rewards(packed,rewards)
        np.testing.assert_array_equal(truth['correct_pair_index'],[1,0,2])
        np.testing.assert_array_equal(truth['correct_roles'],[[2,-1,0],[1,0,-1],[-1,2,1]])
        actions=truth['correct_actions'].copy();snap=[a.copy() for a in (packed,actions,rewards)]
        for rule in e.RULES:
            out=e.settle(packed,actions,rule);row=e.summarize(packed,actions,rule,rewards)
            self.assertEqual(row['full_success_rate'],1)
            self.assertEqual(row['proposal_role_success_rate'],1)
            self.assertEqual(out['executed'].dtype,np.bool_)
            self.assertEqual(out['actual_pair_index'].dtype.kind,'i')
            for pair in e.PAIR_NAMES:self.assertEqual(row['true_pair_strata'][pair]['worlds'],1)
        for actual,expected in zip((packed,actions,rewards),snap):np.testing.assert_array_equal(actual,expected)

    def test_unsigned_actions_and_no_execution_sentinels(self):
        state=np.array([[0,1,6,0,1,2,3,1,2,3]],dtype=np.uint8)
        data=e.settle(state,np.zeros((1,3),dtype=np.uint8),'reciprocal')
        for key in ('executed_roles','proposal_roles','actual_pair_index','executed_site','executed_material','executed_destination'):
            np.testing.assert_array_equal(data[key],-1,key)
        self.assertFalse(data['ignored_proposal'].any())
        self.assertFalse(data['unexecuted_proposal'].any())

    def test_invalid_boundary_and_false_reward_truth_rejected(self):
        state=np.array([[0,1,6,0,1,2,3,1,2,3]],dtype=np.int16);actions=np.array([[2,0,1]],dtype=np.int16)
        invalid_states=[state.astype(float),state[:,:9],np.empty((0,10),int)]
        for column,value in ((0,24),(0,-1),(3,1),(7,2)):
            bad=state.copy();bad[0,column]=value;invalid_states.append(bad)
        for bad in invalid_states:
            with self.assertRaises(ValueError):e.settle(bad,actions,'strict')
        for bad in (actions.astype(float),np.ones((1,3),bool),[[17,0,0]],[[-1,0,0]],[[0,0]],[]):
            with self.assertRaises(ValueError):e.settle(state,bad,'reciprocal')
        with self.assertRaises(ValueError):e.settle(state,actions,'proposal_only')
        native=old_data.reward_terms([e.original.State((0,1,6),(0,1,2,3))])
        for value in (np.zeros((1,24)),np.ones((1,24)),np.full((1,24),np.nan),native[:,:23]):
            with self.assertRaises(ValueError):e.truth_from_rewards(state,value)
        wrong=native.copy();wrong[wrong==1]=0;wrong[0,0]=1
        with self.assertRaises(ValueError):e.truth_from_rewards(state,wrong)


if __name__=='__main__':unittest.main()
