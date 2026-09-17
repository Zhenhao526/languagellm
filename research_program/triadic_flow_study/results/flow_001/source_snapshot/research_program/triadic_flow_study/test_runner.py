"""Finite static/fake fixtures: no saved policies, real neural calls, or optimization."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from . import runner as r
from research_program.triadic_reciprocal_execution_study import kernel
from .test_intervention import fake_forward,COST


def fixture():
    original=r.read(r.ORIGINAL/'prepared.json')['partitions'][r.TARGET]
    complete=r.cases.build_cases(original);ids=np.asarray(complete['group_need_indices'][0]).ravel()
    spec=dict(partition='fixture',state_order=original['state_order'],needs=[original['needs'][int(i)] for i in ids],
        layouts=original['layouts'][:2],private_sites=original['private_sites'][:1],world_count=8)
    return r.previous.make_arrays(spec),r.cases.build_cases(spec)


class RunnerTests(unittest.TestCase):
    def test_static_budget_and_complete_nine_strata(self):
        static=r.prepared();b=static['budget']
        self.assertEqual(b['total_new_module_samples'],48439296)
        self.assertEqual((b['flow_rows'],b['sham_rows']),(8073216,2018304))
        self.assertEqual((b['policy_states'],b['checkpoint_files_read'],b['reused_final_natural_bank_files']),(4,4,16))
        self.assertEqual((b['new_natural_module_samples'],b['training_updates'],b['formal_random_draws']),(0,0,0))
        self.assertEqual(b['generated_data_files'],32)
        for c in static['cases'].values():
            self.assertEqual(c['empty_strata'],[]);self.assertEqual(len(c['strata']),9)
            self.assertEqual(c['flow_cells'],16*c['group_background_count'])
            self.assertEqual(c['sham_cells'],4*c['group_background_count'])

    def test_exact_probability_matches_frozen_kernel_and_all_4913_joint_actions(self):
        arrays,_=fixture();states=arrays['packed_states'][:3];rewards=arrays['rewards'][:3]
        logits=np.sin(np.arange(3*3*17).reshape(3,3,17)*.31)*2
        p,_=r.core.base.policy_distribution(logits);got=r.exact_statistics(p,rewards)
        frozen=kernel.objective_terms(logits,rewards,'reciprocal')
        np.testing.assert_allclose(got['exact_full_success_probability'],frozen['full_success_probability'],atol=1e-16,rtol=1e-13)
        np.testing.assert_allclose(got['expected_native_reward'],frozen['native_expected_reward'],atol=1e-16,rtol=1e-13)
        np.testing.assert_allclose(1-got['execution_probabilities'][:,0],frozen['execution_probability'],atol=1e-15,rtol=1e-13)
        actions=np.asarray(list(product(range(17),repeat=3)),dtype=np.int16)
        self.assertEqual(len(actions),4913)
        for row,state in enumerate(states):
            settlement=r.environment.settle(np.broadcast_to(state,(4913,10)),actions,'reciprocal')
            mass=p[row,0,actions[:,0]]*p[row,1,actions[:,1]]*p[row,2,actions[:,2]]
            self.assertAlmostEqual(got['exact_full_success_probability'][row],float(mass[settlement['greedy_reward']==1].sum()))
            self.assertAlmostEqual(got['expected_native_reward'][row],float(mass@settlement['greedy_reward']))
            for column,pair in enumerate((-1,0,1,2)):
                self.assertAlmostEqual(got['execution_probabilities'][row,column],float(mass[settlement['actual_pair_index']==pair].sum()))
        uniform=np.full((3,3,17),1/17);u=r.exact_statistics(uniform,rewards)
        np.testing.assert_allclose(u['exact_full_success_probability'],1/289)
        np.testing.assert_array_equal(r.environment.settle(states,uniform.argmax(-1),'reciprocal')['greedy_reward'],0)
        correct=r.environment.STRUCTURAL_ACTIONS[(rewards[0]==1).argmax()];spectator=int(np.flatnonzero(correct==0)[0])
        deterministic=np.zeros((1,3,17));deterministic[0,np.arange(3),correct]=1
        deterministic[0,spectator]=0;deterministic[0,spectator,16]=1
        self.assertEqual(r.exact_statistics(deterministic,rewards[:1])['exact_full_success_probability'][0],1)
        for badp,badr in ((uniform[:,:,:16],rewards),(uniform,rewards*2),(uniform,np.zeros_like(rewards))):
            with self.assertRaises(ValueError):r.exact_statistics(badp,badr)

    def test_semantic_vectors_all_focals_layouts_wait_and_no_self_partner(self):
        states=np.array([[0,0,0,0,1,2,3,1,2,3],[0,0,0,3,2,0,1,1,2,3],[0,0,0,2,1,3,0,1,2,3]])
        p=np.broadcast_to(np.arange(1,18)/153.,(3,3,17)).copy();focal=np.arange(3)
        order,got=r.semantic_responses(p,states,focal)
        for row in range(3):
            for position,actor in enumerate(order[row]):
                expected={key:np.zeros(len(value[row,position])) for key,value in got.items()}
                for key in expected:expected[key][0]=p[row,actor,0]
                for action in range(1,17):
                    site=(action-1)//4;destination=((action-1)//2)%2
                    partner=[a for a in range(3) if a!=actor][(action-1)%2];material=int(states[row,3+site])
                    columns=dict(partner=1+list(order[row]).index(partner),site=1+site,material=1+material,
                        kind=1+material//2,length=1+material%2,destination=1+destination)
                    for key,column in columns.items():expected[key][column]+=p[row,actor,action]
                for key,value in got.items():np.testing.assert_allclose(value[row,position],expected[key],atol=1e-15)
                self.assertEqual(got['partner'][row,position,1+position],0.)
        p[:]=0;p[:,:,0]=1
        _,allwait=r.semantic_responses(p,states,focal)
        for value in allwait.values():np.testing.assert_array_equal(value[:,:,0],1);np.testing.assert_array_equal(value[:,:,1:],0)

    @patch.object(r.core.base,'actor_forward',side_effect=fake_forward)
    def test_fake_native_complete_flow_and_metric_storage(self,forward):
        arrays,c=fixture();self.assertEqual(c['group_count'],1)
        native=r.core.rollout(tuple(range(9)),arrays['x_PL'],True);p,_=r.core.base.policy_distribution(native['action_logits'])
        bank=dict(states=arrays['packed_states'],state_indices=np.arange(8),messages=native['messages'],action_probabilities=p,action_indices=p.argmax(-1))
        r.check_bank(arrays,bank);check=r.check_context_packets(c,bank)
        self.assertEqual((check['compared_sender_packets'],check['compared_other_packets']),(4,8))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);record,data=r.evaluate_cells(tuple(range(9)),arrays,bank,c,path/'flow.npz')
            self.assertEqual(forward.call_count,15);self.assertEqual(record['new_module_samples'],192)
            self.assertEqual(record['rows'],32);self.assertEqual(record['native_replay']['checked_rows'],8)
            self.assertEqual(record['native_replay']['max_probability_absolute_error'],0.)
            metrics,file=r.save_metrics(c,data,path/'metrics.npz')
            self.assertEqual(set(metrics),{'S','expected_reward','execution','partner','site','material','kind','length','destination','changed_W1_symbols'})
            self.assertEqual(metrics['partner']['response_shape'],[3,4])
            self.assertEqual(metrics['changed_W1_symbols']['response_shape'],[2])
            with np.load(file['path'],allow_pickle=False) as saved:
                self.assertEqual(len(saved.files),90);self.assertEqual(saved['S__I_given_O0'].shape,(1,2))
                self.assertEqual(saved['partner__interaction'].shape,(1,2,3,4))
            with np.load(path/'flow.npz',allow_pickle=False) as saved:
                self.assertEqual(saved['action_probabilities'].shape,(32,3,17));self.assertEqual(saved['delivered_tokens'].shape,(32,2,3,3,4))
                for viewer in range(3):np.testing.assert_array_equal(saved['delivered_tokens'][:,1,viewer],saved['generated_messages'][:,1])
                np.testing.assert_array_equal(saved['changed_W1_symbols'][saved['incoming']==0,0],0)
                np.testing.assert_array_equal(saved['changed_W1_symbols'][saved['outgoing']==0,1],0)
                self.assertTrue(np.all(saved['changed_W1_symbols']<=8));self.assertTrue(np.all(saved['changed_W1_symbols'][:,1]%2==0))
            before=forward.call_count
            with self.assertRaises(ValueError):r.evaluate_cells(None,arrays,bank,c,path/'flow.npz')
            with self.assertRaises(FileExistsError):r.save_npz(path/'flow.npz',{})
            self.assertEqual(forward.call_count,before)
            broken=deepcopy(bank);idx=c['group_need_indices'][0][1][0]*2;s=c['sender'][0]
            broken['messages'][idx,0,s,0]=(broken['messages'][idx,0,s,0]+1)%8
            with self.assertRaises(ValueError):r.check_context_packets(c,broken)

    def test_primary_uses_signed_exact_S_in_target_only(self):
        rows=[];deltas=[-.2,.1,-.1,.3]
        for seed,d in zip(r.SEEDS,deltas):
            rows.append(dict(seed=seed,checkpoint=6000,parts={r.TARGET:dict(metrics=dict(
                S=dict(S00=.4,S01=.1,S10=.4+d,S11=.8,I_given_O0=d),expected_reward=dict(I_given_O0=.99)))}))
        got=r.primary(list(reversed(rows)));self.assertAlmostEqual(got['mean_S10_minus_S00'],.025)
        self.assertEqual(got['by_seed'][0]['S10_minus_S00'],-.2)
        for broken in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):r.primary(broken)
        broken=deepcopy(rows);broken[0]['parts'][r.TARGET]['metrics']['S']['I_given_O0']=None
        with self.assertRaises(ValueError):r.primary(broken)
        broken=deepcopy(rows);broken[0]['parts'][r.TARGET]['metrics']['S']['I_given_O0']=.2
        with self.assertRaises(ValueError):r.primary(broken)

    def test_sham_exact_discrete_and_tolerant_probability_checks(self):
        p=np.full((1,3,17),.5/16);p[:,:,0]=.5
        bank=dict(messages=np.zeros((1,2,3,4),np.int8),action_indices=np.zeros((1,3),np.int16),action_probabilities=p)
        result=dict(generated_messages=bank['messages'].copy(),action_indices=bank['action_indices'].copy(),action_probabilities=p.copy())
        result['action_probabilities'][0,0,1]+=1e-14;result['action_probabilities'][0,0,2]-=1e-14
        self.assertGreater(r.verify_sham(result,bank,[0]),0.)
        result['action_probabilities'][0,0,1]+=2e-13
        with self.assertRaises(ValueError):r.verify_sham(result,bank,[0])
        result['action_probabilities']=p.copy();result['generated_messages'][0,0,0,0]=1
        with self.assertRaises(ValueError):r.verify_sham(result,bank,[0])

    def test_fixture_prepare_freeze_verify_refuses_overwrite(self):
        source=r.HERE/'__init__.py';sources={str(source):r.sha(source)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'prepared'
            with patch.object(r,'prepared',return_value={'budget':{},'fixture':True}),patch.object(r,'sources',return_value=sources),patch.object(r,'input_manifest',return_value=({},[])):
                result=r.prepare(out);self.assertEqual(result['status'],'prepared_without_policy_forward');r.verify(out)
                with self.assertRaises(ValueError):r.prepare(out)
                (out/'source_snapshot'/source.relative_to(r.ROOT)).write_text('changed')
                with self.assertRaises(ValueError):r.verify(out)
            existing=root/'existing';existing.mkdir();(existing/'execution').mkdir()
            with patch.object(r,'verify',return_value=({},{})),patch.object(r.multiprocessing,'get_context') as spawn:
                with self.assertRaises(FileExistsError):r.execute(existing)
                self.assertEqual(spawn.call_count,0)


if __name__=='__main__':unittest.main()
