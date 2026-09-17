"""Pure synthetic/static audit checks; network replay mocked, no policy loads."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import copy,unittest
import numpy as np
from research_program.triadic_private_partner_study import audit_execution as a


def states():
    return np.array([[12,12,15,0,1,2,3,1,2,3],[12,15,12,0,1,2,3,1,2,3],[15,12,12,0,1,2,3,1,2,3]],np.int16)


def fixture(path,live=True,mode='natural'):
    s=states();ids=np.arange(3,dtype=np.int64);p=np.random.default_rng(901).uniform(.2,1,(3,3,17));p/=p.sum(-1,keepdims=True)
    actions=p.argmax(-1).astype(np.int16);m=np.zeros((3,2,3,4),np.int8);data=dict(states=s,state_indices=ids,messages=m,action_indices=actions,action_probabilities=p)
    data.update(a.prior.settle(s,actions,'reciprocal'));terms=a.prior.conditional_statistics(p,a.env.rewards(s),'reciprocal')
    keys=('conditional_exact_expected_reward','conditional_exact_full_success_probability','conditional_exact_execution_probability','conditional_full_posterior_mass')
    data.update({k:terms[k] for k in keys});np.savez_compressed(path,**data)
    record=a.prior.summarize(s,actions,'reciprocal');record.update(path=str(path),data_sha256=a.sha(path),information='PL',live=live,mode=mode,
        intervention='close_cross_agent_channel_from_window_1' if mode=='closed' else None,
        expected_reward_given_greedy_messages=float(data[keys[0]].mean()),full_probability_given_greedy_messages=float(data[keys[1]].mean()),
        execution_probability_given_greedy_messages=float(data[keys[2]].mean()),full_posterior_mass_given_greedy_messages=float(data[keys[3]].mean()),state_indices_sha256=a.array_sha(ids))
    return record,data

class AuditFixtures(unittest.TestCase):
    def test_fixed_hashes_and_complete_budget(self):
        self.assertEqual(len(a.references()),6)
        original=a.read(a.ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json')['partitions']
        cases={p:a.response.build_cases(s) for p,s in original.items()};b=a.budget(original,cases)
        self.assertEqual(b['final_forward_module_samples'],83607552);self.assertEqual(b['need_response_edge_background_evaluations'],3027456)
        run=a.HERE/'results/private_001';self.assertEqual(a.sha(run/'plan.json'),a.PLAN_SHA)
        prepared=a.read(run/'prepared.json');self.assertEqual(prepared['budget'],b);a.compare(prepared['need_response_cases'],cases)
    def test_private_information_and_route(self):
        x=a.env.features(states(),'PL');self.assertTrue(np.all(x[:,:,53]==0))
        for viewer in range(3):
            for other in range(3):
                if viewer!=other:self.assertTrue(np.all(x[:,viewer,7*other:7*other+7]==0))
        m=np.arange(12,dtype=np.int8).reshape(1,3,4)%8
        live=a.message.route(m,True);silent=a.message.route(m,False)
        self.assertTrue(np.all(live[:,:,96:]==1))
        for actor in range(3):
            np.testing.assert_array_equal(silent[0,actor,96:],np.eye(3)[actor])
            for sender in range(3):self.assertEqual(silent[0,actor,sender*32:(sender+1)*32].sum(),4 if sender==actor else 0)
    def test_saved_monitor_recalculation(self):
        with TemporaryDirectory() as td:
            path=Path(td)/'fixture.npz';entry,data=fixture(path)
            v,r=a.evaluate_saved(entry,path,states(),np.arange(3),True,'natural')
            self.assertEqual(r['independent_network_samples'],0);self.assertEqual(r['worlds'],3)
            bad=copy.deepcopy(entry);bad['live']=False
            with self.assertRaises(Exception):a.evaluate_saved(bad,path,states(),np.arange(3),True,'natural')
            bad=copy.deepcopy(entry);bad['reward_mean']+=.1
            with self.assertRaises(Exception):a.evaluate_saved(bad,path,states(),np.arange(3),True,'natural')
    def test_endpoint_forward_contract_without_nn(self):
        with TemporaryDirectory() as td:
            for live,mode in ((True,'natural'),(False,'closed')):
                path=Path(td)/(mode+'.npz');entry,data=fixture(path,live,mode)
                p=data['action_probabilities'];calls=[]
                def fake(networks,s,information,route):
                    calls.append((information,route,len(s)));return data['messages'],p,np.log(p),0
                with patch.object(a.old,'forward_final',side_effect=fake):
                    _,receipt=a.evaluate_saved(entry,path,states(),np.arange(3),live,mode,networks=object())
                self.assertEqual(calls,[('PL',live,3)]);self.assertEqual(receipt['independent_network_samples'],27)
                with self.assertRaises(Exception):a.evaluate_saved(entry,path,states(),np.arange(3),True,'closed')
    def test_primary_includes_negative_seeds(self):
        values={}
        for k,seed in enumerate(a.SEEDS):
            values[seed,'PL_live']={'need_response':{'Q':(.1,.4,.1,.3)[k]}}
            values[seed,'PL_silent']={'need_response':{'Q':.2}}
        p=a.primary(values);self.assertAlmostEqual(p['mean_difference'],.025);self.assertEqual(p['independent_paired_seed_blocks'],4)
        self.assertLess(p['paired_seeds'][0]['difference'],0)
        values.pop((a.SEEDS[0],'PL_live'))
        with self.assertRaises(Exception):a.primary(values)
    def test_monitor_rejects_q_and_reordered_indices(self):
        with TemporaryDirectory() as td:
            path=Path(td)/'fixture.npz';entry,_=fixture(path)
            with self.assertRaises(Exception):a.evaluate_saved(entry,path,states(),np.array([2,1,0]),True,'natural')
            entry['need_response']={}
            with self.assertRaises(Exception):a.evaluate_saved(entry,path,states(),np.arange(3),True,'natural')

if __name__=='__main__':unittest.main()
