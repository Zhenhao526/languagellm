"""Runner fixtures use static worlds and fake forwards; no saved policy is loaded."""
from copy import deepcopy
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from . import runner as r

COST=dict(fake_forward_calls=0,fake_module_samples=0,real_forward_calls=0,optimizer_updates=0,
    formal_checkpoint_loads=0,formal_NPZ_reads=0)


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


def fixture():
    original=r.read(r.ORIGINAL/'prepared.json')['partitions'][r.TARGET]
    full=r.cases.build_cases(original)
    need_indices=np.asarray(full['group_need_indices'][0]).ravel()
    spec=dict(partition='fixture',state_order=original['state_order'],
        needs=[original['needs'][int(i)] for i in need_indices],layouts=original['layouts'][:2],
        private_sites=original['private_sites'][:1],world_count=8)
    return r.previous.make_arrays(spec),r.cases.build_cases(spec)


class RunnerTests(unittest.TestCase):
    def test_static_budget_and_nine_strata(self):
        static=r.prepared();budget=static['budget']
        self.assertEqual(budget['policy_states'],8);self.assertEqual(budget['training_updates'],0)
        self.assertEqual((budget['cross_rows'],budget['sham_rows']),(8073216,4036608))
        self.assertEqual(budget['new_natural_module_samples'],27869184)
        self.assertEqual(budget['intervention_module_samples'],72658944)
        self.assertEqual(budget['total_new_module_samples'],100528128)
        self.assertEqual(budget['generated_data_files'],112)
        for part,c in static['cases'].items():
            self.assertEqual(c['empty_strata'],[])
            self.assertEqual(c['cross_cells'],8*c['group_background_count'])
            self.assertEqual(c['sham_cells'],4*c['group_background_count'])

    def test_partner_probability_sums_all_eight_proposals_without_success_gate(self):
        p=np.broadcast_to(np.arange(1,18)/153.,(3,3,17)).copy()
        senders=np.arange(3);recipients=np.array([[1,2],[0,2],[0,1]])
        got=r.recipient_probabilities(p,senders,recipients)
        for row,sender in enumerate(senders):
            for column,actor in enumerate(recipients[row]):
                others=[a for a in range(3) if a!=actor]
                expected=sum(p[row,actor,action] for action in range(1,17) if others[(action-1)%2]==sender)
                self.assertAlmostEqual(got[row,column],expected)
        p[:]=0;p[:,:,0]=1
        np.testing.assert_array_equal(r.recipient_probabilities(p,senders,recipients),0)

    @patch.object(r.core.base,'actor_forward',side_effect=fake_forward)
    def test_complete_native_cross_sham_and_saved_bank_fixture(self,forward):
        arrays,case_spec=fixture();self.assertEqual(case_spec['group_count'],1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bank,natural=r.native_bank(tuple(range(9)),arrays,root/'native.npz')
            self.assertEqual(natural['new_module_samples'],72);self.assertEqual(forward.call_count,9)
            self.assertEqual(r.check_context_packets(case_spec,bank)['compared_sender_packets'],4)
            saved,record=r.saved_bank(arrays,dict(path=natural['path'],sha256=natural['sha256']))
            self.assertTrue(record['reused']);self.assertEqual(record['new_module_samples'],0)
            np.testing.assert_array_equal(saved['messages'],bank['messages'])
            with patch.object(r.intervention,'intervene',wraps=r.intervention.intervene) as intervene:
                cross,p=r.evaluate_cells(tuple(range(9)),arrays,bank,case_spec,'cross',root/'cross.npz')
                sham,_=r.evaluate_cells(tuple(range(9)),arrays,bank,case_spec,'sham',root/'sham.npz')
                self.assertTrue(all(call.kwargs['overwrite_windows']==(True,False) for call in intervene.call_args_list))
            self.assertEqual((cross['rows'],sham['rows']),(16,8))
            self.assertEqual(cross['new_module_samples']+sham['new_module_samples'],144)
            self.assertEqual(sham['native_replay']['checked_rows'],8)
            self.assertEqual(sham['native_replay']['max_probability_absolute_error'],0.)
            with np.load(root/'cross.npz',allow_pickle=False) as saved:
                self.assertEqual(saved['delivered_tokens'].shape,(16,2,3,3,4))
                self.assertEqual(saved['recipient_partner_probs'].shape,(16,2))
                for viewer in range(3):
                    np.testing.assert_array_equal(saved['delivered_tokens'][:,1,viewer],saved['generated_messages'][:,1])
                np.testing.assert_array_equal(saved['overwrite_windows'],[True,False])
                # The same donor packet endpoint is used in either receiving context.
                packets=saved['donor_packets'].reshape(1,2,2,2,2,2,4)
                np.testing.assert_array_equal(packets[:,:,0,:,:,0],packets[:,:,1,:,:,0])
            scores=r.save_metrics(case_spec,p,root/'metrics.npz')
            self.assertNotIn('per_group_background',scores)
            with np.load(root/'metrics.npz',allow_pickle=False) as saved:self.assertEqual(saved['delta'].shape,(1,2,2,2))
            before=forward.call_count
            with self.assertRaises(ValueError):r.evaluate_cells(None,arrays,bank,case_spec,'cross',root/'cross.npz')
            with self.assertRaises(ValueError):r.native_bank(None,arrays,root/'native.npz')
            self.assertEqual(forward.call_count,before)
            wrong=deepcopy(bank);wrong['state_indices']=wrong['state_indices'][::-1]
            with self.assertRaises(ValueError):r.check_bank(arrays,wrong)
            wrong=deepcopy(bank);sender=case_spec['sender'][0]
            idx=case_spec['group_need_indices'][0][1][0]*2
            wrong['messages'][idx,0,sender,0]=(wrong['messages'][idx,0,sender,0]+1)%8
            with self.assertRaises(ValueError):r.check_context_packets(case_spec,wrong)

    def test_primary_fixed_final_keeps_initial_and_negative_values_auxiliary(self):
        rows=[];initial=(-.2,-.1,.05,0.);final=(-.05,.1,.2,.15)
        for i,seed in enumerate(r.SEEDS):
            for step in r.CHECKPOINTS:
                value=initial[i] if step==0 else final[i]
                rows.append(dict(seed=seed,checkpoint=step,parts={r.TARGET:dict(metrics={'L':value,'D':value+.1})}))
        result=r.primary(list(reversed(rows)));self.assertAlmostEqual(result['mean_L_final'],.1)
        self.assertEqual(result['by_seed'][0]['L_final'],-.05)
        self.assertAlmostEqual(result['auxiliary']['mean_L_final_minus_initial'],.1625)
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):r.primary(bad)
        bad=deepcopy(rows);bad[0]['parts'][r.TARGET]['metrics']['L']=None
        with self.assertRaises(ValueError):r.primary(bad)

    def test_sham_probability_tolerance_and_exact_symbols_actions(self):
        p=np.full((1,3,17),.5/16);p[:,:,0]=.5
        bank=dict(messages=np.zeros((1,2,3,4),np.int8),action_indices=np.zeros((1,3),np.int16),action_probabilities=p)
        result=dict(generated_messages=bank['messages'].copy(),action_indices=bank['action_indices'].copy(),action_probabilities=p.copy())
        result['action_probabilities'][0,0,1]+=1e-14;result['action_probabilities'][0,0,2]-=1e-14
        self.assertGreater(r.verify_sham(result,bank,[0]),0.)
        result['action_probabilities'][0,0,1]+=2e-13
        with self.assertRaises(ValueError):r.verify_sham(result,bank,[0])
        result['action_probabilities']=p.copy();result['generated_messages'][0,0,0,0]=1
        with self.assertRaises(ValueError):r.verify_sham(result,bank,[0])
        result['generated_messages']=bank['messages'].copy();result['action_indices'][0,0]=1
        with self.assertRaises(ValueError):r.verify_sham(result,bank,[0])

    def test_fixture_prepare_freeze_verify_and_no_overwrite(self):
        source=r.HERE/'__init__.py';sources={str(source):r.sha(source)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'prepared'
            with patch.object(r,'prepared',return_value={'budget':{},'fixture':True}),\
                 patch.object(r,'sources',return_value=sources),patch.object(r,'input_manifest',return_value=({},[])):
                record=r.prepare(out);self.assertEqual(record['status'],'prepared_without_policy_forward')
                r.verify(out)
                with self.assertRaises(ValueError):r.prepare(out)
                snapshot=out/'source_snapshot'/source.relative_to(r.ROOT);snapshot.write_text('changed')
                with self.assertRaises(ValueError):r.verify(out)
            existing=root/'existing';existing.mkdir();(existing/'execution').mkdir()
            with patch.object(r,'verify',return_value=({},{})),patch.object(r.multiprocessing,'get_context') as spawn:
                with self.assertRaises(FileExistsError):r.execute(existing)
                self.assertEqual(spawn.call_count,0)


if __name__=='__main__':unittest.main()
