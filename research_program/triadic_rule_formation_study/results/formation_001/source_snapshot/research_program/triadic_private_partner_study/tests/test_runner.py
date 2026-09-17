"""Small static/mocked fixtures: no policy loading, training, or neural forward."""
from copy import deepcopy
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research_program.triadic_private_partner_study import runner as r


class RunnerTests(unittest.TestCase):
    def spec(self):
        return dict(partition='fixture',state_order='need-major, then layout, then owner',
            needs=[[0,1,6]],layouts=[[0,1,2,3]],private_sites=[[1,2,3]],world_count=1)

    def trace(self,token=0):
        logits=np.full((1,3,17),-20.)
        logits[np.arange(1)[:,None],np.arange(3),[2,1,1]]=20.
        return dict(action_logits=logits,messages=np.full((1,2,3,4),token,dtype=np.int8))

    def test_budget_and_static_full_case_support(self):
        p=r.prepared();b=p['budget']
        self.assertEqual(sum(x['world_count'] for x in p['partitions'].values()),774144)
        self.assertEqual(b['runs'],8);self.assertEqual(b['checkpoints'],48)
        self.assertEqual((b['natural_monitor_files'],b['natural_final_files'],b['closed_final_files']),(192,32,16))
        self.assertEqual(b['training_forward_module_samples'],221184000)
        self.assertEqual(b['evaluation_forward_module_samples'],92897280)
        self.assertEqual(b['aliased_evaluations'],0)
        self.assertEqual(sum(c['state_edges'] for c in p['need_response_cases'].values()),252288)
        for part in r.PARTS:
            c=p['need_response_cases'][part]
            self.assertEqual(c['world_count'],p['partitions'][part]['world_count'])
            self.assertEqual(c['empty_strata'],[])
            self.assertEqual(len(c['strata']),9)

    def test_private_mask_and_public_layout(self):
        a=r.make_arrays(self.spec());x=a['x_PL'];self.assertEqual(set(k for k in a if k.startswith('x_')),{'x_PL'})
        self.assertEqual(x.shape,(1,3,54));self.assertTrue(np.all(x[:,:,53]==0))
        for actor in range(3):
            self.assertEqual(x[0,actor,7*actor],1)
            for other in range(3):
                if other!=actor:self.assertTrue(np.all(x[:,actor,7*other:7*other+7]==0))
            self.assertTrue(np.all(x[0,actor,[21,26,31,36]]==1))
        changed=deepcopy(self.spec());changed['needs']=[[0,2,6]]
        b=r.make_arrays(changed)
        np.testing.assert_array_equal(x[:,0],b['x_PL'][:,0])
        np.testing.assert_array_equal(x[:,2],b['x_PL'][:,2])
        self.assertFalse(np.array_equal(x[:,1],b['x_PL'][:,1]))

    def test_silent_keeps_own_routes_for_both_windows(self):
        for window in range(2):
            tokens=(np.arange(12).reshape(1,3,4)+window)%8
            silent=r.core.routed_window(tokens,False);live=r.core.routed_window(tokens,True)
            for actor in range(3):
                np.testing.assert_array_equal(silent[0,actor,96:],np.eye(3)[actor])
                for sender in range(3):
                    block=slice(sender*32,(sender+1)*32)
                    if sender==actor:np.testing.assert_array_equal(silent[0,actor,block],live[0,actor,block])
                    else:self.assertTrue(np.all(silent[0,actor,block]==0))

    def test_final_closed_repeats_complete_rollout_and_q_uses_executed_pairs(self):
        arrays=r.make_arrays(self.spec());cases=r.cases.build_cases(self.spec())
        actual_metrics=r.cases.metrics
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'live'
            with patch.object(r.core,'rollout',side_effect=[self.trace(1),self.trace(2)]) as rollout,\
                 patch.object(r.cases,'metrics',wraps=actual_metrics) as metrics:
                result=r.evaluate_final(None,arrays,'PL_live',cases,prefix)
            self.assertEqual([c.args[2] for c in rollout.call_args_list],[True,False])
            self.assertEqual(rollout.call_count,2);self.assertEqual(metrics.call_count,2)
            self.assertEqual(set(result),{'natural','closed'})
            for call in metrics.call_args_list:np.testing.assert_array_equal(call.args[1],[1])
            self.assertEqual(result['natural']['full_success_rate'],1.)
            self.assertEqual(result['natural']['proposal_role_success_rate'],0.)
            self.assertIn('need_response',result['natural'])
            self.assertEqual(result['closed']['intervention'],'close_cross_agent_channel_from_window_1')
            with np.load(result['natural']['path'],allow_pickle=False) as z:
                self.assertTrue(np.all(z['messages']==1));np.testing.assert_array_equal(z['actual_pair_index'],[1])
                self.assertEqual(z['action_probabilities'].shape,(1,3,17))
                self.assertIn('conditional_full_posterior_mass',z.files)
            with np.load(result['closed']['path'],allow_pickle=False) as z:self.assertTrue(np.all(z['messages']==2))
            with patch.object(r.core,'rollout',return_value=self.trace()) as rollout:
                silent=r.evaluate_final(None,arrays,'PL_silent',cases,Path(directory)/'silent')
            self.assertEqual(set(silent),{'natural'});self.assertEqual(rollout.call_count,1)
            self.assertFalse(rollout.call_args.args[2])

    def test_monitor_has_no_q_and_invalid_full_domain_rejected_before_forward(self):
        arrays=r.make_arrays(self.spec());case_spec=r.cases.build_cases(self.spec())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'monitor.npz'
            with patch.object(r.core,'rollout',return_value=self.trace()) as rollout,patch.object(r.cases,'metrics') as metrics:
                record=r.evaluate(None,arrays,True,[0],path)
            self.assertEqual(rollout.call_count,1);self.assertEqual(metrics.call_count,0)
            self.assertNotIn('need_response',record)
            for bad in ([-1],[1],[0,0],[.0],[],[[0]]):
                with self.subTest(bad=bad),patch.object(r.core,'rollout') as rollout:
                    with self.assertRaises(ValueError):r.evaluate(None,arrays,True,bad,Path(directory)/'bad.npz')
                    self.assertEqual(rollout.call_count,0)
            larger=deepcopy(case_spec);larger['world_count']=2
            with patch.object(r.core,'rollout') as rollout:
                with self.assertRaises(ValueError):r.evaluate(None,arrays,True,[0],Path(directory)/'partial.npz',larger)
                self.assertEqual(rollout.call_count,0)
            with patch.object(r.core,'rollout') as rollout:
                with self.assertRaises(ValueError):r.evaluate(None,arrays,True,[0],path)
                self.assertEqual(rollout.call_count,0)

    def test_primary_paired_sign_and_seed_identity(self):
        runs=[]
        for j,seed in enumerate(r.SEEDS):
            for condition in r.CONDITIONS:
                q=[.1,.3,.4,.2][j] if condition=='PL_live' else .2
                runs.append(dict(seed=seed,condition=condition,
                    final={r.TARGET:{'natural':{'need_response':{'Q':q}}}}))
        result=r.primary(list(reversed(runs)));self.assertAlmostEqual(result['mean_difference'],.05)
        np.testing.assert_allclose([p['difference'] for p in result['paired_seeds']],[-.1,.1,.2,0.])
        for invalid in (runs[:-1],runs[:-1]+[runs[0]]):
            with self.assertRaises(ValueError):r.primary(invalid)
        invalid=deepcopy(runs);invalid[0]['seed']=99101
        with self.assertRaises(ValueError):r.primary(invalid)
        for value in (None,True,float('nan'),-0.1,1.1):
            invalid=deepcopy(runs);invalid[0]['final'][r.TARGET]['natural']['need_response']['Q']=value
            with self.assertRaises(ValueError):r.primary(invalid)

    def test_prepare_verify_roundtrip_and_overwrite_rejection(self):
        source=r.HERE/'__init__.py';source_map={str(source):r.sha(source)}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'prepared'
            with patch.object(r,'prepared',return_value={'budget':{},'fixture':True}),patch.object(r,'sources',return_value=source_map):
                result=r.prepare(out);self.assertEqual(result['status'],'prepared_without_training')
                r.verify(out)
                with self.assertRaises(ValueError):r.prepare(out)
                snap=out/'source_snapshot'/source.relative_to(r.ROOT);snap.write_text('changed')
                with self.assertRaises(ValueError):r.verify(out)
            existing=Path(directory)/'existing';existing.mkdir();(existing/'execution').mkdir()
            with patch.object(r,'verify',return_value=({},{})),patch.object(r.multiprocessing,'get_context') as pool:
                with self.assertRaises(FileExistsError):r.execute(existing)
                self.assertEqual(pool.call_count,0)


if __name__=='__main__':unittest.main()
