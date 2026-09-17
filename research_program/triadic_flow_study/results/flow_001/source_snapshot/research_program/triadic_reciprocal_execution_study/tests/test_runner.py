"""Pure runner contracts; mocked rollout, no real neural modules or updates."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research_program.triadic_reciprocal_execution_study import runner as r

class RunnerTests(unittest.TestCase):
    def arrays(self):
        states=np.array([[0,1,6,0,1,2,3,1,2,3]],dtype=np.int16)
        choices=np.broadcast_to(r.core.base.JOINT_ACTIONS,(1,24,3)).reshape(24,3)
        rewards=r.old.native(np.repeat(states,24,axis=0),choices)['greedy_reward'][None]
        return dict(packed_states=states,rewards=rewards,x_FI=np.zeros((1,3,54)))

    def test_budget_and_original_partitions(self):
        p=r.prepared();b=p['budget']
        self.assertEqual(sum(x['world_count'] for x in p['partitions'].values()),774144)
        self.assertEqual(b['runs'],8);self.assertEqual(b['checkpoints'],48)
        self.assertEqual(b['actual_monitor_files']+b['actual_final_files'],224)
        self.assertEqual(b['training_forward_module_samples'],221184000)
        self.assertEqual(b['evaluation_forward_module_samples'],65028096)
        self.assertEqual(b['aliased_evaluations'],0)

    def test_primary_keeps_all_paired_differences_and_decomposition(self):
        runs=[]
        for j,seed in enumerate(r.SEEDS):
            for rule in r.RULES:
                full=.2 if rule=='strict' else [.1,.3,.4,.2][j]
                summary={m:full for m in ('full_success_rate','executed_partner_correct_rate','proposal_role_success_rate','reward_mean')}
                summary['cross_settlement']={'strict':dict(full_success_rate=full-.05 if rule=='reciprocal' else full),
                    'reciprocal':dict(full_success_rate=full if rule=='reciprocal' else full+.05)}
                runs.append(dict(seed=seed,rule=rule,final={'new_needs_and_layouts':summary}))
        result=r.primary(runs);self.assertAlmostEqual(result['mean_difference'],.05)
        self.assertEqual(len(result['paired_seeds']),4)
        for x in result['paired_seeds']:
            d=x['decomposition'];self.assertAlmostEqual(d['common_reciprocal_policy_difference']+d['strict_policy_mechanical_release'],x['contrasts']['full_success_rate'])
        with self.assertRaises(AssertionError):r.primary(runs[:-1])

    def test_evaluate_one_rollout_with_two_settlements(self):
        arrays=self.arrays();logits=np.full((1,3,17),-20.);logits[np.arange(1)[:,None],np.arange(3),[2,1,1]]=20.
        trace=dict(action_logits=logits,messages=np.zeros((1,2,3,4),np.int8))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sample.npz'
            with patch.object(r.core,'rollout',return_value=trace) as rollout:
                record=r.evaluate(None,arrays,'reciprocal',np.array([0]),path)
            self.assertEqual(rollout.call_count,1)
            self.assertEqual(record['full_success_rate'],1.)
            self.assertEqual(record['proposal_role_success_rate'],0.)
            self.assertEqual(record['cross_settlement']['strict']['full_success_rate'],0.)
            self.assertEqual(record['cross_settlement']['reciprocal']['full_success_rate'],1.)
            with np.load(path) as z:
                self.assertEqual(z['action_indices'].dtype,np.int16)
                np.testing.assert_array_equal(z['executed'],[[True,False,True]])
            with self.assertRaises(AssertionError):r.evaluate(None,arrays,'reciprocal',[0],path)

    def test_reject_indices_before_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            for bad in ([-1],[1],[0,0],[.0],[],[[0]]):
                with self.subTest(bad=bad),patch.object(r.core,'rollout') as rollout:
                    with self.assertRaises(AssertionError):r.evaluate(None,self.arrays(),'strict',bad,Path(directory)/'bad.npz')
                    self.assertEqual(rollout.call_count,0)

    def test_prepare_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):r.prepare(directory)

if __name__=='__main__':unittest.main()
