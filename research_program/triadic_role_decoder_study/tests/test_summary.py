"""Small in-memory summary fixtures; no formal output or neural computation."""
from copy import deepcopy
from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np

from research_program.triadic_role_decoder_study import summarize_results as s
from research_program.triadic_role_decoder_study import runner


def record(correct_count,n,identity):
    target=np.repeat(np.array([[1,1,0],[2,0,1],[0,2,2]],dtype=np.int8),n//3,axis=0)
    predicted=target.copy();predicted[correct_count:]=0
    correct=predicted==target;joint=correct.all(1)
    true_probability=np.where(correct,.8,.1)
    counts=Counter(map(tuple,predicted.tolist()))
    return dict(path=identity,sha256='f'*64,state_indices_sha256='e'*64,worlds=n,
        joint_accuracy=float(joint.mean()),indiv_accuracy=float(correct.mean()),
        cross_entropy=float(-np.log(true_probability).mean()),
        expected_joint_correct_probability=float(np.prod(true_probability,axis=1).mean()),
        per_actor_accuracy=correct.mean(0).tolist(),
        true_pair_strata={pair:dict(worlds=n//3,joint_accuracy=float(joint[i*n//3:(i+1)*n//3].mean()))
                         for i,pair in enumerate(('AB','AC','BC'))},
        predicted_joint_role_counts=[dict(labels=list(key),worlds=value) for key,value in sorted(counts.items())])


def fixture():
    runs=[]
    for pi,probe in enumerate(s.PROBE_SEEDS):
        for job in runner.jobs(probe):
            if job['view']=='Own':k=2+pi%2
            elif job['view']=='FI':k=6
            elif job['view']=='Initial':k=2+(pi+s.OLD_SEEDS.index(job['old_seed']))%3
            else:k=2+(pi+s.OLD_SEEDS.index(job['old_seed'])+2*s.PAYOFFS.index(job['payoff']))%5
            final={part:record(k,6,job['id']+'/final/'+part) for part in s.PARTS}
            monitor=[dict(update=step,evaluations={part:record((ci+k)%4,3,job['id']+f'/monitor/{step}/'+part)
                                                  for part in s.PARTS}) for ci,step in enumerate(s.STEPS)]
            runs.append(dict(job=job,final=final,monitor=monitor))
    budget={'actual_decoder_runs':42,'logical_decoder_runs':96}
    prepared=dict(budget=budget,source={'partitions':{p:dict(world_count=6,monitor_indices=[0,2,4]) for p in s.PARTS}})
    result=dict(status='completed',budget=budget,runs=runs,logical_results=runner.logical_results(runs),primary=runner.primary(runs))
    old_runs=[]
    for seed in s.OLD_SEEDS:
        for payoff in s.PAYOFFS:
            old_runs.append(dict(seed=seed,payoff=payoff,condition='PL_live',final={part:dict(natural=dict(
                worlds=6,role_success_rate=1/3,full_success_rate=1/6,reward_mean=.5,utility_mean=.4,
                path=f'old/{seed}/{payoff}/{part}',data_sha256='d'*64,
                raw_joint_role_counts=[dict(partner_indices=[2,-1,0],worlds=6)])) for part in s.PARTS}))
    return result,prepared,dict(status='completed',runs=old_runs)


class SummaryTests(unittest.TestCase):
    def test_all_actual_logical_and_nested_means_match_fixed_primary(self):
        result,prepared,old=fixture()
        with patch.object(runner.learner,'prediction_terms',side_effect=AssertionError('No neural call')):
            summary=s.extract_summary(result,prepared,old)
        self.assertEqual(summary['primary'],runner.primary(result['runs']))
        self.assertEqual(summary['primary'],result['primary'])
        self.assertEqual(len(summary['actual_jobs']),42)
        self.assertEqual(len(summary['logical_results']),96)
        self.assertEqual(summary['counts']['actual_evaluation_records'],1176)
        self.assertEqual(summary['counts']['logical_evaluation_references'],2688)
        self.assertEqual(summary['old_policy_checks'],dict(policies=8,fixed_pairs=8,all_partition_one_third=8))
        self.assertEqual(len(summary['protocol_payoff_cells']),8)
        self.assertEqual(len(summary['protocol_seed_blocks']),4)
        for cell in summary['protocol_payoff_cells']:
            self.assertEqual(set(cell['final']),set(s.PARTS))
            self.assertEqual(tuple(map(int,cell['monitor'])),s.STEPS)
            endpoint=cell['final'][s.TARGET]
            self.assertEqual(len(endpoint['probes']),3)
            self.assertAlmostEqual(endpoint['final_minus_initial'],np.mean([p['final_minus_initial'] for p in endpoint['probes']]))
            self.assertIn('per_view_minus_23_over_62',endpoint)
            self.assertNotIn('per_view_minus_23_over_62',cell['final']['train'])
            for view in s.VIEWS:
                distribution=endpoint['mean_over_probe_seeds'][view]['predicted_joint_role_distribution']
                self.assertEqual(len(distribution),27)
                self.assertAlmostEqual(sum(row['mean_proportion'] for row in distribution),1)
        self.assertEqual(len(summary['grand_descriptive_means']['monitor']),6)
        self.assertEqual(summary['grand_descriptive_means']['final'][s.TARGET]['mean_over_protocol_seeds_after_payoff_and_probe']['Final']['worlds_per_actual_evaluation'],6)
        self.assertEqual(summary['grand_descriptive_means']['monitor']['6000'][s.TARGET]['mean_over_protocol_seeds_after_payoff_and_probe']['Final']['worlds_per_actual_evaluation'],3)
        self.assertFalse(summary['scope']['probability_arrays_recomputed'])

    def test_bad_coverage_primary_histogram_and_alias_are_rejected(self):
        result,prepared,old=fixture()
        bad=deepcopy(result);bad['runs'].pop()
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared,old)
        bad=deepcopy(result);bad['logical_results'][0]['actual_job_id']='missing'
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared,old)
        bad=deepcopy(result);bad['primary']['mean_difference']+=.01
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared,old)
        bad=deepcopy(result);bad['runs'][0]['final'][s.TARGET]['predicted_joint_role_counts'][0]['worlds']+=1
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared,old)
        bad=deepcopy(result);bad['status']='running'
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared,old)

    def test_old_fixed_pair_is_measured_not_hard_coded(self):
        _,_,old=fixture()
        changed=deepcopy(old)
        rec=changed['runs'][0]['final']['train']['natural']
        rec['raw_joint_role_counts']=[dict(partner_indices=[2,-1,0],worlds=3),dict(partner_indices=[1,0,-1],worlds=3)]
        rec['role_success_rate']=.5
        rows=s.native_reference(changed)
        self.assertFalse(rows[0]['fixed_mutual_pair_over_all_worlds'])
        self.assertFalse(rows[0]['all_partitions_role_accuracy_equals_one_third'])
        self.assertEqual(len(rows[0]['actual_role_patterns']),2)
        self.assertTrue(all(r['fixed_mutual_pair_over_all_worlds'] for r in rows[1:]))


if __name__=='__main__':unittest.main()
