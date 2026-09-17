"""Synthetic JSON fixtures only; no production statistics or neural imports."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from research_program.triadic_reciprocal_execution_study import summarize_results as s


def settlement(rule,full):
    n=12;physical=6 if rule=='strict' else 8;ignored=0 if rule=='strict' else 2
    scalars={k:physical/n for k in s.SCALARS}
    scalars.update(reward_mean=full/n,full_success_rate=full/n,proposal_role_success_rate=1/3,
        ignored_proposal_world_rate=ignored/n,ignored_proposal_agent_rate=ignored/(3*n),unexecuted_proposal_agent_rate=1/3)
    return dict(worlds=n,rule=rule,**scalars,
        reward_counts={'0.0':n-full,'0.5':0,'1.0':full},
        raw_joint_action_counts=[dict(action_indices=[1,1,1],worlds=n)],
        raw_proposal_role_counts=[dict(partner_indices=[1,0,0],worlds=n)],
        raw_executed_role_counts=[dict(partner_indices=[-1,-1,-1],worlds=n-physical),dict(partner_indices=[1,0,-1],worlds=physical)],
        actual_pair_counts=dict(none=n-physical,AB=physical,AC=0,BC=0),ignored_proposal_counts_by_actor=[0,0,ignored],
        true_pair_strata={pair:dict(worlds=n if pair=='AB' else 0,**{k:v if pair=='AB' else None for k,v in scalars.items()}) for pair in s.PAIRS},
        metric_scope={'fixture':'Synthetic aggregate JSON, not independently computed proposals.'})


def fixture():
    runs=[]
    for index,seed in enumerate(s.SEEDS):
        for rule in s.RULES:
            ss=[3,4,2,3][index];rr=[5,3,5,3][index]
            counts={'strict':ss,'reciprocal':ss+1} if rule=='strict' else {'strict':rr-2,'reciprocal':rr}
            def record(part,phase,step):
                cross={r:settlement(r,counts[r]) for r in s.RULES};row=deepcopy(cross[rule])
                row.update(cross_settlement=cross,information='FI',live=False,
                    **{k:.5 for k in s.CONDITIONAL},path=f'nonexistent/{seed}_{rule}_{phase}_{step}_{part}.npz',
                    data_sha256='0'*64,state_indices_sha256='1'*64)
                return row
            runs.append(dict(seed=seed,rule=rule,condition='FI_silent',updates=6000,
                monitor=[dict(update=step,monitor={p:record(p,'monitor',step) for p in s.PARTS}) for step in s.STEPS],
                final={p:record(p,'final',6000) for p in s.PARTS},uninterpreted_run_metadata={'retained':True}))
    budget=dict(runs=8,actual_monitor_files=192,actual_final_files=32,neural_fixture_calls=0)
    prepared=dict(budget=budget,partitions={p:dict(world_count=12,monitor_indices=list(range(12))) for p in s.PARTS})
    result=dict(status='completed',budget=budget,runs=runs,primary=s.primary(runs))
    return result,prepared


def write_fixture(directory):
    root=Path(directory);(root/'execution').mkdir();result,prepared=fixture()
    def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')
    write(root/'prepared.json',prepared)
    write(root/'plan.json',dict(prepared_sha256=s.sha(root/'prepared.json'),
        config=dict(seeds=list(s.SEEDS),execution_rules=list(s.RULES))))
    result['plan_sha256']=s.sha(root/'plan.json')
    write(root/'freeze.json',dict(plan_sha256=result['plan_sha256']))
    write(root/'execution/results.json',result);write(root/'execution/status.json',dict(status='completed'))
    return result,prepared


class SummaryTests(unittest.TestCase):
    def test_complete_records_exact_primary_two_decompositions_and_role_counts(self):
        result,prepared=fixture();before=deepcopy(result)
        output=s.extract_summary(result,prepared)
        self.assertEqual(output['actual_runs'],result['runs']);self.assertEqual(result,before)
        self.assertEqual(output['primary'],result['primary'])
        self.assertEqual(output['counts']['actual_evaluation_records'],224)
        self.assertEqual(output['counts']['cross_settlement_summary_references'],448)
        expected=[2/12,-1/12,3/12,0.]
        for row,value in zip(output['primary']['paired_seeds'],expected):
            self.assertAlmostEqual(row['contrasts']['full_success_rate'],value)
            d=row['decomposition']
            self.assertAlmostEqual(value,d['common_reciprocal_policy_difference']+d['strict_policy_mechanical_release'])
            self.assertAlmostEqual(value,d['common_strict_policy_difference']+d['reciprocal_policy_mechanical_release'])
        self.assertAlmostEqual(output['primary']['mean_difference'],1/12)
        self.assertEqual(set(output['monitor_subsets_by_update']),{str(x) for x in s.STEPS})
        self.assertEqual(set(output['full_endpoints']),set(s.PARTS))
        self.assertEqual(len(output['final_policy_behavior']),8)
        for row in output['final_policy_behavior']:
            self.assertEqual(row['worlds'],48);self.assertTrue(row['one_pair_when_execution_occurs'])
            self.assertFalse(row['same_pair_executes_in_every_world']);self.assertEqual(row['fixed_executed_pair'],'AB')
            self.assertGreater(row['no_execution_worlds'],0)
            self.assertEqual(len(row['raw_proposal_role_counts']),27);self.assertEqual(len(row['raw_executed_role_counts']),27)
            self.assertEqual(sum(r['worlds'] for r in row['raw_executed_role_counts']),48)
        json.dumps(output,allow_nan=False)
        record=deepcopy(result['runs'][0]['final']['train'])
        record['execution_probability_given_greedy_messages']=1.0000000000000002
        s.checked_record(record,12,'strict')

    def test_fixedness_uses_counts_and_distinguishes_no_execution(self):
        result,_=fixture();run=deepcopy(result['runs'][0])
        first=run['final'][s.PARTS[0]]
        first['actual_pair_counts']=dict(none=0,AB=0,AC=12,BC=0)
        first['raw_executed_role_counts']=[dict(partner_indices=[2,-1,0],worlds=12)]
        mixed=s.policy_behavior(run)
        self.assertFalse(mixed['one_pair_when_execution_occurs']);self.assertIsNone(mixed['fixed_executed_pair'])
        self.assertEqual(mixed['observed_executed_pairs'],['AB','AC'])
        for record in run['final'].values():
            record['actual_pair_counts']=dict(none=12,AB=0,AC=0,BC=0)
            record['raw_executed_role_counts']=[dict(partner_indices=[-1,-1,-1],worlds=12)]
        inactive=s.policy_behavior(run)
        self.assertFalse(inactive['one_pair_when_execution_occurs']);self.assertEqual(inactive['executing_worlds'],0)
        self.assertEqual(inactive['no_execution_worlds'],48)

    def test_reject_incomplete_coverage_primary_or_count_tampering(self):
        result,prepared=fixture()
        bad=deepcopy(result);bad['runs'].pop()
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared)
        bad=deepcopy(result);bad['runs'][0]['monitor'].pop()
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared)
        bad=deepcopy(result);bad['primary']['mean_difference']+=1e-10
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared)
        bad=deepcopy(result);bad['runs'][0]['final']['train']['actual_pair_counts']['none']+=1
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared)
        bad=deepcopy(result);bad['runs'][0]['final']['train']['cross_settlement']['reciprocal']['raw_proposal_role_counts'][0]['partner_indices']=[1,0,-1]
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared)
        bad=deepcopy(result);bad['status']='running'
        with self.assertRaises(ValueError):s.extract_summary(bad,prepared)

    def test_json_only_sources_output_receipt_refusal_and_hash_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            result,_=write_fixture(directory);root=Path(directory)
            out=s.execute(root)
            self.assertEqual(out['status'],'completed')
            receipt=s.read(root/'summary_001/receipt.json')
            self.assertEqual(receipt['primary'],result['primary']);self.assertEqual(receipt['scope']['npz_files_opened'],0)
            self.assertEqual(len(receipt['source_sha256']),5)
            self.assertTrue(all(Path(p).suffix=='.json' for p in receipt['source_sha256']))
            for path,digest in receipt['outputs'].items():self.assertEqual(s.sha(path),digest)
            with self.assertRaises(ValueError):s.execute(root)
            (root/'execution/status.json').write_text('{"status":"running"}')
            with self.assertRaises(ValueError):s.load_completed(root)
            (root/'execution/status.json').write_text('{"status":"completed"}')
            (root/'prepared.json').write_text((root/'prepared.json').read_text()+' ')
            with self.assertRaises(ValueError):s.load_completed(root)


if __name__=='__main__':unittest.main()
