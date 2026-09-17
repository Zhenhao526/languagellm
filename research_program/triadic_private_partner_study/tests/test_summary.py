"""Two JSON/arithmetic fixtures; no model, production cases, or NPZ import."""
from copy import deepcopy
from pathlib import Path
from itertools import product
import json,tempfile,unittest

from research_program.triadic_private_partner_study import summarize_results as s


def fixture():
    cases=dict(world_count=12,need_edges=36,state_edges=36,n_backgrounds=1,
        strata=[dict(changed_person=p,axis=a,need_edges=4,state_edges=4) for p,a in product(range(3),s.AXES)])
    budget=dict(runs=8,actual_monitor_files=192,actual_final_files=48)
    prepared=dict(budget=budget,partitions={p:dict(world_count=12,monitor_indices=list(range(12))) for p in s.PARTS},
        need_response_cases={p:deepcopy(cases) for p in s.PARTS})
    runs=[]
    for i,seed in enumerate(s.SEEDS):
        for condition in s.CONDITIONS:
            def record(part,mode,phase,step):
                task={k:.75 for k in s.TASK};task.update(physical_execution_rate=1,full_success_rate=.25,reward_mean=.5,
                    ignored_proposal_world_rate=1,ignored_proposal_agent_rate=1/3,unexecuted_proposal_agent_rate=1/3)
                row=dict(worlds=12,rule='reciprocal',information='PL',live=condition=='PL_live' and mode=='natural',mode=mode,
                    path=f'not_present/{seed}/{condition}/{phase}_{step}_{part}_{mode}.npz',data_sha256='a'*64,
                    **task,**{k:.5 for k in s.CONDITIONAL},actual_pair_counts=dict(none=0,AB=4,AC=4,BC=4),
                    raw_joint_action_counts=[dict(action_indices=[2,1,1],worlds=12)],
                    raw_proposal_role_counts=[dict(partner_indices=[2,0,0],worlds=12)],
                    raw_executed_role_counts=[dict(partner_indices=v,worlds=4) for v in ([1,0,-1],[2,-1,0],[-1,2,1])],
                    true_pair_strata={p:dict(worlds=4,**task) for p in ('AB','AC','BC')})
                if phase=='final':
                    q=([.5,0,.75,.25][i] if mode=='natural' else [.25,0,.5,0][i]) if condition=='PL_live' else .25
                    values=dict(Q=q,Q_shuffle=4/33,Q_excess=q-4/33,both_correct_rate=q,only_first_correct_rate=.25,
                        only_second_correct_rate=0.,one_correct_rate=.25,neither_correct_rate=.75-q,
                        partner_change_rate=1.,both_executed_pair_change_rate=1.,execution_status_change_rate=0.)
                    raw_counts=dict(both_correct=int(4*q),one_correct=1,neither_correct=int(3-4*q))
                    strata=[dict(changed_person=p,axis=a,need_edges=4,state_edges=4,raw_counts=raw_counts,
                        backgrounds=[dict(background_index=0,edges=4,shuffle_numerator_sum=64,shuffle_denominator=528,
                                          raw_counts=raw_counts,**values)],**values) for p,a in product(range(3),s.AXES)]
                    row['need_response']=dict(worlds=12,need_edges=36,state_edges=36,complete_nine_strata=True,
                        empty_strata=[],strata=strata,backgrounds=[dict(background_index=0,actual_pair_counts=dict(none=0,AB=4,AC=4,BC=4))],**values)
                return row
            runs.append(dict(seed=seed,condition=condition,rule='reciprocal',updates=6000,
                final={part:{mode:record(part,mode,'final',6000) for mode in (('natural','closed') if condition=='PL_live' else ('natural',))} for part in s.PARTS},
                monitor=[dict(update=step,monitor={part:record(part,'natural','monitor',step) for part in s.PARTS}) for step in s.STEPS]))
    return dict(status='completed',budget=budget,runs=runs,primary=s.primary(runs)),prepared


class SummaryTests(unittest.TestCase):
    def test_arithmetic_complete_views_negative_seed_and_nine_strata_retained(self):
        result,prepared=fixture();before=deepcopy(result);out=s.extract_summary(result,prepared)
        self.assertEqual(result,before);self.assertEqual(out['actual_runs'],result['runs'])
        self.assertEqual(out['primary'],result['primary']);self.assertEqual(out['primary']['mean_difference'],.125)
        self.assertEqual([p['difference'] for p in out['primary']['paired_seeds']],[.25,-.25,.5,0.])
        self.assertEqual(out['counts']['actual_evaluation_records'],240)
        self.assertEqual(out['counts']['live_policy_closed_final_records'],16)
        self.assertEqual(out['counts']['full_need_response_records'],48)
        for part in s.PARTS:
            final=out['full_endpoints'][part]
            self.assertEqual(final['mean_live_minus_silent']['need_response']['Q'],.125)
            self.assertEqual(final['mean_live_minus_closed']['need_response']['Q'],.1875)
            self.assertEqual(set(final['means_by_policy_view']),set(s.VIEWS))
            self.assertEqual(len(final['means_by_policy_view']['PL_live_natural']['need_response']['strata']),9)
        for step in s.STEPS:
            for part in s.PARTS:
                self.assertNotIn('need_response',out['monitor_subsets_by_update'][str(step)][part]['mean_live_minus_silent'])
        json.dumps(out,allow_nan=False)
        changed=deepcopy(result);changed['runs'][0]['final']['train'].pop('closed')
        with self.assertRaises(ValueError):s.extract_summary(changed,prepared)
        changed=deepcopy(result);changed['primary']['mean_difference']+=1e-10
        with self.assertRaises(ValueError):s.extract_summary(changed,prepared)

    def test_json_only_completed_hashes_receipt_and_refusal(self):
        result,prepared=fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'execution').mkdir()
            write=lambda p,v:p.write_text(json.dumps(v),encoding='utf-8')
            write(root/'prepared.json',prepared)
            plan=dict(prepared_sha256=s.sha(root/'prepared.json'),config=dict(seeds=list(s.SEEDS),conditions=list(s.CONDITIONS),execution_rule='reciprocal'))
            write(root/'plan.json',plan);result['plan_sha256']=s.sha(root/'plan.json')
            write(root/'freeze.json',dict(plan_sha256=result['plan_sha256'],prepared_sha256=plan['prepared_sha256']))
            write(root/'execution/results.json',result)
            status=dict(status='completed',results_sha256=s.sha(root/'execution/results.json'));write(root/'execution/status.json',status)
            answer=s.execute(root);self.assertEqual(answer['primary'],.125)
            receipt=s.read(root/'summary_001/receipt.json');self.assertTrue(receipt['scope']['primary_exact_equals_main'])
            self.assertEqual(receipt['scope']['npz_files_opened'],0)
            self.assertEqual(len(receipt['source_sha256']),5)
            for path,digest in receipt['outputs'].items():self.assertEqual(s.sha(path),digest)
            with self.assertRaises(ValueError):s.execute(root)
            write(root/'execution/status.json',dict(status='running'))
            with self.assertRaises(ValueError):s.load_completed(root)
            write(root/'execution/status.json',status)
            (root/'execution/results.json').write_text((root/'execution/results.json').read_text()+' ')
            with self.assertRaises(ValueError):s.load_completed(root)


if __name__=='__main__':unittest.main()
