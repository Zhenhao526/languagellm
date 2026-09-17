"""Synthetic-only consumer tests; no real execution records or model calls."""
import csv
from io import StringIO
from itertools import permutations, product
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research_program.triadic_semantic_probe_study import summarize_results as s
from research_program.triadic_semantic_probe_study import plot_results as p


def fixture():
    sl=list(permutations(range(3),2));n=18
    labels=dict(success_action_masks=np.tile([2,4],(n,1)).astype(np.uint32),
        sender=np.tile([v[0] for v in sl],3).astype(np.int8),listener=np.tile([v[1] for v in sl],3).astype(np.int8),
        axis=np.repeat(np.arange(3),6).astype(np.int8),classification=np.ones(n,dtype=np.int8),
        content_within_axis_weight=np.full(n,1/6),role_within_axis_weight=np.zeros(n),
        endpoint_indices=np.arange(n*2).reshape(n,2),case_index=np.arange(n,dtype=np.int32))
    labels['listener_partners']=np.repeat(labels['sender'][:,None],2,axis=1)
    actions=np.zeros((n*2,3),dtype=np.int16);probs=np.eye(17)[actions]
    values=dict(states=np.zeros((n*2,10),dtype=np.int16),state_indices=np.arange(n*2),
        action_indices=actions,action_probabilities=probs,greedy_reward=np.zeros(n*2))
    return labels,values


def synthetic_compact_summary():
    records=[]
    for seed,condition,checkpoint,part in product(s.SEEDS,s.CONDITIONS,s.CHECKPOINTS,s.PARTITIONS):
        value=0. if condition.endswith('silent') else .1*(s.SEEDS.index(seed)+1)*checkpoint/6000
        content=dict(macro={'both_endpoints_apt':value},by_axis={a:{'means':{'both_endpoints_apt':value}} for a in s.metrics.AXES})
        records.append(dict(kind='natural',seed=seed,condition=condition,checkpoint=checkpoint,partition=part,mode='natural',content=content))
    for seed,condition,part,mode in product(s.SEEDS,('PI_silent','PI_live'),s.PARTITIONS,s.MODES):
        value=.2 if 'same' in mode else .1
        records.append(dict(kind='intervention',seed=seed,condition=condition,checkpoint=6000,partition=part,mode=mode,
            content={'macro':{'direction_mean_counterfactual_apt':value}}))
    for seed,condition,part,window in product(s.SEEDS,('PI_silent','PI_live'),s.PARTITIONS,s.WINDOWS):
        value=0. if condition=='PI_silent' else -.1
        records.append(dict(kind='remote_contrast',seed=seed,condition=condition,checkpoint=6000,partition=part,mode=window,
            content={'macro':{'direction_mean_target_apt_opposite_minus_same':value}}))
    return dict(status='completed_read_only_summary',compact_records=records)


class SummaryPlotTests(unittest.TestCase):
    def test_natural_consumes_two_endpoints_not_independent_rows(self):
        arrays,values=fixture();report=s.natural_report(values,arrays)
        self.assertEqual(report['content']['macro']['both_endpoints_apt'],0.)
        self.assertEqual(report['content']['by_axis'][s.metrics.AXES[0]]['n_case_worlds'],6)
        self.assertEqual(report['row_values']['native_reward'].shape,(18,2))
        self.assertEqual(report['content']['macro']['both_team_full_success'],0.)

    def test_directions_reorder_and_bad_recipient_rejected(self):
        arrays,values=fixture();rows=np.arange(18)
        pair={b:dict(dataset_rows=rows.copy(),recipient_indices=arrays['endpoint_indices'][:,b].copy(),
            action_indices=values['action_indices'][arrays['endpoint_indices'][:,b]],
            action_probabilities=values['action_probabilities'][arrays['endpoint_indices'][:,b]],
            greedy_reward=np.full(18,.5*b)) for b in (1,0)}
        result=s.combine_directions(pair,arrays,rows)
        self.assertTrue(np.array_equal(result['greedy_reward'],np.tile([0,.5],(18,1))))
        pair[0]['recipient_indices'][0]+=1
        with self.assertRaises(AssertionError):s.combine_directions(pair,arrays,rows)

    def test_full_report_row_values_and_strata_saved(self):
        arrays,values=fixture();report=s.natural_report(values,arrays)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);(out/'reports').mkdir();(out/'row_metrics').mkdir();stream=StringIO()
            metadata=dict(kind='natural',seed=49101,condition='PI_silent',checkpoint=0,partition='train',mode='natural')
            entry=s.save_report(out,'fake',report,metadata,arrays,np.arange(18),csv.writer(stream))
            saved=json.loads((out/entry['report_file']).read_text())
            self.assertNotIn('row_values',saved['metrics']);self.assertNotIn('_labels',saved['metrics'])
            self.assertEqual(len(saved['metrics']['content']['strata']['ordered_sender_listener']),6)
            with np.load(out/entry['row_metrics']['path'],allow_pickle=False) as z:
                self.assertEqual(z['both_endpoints_apt'].shape,(18,))
                self.assertEqual(z['native_reward'].shape,(18,2))
            self.assertIn('both_endpoints_apt',stream.getvalue())

    def test_flatten_keeps_zero_negative_and_vector_directions(self):
        output=dict(s.flatten_numbers({'zero':0,'negative':-.25,'directions':[0.,-.1],'empty':None}))
        self.assertEqual(output,{'zero':0,'negative':-.25,'directions/0':0.,'directions/1':-.1,'empty':None})

    def test_primary_equal_four_seed_and_three_axis_inputs(self):
        summary=synthetic_compact_summary();answer=s.primary_comparison(summary['compact_records'])
        self.assertEqual(len(answer['seeds']),4)
        self.assertAlmostEqual(answer['equal_seed_mean']['PI_live_minus_silent'],.25)
        self.assertTrue(all(len(r['axis_rates'])==3 for r in answer['seeds']))

    def test_figure_data_keeps_all_seeds_checkpoints_negative_remote(self):
        x=p.figure_data(synthetic_compact_summary())
        self.assertEqual(np.asarray(x['trajectory']['train']['PI_live']).shape,(4,6))
        self.assertEqual(len(x['endpoint']),4)
        self.assertEqual(x['transfer']['both']['PI_live']['difference'],[-.1]*4)
        self.assertEqual(x['transfer']['both']['PI_silent']['difference'],[0.]*4)

    def test_record_sha_and_completed_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);path=out/'fake.npz';np.savez(path,x=np.array([1]))
            r=dict(path=str(path),data_sha256=s.sha(path));hashes={}
            self.assertTrue(np.array_equal(s.load_record(r,out/'results.json',hashes)['x'],[1]))
            r['data_sha256']='0'*64
            with self.assertRaises(AssertionError):s.load_record(r,out/'results.json',{})
            result=out/'results.json';result.write_text(json.dumps({'status':'running'}))
            with self.assertRaises(AssertionError):s.summarize(result,out,out/'summary')
            self.assertFalse((out/'summary').exists())


if __name__=='__main__':unittest.main()
