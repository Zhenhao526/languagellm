"""Two JSON-only fixture tests; no production result or array files."""
from copy import deepcopy
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from . import analysis as a


def fixture():
    records=[]
    for si,seed in enumerate(a.SEEDS):
        for step in a.STEPS:
            parts={}
            for pi,part in enumerate(a.PARTS):
                strata=[]
                for i,(sender,axis) in enumerate(product(range(3),range(3))):
                    value=-.08+.01*si+.001*pi+.005*(step==6000)+.02*i
                    g=i+1;positive=g if value>0 else 0;negative=g if value<0 else 0
                    strata.append(dict(sender=sender,axis_index=axis,axis=a.AXES[axis],empty=False,
                        group_count=g,group_background_count=g,L=value,D=value+.04,
                        mean_compatible_probability=.5+value/2,mean_incompatible_probability=.5-value/2,
                        strict_positive_L_fraction=float(value>0),positive_L_group_backgrounds=positive,
                        negative_L_group_backgrounds=negative,zero_L_group_backgrounds=g-positive-negative,
                        mean_four_deltas=[[value,value+.04],[value+.04,value+.08]]))
                g=sum(s['group_count'] for s in strata)
                metric=dict(complete_nine_strata=True,empty_strata=[],strata=strata,group_count=g,n_backgrounds=1,
                    group_background_count=g,cross_cells=g*8,
                    raw_group_background_counts={k:sum(s[k+'_group_backgrounds'] for s in strata) for k in a.COUNTS},
                    per_group_background_file=dict(path='fixture_not_opened.npz',sha256='0'*64),
                    **{k:float(np.mean([s[k] for s in strata])) for k in a.METRICS})
                parts[part]=dict(metrics=metric,context_packet_check=dict(all_equal=True,compared_sender_packets=g*2),
                    natural_bank=dict(new_module_samples=90 if step==0 else 0,reused=step!=0),
                    cross=dict(rows=g*8,new_module_samples=g*8*6,physical_summary={'fixture_marker':seed+step}),
                    sham=dict(rows=g*4,new_module_samples=g*4*6,native_replay=dict(all_messages_equal=True,all_actions_equal=True,
                        checked_rows=g*4,max_probability_absolute_error=0.)))
            records.append(dict(seed=seed,checkpoint=step,parts=parts))
    costs=dict(new_natural_module_samples=1440,intervention_module_samples=32*45*12*6)
    costs['total_new_module_samples']=sum(costs.values())
    return dict(status='completed',policy_states=records,primary=a.primary(records),measured_module_samples=costs,budget=costs.copy())


class AnalysisTests(unittest.TestCase):
    def test_group_then_background_mean_roundoff_in_positive_fraction(self):
        record=deepcopy(fixture()['policy_states'][0]['parts']['train']);m=record['metrics']
        m['n_backgrounds']=3;m['group_background_count']*=3;m['cross_cells']*=3
        for s in m['strata']:
            s['group_background_count']*=3
            for key in a.COUNTS:s[key+'_group_backgrounds']*=3
        row=m['strata'][2]
        row.update(positive_L_group_backgrounds=5,zero_L_group_backgrounds=0,negative_L_group_backgrounds=4,
                   strict_positive_L_fraction=float(np.mean([1/3,2/3,2/3])))
        self.assertNotEqual(row['strict_positive_L_fraction'],5/9)
        m['strict_positive_L_fraction']=float(np.mean([s['strict_positive_L_fraction'] for s in m['strata']]))
        m['raw_group_background_counts']={k:sum(s[k+'_group_backgrounds'] for s in m['strata']) for k in a.COUNTS}
        record['cross']['rows']*=3;record['sham']['rows']*=3
        record['sham']['native_replay']['checked_rows']*=3;record['context_packet_check']['compared_sender_packets']*=3
        a.validate_metric(record)
        row['strict_positive_L_fraction']+=.01
        m['strict_positive_L_fraction']=float(np.mean([s['strict_positive_L_fraction'] for s in m['strata']]))
        with self.assertRaisesRegex(ValueError,'stratum positive fraction'):a.validate_metric(record)

    def test_full_grid_primary_weighting_marginals_and_raw_preservation(self):
        data=fixture();original=deepcopy(data)
        with patch.object(np,'load',side_effect=AssertionError('No array files may be read')):
            out=a.summarize(data)
        self.assertEqual(data,original);self.assertEqual(out['policy_states'],data['policy_states'])
        self.assertEqual(out['primary'],data['primary']);self.assertAlmostEqual(out['primary']['mean_L_final'],.023)
        self.assertAlmostEqual(out['primary']['auxiliary']['mean_L_final_minus_initial'],.005)
        self.assertEqual(len(out['time_partition_summaries']),8);self.assertEqual(len(out['all_time_partition_strata']),72)
        self.assertEqual(sum(len(s['by_seed']) for s in out['all_time_partition_strata']),288)
        self.assertEqual(len(out['paired_seed_partitions']),16);self.assertEqual(len(out['target_sender_and_axis_marginals']),12)
        final=next(r for r in out['time_partition_summaries'] if r['checkpoint']==6000 and r['partition']==a.TARGET)
        self.assertEqual(final['means']['L'],out['primary']['mean_L_final'])
        row=final['by_seed'][0]
        self.assertAlmostEqual(row['strict_positive_L_fraction'],5/9)
        self.assertNotEqual(row['strict_positive_L_fraction'],row['raw_pooled_positive_L_fraction'])
        marginal=next(r for r in out['target_sender_and_axis_marginals'] if r['checkpoint']==6000 and r['dimension']=='sender' and r['level']==0)
        self.assertAlmostEqual(marginal['means']['L'],-.037)

    def test_rejections_and_completed_file_binding_refuse_overwrite(self):
        for mutation in (
            lambda x:x.update(status='running'),
            lambda x:x['policy_states'].pop(),
            lambda x:x['policy_states'].reverse(),
            lambda x:x['primary'].update(mean_L_final=.7),
            lambda x:x['policy_states'][0]['parts']['train']['metrics'].update(L=.7),
            lambda x:x['policy_states'][0]['parts']['train']['sham']['native_replay'].update(all_messages_equal=False),
        ):
            bad=fixture();mutation(bad)
            with self.assertRaises(ValueError):a.summarize(bad)
        with TemporaryDirectory() as temp:
            run=Path(temp).resolve();(run/'execution').mkdir();result=fixture()
            a.write(run/'prepared.json',{'budget':result['budget']})
            plan=dict(prepared_sha256=a.sha(run/'prepared.json'),config=dict(primary=a.PRIMARY,seeds=list(a.SEEDS),
                checkpoints=list(a.STEPS),partitions=list(a.PARTS)))
            a.write(run/'plan.json',plan)
            result['plan_sha256']=a.sha(run/'plan.json')
            a.write(run/'freeze.json',dict(plan_sha256=result['plan_sha256'],prepared_sha256=plan['prepared_sha256']))
            a.write(run/'execution/results.json',result)
            with patch.object(np,'load',side_effect=AssertionError('No array files may be read')):
                answer=a.execute(run)
            self.assertEqual(answer['status'],'passed')
            receipt=a.read(run/'analysis_001/receipt.json')
            self.assertTrue(receipt['primary_exact_match']);self.assertEqual(receipt['npz_reads'],0)
            self.assertEqual(receipt['outputs_sha256'][str(run/'analysis_001/summary.json')],a.sha(run/'analysis_001/summary.json'))
            with self.assertRaisesRegex(ValueError,'overwrite'):a.execute(run)


if __name__=='__main__':unittest.main()
