"""Small arithmetic fixtures only; no formal results, NPZ or models."""
from copy import deepcopy
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from . import summarize as s
from . import metrics
from .test_metrics import runs_fixture


def fixture():
    worlds=dict(zip(s.PARTS,(419904,160704,139968,53568)))
    runs=runs_fixture()
    for run in runs:run['condition']=f'{run["rule"]}_PL_{"live" if run["live"] else "silent"}'
    runs.sort(key=lambda r:(r['seed'],s.CONDITIONS.index(r['condition'])))
    def evaluation(source,run,part,update):
        out=deepcopy(source);n=worlds[part]
        for view in s.VIEWS:
            q=out['need_response'][view];q['Q_shuffle']=q['Q']-q['Q_excess']
            q.update(complete_nine_strata=True,empty_strata=[],
                strata=[dict(changed_person=a,axis_index=b,**{k:q[k] for k in s.RESPONSES},
                    backgrounds=[dict(background_index=0,raw_example_count=7)]) for a,b in product(range(3),range(3))])
            out[view]={k:.4+index*.01 for index,k in enumerate(s.TASK)}
            out[view]['raw_actual_pair_counts']={'none':17,'AB':11,'AC':9,'BC':3}
        out.update(path=f'fixture/{run["seed"]}/{run["condition"]}/{part}/{update}.npz',data_sha256='fixture',
            worlds=n,forward_module_samples=n*9,scope='complete_partition')
        return out
    for run in runs:
        run['updates']=6000
        for time,row in enumerate(run['trajectory']):
            row['evaluation']=evaluation(row['evaluation'],run,s.TARGET,row['update'])
            panels=[];transitions=[]
            for actor,window in product(range(3),range(2)):
                h=.1*actor+.01*window+.2*time+.003*(run['seed']-s.SEEDS[0])
                backgrounds=[dict(background_index=b,entropy_bits=h,constant_code=h==0) for b in range(36)]
                panels.append(dict(actor=actor,window=window,conditional_entropy_bits=h,
                    conditional_effective_packet_count=2**h,backgrounds=backgrounds))
                transitions.append(dict(actor=actor,window=window,packet_equal_rate=.7,token_hamming_fraction=.2,
                    global_ARI=.3,mean_background_ARI=.1,conditional_entropy_bits_before=h-.2,
                    conditional_entropy_bits_after=h,backgrounds=[dict(background_index=b,ARI=.1) for b in range(36)]))
            row['message_snapshot']=dict(background_count=36,panels=panels)
            row['message_transition']=None if time==0 else dict(background_count=36,panels=transitions)
        run['final']={p:evaluation(run['trajectory'][-1]['evaluation'],run,p,6000) for p in s.PARTS if p!=s.TARGET}
        run['final'][s.TARGET]=dict(run['trajectory'][-1]['evaluation'],alias_of='trajectory_update_6000',additional_forward_module_samples=0)
    eval_worlds=64*(worlds[s.TARGET]*6+sum(worlds[p] for p in s.PARTS if p!=s.TARGET))
    budget=dict(training_forward_module_samples=64*6000*256*2*9,evaluation_forward_module_samples=eval_worlds*9,
        checkpoints=384,evaluation_files=576,evaluation_worlds=eval_worlds,final_target_aliases=64)
    return dict(status='completed',runs=runs,primary=metrics.primary(runs),budget=budget,measured_budget=dict(budget))


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.source=fixture()

    def test_complete_grid_independent_arithmetic_and_six_separate_panels(self):
        source=deepcopy(self.source)
        with patch.object(np,'load',side_effect=AssertionError('No NPZ allowed')):out=s.summarize(source)
        self.assertEqual(source,self.source)
        self.assertEqual(out['primary'],self.source['primary'])
        self.assertAlmostEqual(out['primary']['mean_centered_AUC'],.085)
        self.assertEqual(out['counts']['actual_evaluations'],576)
        self.assertEqual(len(out['target_trajectory_summaries']),24)
        self.assertEqual(len(out['complete_endpoint_summaries']),16)
        self.assertEqual(len(out['message_entropy_curves']),24)
        self.assertEqual(len(out['message_transition_curves']),24)
        self.assertEqual(out['runs'],source['runs'])
        self.assertEqual(len(out['message_entropy_curves'][0]['by_seed']),16)
        self.assertAlmostEqual(out['message_entropy_curves'][0]['means']['conditional_entropy_bits'][-1],1+.003*7.5)
        self.assertNotEqual(out['message_entropy_curves'][0]['means']['conditional_entropy_bits'],
                            out['message_entropy_curves'][1]['means']['conditional_entropy_bits'])
        strata=out['complete_endpoint_summaries'][-1]['scoring']['native']['strata']
        self.assertEqual(len(strata),9);self.assertEqual(len(strata[0]['by_seed']),16)
        self.assertEqual(strata[0]['by_seed'][0]['backgrounds'][0]['raw_example_count'],7)
        from .plot_results import plotted_data
        out['audit_status']='passed';plot=plotted_data(out)
        self.assertAlmostEqual(plot['primary_mean_pp'],8.5)
        self.assertEqual(len(plot['native_AUC_pp']),16);self.assertEqual(len(plot['entropy_bits']),24)
        self.assertEqual(set(plot['Q_percent']['native']),set(s.CONDITIONS))
        self.assertEqual(plot['centered_DiD_pp']['native_Q']['mean'][0],0)
        self.assertEqual(plot['primary_interval_pp'],[100*out['primary']['statistics'][k] for k in ('ci95_lower','ci95_upper')])

    def test_grid_alias_and_audit_rejections(self):
        for mutate in (
            lambda x:x.update(status='running'),
            lambda x:x['runs'].__setitem__(1,x['runs'][0]),
            lambda x:x['runs'][0]['final'][s.TARGET].update(additional_forward_module_samples=9),
            lambda x:x['runs'][0]['trajectory'][1].update(update=101),
            lambda x:x['primary'].update(mean_centered_AUC=1),
        ):
            bad=deepcopy(self.source);mutate(bad)
            with self.assertRaises(ValueError):s.summarize(bad)
        audit=deepcopy(self.source['primary']);audit['statistics'].pop('interval_method')
        s.compare_audit(self.source['primary'],audit)
        audit['statistics'].pop('mean')
        with self.assertRaises(ValueError):s.compare_audit(self.source['primary'],audit)

    def test_json_only_execution_source_binding_and_refuse_overwrite(self):
        with TemporaryDirectory() as temp:
            root=Path(temp).resolve();(root/'execution').mkdir();(root/'audit_execution_001').mkdir()
            s.write(root/'prepared.json',dict(budget=self.source['budget']))
            s.write(root/'plan.json',dict(prepared_sha256=s.sha(root/'prepared.json')))
            s.write(root/'freeze.json',dict(prepared_sha256=s.sha(root/'prepared.json'),plan_sha256=s.sha(root/'plan.json')))
            main=dict(self.source,plan_sha256=s.sha(root/'plan.json'));s.write(root/'execution/results.json',main)
            s.write(root/'audit_execution_001/verification.json',dict(status='passed',plan_sha256=main['plan_sha256'],
                primary=main['primary'],artifacts_sha256={str(root/'execution/results.json'):s.sha(root/'execution/results.json')}))
            with patch.object(np,'load',side_effect=AssertionError('No NPZ allowed')):out=s.execute(root)
            self.assertEqual(out['status'],'passed')
            self.assertEqual(s.read(root/'summary_001/receipt.json')['model_forwards'],0)
            with self.assertRaisesRegex(ValueError,'overwrite'):s.execute(root)
            verification=s.read(root/'audit_execution_001/verification.json')
            verification['artifacts_sha256'][str(root/'execution/results.json')]='wrong'
            with patch.object(s,'read',side_effect=lambda p:verification if Path(p).name=='verification.json' else __import__('json').loads(Path(p).read_text())):
                with self.assertRaisesRegex(ValueError,'binding'):s.execute(root,output='summary_002')


if __name__=='__main__':unittest.main()
