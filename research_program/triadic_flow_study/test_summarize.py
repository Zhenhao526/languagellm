"""Two synthetic JSON fixtures, without model or policy-array access."""
from copy import deepcopy
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from . import summarize as s


def fixture():
    records=[]
    for si,seed in enumerate(s.SEEDS):
        parts={}
        for pi,part in enumerate(s.PARTS):
            metrics={}
            for response,shape in s.SHAPES.items():
                strata=[]
                for j,(sender,axis) in enumerate(product(range(3),range(3))):
                    arms={}
                    for i,o in product(range(2),range(2)):
                        base=.3+.01*si+.001*pi+.0001*j;value=base-(.05+.01*si)*i-.02*o+.01*i*o
                        change=.01*i-.005*o
                        if response=='S':value=np.asarray(value)
                        elif response=='expected_reward':value=np.asarray(value+.1)
                        elif response=='execution':value=np.array([.3-change,.2+change,.25,.25])
                        elif response=='changed_W1_symbols':value=np.array([4*i,6*o],float)
                        elif response=='partner':
                            value=np.full((3,4),.3);value[:,0]=.4
                            value[np.arange(3),1+np.arange(3)]=0
                        else:
                            value=np.full(shape,1/shape[-1]);value[:,0]+=change;value[:,1]-=change
                        arms[f'S{i}{o}']=value
                    contrasts=dict(I_given_O0=arms['S10']-arms['S00'],I_given_O1=arms['S11']-arms['S01'],
                        O_given_I0=arms['S01']-arms['S00'],O_given_I1=arms['S11']-arms['S10'])
                    contrasts['interaction']=contrasts['I_given_O1']-contrasts['I_given_O0']
                    strata.append(dict(sender=sender,axis_index=axis,axis=s.AXES[axis],empty=False,group_count=j+1,
                        group_background_count=j+1,**{k:s.number(v) for k,v in {**arms,**contrasts}.items()}))
                metrics[response]=dict(response_shape=list(shape),strata=strata,complete_nine_strata=True,empty_strata=[],
                    group_count=45,n_backgrounds=1,group_background_count=45,flow_cells=720,arm_cells=180,
                    per_group_background_file=dict(path='fixture_never_opened.npz',sha256='0'*64),
                    **{k:s.average([row[k] for row in strata]) for k in s.VALUES})
            parts[part]=dict(metrics=metrics,natural_bank=dict(reused=True,new_module_samples=0),
                flow=dict(rows=720,new_module_samples=4320,physical_summary={'pooled_fixture_marker':seed},
                    native_replay=dict(all_messages_equal=True,all_actions_equal=True,checked_rows=180,max_probability_absolute_error=0.)))
        records.append(dict(seed=seed,checkpoint=6000,parts=parts))
    costs=dict(new_natural_module_samples=0,intervention_module_samples=69120,total_new_module_samples=69120)
    return dict(status='completed',policy_states=records,primary=s.primary(records),measured_module_samples=costs,
        budget=dict(**costs,flow_rows=11520,sham_rows=2880))


class SummaryTests(unittest.TestCase):
    def test_complete_grid_vectors_wait_strata_and_exact_primary(self):
        data=fixture();before=deepcopy(data)
        with patch.object(np,'load',side_effect=AssertionError('NPZ forbidden')):out=s.summarize(data)
        self.assertEqual(data,before);self.assertEqual(out['policy_states'],data['policy_states'])
        self.assertEqual(out['primary'],data['primary']);self.assertAlmostEqual(out['primary']['mean_S10_minus_S00'],-.065)
        self.assertEqual(len(out['arm_records']),64);self.assertEqual(len(out['contrast_records']),80)
        self.assertEqual(len(out['all_partition_strata']),36)
        self.assertEqual(sum(len(r['responses'])*4 for r in out['all_partition_strata']),1440)
        row=next(r for r in out['partition_summaries'] if r['partition']==s.TARGET)
        self.assertEqual(row['responses']['S']['means']['I_given_O0'],out['primary']['mean_S10_minus_S00'])
        self.assertTrue(np.allclose(row['responses']['partner']['means']['S00'][0],[.4,0.,.3,.3],rtol=0,atol=1e-15))
        self.assertEqual(row['responses']['changed_W1_symbols']['means']['S11'],[4,6])
        self.assertEqual(np.asarray(row['responses']['material']['means']['S10']).shape,(3,5))
        # Unequal stratum sizes remain equally weighted in stored response means.
        m=data['policy_states'][0]['parts']['train']['metrics']['S']
        pooled=np.average([r['S00'] for r in m['strata']],weights=[r['group_count'] for r in m['strata']])
        self.assertNotAlmostEqual(m['S00'],pooled)

    def test_rejections_audit_binding_and_refuse_overwrite(self):
        for mutate in (
            lambda d:d.update(status='running'),lambda d:d['policy_states'].reverse(),
            lambda d:d['primary'].update(mean_S10_minus_S00=.7),
            lambda d:d['policy_states'][0]['parts']['train']['metrics'].pop('destination'),
            lambda d:d['policy_states'][0]['parts']['train']['flow']['native_replay'].update(all_messages_equal=False),
        ):
            bad=fixture();mutate(bad)
            with self.assertRaises(ValueError):s.summarize(bad)
        with TemporaryDirectory() as temporary:
            run=Path(temporary).resolve();(run/'execution').mkdir();(run/'audit_execution_001').mkdir();data=fixture()
            s.write(run/'prepared.json',dict(budget=data['budget']))
            plan=dict(prepared_sha256=s.sha(run/'prepared.json'),config=dict(primary=s.PRIMARY,seeds=list(s.SEEDS),checkpoints=[6000],partitions=list(s.PARTS)))
            s.write(run/'plan.json',plan);data['plan_sha256']=s.sha(run/'plan.json')
            s.write(run/'freeze.json',dict(plan_sha256=data['plan_sha256'],prepared_sha256=plan['prepared_sha256']))
            s.write(run/'execution/results.json',data)
            audit=dict(status='passed',plan_sha256=data['plan_sha256'],primary={k:v for k,v in data['primary'].items() if k!='interpretation'},
                artifacts_sha256={str(run/'execution/results.json'):s.sha(run/'execution/results.json')})
            s.write(run/'audit_execution_001/verification.json',audit)
            with patch.object(np,'load',side_effect=AssertionError('NPZ forbidden')):answer=s.execute(run)
            self.assertEqual(answer['status'],'passed');out=s.read(run/'summary_001/summary.json')
            self.assertEqual(out['audit_status'],'passed');self.assertEqual(out['audit_primary'],audit['primary'])
            receipt=s.read(run/'summary_001/receipt.json');self.assertEqual(receipt['npz_reads'],0)
            self.assertEqual(receipt['outputs_sha256'][str(run/'summary_001/summary.json')],s.sha(run/'summary_001/summary.json'))
            with self.assertRaisesRegex(ValueError,'overwrite'):s.execute(run)
            audit['primary']['mean_S10_minus_S00']+=.01
            with self.assertRaisesRegex(ValueError,'number mismatch'):s.close_tree(out['audit_primary'],audit['primary'])


if __name__=='__main__':unittest.main()
