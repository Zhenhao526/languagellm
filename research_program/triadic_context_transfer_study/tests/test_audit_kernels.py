"""Synthetic schedules only: no parameters, network initialization or forward."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from itertools import product
import numpy as np
from research_program.triadic_context_transfer_study import audit_execution as a

class KernelTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(112233)

    def fake_second(self,inputs):
        # A deterministic finite lookup, deliberately dependent on own need and
        # received payload. This is not a neural model or a learned policy.
        n=len(inputs);answer=np.zeros((n,3,4),dtype=np.int8)
        for who,pos in product(range(3),range(4)):
            value=inputs[:,who,:54].sum(-1).astype(int)
            for sender in range(3):
                block=inputs[:,who,54+32*sender+8*pos:54+32*sender+8*(pos+1)]
                value+=block.argmax(-1)*(sender+1)
            answer[:,who,pos]=value%8
        return answer

    def test_route_outward_only_and_visibility(self):
        m=self.rng.integers(0,8,(6,3,4),dtype=np.int8);s=np.arange(6)%3;p=(m[np.arange(6),s]+1)%8
        before=a.routes(m,True);after=a.routes(m,True,s,p)
        for row,viewer,sender in product(range(6),range(3),range(3)):
            chosen=p[row] if sender==s[row] and sender!=viewer else m[row,sender]
            np.testing.assert_array_equal(after[row,viewer,32*sender:32*(sender+1)],np.eye(8)[chosen].reshape(-1))
        np.testing.assert_array_equal(after[:,:,96:],before[:,:,96:])
        np.testing.assert_array_equal(after[np.arange(6),s],before[np.arange(6),s])
        np.testing.assert_array_equal(a.routes(m,False,s,p),a.routes(m,False))

    def test_sham_and_silent_are_input_identities(self):
        x=self.rng.integers(0,2,(9,3,54)).astype(float);s=np.arange(9)%3;m=self.rng.integers(0,8,(9,2,3,4),dtype=np.int8)
        for live in (True,False):
            w2=self.fake_second(np.concatenate((x,a.routes(m[:,0],live)),axis=-1));m[:,1]=w2
            p=m[np.arange(9),:,s,:]
            result=a.causal_inputs(x,m,s,p,live,self.fake_second)
            natural=np.concatenate((x,a.routes(m[:,0],live),a.routes(w2,live)),axis=-1)
            np.testing.assert_array_equal(result['action_inputs'],natural)
            if not live:
                altered=a.causal_inputs(x,m,s,(p+3)%8,False,self.fake_second)
                np.testing.assert_array_equal(altered['action_inputs'],natural)

    def test_local_both_and_remote_endpoint_swap_non_sender_inputs(self):
        n=9;s=np.arange(n)%3;x=self.rng.integers(0,2,(n,3,54)).astype(float)
        opposite=x.copy();opposite[np.arange(n),s,0]+=1
        w1=self.rng.integers(0,8,(n,3,4),dtype=np.int8);w1_other=w1.copy();w1_other[np.arange(n),s]=(w1_other[np.arange(n),s]+1)%8
        def natural(xx,first):
            second=self.fake_second(np.concatenate((xx,a.routes(first,True)),axis=-1))
            return np.stack((first,second),axis=1)
        m0,m1=natural(x,w1),natural(opposite,w1_other)
        p1=m1[np.arange(n),:,s,:]
        local=a.causal_inputs(x,m0,s,p1,True,self.fake_second)
        donor=np.concatenate((opposite,a.routes(m1[:,0],True),a.routes(m1[:,1],True)),axis=-1)
        packets=self.rng.integers(0,8,(n,2,4),dtype=np.int8)
        # Opposite at endpoint0 and same at endpoint1 take the same remote packet.
        first=a.causal_inputs(x,m0,s,packets,True,self.fake_second)
        second=a.causal_inputs(opposite,m1,s,packets,True,self.fake_second)
        for row,who in product(range(n),range(3)):
            if who!=s[row]:
                np.testing.assert_array_equal(local['action_inputs'][row,who],donor[row,who])
                np.testing.assert_array_equal(first['action_inputs'][row,who],second['action_inputs'][row,who])
        self.assertTrue(np.any(first['action_inputs'][np.arange(n),s]!=second['action_inputs'][np.arange(n),s]))

    def test_receivers_second_messages_recomputed_and_old_packets_not_overwritten(self):
        n=6;x=np.zeros((n,3,54));s=np.arange(n)%3;m=np.zeros((n,2,3,4),dtype=np.int8);saved=m.copy()
        packets=np.full((n,2,4),3,dtype=np.int8)
        out=a.causal_inputs(x,m,s,packets,True,self.fake_second)
        np.testing.assert_array_equal(m,saved)
        np.testing.assert_array_equal(out['generated_messages'][:,0],m[:,0])
        self.assertTrue(np.any(out['generated_messages'][:,1]!=m[:,1]))
        for row in range(n):
            np.testing.assert_array_equal(out['second_routes'][row,s[row],32*s[row]:32*(s[row]+1)],np.eye(8)[out['generated_messages'][row,1,s[row]]].reshape(-1))

    def test_complete_static_indices_truth_eligibility_and_weights(self):
        from research_program.triadic_context_transfer_study import dataset
        context,_=a.references();specs,_,_=context.independent_specs();worlds=files=0
        for part,spec in specs.items():
            spec=dict(spec,partition=part)
            expected,meta=a.independent_dataset(spec);actual=dataset.make_spec(spec)['arrays']
            self.assertEqual(set(actual),set(expected))
            for key in actual:
                if actual[key].dtype.kind=='f':np.testing.assert_allclose(actual[key],expected[key],atol=1e-18,rtol=0)
                else:np.testing.assert_array_equal(actual[key],expected[key])
            for axis in range(3):
                mask=expected['axis']==axis;weights=expected['base_within_axis_weight'][mask]
                self.assertAlmostEqual(weights.sum(),1.,places=12)
                self.assertAlmostEqual((weights*expected['eligible'][:,mask]/expected['eligible_donor_count'][mask]).sum(),1.,places=12)
            self.assertTrue(np.all(expected['eligible_donor_count']>0))
            worlds+=8*4*len(spec['layouts'])*meta['n_rows'];files+=8*4*len(spec['layouts'])
        self.assertEqual(worlds,184135680);self.assertEqual(files,1536)

    def test_actual_silent_record_schema_without_neural_calls(self):
        from research_program.triadic_context_transfer_study import runner
        from unittest.mock import patch
        context,_=a.references();specs,_,_=context.independent_specs();part=specs['new_needs_and_layouts']
        spec,_=a.independent_dataset(part)
        for key,value in list(spec.items()):
            if key=='donor_layout_index':continue
            if key in ('donor_endpoint_indices','donor_correct_actions','target_site_changed','eligible','view_equal_LL'):spec[key]=value[:,:5]
            else:spec[key]=value[:5]
        states=context.packed(part);n=len(states)
        pool=dict(states=states,messages=np.zeros((n,2,3,4),dtype=np.int8),action_indices=np.zeros((n,3),dtype=np.int16),
            action_probabilities=np.zeros((n,3,17)),greedy_reward=np.zeros(n),executed=np.zeros((n,3),bool),satisfied=np.zeros((n,3),bool))
        pool['action_probabilities'][:,:,0]=1
        x=context.features(states,'PL')
        with TemporaryDirectory() as directory,patch.object(runner.core.base,'actor_forward',side_effect=AssertionError('No neural forward')):
            for direction in (0,1):
                path=Path(directory)/f'{direction}.npz'
                record=runner.execute_cell(None,x,pool,spec,'remote_opposite_both',0,direction,False,path)
                record.update(condition='PL_silent')
                checked=a.audit_cell(record,path,pool,x,spec,None,False)
                self.assertEqual(checked['independent_module_samples'],0)
                self.assertIsNone(checked['path'])
                self.assertFalse(path.exists())

    def test_aggregate_raw_swap_does_not_force_direction_asymmetric_C_identity(self):
        records=[]
        for part in a.PARTS:
            for mode,b in product(a.MODES,(0,1)):
                opposite=mode=='remote_opposite_both';target=.4 if opposite else .1;current=.1 if opposite else .4
                values=dict(weight_mass=.5,counterfactual_apt=target,current_apt=current,
                    counterfactual_probability=target,current_probability=current,
                    C_gate_mass=.5 if b else 0.,C_counterfactual_apt=target if b else 0.)
                records.append(dict(partition=part,mode=mode,shift_index=0,direction=b,
                    scalars={layer:{str(axis):values.copy() for axis in range(3)} for layer in ('eligible','all_other')}))
        result=a.aggregate(records)
        for part in a.PARTS:
            for layer in ('eligible','all_other'):
                self.assertAlmostEqual(result[part][layer]['macro_contrast']['counterfactual_apt'],.6)
                self.assertAlmostEqual(result[part][layer]['macro_contrast']['C_counterfactual_apt'],.3)

if __name__=='__main__':unittest.main()
