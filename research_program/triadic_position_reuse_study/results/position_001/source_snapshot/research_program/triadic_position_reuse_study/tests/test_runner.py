import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research_program.triadic_position_reuse_study import runner as r


class RunnerTests(unittest.TestCase):
    def test_budget_and_schedule(self):
        cells=list(r.cell_specs())
        self.assertEqual(len(cells),36)
        self.assertEqual(len(set(cells)),36)
        self.assertEqual(sum(144*8 for _ in cells),41472)
        self.assertEqual(sum(144*8*(6 if u<4 else 3) for u,a,d in cells),186624)
        self.assertEqual(sum(a=='sham' for u,a,d in cells),4)

    def test_direction_and_sham_indices(self):
        spec=dict(endpoint_indices=np.array([[1,2],[3,4]]),donor_endpoint_indices=np.array([[5,6],[7,8]]))
        for d in (0,1):
            np.testing.assert_array_equal(r.indices(spec,'sham',d)[1],spec['endpoint_indices'][:,d])
            np.testing.assert_array_equal(r.indices(spec,'same',d)[1],spec['donor_endpoint_indices'][:,d])
            np.testing.assert_array_equal(r.indices(spec,'opposite',d)[1],spec['donor_endpoint_indices'][:,1-d])
        with self.assertRaises(AssertionError):r.indices(spec,'other',0)

    def test_refuse_preparation_overwrite_before_reading_policies(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(r,'sources',side_effect=RuntimeError('must not read')):
                with self.assertRaises(AssertionError):r.prepare(td)

    def test_reference_gathers_per_row_shift_and_checks_alignment(self):
        # Different old shifts and nontrivial old row order are essential: a
        # single-shift shortcut would silently give the wrong matched baseline.
        rng=np.random.default_rng(91);n=4
        spec=dict(sender=np.array([0,1,2,0]),endpoint_indices=np.array([[0,1],[2,3],[4,5],[6,7]]),
            donor_endpoint_indices=np.array([[8,9],[10,11],[12,13],[14,15]]),
            donor_shift_index=np.array([1,0,1,0]),source_spec_row=np.array([2,3,0,1]))
        pool=dict(messages=rng.integers(0,8,(16,2,3,4),dtype=np.int8))
        def base(*args):
            return dict(messages=np.zeros((n,2,3,4),np.int8),action_indices=np.zeros((n,3),np.int16),
                action_probabilities=np.zeros((n,3,17)),greedy_reward=np.zeros(n),executed=np.zeros((n,3),bool),
                satisfied=np.zeros((n,3),bool),counterfactual_reward=np.zeros(n),counterfactual_satisfied=np.zeros((n,3),bool))
        with tempfile.TemporaryDirectory() as td:
            refs=[]
            for shift in (0,1):
                old=base();old.update(recipient_indices=np.full(n,-1),donor_indices=np.full(n,-1),donor_packets=np.zeros((n,2,4),np.int8))
                take=np.flatnonzero(spec['donor_shift_index']==shift);rows=spec['source_spec_row'][take]
                old['recipient_indices'][rows]=spec['endpoint_indices'][take,0]
                old['donor_indices'][rows]=spec['donor_endpoint_indices'][take,1]
                old['donor_packets'][rows]=pool['messages'][old['donor_indices'][rows],:,spec['sender'][take],:]
                old['action_indices'][rows]=np.array(take)[:,None]+1
                path=Path(td)/f'{shift}.npz';np.savez(path,**old)
                refs.append(dict(path=str(path),data_sha256=r.sha(path),shift_index=shift,direction=0,mode='remote_opposite_both'))
            policy=dict(condition='PL_live',whole_reference_records=refs)
            with patch.object(r,'natural_data',side_effect=base),patch.object(r,'settle_data',side_effect=lambda data,*_:data):
                result,used=r.whole_reference(policy,pool,spec,'opposite',0)
                np.testing.assert_array_equal(result['action_indices'],np.broadcast_to(np.arange(1,5)[:,None],(4,3)))
                self.assertEqual(sum(v['selected_rows'] for v in used),4)
                spec['source_spec_row'][0]=1
                with self.assertRaises(AssertionError):r.whole_reference(policy,pool,spec,'opposite',0)

if __name__=='__main__':unittest.main()
