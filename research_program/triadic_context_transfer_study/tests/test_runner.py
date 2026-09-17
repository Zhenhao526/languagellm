"""Orchestration tests using artificial arrays, never real parameters."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research_program.triadic_context_transfer_study import runner as r

class RunnerTests(unittest.TestCase):
    def test_donor_orientation_and_controls(self):
        spec={'endpoint_indices':np.array([[1,2],[3,4]]),
              'donor_endpoint_indices':np.array([[[5,6],[7,8]],[[9,10],[11,12]]])}
        for direction in (0,1):
            np.testing.assert_array_equal(r.donor_indices(spec,'sham_both',-1,direction),spec['endpoint_indices'][:,direction])
            np.testing.assert_array_equal(r.donor_indices(spec,'local_opposite_both',-1,direction),spec['endpoint_indices'][:,1-direction])
            for shift in range(2):
                np.testing.assert_array_equal(r.donor_indices(spec,'remote_same_both',shift,direction),spec['donor_endpoint_indices'][shift,:,direction])
                np.testing.assert_array_equal(r.donor_indices(spec,'remote_opposite_both',shift,direction),spec['donor_endpoint_indices'][shift,:,1-direction])
        with self.assertRaises(AssertionError):r.donor_indices(spec,'sham_both',0,0)

    def test_modes_and_complete_budget(self):
        rows=(209952,77760,69984,25920);ks=(17,17,5,5)
        worlds=sum(n*2*(2+2*k) for n,k in zip(rows,ks))*8
        files=sum(2*(2+2*k) for k in ks)*8
        self.assertEqual(worlds,r.CONFIG['new_intervention_worlds'])
        self.assertEqual(6*worlds,r.CONFIG['new_network_samples'])
        self.assertEqual(files,r.CONFIG['actual_intervention_files'])
        for k in ks:
            modes=list(r.modes({'donor_endpoint_indices':np.empty((k,1,2))}))
            self.assertEqual(len(modes),2+2*k)
            self.assertEqual(len(set(modes)),len(modes))
            self.assertEqual(modes[:2],[('sham_both',-1),('local_opposite_both',-1)])

    def test_silent_cell_no_forward_no_output_npz(self):
        rng=np.random.default_rng(731)
        x=rng.normal(size=(12,3,54));m=rng.integers(0,8,size=(12,2,3,4),dtype=np.int8)
        spec={'endpoint_indices':np.array([[0,1],[2,3],[4,5]]),
              'donor_endpoint_indices':np.array([[[6,7],[8,9],[10,11]]]),'sender':np.arange(3)}
        pool={'messages':m}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'never.npz'
            with patch.object(r.ch,'intervene',side_effect=AssertionError('No forward allowed')):
                result=r.execute_cell(None,x,pool,spec,'remote_opposite_both',0,0,False,path)
            self.assertFalse(path.exists());self.assertTrue(result['is_silent_alias'])
            self.assertEqual(result['new_forward_worlds'],0);self.assertEqual(result['new_network_samples'],0)
            expected=r.ch.natural_action_inputs(x[[0,2,4]],m[[0,2,4]],False)
            self.assertEqual(result['batches'][0]['action_inputs_sha256'],r.core.array_sha(expected))
            self.assertEqual(result['donor_indices_sha256'],r.core.array_sha(np.array([7,9,11])))

    def test_preparation_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(r,'collect_inputs',side_effect=AssertionError('must refuse before input reads')) as mocked:
                with self.assertRaises(AssertionError):r.prepare(temp)
                mocked.assert_not_called()

    def test_json_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'receipt.json';r.write(p,{'n':1})
            with self.assertRaises(FileExistsError):r.write(p,{'n':2})
            self.assertEqual(r.read(p),{'n':1})

if __name__=='__main__':unittest.main()
