"""Static and stub tests; no real networks or formal model files."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research_program.triadic_role_decoder_study import runner as r

class RunnerTests(unittest.TestCase):
    def fake_runs(self):
        result=[];diff=(-.1,.02,.08,.01)
        for ps in r.PROBE_SEEDS:
            for job in r.jobs(ps):
                value=.25 if job['view']=='Own' else .98 if job['view']=='FI' else .50
                if job['view']=='Final':value+=diff[r.OLD_SEEDS.index(job['old_seed'])]+(.01 if job['payoff']=='a50' else -.01)
                result.append(dict(job=job,final={'new_needs_and_layouts':dict(joint_accuracy=value)}))
        return result

    def test_actual_logical_reuse_and_primary_nesting(self):
        runs=self.fake_runs();logical=r.logical_results(runs)
        self.assertEqual(len(logical),96);self.assertEqual(len({x['actual_job_id'] for x in logical}),42)
        p=r.primary(runs);self.assertAlmostEqual(p['mean_difference'],.0025)
        for row,expected in zip(p['protocol_seed_blocks'],(-.1,.02,.08,.01)):
            self.assertAlmostEqual(row['mean_final_minus_initial'],expected)
            self.assertEqual(len(row['payoffs']),2)
            for cell in row['payoffs']:
                self.assertEqual(len(cell['probes']),3)
                for sub in cell['probes']:
                    self.assertAlmostEqual(sub['accuracies']['Own'],.25)
                    self.assertAlmostEqual(sub['initial_minus_bound'],.5-23/62)
        self.assertEqual(r.primary(list(reversed(runs))),p)
        with self.assertRaises(AssertionError):r.primary(runs[:-1])
        with self.assertRaises(AssertionError):r.primary(runs[:-1]+[runs[0]])

    def test_budget_counts_forward_vs_reuse(self):
        parts={p:dict(world_count=n,monitor_indices=range(m)) for p,n,m in zip(r.PARTS,(419904,160704,139968,53568),(7776,2976,7776,2976))}
        b=r.budget(parts)
        expected=dict(actual_decoder_runs=42,logical_decoder_runs=96,aliased_decoder_runs=54,updates=252000,
            training_world_samples=64512000,training_forward_module_samples=193536000,
            checkpoints=252,actual_monitor_files=1008,actual_final_files=168,aliased_evaluation_records=1512,
            initial_sender_module_samples=18579456,decoder_evaluation_module_samples=113799168,
            actual_final_worlds=32514048,actual_monitor_worlds=5419008)
        for key,v in expected.items():self.assertEqual(b[key],v,key)

    def test_evaluation_unprojected_roles_and_stable_loss(self):
        labels=np.asarray([[1,1,0],[2,0,1],[0,2,2]],dtype=np.int8)
        pred=np.asarray([[1,1,0],[2,0,1],[0,0,0]],dtype=np.int8)
        p=np.full((3,3,3),.1);p[np.arange(3)[:,None],np.arange(3),pred]=.8
        arrays=dict(labels=labels,packed_states=np.zeros((3,10),dtype=np.int16))
        job={'view':'Own'}
        with tempfile.TemporaryDirectory() as td,patch.object(r.dataset,'build_inputs',return_value=np.zeros((3,3,252))) as build,patch.object(r.learner,'prediction_terms',return_value=(p,np.log(p))) as call:
            path=Path(td)/'result.npz';v=r.evaluate(None,arrays,job,None,np.arange(3),path)
            self.assertAlmostEqual(v['joint_accuracy'],2/3);self.assertAlmostEqual(v['indiv_accuracy'],7/9)
            self.assertAlmostEqual(v['cross_entropy'],-(7*np.log(.8)+2*np.log(.1))/9)
            self.assertEqual(v['true_pair_strata']['BC']['joint_accuracy'],0)
            with np.load(path) as z:np.testing.assert_array_equal(z['predictions'],pred)
            with self.assertRaises(AssertionError):r.evaluate(None,arrays,job,None,np.arange(3),path)
            for i,ids in enumerate(([-1],[3],[],[0,0],[.5],[[0,1]])):
                with self.subTest(ids=ids),self.assertRaises(AssertionError):r.evaluate(None,arrays,job,None,ids,Path(td)/f'bad{i}.npz')
            self.assertEqual(call.call_count,1)

    def test_existing_execution_guard_precedes_forward(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'execution').mkdir()
            with patch.object(r,'verify',return_value=({},{})),patch.object(r,'make_initial',side_effect=AssertionError('no forward')):
                with self.assertRaises(FileExistsError):r.execute(out)

    def test_prepare_guard_and_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);sourcefile=root/'source.txt';sourcefile.write_text('frozen')
            source=dict(partitions={},source_sha256={str(sourcefile):r.sha(sourcefile)})
            with patch.object(r,'ROOT',root),patch.object(r.dataset,'source_manifest',return_value=source),patch.object(r,'budget',return_value={}),patch.object(r,'sources',return_value={str(sourcefile):r.sha(sourcefile)}):
                out=root/'run';r.prepare(out);r.verify(out)
                with self.assertRaises(AssertionError):r.prepare(out)
                (out/'source_snapshot/source.txt').write_text('bad')
                with self.assertRaises(AssertionError):r.verify(out)

if __name__=='__main__':unittest.main()
