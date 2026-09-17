import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from research_program.triadic_action_dependency_study import runner as r


class RunnerTests(unittest.TestCase):
    def test_native_all_joint_actions_two_expanded_states(self):
        choices=np.asarray(list(__import__('itertools').product(range(17),repeat=3)),dtype=np.int16)
        needs=(0,4,12)  # wood/L, fiber/R, wood-short/L; only A-C succeeds.
        for layout in ((0,1,2,3),(3,1,0,2)):
            state=r.env.State(needs,layout,(2,3,1))
            packed=np.tile(np.asarray(state.needs+state.layout+state.private_sites,dtype=np.int16),(len(choices),1))
            actual=r.native(packed,choices)
            for i,row in enumerate(choices):
                expected=r.env.settle(state,{a:r.env.all_actions(a)[row[j]] for j,a in enumerate(r.env.AGENTS)})
                self.assertEqual(actual['greedy_reward'][i],expected['reward'])
                for j,a in enumerate(r.env.AGENTS):
                    self.assertEqual(actual['executed'][i,j],expected['individual_feedback'][a]['executed'])
                    self.assertEqual(actual['satisfied'][i,j],expected['individual_feedback'][a]['own_need_satisfied'])

    def test_prepare_verifies_snapshot_without_models(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'fixture.txt';source.write_text('static fixture')
            prepared={'partitions':{},'execution_budget':{'runs':24}}
            with patch.object(r,'ROOT',root),patch.object(r,'sources',side_effect=lambda:{'fixture.txt':r.sha(source)}),patch.object(r,'make_prepared',return_value=prepared),patch.object(r.core,'make_networks',side_effect=AssertionError('model call')),patch.object(r.dataset,'make_arrays',side_effect=AssertionError('array call')):
                output=root/'run';self.assertEqual(r.prepare(output)['status'],'prepared_without_training')
                r.verify(output)
                with self.assertRaises(AssertionError):r.prepare(output)
                (output/'source_snapshot/fixture.txt').write_text('changed')
                with self.assertRaises(AssertionError):r.verify(output)

    def test_existing_execution_refuses_before_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp);(output/'execution').mkdir()
            with patch.object(r,'verify',return_value=({},{})),patch.object(r.multiprocessing,'get_context',side_effect=AssertionError('spawn call')):
                with self.assertRaises(FileExistsError):r.execute(output)

    def test_evaluator_saves_only_actual_stub_outputs(self):
        needs=(0,4,12)
        spec=dict(needs=[needs],layouts=[(0,1,2,3)],private_sites=[(1,2,3)],world_count=1)
        arrays=r.dataset.make_arrays(spec)
        def fake(networks,x,live):
            return dict(messages=np.zeros((len(x),2,3,4),dtype=np.int8),action_logits=np.zeros((len(x),3,17)))
        with tempfile.TemporaryDirectory() as temp,patch.object(r.core,'rollout',side_effect=fake):
            path=Path(temp)/'eval.npz'
            result=r.evaluate(None,arrays,'PL',True,np.array([0]),path)
            self.assertEqual(result['full_success_rate'],0)
            self.assertEqual(result['physical_execution_rate'],0)
            with np.load(path) as z:
                np.testing.assert_array_equal(z['action_indices'],0)
                np.testing.assert_allclose(z['action_probabilities'],1/17)
            self.assertEqual(result['data_sha256'],r.sha(path))

    def test_silent_reuses_record_but_live_actually_closes(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'evaluate',return_value={'reused_natural':False}) as evaluate:
            silent=r.evaluate_modes(None,None,'PL_silent',[],Path(temp)/'s')
            self.assertTrue(silent['closed']['reused_natural']);self.assertEqual(evaluate.call_count,1)
            evaluate.reset_mock()
            live=r.evaluate_modes(None,None,'LL_live',[],Path(temp)/'l')
            self.assertFalse(live['closed']['reused_natural']);self.assertEqual(evaluate.call_count,2)
            self.assertTrue(evaluate.call_args_list[0].args[3]);self.assertFalse(evaluate.call_args_list[1].args[3])


if __name__=='__main__':unittest.main()
