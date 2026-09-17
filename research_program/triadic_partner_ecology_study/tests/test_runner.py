"""New ecology wrapper boundaries only: no real actor, gradient or optimizer.

Tests use actual saved numeric arrays and fake policy evaluations. They do not
train a task policy, call neural forward, or inspect formal experimental scores.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from research_program.triadic_partner_ecology_study import runner as r


class WrapperTests(unittest.TestCase):
    def test_prepared_exact_budget_and_unchanged_learning_configuration(self):
        old=deepcopy(r.r.CONFIG)
        p=r.make_prepared()
        self.assertEqual(r.r.CONFIG,old)
        self.assertEqual(p['training_state_samples_total'],49152000)
        self.assertEqual(p['sampled_complete_message_trajectories_total'],98304000)
        self.assertEqual(p['categorical_message_samples_total'],2359296000)
        self.assertEqual(p['offline_reward_table_entries_actual'],18247680)
        self.assertEqual(p['planned_checkpoint_files'],192)
        self.assertEqual(p['planned_natural_monitor_files']+p['planned_closed_monitor_forward_files'],576)
        self.assertEqual(p['planned_natural_final_files']+p['planned_closed_final_forward_files'],96)
        self.assertEqual(p['all_final_forward_worlds_total'],4561920)
        self.assertEqual(p['monitor_forward_worlds_total'],387072)
        for k in ('updates','batch_size','learning_rate','adam_beta1','adam_beta2','adam_epsilon',
                  'global_gradient_clip','entropy_initial','entropy_zero_after_updates','dimensions',
                  'sender_windows','sender_tokens_per_window','alphabet','trajectories_per_state','message_stream'):
            self.assertEqual(r.CONFIG[k],old[k])
        self.assertEqual([(x['seed'],x['ecology'],x['condition']) for x in p['runs']],
                         [(s,e,c) for s in r.SEEDS for e in r.ECOLOGIES for c in r.CONDITIONS])

    def test_two_weight_files_actual_arrays_and_container_sha(self):
        spec=r.design.make_prepared()['partitions']['unique']['heldout_layouts']
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'worker1';b=root/'worker2';a.mkdir();b.mkdir()
            left=r.save_weights(a/'weights.npz',spec,root);right=r.save_weights(b/'weights.npz',spec,root)
            self.assertNotEqual(left['file'],right['file'])
            self.assertEqual(left['file_sha256'],right['file_sha256'])
            self.assertEqual({k:v for k,v in left.items() if k!='file'},
                             {k:v for k,v in right.items() if k!='file'})
            l={e:{p:{'weights':deepcopy(left),'state_hash':'same'} for p in r.PARTITIONS} for e in r.ECOLOGIES}
            rr={e:{p:{'weights':deepcopy(right),'state_hash':'same'} for p in r.PARTITIONS} for e in r.ECOLOGIES}
            self.assertEqual(r.normalized_inputs(l),r.normalized_inputs(rr))
            self.assertIn('file',l['unique']['train']['weights'])
            with np.load(a/'weights.npz',allow_pickle=False) as data:
                self.assertAlmostEqual(data['full_weights'].sum(),1)
                self.assertEqual(data['monitor_weights'].shape,(672,))
                self.assertEqual(r.array_sha(data['monitor_indices']),left['monitor_indices_sha256'])
            with self.assertRaisesRegex(ValueError,'Weight output exists'):r.save_weights(a/'weights.npz',spec,root)

    def _fake_eval(self,calls):
        actions=np.zeros((2,3),dtype=np.int16)
        for i,j in ((0,1),(1,0)):
            actions[1,i]=r.r.base.ACTIONS[i].index(dict(kind='transport',site='S0',destination='L',partner=r.r.base.AGENTS[j]))
        states=np.array([[0,0,0,0,1,2,3,1,2,3]]*2,dtype=np.int16)
        def evaluate(networks,arrays,condition,indices,*,save_path):
            calls.append((condition,indices,save_path))
            np.savez_compressed(save_path,states=states,action_indices=actions,greedy_reward=np.array([0.,1.]),
                conditional_exact_expected_reward=np.array([.2,.6]),conditional_exact_full_success_probability=np.array([.1,.5]),
                conditional_exact_execution_probability=np.array([.3,.8]),
                messages=np.full((2,2,3,4),int(condition.endswith('_live')),dtype=np.int8))
            return dict(worlds=2,condition=condition,greedy_full_success_rate=.5)
        return evaluate

    def test_live_closed_rerolls_whole_policy_and_weighted_not_uniform(self):
        calls=[];weights=np.array([.25,.75])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);receipt=dict(file='weights.npz',file_sha256='file_hash',full_weights_sha256=r.array_sha(weights))
            with (patch.object(r.r,'evaluate',side_effect=self._fake_eval(calls)),
                  patch.object(r.design,'evaluation_weights',return_value=weights)):
                result=r.evaluate_modes([],{},'PI_live',{'population_weighting':'fixture'},None,root/'final',receipt,root)
            self.assertEqual([c[0] for c in calls],['PI_live','PI_silent'])
            self.assertEqual(result['natural']['uniform']['greedy_full_success_rate'],.5)
            self.assertEqual(result['natural']['weighted']['full_success_rate'],.75)
            self.assertAlmostEqual(result['natural']['weighted']['conditional_exact_expected_reward_mean'],.5)
            self.assertNotEqual(result['natural']['data_file'],result['closed']['data_file'])
            self.assertFalse(result['closed']['reused_natural'])
            with np.load(root/result['closed']['data_file'],allow_pickle=False) as data:
                np.testing.assert_array_equal(data['messages'],0)
            self.assertEqual(result['natural']['weights_sha256'],result['closed']['weights_sha256'])

    def test_silent_closed_reuses_real_file_without_forward_or_alias_mutation(self):
        calls=[];weights=np.array([.25,.75])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);receipt=dict(file='weights.npz',file_sha256='file_hash',full_weights_sha256=r.array_sha(weights))
            with (patch.object(r.r,'evaluate',side_effect=self._fake_eval(calls)),
                  patch.object(r.design,'evaluation_weights',return_value=weights)):
                result=r.evaluate_modes([],{},'FI_silent',{'population_weighting':'fixture'},None,root/'final',receipt,root)
            self.assertEqual(len(calls),1)
            self.assertTrue(result['closed']['reused_natural'])
            self.assertEqual(result['natural']['data_file'],result['closed']['data_file'])
            self.assertEqual(result['natural']['data_sha256'],result['closed']['data_sha256'])
            result['closed']['weighted']['full_success_rate']=0
            self.assertEqual(result['natural']['weighted']['full_success_rate'],.75)
            self.assertFalse((root/'final_closed.npz').exists())

    def _pairing_fixture(self,path,change=None):
        results=[]
        for s in r.SEEDS:
            for e in r.ECOLOGIES:
                for c in r.CONDITIONS:
                    results.append(dict(seed=s,ecology=e,condition=c,initial_parameter_sha256=f'init{s}'))
                    folder=path/f'seed_{s}_{e}_{c}';folder.mkdir()
                    with (folder/'training.jsonl').open('w') as f:
                        for u in (1,2,3):
                            row=dict(seed=s,ecology=e,condition=c,update=u,world_uniforms_sha256=f'u{u}',
                                destinations_sha256=f'D{u}',layout_indices_sha256=f'L{u}',owner_indices_sha256=f'O{u}',
                                sample_uniforms_sha256=f'M{u}',entropy_coefficient=.001,
                                batch_indices_sha256=f'{e}ids{u}',batch_states_sha256=f'{e}states{u}')
                            if change and (s,e,c,u)==(r.SEEDS[0],'multiple','PI_live',2):row[change]='tampered'
                            f.write(json.dumps(row)+'\n')
        return results

    def test_pairing_allows_cross_ecology_states_but_not_external_context_mismatch(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(r.CONFIG,updates=3):
            root=Path(d);results=self._pairing_fixture(root)
            check=r.check_pairing(root,results)
            self.assertEqual(len(check),4)
            self.assertTrue(all(x['common_uniform_context_updates']==3 for x in check))
        for field,message in [('destinations_sha256','common uniform/context'),('batch_states_sha256','Within-ecology')]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d,patch.dict(r.CONFIG,updates=3):
                root=Path(d);results=self._pairing_fixture(root,field)
                with self.assertRaisesRegex(ValueError,message):r.check_pairing(root,results)

    def test_prepare_verify_no_network_or_training_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'run'
            with (patch.object(r.r,'make_networks',side_effect=AssertionError('no actor')),
                  patch.object(r.r,'build_arrays',side_effect=AssertionError('no model inputs')),
                  patch.object(r,'train_run',side_effect=AssertionError('no training'))):
                r.prepare(output);r.verify(output)
                with self.assertRaisesRegex(ValueError,'overwrite'):r.prepare(output)
            self.assertFalse((output/'execution').exists())
            plan=r.read(output/'plan.json')
            self.assertIn('research_program/triadic_partner_ecology_study/seed_selection_receipt.json',plan['sources'])
            self.assertIn('research_program/triadic_partner_ecology_study/design_test_receipt.json',plan['sources'])

    def test_tamper_or_started_execution_rejected_before_spawn(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'run';r.prepare(output)
            with patch.object(r.multiprocessing,'get_context',side_effect=AssertionError('no workers')):
                (output/'execution').mkdir()
                with self.assertRaisesRegex(ValueError,'Never resume'):r.execute(output)
                (output/'execution').rmdir()
                source=output/'source_snapshot/research_program/triadic_partner_ecology_study/design.py'
                source.write_text(source.read_text()+'\n# changed snapshot\n')
                with self.assertRaisesRegex(ValueError,'snapshot changed'):r.execute(output)
                self.assertFalse((output/'execution').exists())

    def test_mock_worker_failure_stops_only_batch_handles_and_leaves_receipt(self):
        class Process:
            next_pid=100
            def __init__(self,*,target,args,name):
                self.name=name;self.pid=Process.next_pid;Process.next_pid+=1;self.exitcode=None;self.events=[]
            def start(self):
                self.events.append('start')
                if self.pid==100:self.exitcode=1
            def is_alive(self):return self.exitcode is None
            def terminate(self):self.events.append('terminate');self.exitcode=-15
            def join(self,timeout=None):self.events.append('join')
            def kill(self):self.events.append('kill');self.exitcode=-9
        class Context:
            def __init__(self):self.processes=[]
            def Process(self,**kwargs):
                value=Process(**kwargs);self.processes.append(value);return value
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'run';r.prepare(output);context=Context()
            outsider=Process(target=None,args=(),name='unrelated');Process.next_pid=100
            with patch.object(r.multiprocessing,'get_context',return_value=context) as spawn:
                with self.assertRaisesRegex(RuntimeError,'worker failed'):r.execute(output)
            spawn.assert_called_once_with('spawn')
            self.assertEqual(len(context.processes),4)
            self.assertTrue(all('terminate' in p.events for p in context.processes[1:]))
            self.assertEqual(outsider.events,[])
            self.assertEqual(r.read(output/'execution/status.json')['status'],'failed')
            self.assertTrue((output/'execution/failure.json').is_file())
            self.assertFalse((output/'execution/results.json').exists())


if __name__=='__main__':unittest.main()
