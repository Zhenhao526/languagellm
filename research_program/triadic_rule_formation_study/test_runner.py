"""Bounded static/fake fixtures; no trained policies or optimizer/NN calls."""
from copy import deepcopy
from pathlib import Path
import json,tempfile,unittest
from unittest.mock import patch
import numpy as np
from . import runner as r
from research_program.triadic_need_response_study.test_cases import spec

COST=dict(fake_forward_calls=0,fake_module_samples=0,real_forward_calls=0,real_module_samples=0,
          formal_checkpoint_loads=0,formal_result_reads=0,optimizer_calls=0,training_updates=0)


def fake_forward(net,x):
    COST['fake_forward_calls']+=1;COST['fake_module_samples']+=len(x)
    if net%3==2:
        # AB match while C proposes: strict cannot execute, reciprocal can.
        choice=16 if net==8 else 1;logits=np.full((len(x),17),-4.);logits[:,choice]=4.
    else:
        value=np.rint(x@(1+np.arange(x.shape[1])%7)).astype(int);logits=np.full((len(x),4,8),-4.)
        for token in range(4):logits[np.arange(len(x)),token,(value+net+token)%8]=4.
        logits=logits.reshape(len(x),32)
    return logits,None


def fixture():
    source=spec();return source,r.make_arrays(source),r.cases.build_cases(source)


def fake_records():
    rows=[]
    # Full frozen support counts, without reading any learned output.
    counts={p:s['world_count'] for p,s in r.read(r.ORIGINAL/'prepared.json')['partitions'].items()}
    for seed in r.SEEDS:
        for condition in r.CONDITIONS:
            rule,live=r.split_condition(condition);trajectory=[]
            for update in r.STEPS:
                t=update/6000
                q=.2+(.1+.3*t if rule=='reciprocal' and live else .05+.1*t if rule=='strict' and live else 0.)
                evaluation=dict(path=f'{seed}/{condition}/trajectory_{update}',data_sha256=f'{seed}_{condition}_{update}',rule=rule,live=live,
                    worlds=counts[r.TARGET],forward_module_samples=9*counts[r.TARGET],
                    need_response=dict(native=dict(Q=q,Q_excess=-.1),common_reciprocal=dict(Q=.3,Q_excess=.1)))
                trajectory.append(dict(update=update,evaluation=evaluation))
            final={p:dict(path=f'{seed}/{condition}/final_{p}',data_sha256='fixture',worlds=counts[p],forward_module_samples=9*counts[p]) for p in r.PARTS if p!=r.TARGET}
            final[r.TARGET]=dict(trajectory[-1]['evaluation'],alias_of='trajectory_update_6000',additional_forward_module_samples=0)
            rows.append(dict(seed=seed,condition=condition,rule=rule,live=live,updates=6000,trajectory=trajectory,final=final))
    return rows


class RunnerTests(unittest.TestCase):
    def test_complete_fixed_budget_support_and_grid(self):
        static=r.prepared();b=static['budget']
        self.assertEqual(r.SEEDS,tuple(range(60101,60117)));self.assertEqual(len(r.CONDITIONS),4)
        self.assertEqual((b['training_updates'],b['checkpoints'],b['evaluation_files']),(384000,384,576))
        self.assertEqual((b['training_forward_module_samples'],b['evaluation_forward_module_samples'],b['total_forward_module_samples']),
                         (1769472000,600182784,2369654784))
        self.assertEqual(b['evaluation_worlds'],66686976);self.assertEqual(b['generated_NPZ_files'],960)
        self.assertEqual((b['closed_files'],b['monitor_subset_files'],b['additional_final_target_forward_samples']),(0,0,0))
        self.assertEqual(b['final_target_aliases'],64)
        for c in static['need_response_cases'].values():self.assertEqual(c['empty_strata'],[])
        for condition,expected in zip(r.CONDITIONS,[('strict',True),('strict',False),('reciprocal',True),('reciprocal',False)]):
            self.assertEqual(r.split_condition(condition),expected)
        with self.assertRaises(ValueError):r.split_condition('FI_live')

    def test_pl_mask_and_fresh_paired_random_streams(self):
        source,arrays,_=fixture();x=arrays['x_PL']
        self.assertEqual(x.shape,(source['world_count'],3,54));np.testing.assert_array_equal(x[:,:,53],0)
        for actor in range(3):
            for other in range(3):
                if actor!=other:np.testing.assert_array_equal(x[:,actor,7*other:7*other+7],0)
        streams=[r.make_random_streams(r.SEEDS[0]) for _ in r.CONDITIONS]
        self.assertEqual(len({id(w) for w,m in streams}),4)
        for key in streams[0][1]:self.assertEqual(len({id(m[key]) for w,m in streams}),4)
        old_world=None
        for _ in range(3):
            worlds=[w.random((7,3)) for w,m in streams];messages=[r.core.draw_uniforms(m,7) for w,m in streams]
            for v in worlds[1:]:np.testing.assert_array_equal(v,worlds[0])
            for v in messages[1:]:np.testing.assert_array_equal(v,messages[0])
            self.assertEqual(messages[0].shape,(2,7,2,3,4))
            ids=[r.dataset.sample_indices(source,v) for v in worlds]
            for v in ids[1:]:np.testing.assert_array_equal(v,ids[0])
            if old_world is not None:self.assertFalse(np.array_equal(old_world,worlds[0]))
            old_world=worlds[0]

    @patch.object(r.core.base,'actor_forward',side_effect=fake_forward)
    def test_complete_fake_evaluation_dual_settlement_and_silent_own_routes(self,forward):
        source,arrays,cases=fixture();n=source['world_count'];snapshots=[];messages=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for condition in r.CONDITIONS:
                rule,live=r.split_condition(condition)
                with patch.object(r.core,'rollout',wraps=r.core.rollout) as rollout:
                    record,transcript=r.evaluate(tuple(range(9)),arrays,live,rule,np.arange(n),root/f'{condition}.npz',cases)
                    self.assertEqual(rollout.call_count,1);self.assertIs(rollout.call_args.args[2],live)
                self.assertEqual(record['worlds'],n);self.assertEqual(record['forward_module_samples'],9*n)
                self.assertEqual(record['native']['physical_execution_rate'],float(rule=='reciprocal'))
                self.assertEqual(record['strict']['physical_execution_rate'],0.)
                self.assertEqual(record['common_reciprocal']['physical_execution_rate'],1.)
                with np.load(record['path'],allow_pickle=False) as saved:
                    self.assertEqual(saved['action_probabilities'].shape,(n,3,17))
                    np.testing.assert_array_equal(saved['messages'],transcript)
                    np.testing.assert_array_equal(saved['strict__actual_pair_index'],-1)
                    np.testing.assert_array_equal(saved['common_reciprocal__actual_pair_index'],0)
                    np.testing.assert_array_equal(saved['actual_pair_index'],-1 if rule=='strict' else 0)
                    p=saved['action_probabilities'];full=(arrays['rewards']==1).argmax(-1)
                    plan=r.environment.STRUCTURAL_ACTIONS[full];expected=np.ones(n)
                    for actor in range(3):
                        factor=p[np.arange(n),actor,plan[:,actor]]
                        if rule=='reciprocal':factor=np.where(plan[:,actor]>0,factor,1.)
                        expected*=factor
                    np.testing.assert_allclose(saved['conditional_exact_full_success_probability'],expected,rtol=1e-13,atol=1e-16)
                snapshots.append(r.metrics.message_snapshot(transcript,source));messages.append(transcript)
            self.assertEqual(forward.call_count,36)
            self.assertEqual(len(snapshots),4)
            self.assertEqual(len(r.metrics.message_transition(messages[0],messages[1],source)['panels']),6)
            before=forward.call_count
            with self.assertRaises(ValueError):r.evaluate(None,arrays,True,'strict',np.arange(n-1),root/'subset.npz',cases)
            with self.assertRaises(ValueError):r.evaluate(None,arrays,True,'strict',np.arange(n)[::-1],root/'reorder.npz',cases)
            with self.assertRaises(ValueError):r.evaluate(None,arrays,True,'strict',np.arange(n),root/f'{r.CONDITIONS[0]}.npz',cases)
            self.assertEqual(forward.call_count,before)
        for live in (True,False):
            routes=r.core.routed_window(np.arange(12).reshape(1,3,4)%8,live)
            self.assertEqual(routes.shape,(1,3,99))
            for actor in range(3):
                self.assertEqual(routes[0,actor,96+actor],1.)
                self.assertGreater(routes[0,actor,32*actor:32*(actor+1)].sum(),0)
                if not live:
                    for other in range(3):
                        if actor!=other:
                            np.testing.assert_array_equal(routes[0,actor,32*other:32*(other+1)],0)
                            self.assertEqual(routes[0,actor,96+other],0.)

    def test_other_final_only_three_calls_target_alias_and_budget_count(self):
        source,arrays,cases=fixture();aa={p:arrays for p in r.PARTS};cc={p:cases for p in r.PARTS}
        endpoint=dict(path='/fixture/target6000',data_sha256='saved',rule='strict',live=True,worlds=len(arrays['packed_states']))
        with tempfile.TemporaryDirectory() as directory,patch.object(r,'evaluate',return_value=({'fixture':True},np.zeros((1,2,3,4)))) as evaluate:
            final=r.evaluate_other_final(None,aa,'strict_PL_live',cc,Path(directory),endpoint)
            self.assertEqual(evaluate.call_count,3)
            self.assertTrue(all(r.TARGET not in str(call.args[5]) for call in evaluate.call_args_list))
            self.assertEqual(final[r.TARGET]['path'],endpoint['path']);self.assertEqual(final[r.TARGET]['data_sha256'],endpoint['data_sha256'])
            self.assertEqual(final[r.TARGET]['additional_forward_module_samples'],0)
        rows=fake_records();measured=r.measured_budget(rows);budget=r.prepared()['budget']
        self.assertTrue(all(budget[k]==v for k,v in measured.items()))
        broken=deepcopy(rows);broken[0]['final'][r.TARGET]['data_sha256']='different'
        with self.assertRaises(ValueError):r.measured_budget(broken)

    def test_statistics_schema_uses_native_baseline_centered_DiD(self):
        answer=r.primary(fake_records())
        self.assertAlmostEqual(answer['mean_centered_AUC'],.1)
        self.assertEqual(answer['statistics']['n'],16)
        for block in answer['by_seed']:
            self.assertAlmostEqual(block['measures']['native_Q']['raw_AUC'],.15)
            self.assertAlmostEqual(block['measures']['native_Q']['endpoint_centered_DiD'],.2)
            self.assertAlmostEqual(block['measures']['common_reciprocal_Q']['centered_AUC'],0.)
        self.assertIs(r.primary,r.metrics.primary)

    def test_four_log_pairing_and_complete_update_requirement(self):
        seed=r.SEEDS[0];runs=[dict(seed=seed,condition=c,initial_parameter_sha256='same') for c in r.CONDITIONS]
        with tempfile.TemporaryDirectory() as directory,patch.object(r,'SEEDS',(seed,)):
            root=Path(directory)
            for condition in r.CONDITIONS:
                path=root/r.name(seed,condition);path.mkdir();rule,_=r.split_condition(condition)
                with (path/'training.jsonl').open('x') as stream:
                    for update in range(1,6001):
                        row=dict(update=update,seed=seed,condition=condition,rule=rule,entropy_coefficient=.01)
                        row.update({k:'paired' for k in ['world_uniforms_sha256','batch_indices_sha256','batch_states_sha256','sample_uniforms_sha256']})
                        stream.write(json.dumps(row)+'\n')
            r.verify_pairing(root,runs)
            path=root/r.name(seed,r.CONDITIONS[-1])/'training.jsonl';lines=path.read_text().splitlines()
            first=json.loads(lines[0]);first['sample_uniforms_sha256']='not_paired';lines[0]=json.dumps(first);path.write_text('\n'.join(lines)+'\n')
            with self.assertRaises(ValueError):r.verify_pairing(root,runs)
            runs[-1]['initial_parameter_sha256']='different'
            with self.assertRaises(ValueError):r.verify_pairing(root,runs)

    def test_fixture_prepare_freeze_and_existing_output_rejection(self):
        source=r.HERE/'runner.py';sources={str(source):r.sha(source)}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'prepared'
            with patch.object(r,'prepared',return_value={'budget':{},'fixture':True}),patch.object(r,'sources',return_value=sources):
                result=r.prepare(out);self.assertEqual(result['status'],'prepared_without_training');r.verify(out)
                with self.assertRaises(ValueError):r.prepare(out)
                (out/'source_snapshot'/source.relative_to(r.ROOT)).write_text('changed')
                with self.assertRaises(ValueError):r.verify(out)
            existing=root/'existing';existing.mkdir();(existing/'execution').mkdir()
            with patch.object(r,'verify',return_value=({},{})),patch.object(r.multiprocessing,'get_context') as spawn:
                with self.assertRaises(FileExistsError):r.execute(existing)
                self.assertEqual(spawn.call_count,0)


if __name__=='__main__':unittest.main()
