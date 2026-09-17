"""Pure static/synthetic audit fixtures; zero trained policy or result reads."""
import copy,itertools,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from . import audit_execution as a


def fixture_spec():
    spec=a.read(a.ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json')['partitions']['new_needs_and_layouts']
    spec=copy.deepcopy(spec);spec['needs']=spec['needs'][:4];spec['layouts']=spec['layouts'][:1];spec['private_sites']=spec['private_sites'][:1];spec['world_count']=4;return spec


def fixture_runs():
    output=[]
    for index,seed in enumerate(a.SEEDS):
        for rule,live in itertools.product(a.RULES,a.LIVES):
            t=np.asarray(a.STEPS)/6000;value=.2+(index-7.5)*.008*t if rule=='reciprocal' and live else np.full(6,.2)
            trajectory=[dict(update=step,evaluation=dict(need_response={s:dict(Q=float(value[k]),Q_excess=float(value[k]-.05)) for s in ('native','common_reciprocal')})) for k,step in enumerate(a.STEPS)]
            output.append(dict(seed=seed,rule=rule,live=live,trajectory=trajectory))
    return output

class AuditFixtures(unittest.TestCase):
    def test_pinned_helpers_complete_support_and_budget(self):
        self.assertEqual(len(a.references()),6);specs=a.read(a.ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json')['partitions']
        cases={p:a.response.build_cases(s) for p,s in specs.items()};b=a.budget(specs,cases)
        self.assertEqual(b['evaluation_forward_module_samples'],600182784);self.assertEqual(b['training_forward_module_samples'],1769472000);self.assertEqual(b['generated_NPZ_files'],960)
        for p,c in cases.items():self.assertFalse(c['empty_strata']);self.assertEqual(len(c['strata']),9)
    def test_private_features_and_silent_self_routes(self):
        x=a.env.features(a.env.packed(fixture_spec()),'PL');self.assertTrue(np.all(x[:,:,53]==0))
        for viewer,other in itertools.product(range(3),repeat=2):
            if viewer!=other:self.assertTrue(np.all(x[:,viewer,7*other:7*other+7]==0))
        m=np.arange(12,dtype=np.int8).reshape(1,3,4)%8;silent=a.message.route(m,False)
        for actor in range(3):
            np.testing.assert_array_equal(silent[0,actor,96:],np.eye(3)[actor])
            for source in range(3):self.assertEqual(silent[0,actor,32*source:32*(source+1)].sum(),4 if actor==source else 0)
    def test_unequal_update_trapezoid_and_baseline(self):
        t=np.asarray(a.STEPS)/6000;cells={(r,l):np.zeros(6) for r,l in itertools.product(a.RULES,a.LIVES)};cells['reciprocal',True]=.2+.4*t
        got=a.trajectory_contrast(cells);self.assertAlmostEqual(got['centered_AUC'],.2);self.assertAlmostEqual(got['raw_AUC'],.4);self.assertAlmostEqual(got['baseline_DiD'],.2)
        self.assertNotAlmostEqual(np.mean(got['centered_DiD']),got['centered_AUC'])
        cells['reciprocal',True]=np.full(6,.2);self.assertEqual(a.trajectory_contrast(cells)['centered_AUC'],0.)
    def test_extreme_centered_did_is_not_clipped(self):
        ones=np.ones(6);zero=np.zeros(6);cells={('reciprocal',True):ones.copy(),('reciprocal',False):zero.copy(),('strict',True):zero.copy(),('strict',False):ones.copy()}
        for key in cells:cells[key][0]=1-cells[key][0]
        got=a.trajectory_contrast(cells);self.assertEqual(got['centered_DiD'][1],4.);self.assertGreater(got['centered_AUC'],3.)
    def test_primary_16_units_signed_and_production_agreement(self):
        from . import metrics
        runs=fixture_runs();own=a.primary(runs);declared=metrics.primary(runs);a.compare(declared,own,'Independent synthetic primary/')
        self.assertEqual(own['statistics']['df'],15);self.assertAlmostEqual(own['mean_centered_AUC'],0.)
        values=[r['measures']['native_Q']['centered_AUC'] for r in own['by_seed']];self.assertLess(min(values),0);self.assertGreater(max(values),0)
        self.assertAlmostEqual(own['statistics']['standard_error'],np.std(values,ddof=1)/4)
        runs.pop()
        with self.assertRaises(Exception):a.primary(runs)
    def test_ari_relabeling_and_degenerate_partitions(self):
        self.assertEqual(a.adjusted_rand([0,0,1,1],[7,7,4,4]),1.)
        self.assertAlmostEqual(a.adjusted_rand([0,0,1,1],[0,1,0,1]),-.5)
        self.assertEqual(a.adjusted_rand([0,0,0],[7,7,7]),1.)
        self.assertEqual(a.adjusted_rand([0,0,0],[1,2,3]),0.)
        self.assertEqual(a.adjusted_rand([1,2,3],[3,8,4]),1.);self.assertEqual(a.adjusted_rand([1],[2]),1.)
    def test_conditional_entropy_and_raw_form_are_distinct(self):
        from . import metrics
        spec=dict(world_count=8,needs=list(range(4)),layouts=[0,1],private_sites=[0],state_order='need-major, then layout, then owner')
        before=np.zeros((8,2,3,4),np.int8);before[1::2]=1;after=(before+3).astype(np.int8)
        snap=a.message_snapshot(before,spec);transition=a.message_transition(before,after,spec)
        a.compare(metrics.message_snapshot(before,spec),snap,'Independent synthetic entropy/');a.compare(metrics.message_transition(before,after,spec),transition,'Independent synthetic ARI/')
        self.assertEqual(len(snap['panels']),6)
        for s,t in zip(snap['panels'],transition['panels']):
            self.assertEqual(s['conditional_entropy_bits'],0);self.assertEqual(s['global_observed_packet_count'],2)
            self.assertEqual(t['packet_equal_rate'],0);self.assertEqual(t['token_hamming_fraction'],1);self.assertEqual(t['global_ARI'],1);self.assertEqual(t['mean_background_ARI'],1)
        changing=before.copy();changing[::4]+=2
        variation=a.message_transition(before,changing,spec);a.compare(metrics.message_transition(before,changing,spec),variation,'Independent nonzero transition entropy/')
        for panel in variation['panels']:
            self.assertEqual(panel['conditional_entropy_bits_before'],0.);self.assertAlmostEqual(panel['conditional_entropy_bits_after'],.5)
    def test_complete_saved_evaluation_with_mock_forward(self):
        spec=fixture_spec();states=a.env.packed(spec);cases=a.response.build_cases(spec);n=len(states);ids=np.arange(n,dtype=np.int64)
        p=np.random.default_rng(1902).uniform(.2,1,(n,3,17));p/=p.sum(-1,keepdims=True);act=p.argmax(-1).astype(np.int16);messages=np.zeros((n,2,3,4),np.int8)
        with TemporaryDirectory() as tmp:
            for rule,live in itertools.product(a.RULES,a.LIVES):
                data=dict(states=states,state_indices=ids,messages=messages,action_probabilities=p,action_indices=act)
                native=a.prior.settle(states,act,rule);common=a.prior.settle(states,act,'reciprocal');strict=a.prior.settle(states,act,'strict');data.update(native)
                for prefix,settled in (('common_reciprocal__',common),('strict__',strict)):data.update({prefix+k:v for k,v in settled.items()})
                terms=a.prior.conditional_statistics(p,a.env.rewards(states),rule);data.update({k:terms[k] for k in a.CONDITIONAL})
                path=Path(tmp)/f'{rule}_{live}.npz';np.savez_compressed(path,**data)
                entry=dict(path=str(path),data_sha256=a.sha(path),information='PL',live=live,rule=rule,scope='complete_partition',worlds=n,forward_module_samples=n*9,state_indices_sha256=a.array_sha(ids),
                    native=a.prior.summarize(states,act,rule),strict=a.prior.summarize(states,act,'strict'),common_reciprocal=a.prior.summarize(states,act,'reciprocal'),need_response=dict(native=a.response.metrics(cases,native['actual_pair_index']),common_reciprocal=a.response.metrics(cases,common['actual_pair_index'])))
                for key,field in zip(a.CONDITIONAL,('expected_reward_given_greedy_messages','full_probability_given_greedy_messages','execution_probability_given_greedy_messages','full_posterior_mass_given_greedy_messages')):entry[field]=float(data[key].mean())
                calls=[]
                def fake(nets,s,info,route):calls.append((len(s),info,route));return messages,p,np.log(p),0
                with patch.object(a.old,'forward_final',side_effect=fake):_,receipt,_=a.evaluate_saved(entry,path,states,rule,live,object(),cases)
                self.assertEqual(calls,[(n,'PL',live)]);self.assertEqual(receipt['independent_module_samples'],n*9);self.assertEqual(receipt['max_errors']['probability'],0)
                bad=copy.deepcopy(entry);bad['live']=not live
                with patch.object(a.old,'forward_final',side_effect=fake):
                    with self.assertRaises(Exception):a.evaluate_saved(bad,path,states,rule,live,object(),cases)

    def test_t15_quantile_independent_density_quadrature(self):
        self.assertAlmostEqual(a.t15_cdf(a.T15),.975,places=14)
        self.assertLess(a.t15_cdf(a.T15-1e-8),.975);self.assertGreater(a.t15_cdf(a.T15+1e-8),.975)
    def test_live_training_rows_keep_nonzero_sender_terms(self):
        expected={k:'a'*64 for k in ('world_uniforms_sha256','batch_indices_sha256','batch_states_sha256','sample_uniforms_sha256')}
        for rule,live in itertools.product(a.RULES,a.LIVES):
            row=dict(seed=60101,rule=rule,condition=a.condition(rule,live),update=1,settlement_rule=rule,partial_success_utility=.5,entropy_coefficient=.001,mean_log_expected_utility=-1.4,mean_actor_entropy=1.2,mean_F=-1.3988,receiver_loss=1.3988,mean_expected_utility=.25,mean_native_expected_reward=.25,mean_full_success_probability=.1,mean_partial_success_probability=.3,mean_execution_probability=.7,mean_full_success_posterior_mass=.4,mean_partial_success_posterior_mass=.6,min_log_expected_utility=-2.,max_log_expected_utility=-1.,zero_float_expected_utility_states=0,sender_advantage_mean=0.,sender_advantage_abs_mean=.3,sender_advantage_max_abs=.5,sender_advantage_squared_mean=.11,sender_mean_complete_log_score=-20.,sampled_messages_sha256='b'*64,gradient_norm=10.,gradient_clip_scale=.5,**expected)
            a.training_row(row,60101,rule,live,1,expected)
            wrong=copy.deepcopy(row);wrong['sender_advantage_mean']=.1
            with self.assertRaises(Exception):a.training_row(wrong,60101,rule,live,1,expected)

    def test_prepared_complete_schema_support_and_source_bindings(self):
        run=a.HERE/'results/formation_001';self.assertEqual(a.sha(run/'plan.json'),'b3ae053bca41e5acf60fa9108835cfe5bec85f2c6638cc48b2a7db9e5f00354f')
        plan=a.read(run/'plan.json');prepared=a.read(run/'prepared.json');self.assertEqual(a.sha(run/'prepared.json'),plan['prepared_sha256'])
        for p,digest in plan['source_sha256'].items():
            self.assertEqual(a.sha(p),digest);self.assertEqual(a.sha(run/'source_snapshot'/Path(p).relative_to(a.ROOT)),digest)
        cases={p:a.response.build_cases(s) for p,s in prepared['partitions'].items()};a.compare(prepared['need_response_cases'],cases,'Independent prepared Q cases/')
        self.assertEqual(prepared['budget'],a.budget(prepared['partitions'],cases));config=plan['config']
        self.assertEqual(config['seeds'],list(a.SEEDS));self.assertEqual(config['conditions'],[a.condition(r,l) for r,l in itertools.product(a.RULES,a.LIVES)])
        self.assertEqual(config['checkpoints'],list(a.STEPS));self.assertEqual(config['evaluation_checkpoints'],list(a.STEPS));self.assertEqual(config['partitions'],list(a.PARTS))
        self.assertEqual(config['updates'],6000);self.assertEqual(config['batch_size'],256);self.assertFalse(config['automatic_followon_experiment'])
        self.assertEqual(config['primary'],'six_point_native_Q_communication_rule_DiD_minus_initial_trapezoid_over6000_then_mean16_blocks')

if __name__=='__main__':unittest.main()
