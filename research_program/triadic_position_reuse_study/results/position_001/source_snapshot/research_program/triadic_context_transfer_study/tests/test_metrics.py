"""Pure fixture tests: no checkpoint loading, actor forward, or training."""
from copy import deepcopy
from unittest import TestCase
import numpy as np
from research_program.triadic_context_transfer_study import metrics as m
from research_program.triadic_action_dependency_study import environment as env


def fixture(information='PL'):
    # C changes short -> long; A remains wood/L, B wood/R. Unique pair A/C.
    # Every corresponding target changes site at the donor layout.
    states = np.asarray([n + layout + (1, 2, 3)
        for n in ((0, 1, 6), (0, 1, 9))
        for layout in ((0, 1, 2, 3), (2, 3, 0, 1))], dtype=np.int16)
    actions = np.zeros((4, 3), dtype=np.int16)
    physics = m.native.settle_arrays(states, actions)
    pool = dict(states=states, action_indices=actions,
        action_probabilities=np.eye(17)[actions],
        messages=np.broadcast_to(np.arange(4)[:, None, None, None], (4, 2, 3, 4)).copy(),
        greedy_reward=physics['reward'], executed=physics['executed'], satisfied=physics['satisfied'])
    context = m.build_pool_context(pool, information)
    truth = context['truth_actions']
    spec = dict(endpoint_indices=np.array([[0, 2]]), correct_actions=truth[[[0, 2]]],
        donor_endpoint_indices=np.array([[[1, 3]]]), donor_correct_actions=truth[[[[1, 3]]]],
        listener=np.array([0]), sender=np.array([2]), axis=np.array([1]),
        eligible=np.array([[True]]), eligible_donor_count=np.array([1]),
        base_within_axis_weight=np.array([1.]), view_equal_LL=np.array([[False]]))
    return spec, context


def intervention_data(spec, context, mode, direction, choices, shift=0, markers=None):
    n = len(spec['endpoint_indices']); ids = spec['endpoint_indices'][:, direction]
    cfids = spec['endpoint_indices'][:, 1-direction]
    dpair = spec['donor_endpoint_indices'][shift] if shift >= 0 else spec['endpoint_indices']
    donor = dpair[:, direction if mode in ('remote_same_both', 'sham_both') else 1-direction]
    p = m.native.settle_arrays(context['pool']['states'][ids], choices)
    cf = m.native.settle_arrays(context['pool']['states'][cfids], choices)
    return dict(dataset_rows=np.arange(n), recipient_indices=ids, counterfactual_recipient_indices=cfids,
        donor_indices=donor.copy(), action_indices=np.asarray(choices), action_probabilities=np.eye(17)[choices],
        messages=context['pool']['messages'][ids].copy(),
        action_input_equals_donor=np.zeros((n, 3, 2), dtype=bool) if markers is None else markers,
        greedy_reward=p['reward'], executed=p['executed'], satisfied=p['satisfied'],
        counterfactual_reward=cf['reward'], counterfactual_satisfied=cf['satisfied'])


def weights_fixture():
    # Unequal eligible donor counts must not give easier rows more mass.
    eligible = np.tile(np.array([[1,1],[0,1],[0,0]],dtype=bool),(1,3))
    return dict(endpoint_indices=np.zeros((6,2),dtype=int), axis=np.repeat(np.arange(3),2),
        donor_endpoint_indices=np.zeros((3,6,2),dtype=int), eligible=eligible,
        eligible_donor_count=eligible.sum(0), base_within_axis_weight=np.full(6,.5))


class MetricsTests(TestCase):
    def test_actual_layout_truth_and_native_vs_counterfactual(self):
        spec, ctx = fixture()
        # A/C move wood/short at S0 or wood/long at S1, donor S2/S3.
        np.testing.assert_array_equal(ctx['truth_actions'][:,0],[2,10,6,14])
        choices=ctx['truth_actions'][[2]]
        data=intervention_data(spec,ctx,'remote_opposite_both',0,choices)
        result=m.cell_values(spec,ctx,data,mode='remote_opposite_both',shift=0,direction=0,live=True)
        self.assertEqual(result['target_apt'].tolist(),[True])
        self.assertEqual(result['current_apt'].tolist(),[False])
        self.assertEqual(result['native_reward'].tolist(),[.5])
        self.assertEqual(result['counterfactual_reward'].tolist(),[1.])
        self.assertEqual(result['conservative_target_apt'].tolist(),[True])
        s=env.State((0,1,6),(0,1,2,3),(1,2,3))
        a={name: env.all_actions(name)[int(choices[0,j])] for j,name in enumerate(env.AGENTS)}
        self.assertEqual(result['native_reward'][0],env.settle(s,a)['reward'])
        broken=deepcopy(spec);broken['correct_actions'][0,0,0]=10
        with self.assertRaisesRegex(ValueError,'Recipient truth'):
            m.cell_values(broken,ctx,data,mode='remote_opposite_both',shift=0,direction=0,live=True)

    def test_fixed_endpoint_identity_is_remapped_by_direction(self):
        spec, ctx = fixture(); pool=ctx['pool']
        # At direction1, marker endpoint1 is SAME, not opposite.
        target=int(ctx['truth_actions'][0,0])
        pool['action_indices'][3,0]=target
        pool['action_probabilities'][3,0]=np.eye(17)[target]
        marks=np.zeros((1,3,2),bool);marks[0,0,1]=True
        choices=np.array([[target,0,0]],dtype=int)
        data=intervention_data(spec,ctx,'remote_opposite_both',1,choices,markers=marks)
        got=m.cell_values(spec,ctx,data,mode='remote_opposite_both',shift=0,direction=1,live=True)
        self.assertTrue(got['input_equal_donor_same'][0])
        self.assertFalse(got['input_equal_donor_opposite'][0])
        self.assertFalse(got['conservative_gate'][0])
        self.assertTrue(got['target_apt'][0])

    def test_silent_alias_structural_zero_and_no_mutation(self):
        spec,ctx=fixture('LL'); before=deepcopy(ctx['pool'])
        for direction in (0,1):
            a=m.cell_values(spec,ctx,None,mode='remote_same_both',shift=0,direction=direction,live=False)
            b=m.cell_values(spec,ctx,None,mode='remote_opposite_both',shift=0,direction=direction,live=False)
            c=m.contrast_values(a,b)
            for key,value in c.items():
                if key.endswith('_gain'):np.testing.assert_array_equal(value,0)
        for key in before:np.testing.assert_array_equal(before[key],ctx['pool'][key])

    def test_shared_gate_includes_wrong_donor_copy_without_filtering(self):
        spec,ctx=fixture();target=int(ctx['truth_actions'][2,0])
        # A wrong donor-same action can equal the recipient counterfactual answer.
        ctx['pool']['action_indices'][1,0]=target
        ctx['pool']['action_probabilities'][1,0]=np.eye(17)[target]
        choices=ctx['truth_actions'][[2]]
        data=intervention_data(spec,ctx,'remote_opposite_both',0,choices)
        got=m.cell_values(spec,ctx,data,mode='remote_opposite_both',shift=0,direction=0,live=True)
        self.assertFalse(got['donor_same_correct'][0])
        self.assertTrue(got['copy_donor_same_target_apt'][0])
        self.assertTrue(got['target_apt'][0]);self.assertFalse(got['conservative_target_apt'][0])
        same=deepcopy(got);same['target_apt']=np.array([False]);same['conservative_target_apt']=np.array([False])
        delta=m.contrast_values(same,got)
        self.assertEqual(delta['target_apt_gain'][0],1)
        self.assertEqual(delta['conservative_target_apt_gain'][0],0)
        same['conservative_gate']=np.array([True])
        with self.assertRaisesRegex(ValueError,'shared quantity'):m.contrast_values(same,got)

    def test_eligible_uniform_within_recipient_and_axis_macro(self):
        spec=weights_fixture();acc=m.WeightedAccumulator();mass=np.zeros(6)
        values={'apt':np.tile([1.,0.],3)}
        for shift in range(3):
            for direction in range(2):
                w=m.support_weights(spec,shift,'eligible');mass+=w
                acc.update(values,spec['axis'],w)
        np.testing.assert_array_equal(mass,np.full(6,.5))
        got=acc.finish();self.assertEqual(got['macro']['apt'],.5)
        self.assertEqual(got['by_axis']['kind']['row_count'],12)
        # Multiplying a row's value by a gate keeps its denominator, even all-zero.
        acc=m.WeightedAccumulator()
        for shift in range(3):
            for direction in range(2):
                acc.update({'gated_apt':np.zeros(6)},spec['axis'],m.support_weights(spec,shift,'eligible'))
        self.assertEqual(acc.finish()['macro']['gated_apt'],0.)

    def test_all_donor_and_streaming_weights_missing_direction_refused(self):
        spec=weights_fixture();whole=m.WeightedAccumulator();chunk=m.WeightedAccumulator()
        vals={'one':np.ones(6),'signed':np.arange(6)/5-.5}
        for shift in range(3):
            for direction in range(2):
                whole.update(vals,spec['axis'],m.support_weights(spec,shift,'all_other'))
                for sl in (slice(0,3),slice(3,6)):
                    chunk.update({k:v[sl] for k,v in vals.items()},spec['axis'][sl],m.support_weights(spec,shift,'all_other',sl))
        a,b=whole.finish(),chunk.finish()
        for k in a['macro']:self.assertAlmostEqual(a['macro'][k],b['macro'][k],places=15)
        missing=m.WeightedAccumulator();missing.update({'one':np.ones(6)},spec['axis'],m.support_weights(spec,-1,'control'))
        with self.assertRaisesRegex(ValueError,'Incomplete'):missing.finish()
        bad=deepcopy(spec);bad['eligible_donor_count'][0]=0
        with self.assertRaisesRegex(ValueError,'lacking eligible'):m.support_weights(bad,0,'eligible')

    def test_cell_slice_retains_global_row_identity(self):
        spec,ctx=fixture()
        for key in ('endpoint_indices','correct_actions','listener','sender','axis','eligible_donor_count','base_within_axis_weight'):
            spec[key]=np.repeat(spec[key],3,axis=0)
        for key in ('donor_endpoint_indices','donor_correct_actions','eligible','view_equal_LL'):
            spec[key]=np.repeat(spec[key],3,axis=1)
        data=intervention_data(spec,ctx,'remote_same_both',0,np.zeros((3,3),dtype=int))
        full=m.cell_values(spec,ctx,data,mode='remote_same_both',shift=0,direction=0,live=True)
        piece=m.cell_values(spec,ctx,data,mode='remote_same_both',shift=0,direction=0,live=True,rows=slice(1,3))
        for key in full:np.testing.assert_array_equal(piece[key],full[key][1:3])
        data['dataset_rows'][1]=0
        with self.assertRaisesRegex(ValueError,'row identity'):
            m.cell_values(spec,ctx,data,mode='remote_same_both',shift=0,direction=0,live=True,rows=slice(1,3))

    def test_primary_requires_all_policies_preserves_negative_seeds(self):
        records=[]
        for i,seed in enumerate(m.SEEDS):
            for condition in m.CONDITIONS:
                value=0. if condition.endswith('silent') else ([.1,.2,.3,.4][i] if condition=='PL_live' else [.3,.1,.2,.5][i])
                contrast=dict(macro={'target_apt_gain':value,'conservative_target_apt_gain':value/2},
                    by_axis={a:{'weight_sum':1.} for a in m.AXES})
                records.append(dict(seed=seed,condition=condition,partitions={p:{s:{'contrast':deepcopy(contrast)}
                    for s in ('eligible','all_other')} for p in m.PARTITIONS}))
        got=m.primary_comparison(records)['primary']
        np.testing.assert_allclose([x['PL_minus_LL'] for x in got['seed_pairs']],[-.2,.1,.1,-.1])
        self.assertAlmostEqual(got['equal_seed_mean_PL_minus_LL'],-.025)
        with self.assertRaisesRegex(ValueError,'Exactly'):m.primary_comparison(records[:-1])
        records[0]['partitions'][m.PARTITIONS[-1]]['eligible']['contrast']['macro']['target_apt_gain']=.1
        with self.assertRaisesRegex(ValueError,'structural zero'):m.primary_comparison(records)

    def test_actual_generated_message_change_is_separate_from_action_change(self):
        spec,ctx=fixture()
        data=intervention_data(spec,ctx,'remote_same_both',0,np.zeros((1,3),dtype=int))
        data['messages'][0,1,0,0]=7
        got=m.cell_values(spec,ctx,data,mode='remote_same_both',shift=0,direction=0,live=True)
        self.assertTrue(got['any_generated_message_changed'][0])
        self.assertTrue(got['listener_own_second_message_changed'][0])
        self.assertFalse(got['listener_action_changed'][0])
        silent=m.cell_values(spec,ctx,None,mode='remote_same_both',shift=0,direction=0,live=False)
        self.assertFalse(silent['any_generated_message_changed'][0])
        self.assertFalse(silent['listener_own_second_message_changed'][0])

    def test_nonfinite_bad_saved_physics_or_donor_mapping_refused(self):
        spec,ctx=fixture();bad=deepcopy(ctx['pool']);bad['greedy_reward'][0]=1
        with self.assertRaisesRegex(ValueError,'physics mismatch'):m.build_pool_context(bad,'PL')
        data=intervention_data(spec,ctx,'remote_same_both',0,np.zeros((1,3),dtype=int))
        data['donor_indices'][0]=3
        with self.assertRaisesRegex(ValueError,'donor_indices'):
            m.cell_values(spec,ctx,data,mode='remote_same_both',shift=0,direction=0,live=True)
        a=m.WeightedAccumulator()
        with self.assertRaisesRegex(ValueError,'Invalid measurement'):
            a.update({'x':np.array([float('nan')])},np.array([0]),np.array([1.]))
