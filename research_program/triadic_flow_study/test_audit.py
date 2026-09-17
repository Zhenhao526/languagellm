"""Pure support and synthetic numerical fixtures; no trained policy reads."""
import itertools,unittest
from unittest.mock import patch
import numpy as np
from . import audit_execution as a


def toy(extra=False):
    pairs=list(itertools.product(range(3),repeat=2))
    if extra:pairs.insert(0,(0,0))
    return dict(partition='fixture',group_count=len(pairs),n_backgrounds=1,sender=[v[0] for v in pairs],axis_index=[v[1] for v in pairs])

class AuditFixtures(unittest.TestCase):
    def test_full_static_support_and_production_metadata(self):
        from . import cases as declared
        self.assertEqual(len(a.references()),7)
        spec=a.read(a.ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json')['partitions'];total=0
        for p,s in spec.items():
            independent=a.build_cases(s);production=declared.build_cases(s);a.compare(production,independent,'All structural fields '+p)
            actual=a.expand(independent);expected=declared.expand(production);self.assertEqual(set(actual),set(expected))
            for key in actual:np.testing.assert_array_equal(actual[key],expected[key])
            partial=a.expand(independent,7,39)
            for key,v in partial.items():np.testing.assert_array_equal(v,actual[key][7:39])
            B=independent['n_backgrounds'];bg=actual['receiver_state_indices']%B
            for key in ('incoming_donor_state_indices','outgoing_donor_state_indices'):np.testing.assert_array_equal(actual[key]%B,bg)
            self.assertEqual(np.count_nonzero(actual['sham']),independent['flow_cells']//4);total+=independent['flow_cells']
        self.assertEqual(total,2018304);self.assertEqual(total*4*6,48439296)
    def test_route_preserves_self_and_sender_identity(self):
        native=np.zeros((4,2,3,4),np.int8);native[:,0,1]=1;native[:,0,2]=2
        incoming=np.broadcast_to(np.arange(3)[None,:,None]+3,(4,3,4)).astype(np.int8);outgoing=np.full((4,4),7,np.int8)
        ii=np.array([0,0,1,1]);oo=np.array([0,1,0,1]);route,delivery=a.first_routes(native,np.zeros(4,np.int8),incoming,outgoing,ii,oo)
        for row in range(4):
            for viewer,sender in itertools.product(range(3),repeat=2):
                target=native[row,0,sender]
                if viewer==0 and sender!=0 and ii[row]:target=incoming[row,sender]
                if viewer!=0 and sender==0 and oo[row]:target=outgoing[row]
                np.testing.assert_array_equal(delivery[row,viewer,sender],target)
                self.assertEqual(route[row,viewer,32*sender:32*(sender+1)].sum(),4)
        self.assertTrue(np.all(route[:,:,96:]==1));np.testing.assert_array_equal(route[0],a.message.route(native[:1,0],True)[0])
    def test_mock_six_head_contract(self):
        s=np.array([[12,12,15,0,1,2,3,1,2,3]],np.int16);native=np.zeros((1,2,3,4),np.int8);calls=[]
        def sender(net,x):calls.append(('sender',x.copy()));p=np.zeros((len(x),4,8));p[:,:,6]=1;return p
        def action(net,x):calls.append(('action',x.copy()));p=np.full((len(x),17),1/17);return p,np.log(p)
        with patch.object(a.message,'probabilities',side_effect=sender),patch.object(a,'action_distribution',side_effect=action):
            replay=a.replay([object() for _ in range(9)],s,native,np.array([0]),np.full((1,3,4),4,np.int8),np.full((1,4),5,np.int8),np.array([1]),np.array([1]))
        self.assertEqual([c[0] for c in calls],['sender']*3+['action']*3);self.assertEqual(replay['independent_module_samples'],6)
        self.assertTrue(np.all(replay['generated_messages'][:,1]==6));self.assertTrue(np.all(replay['delivered_tokens'][:,1]==6))
    def test_exact_success_integrates_third_actor_and_event_exhaustion(self):
        spec=a.read(a.ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json')['partitions']['new_needs_and_layouts']
        states=a.env.packed(spec)[:3];p=np.full((3,3,17),1/17);stats,target=a.exact_responses(states,p)
        independent=a.prior.conditional_statistics(p,a.env.rewards(states),'reciprocal')
        np.testing.assert_allclose(stats['full_success_probability'],1/289);np.testing.assert_allclose(stats['execution'].sum(-1),1)
        np.testing.assert_allclose(stats['full_success_probability'],independent['conditional_exact_full_success_probability'])
        np.testing.assert_allclose(stats['expected_reward'],independent['conditional_exact_expected_reward'])
        for row in range(3):
            actions=a.env.JOINT[target[row]];p[row]=0
            for actor in range(3):p[row,actor,int(actions[actor]) if actions[actor] else 16]=1
        stats,_=a.exact_responses(states,p);np.testing.assert_array_equal(stats['full_success_probability'],1)
    def test_wait_preserved_and_material_uses_original_layout(self):
        states=np.array([[12,12,15,3,2,1,0,1,2,3]],np.int16);p=np.zeros((1,3,17));p[:,:,0]=.25;p[:,:,1]=.75
        result=a.action_responses(states,p,np.array([1]))
        for key,x in result.items():np.testing.assert_allclose(x[:,:,0],.25);np.testing.assert_allclose(x.sum(-1),1)
        np.testing.assert_allclose(result['material'][:,:,4],.75);np.testing.assert_allclose(result['kind'][:,:,2],.75)
        for slot in range(3):self.assertEqual(result['partner'][0,slot,slot+1],0)
    def test_pairing_and_equal_strata_not_pooled(self):
        cases=toy(True);v=np.zeros((10,1,2,2,2,2,2));v[:,:,:,:,1,:,0]=1;v[:,:,:,:,:,1,1]=2;v[:2]=0
        result,arrays=a.score(cases,v.reshape(-1,2));np.testing.assert_allclose(result['I_given_O0'],[8/9,0]);np.testing.assert_allclose(result['O_given_I0'],[0,16/9]);np.testing.assert_array_equal(result['interaction'],[0,0])
        self.assertEqual(arrays['S00'].shape,(10,1,2));self.assertNotAlmostEqual(result['I_given_O0'][0],.8)

    def test_complete_synthetic_record_wrapper(self):
        cases=toy();cases.update(group_need_indices=[[[0,1],[2,3]]]*9,truth_pair_indices=[[[0,1],[1,0]]]*9,recipients=[[j for j in range(3) if j!=who] for who in cases['sender']],compatible_d=[[[0,1],[1,0]]]*9,sham_cells=36)
        spec=a.read(a.ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json')['partitions']['new_needs_and_layouts'];states=a.env.packed(spec)[:4]
        native=np.zeros((4,2,3,4),np.int8);bp=np.full((4,3,17),1/17);bank=dict(states=states,messages=native,action_probabilities=bp,action_indices=np.zeros((4,3),np.int16))
        meta=a.expand(cases);ri=meta['receiver_state_indices'];n=len(ri);p=bp[ri];act=bank['action_indices'][ri];worlds=states[ri];exact,_=a.exact_responses(worlds,p)
        data=dict(meta);data.update(receiver_states=worlds,generated_messages=native[ri],delivered_tokens=np.zeros((n,2,3,3,4),np.int8),delivery_visibility=np.ones((n,2,3,3),bool),incoming_donor_W1=np.zeros((n,3,4),np.int8),outgoing_donor_W1=np.zeros((n,4),np.int8),action_probabilities=p,action_indices=act,exact_full_success_probability=exact['full_success_probability'],expected_native_reward=exact['expected_reward'],execution_probabilities=exact['execution'],changed_W1_symbols=np.zeros((n,2),np.int8),response_actor_agents=np.array([[0,1,2],[1,0,2],[2,0,1]],np.int8)[meta['focus_actor']]);data.update(a.prior.settle(worlds,act,'reciprocal'))
        entry=dict(rows=n,new_module_samples=n*6,physical_summary=a.prior.summarize(worlds,act,'reciprocal'),native_replay=dict(all_messages_equal=True,all_actions_equal=True,max_probability_absolute_error=0.,checked_rows=n//4))
        def sender(net,x):out=np.zeros((len(x),4,8));out[:,:,0]=1;return out
        def action(net,x):out=np.full((len(x),17),1/17);return out,np.log(out)
        with patch.object(a.directed,'npz_record',return_value=data),patch.object(a.message,'probabilities',side_effect=sender),patch.object(a,'action_distribution',side_effect=action):values,receipt=a.flow_record(entry,'unused',cases,bank,[object() for _ in range(9)])
        self.assertEqual(len(values),10);self.assertEqual(receipt['rows'],144);self.assertEqual(receipt['independent_module_samples'],864);self.assertEqual(receipt['max_errors']['probability'],0.)
    def test_primary_is_signed_incoming_not_largest_effect(self):
        values={s:dict(S00=.2,S01=.9,S10=.1,S11=.8,I_given_O0=-.1,I_given_O1=-.1,O_given_I0=.7,O_given_I1=.7,interaction=0.) for s in a.SEEDS}
        result=a.primary(values);self.assertAlmostEqual(result['mean_S10_minus_S00'],-.1);self.assertEqual(result['contrast'],'I_given_O0');self.assertEqual(len(result['by_seed']),4)
        values.pop(a.SEEDS[0])
        with self.assertRaises(Exception):a.primary(values)
    def test_prepared_plan_static_cases_budget(self):
        run=a.HERE/'results/flow_001';self.assertEqual(a.sha(run/'plan.json'),'61a8e7ceacb9ab2aec4f3b20fc2c7b8988a29dd1da9c83a94baae5ddce31c544')
        prepared=a.read(run/'prepared.json');cc={p:a.build_cases(s) for p,s in prepared['partitions'].items()}
        a.compare(prepared['cases'],cc,'Prepared independent flow metadata/');self.assertEqual(prepared['budget'],a.budget(prepared['partitions'],cc))

if __name__=='__main__':unittest.main()
