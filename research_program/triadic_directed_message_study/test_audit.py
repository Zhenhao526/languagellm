"""Static/synthetic audit fixtures. No trained networks or model calls."""
import copy,itertools,unittest
from unittest.mock import patch
import numpy as np
from . import audit_execution as a


def toy(extra=False):
    strata=list(itertools.product(range(3),range(3)))
    if extra:strata.insert(0,(0,0))
    G=len(strata)
    return dict(partition='fixture',group_count=G,n_backgrounds=1,sender=[x[0] for x in strata],axis_index=[x[1] for x in strata],
        compatible_d=[[[0,1],[1,0]] for _ in strata])


def selective(cases):
    p=np.empty((cases['group_count'],1,2,2,2,2))
    for g,c,r,d,k in itertools.product(range(cases['group_count']),range(2),range(2),range(2),range(2)):
        p[g,0,c,r,d,k]=float(d==cases['compatible_d'][g][c][k])
    return p

class AuditFixtures(unittest.TestCase):
    def test_static_support_and_budget(self):
        self.assertEqual(len(a.references()),6)
        spec=a.read(a.ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json')['partitions']
        cc={p:a.build_cases(s) for p,s in spec.items()};b=a.budget(spec,cc)
        self.assertEqual(b['total_new_module_samples'],100528128);self.assertEqual(b['intervention_rows'],12109824)
        for part,c in cc.items():
            held=part in ('new_needs','new_needs_and_layouts');self.assertEqual(c['group_count'],168 if held else 708)
            counts=[sum(s==actor and x==axis for s,x in zip(c['sender'],c['axis_index'])) for actor,axis in itertools.product(range(3),repeat=2)]
            self.assertEqual(counts,([20,20,16] if held else [76,76,84])*3)
            for mode in ('cross','sham'):
                full=a.expand(c,mode);n=len(full['sender']);chunk=a.expand(c,mode,7,31)
                for key in chunk:np.testing.assert_array_equal(chunk[key],full[key][7:31])
                if mode=='sham':np.testing.assert_array_equal(full['receiver_state_indices'],full['donor_state_indices'])
                else:self.assertTrue(np.all(full['donor_background_index']!=full['background_index']))
    def test_prepared_all_partition_case_metadata(self):
        run=a.HERE/'results/directed_001'
        self.assertEqual(a.sha(run/'plan.json'),'0e880e120b264a7acebb06ec149eab0dc620810ba33a4939bbeaa842a7b05e1b')
        prepared=a.read(run/'prepared.json')
        for part,spec in prepared['partitions'].items():
            independent=a.build_cases(spec)
            a.compare(prepared['cases'][part],independent,'Independent full structural metadata '+part+'/')
        self.assertEqual(prepared['budget'],a.budget(prepared['partitions'],prepared['cases']))
    def test_four_way_min_after_own_endpoint_mean(self):
        c=toy();p=selective(c);summary,arrays=a.score(c,p.reshape(-1,2));self.assertEqual(summary['L'],1);self.assertEqual(summary['D'],1)
        p[:,:,:,1,:,:]=1-p[:,:,:,1,:,:]
        summary,_=a.score(c,p.reshape(-1,2));self.assertEqual(summary['L'],0);self.assertEqual(summary['D'],0)
    def test_monotone_suppression_counterexample(self):
        c=toy();p=np.empty((9,1,2,2,2,2));table=np.array([[[.8,.2],[.4,.3]],[[.4,.3],[.8,.2]]])
        for g,context,r,d,k in itertools.product(range(9),range(2),range(2),range(2),range(2)):p[g,0,context,r,d,k]=table[context,k,d]
        summary,arrays=a.score(c,p.reshape(-1,2));self.assertAlmostEqual(summary['D'],.25);self.assertAlmostEqual(summary['L'],-.1)
        np.testing.assert_allclose(arrays['delta'][0,0],[[.6,-.1],[-.1,.6]])
    def test_nine_strata_equal_not_pooled(self):
        c=toy(True);p=selective(c);p[:2]=.5;summary,_=a.score(c,p.reshape(-1,2))
        self.assertAlmostEqual(summary['L'],8/9);self.assertNotAlmostEqual(summary['L'],.8)
    def test_pure_route_and_mock_six_head_contract(self):
        state=np.array([[12,12,15,0,1,2,3,1,2,3]],np.int16);native=np.zeros((1,2,3,4),np.int8);sender=np.array([0]);cue=np.full((1,4),6,np.int8)
        route=a.first_routes(native,sender,cue);np.testing.assert_array_equal(route[0,0],a.message.route(native[:,0],True)[0,0])
        for recipient in (1,2):
            for pos in range(4):self.assertEqual(route[0,recipient,pos*8+6],1)
        second_calls=[];action_calls=[]
        def fake_sender(net,x):
            second_calls.append(x.copy());p=np.zeros((len(x),4,8));p[:,:,7]=1;return p
        def fake_action(net,x):
            action_calls.append(x.copy());p=np.full((len(x),17),1/17);return p,np.log(p)
        with patch.object(a.message,'probabilities',side_effect=fake_sender),patch.object(a,'action_distribution',side_effect=fake_action):
            result=a.replay([object() for _ in range(9)],state,native,sender,cue)
        self.assertEqual(len(second_calls),3);self.assertEqual(len(action_calls),3);self.assertEqual(result['independent_module_samples'],6)
        self.assertTrue(np.all(result['generated_messages'][:,1]==7));self.assertTrue(np.all(result['delivered_tokens'][:,1]==7))
        self.assertTrue(np.all(result['delivery_visibility']));np.testing.assert_array_equal(native,np.zeros_like(native))
    def test_recipient_event_excludes_wait(self):
        rng=np.random.default_rng(123);p=rng.uniform(.2,1,(6,3,17));p/=p.sum(-1,keepdims=True)
        sender=np.array([0,1,2,0,1,2]);recipients=np.array([[j for j in range(3) if j!=s] for s in sender]);got=a.recipient_probabilities(p,sender,recipients)
        for row in range(6):
            for slot,j in enumerate(recipients[row]):
                total=0
                for choice in range(1,17):
                    peer=[k for k in range(3) if k!=j][(choice-1)%2]
                    if peer==sender[row]:total+=p[row,j,choice]
                self.assertAlmostEqual(got[row,slot],total)
        p[:]=0;p[:,:,0]=1;np.testing.assert_array_equal(a.recipient_probabilities(p,sender,recipients),0)
    def test_unique_primary_is_final_L_not_change_or_D(self):
        values={(s,t):dict(L=-.2 if t==0 else (-.03 if i%2 else .01),D=.9) for i,s in enumerate(a.SEEDS) for t in a.TIMES}
        result=a.primary(values);self.assertAlmostEqual(result['mean_L_final'],-.01);self.assertAlmostEqual(result['auxiliary']['mean_L_final_minus_initial'],.19)
        self.assertEqual(len(result['by_seed']),4);values.pop((a.SEEDS[0],0))
        with self.assertRaises(Exception):a.primary(values)

if __name__=='__main__':unittest.main()
