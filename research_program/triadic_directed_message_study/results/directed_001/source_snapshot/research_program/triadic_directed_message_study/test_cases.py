"""Native small-support and arithmetic fixtures; no policy reads or forwards."""
from copy import deepcopy
from itertools import product
import json
import unittest

import numpy as np

from . import cases as c
from research_program.triadic_need_response_study.test_cases import NEED_EDGES
from research_program.triadic_action_dependency_study import dataset as native


def spec(edges=NEED_EDGES):
    rows=set()
    for before,after in edges:
        sender=next(a for a in range(3) if before[a]!=after[a])
        others=[a for a in range(3) if a!=sender]
        for endpoint in (before,after):
            rows.add(endpoint);mirror=list(endpoint)
            mirror[others[0]],mirror[others[1]]=mirror[others[1]],mirror[others[0]]
            rows.add(tuple(mirror))
    needs=sorted(rows)
    return dict(partition='fixture',needs=[list(n) for n in needs],layouts=[[0,1,2,3],[3,1,0,2]],
                private_sites=[[1,2,3],[3,2,1]],world_count=len(needs)*4,state_order=c.STATE_ORDER)


def probabilities(built,deltas):
    """Realize given [G,B,context,recipient] contrasts around probability .5."""
    g=built['group_count'];b=built['n_backgrounds']
    delta=np.broadcast_to(deltas,(g,b,2,2))
    out=np.empty((g,b,2,2,2,2),float)
    for gi,bg,ctx,r,d,k in product(range(g),range(b),range(2),range(2),range(2),range(2)):
        out[gi,bg,ctx,r,d,k]=.5+(.5 if d==built['compatible_d'][gi][ctx][k] else -.5)*delta[gi,bg,ctx,k]
    return out.reshape(-1,2)


class DirectedCasesTests(unittest.TestCase):
    def test_mirror_groups_complete_unique_and_compatible_truth(self):
        built=c.build_cases(spec());self.assertEqual(built['empty_strata'],[])
        old=c.original.build_cases(spec());seen=set();needs=built['needs']
        for g,(contexts,sender,recipients) in enumerate(zip(built['group_need_indices'],built['sender'],built['recipients'])):
            self.assertEqual(recipients,[a for a in range(3) if a!=sender])
            self.assertEqual(len(set(np.asarray(contexts).ravel())),4)
            truth=built['truth_pair_indices'][g];self.assertEqual(truth[0],truth[1][::-1])
            self.assertNotEqual(*truth[0])
            for ctx in range(2):
                a,z=(needs[i] for i in contexts[ctx]);self.assertLess(a[sender],z[sender])
                self.assertEqual([p for p in range(3) if a[p]!=z[p]],[sender])
                seen.add((tuple(contexts[ctx]),sender,built['axis_index'][g]))
                mirror=[needs[i] for i in contexts[1-ctx]]
                for endpoint,row in enumerate((a,z)):
                    changed=list(row);changed[recipients[0]],changed[recipients[1]]=changed[recipients[1]],changed[recipients[0]]
                    self.assertEqual(changed,mirror[endpoint])
                for k,recipient in enumerate(recipients):
                    compatible=built['compatible_d'][g][ctx][k]
                    self.assertIn(recipient,c.PAIRS[truth[ctx][compatible]])
                    self.assertNotIn(recipient,c.PAIRS[truth[ctx][1-compatible]])
        expected={(tuple(e),s,a) for e,s,a in zip(old['edge_need_indices'],old['changed_person'],old['axis_index'])}
        self.assertEqual(seen,expected);self.assertEqual(2*built['group_count'],len(seen))
        self.assertEqual(sum(s['group_count'] for s in built['strata']),built['group_count'])
        json.dumps(built,allow_nan=False)

    def test_cross_sham_order_donors_and_full_partition_validation(self):
        source=spec();built=c.build_cases(source);b=built['n_backgrounds'];o=len(built['private_sites'])
        states=native.pack_states(source);indices=np.arange(len(states))
        self.assertEqual(c.validate_partition_arrays(built,states,indices)['status'],'valid')
        for mode,ds in (('cross',range(2)),('sham',(None,))):
            expanded=c.expand(built,mode);i=0
            for g,bg,ctx,r,d in product(range(built['group_count']),range(b),range(2),range(2),ds):
                packet=r if d is None else d;receiver=built['group_need_indices'][g][ctx][r]*b+bg
                db=built['donor_background_indices'][bg]
                donor=receiver if mode=='sham' else built['group_need_indices'][g][0][packet]*b+db
                for name,value in (('group_index',g),('background_index',bg),('context_index',ctx),('self_endpoint',r),
                                   ('packet_endpoint',packet),('receiver_state_indices',receiver),('donor_state_indices',donor)):
                    self.assertEqual(expanded[name][i],value)
                self.assertEqual(expanded['recipient_agents'][i].tolist(),built['recipients'][g])
                if mode=='cross':
                    self.assertEqual(db%o,bg%o);self.assertNotEqual(db//o,bg//o)
                    self.assertEqual(built['donor_background_indices'][db],bg)
                i+=1
            self.assertEqual(i,built[mode+'_cells']);self.assertTrue(all(len(v)==i for v in expanded.values()))
        for bad_states,bad_indices in ((states[::-1],indices),(states,indices[::-1]),(states[:-1],indices[:-1]),
                                      (states.astype(float),indices),(states,indices.astype(float))):
            with self.assertRaises(ValueError):c.validate_partition_arrays(built,bad_states,bad_indices)

    def test_perfect_identical_and_general_suppression_counterexample(self):
        built=c.build_cases(spec());g=built['group_count'];b=built['n_backgrounds']
        answer=c.metrics(built,probabilities(built,1));self.assertEqual(answer['L'],1);self.assertEqual(answer['D'],1)
        self.assertEqual(answer['mean_compatible_probability'],1);self.assertEqual(answer['mean_incompatible_probability'],0)
        self.assertEqual(answer['raw_group_background_counts']['positive_L'],g*b)
        equal=c.metrics(built,np.full((built['cross_cells'],2),.37));self.assertEqual(equal['L'],0);self.assertEqual(equal['D'],0)
        # Packet0 depresses both recipients in both contexts. Variable need
        # sensitivity makes D positive although no effect ever reverses sign.
        probs=np.zeros((g,b,2,2,2,2))
        for gi,ctx,k in product(range(g),range(2),range(2)):
            probs[gi,:,ctx,:,1,k]=.9 if built['compatible_d'][gi][ctx][k]==1 else .1
        out=c.metrics(built,probs.reshape(-1,2));self.assertAlmostEqual(out['D'],.4);self.assertAlmostEqual(out['L'],-.1)
        self.assertEqual(out['raw_group_background_counts']['negative_L'],g*b)
        self.assertEqual(set(out['per_group_background']),{'delta','compatible_probability','incompatible_probability','L','D'})
        self.assertEqual(np.asarray(out['per_group_background']['delta']).shape,(g,b,2,2));json.dumps(out,allow_nan=False)

    def test_average_endpoint_before_min_and_equal_nine_strata(self):
        built=c.build_cases(spec());g=built['group_count'];b=built['n_backgrounds']
        positive=probabilities(built,1).reshape(g,b,2,2,2,2)
        negative=probabilities(built,-1).reshape(g,b,2,2,2,2)
        positive[:,:,:,1]=negative[:,:,:,1]
        out=c.metrics(built,positive.reshape(-1,2));self.assertEqual(out['L'],0);self.assertEqual(out['D'],0)
        uneven=c.build_cases(spec(NEED_EDGES+[((0,14,23),(3,14,23))]))
        self.assertGreater(len(set(s['group_count'] for s in uneven['strata'])),1)
        delta=np.zeros((uneven['group_count'],uneven['n_backgrounds'],2,2))
        delta[uneven['strata'][0]['group_indices']]=1
        weighted=c.metrics(uneven,probabilities(uneven,delta));self.assertAlmostEqual(weighted['L'],1/9)
        self.assertAlmostEqual(weighted['D'],1/9);self.assertNotAlmostEqual(weighted['L'],delta.mean())
        # Background weights also precede the stratum mean.
        delta[:]=0;delta[:,0]=1
        weighted=c.metrics(uneven,probabilities(uneven,delta));self.assertEqual(weighted['L'],.25)

    def test_invalid_inputs_missing_mirror_and_empty_strata(self):
        built=c.build_cases(spec());n=built['cross_cells']
        for bad in (np.zeros((n-1,2)),np.zeros((n,3)),np.ones((n,2),bool),np.full((n,2),np.nan),
                    np.full((n,2),1.01),np.full((n,2),-.01)):
            with self.assertRaises(ValueError):c.metrics(built,bad)
        with self.assertRaises(ValueError):c.expand(built,'unknown')
        bad=spec();bad['layouts']=bad['layouts'][:1];bad['world_count']//=2
        with self.assertRaises(ValueError):c.build_cases(bad)
        bad=spec();bad['needs'].pop();bad['world_count']-=4
        with self.assertRaises(ValueError):c.build_cases(bad)
        sparse=c.build_cases(spec(NEED_EDGES[:1]));out=c.metrics(sparse,np.zeros((sparse['cross_cells'],2)))
        self.assertIsNone(out['L']);self.assertIsNone(out['D']);self.assertEqual(len(out['strata']),9)
        empty=deepcopy(spec(NEED_EDGES[:1]));empty['needs']=empty['needs'][:1];empty['world_count']=4
        empty=c.build_cases(empty);self.assertEqual(c.expand(empty)['receiver_state_indices'].shape,(0,))
        out=c.metrics(empty,np.zeros((0,2)));self.assertIsNone(out['L']);json.dumps(out,allow_nan=False)


if __name__=='__main__':unittest.main()
