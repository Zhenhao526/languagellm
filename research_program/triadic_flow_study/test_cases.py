"""Small native static fixtures. No model calls or learned-result reads."""
from copy import deepcopy
from itertools import product
import json
import unittest

import numpy as np

from . import cases as c
from research_program.triadic_directed_message_study.test_cases import spec
from research_program.triadic_action_dependency_study import dataset as native


class FlowCasesTests(unittest.TestCase):
    def test_complete_grid_order_same_background_donors_and_sham(self):
        source=spec();built=c.build_cases(source);meta=c.expand(built);g=built['group_count'];b=built['n_backgrounds']
        self.assertNotIn('cross_cells',built);self.assertNotIn('donor_layout_indices',built)
        self.assertEqual(built['flow_cells'],16*g*b);self.assertEqual(int(meta['sham'].sum()),4*g*b)
        self.assertEqual(sum(s['flow_cells'] for s in built['strata']),built['flow_cells']);self.assertEqual(built['empty_strata'],[])
        states=native.pack_states(source);self.assertEqual(c.validate_partition_arrays(built,states,np.arange(len(states)))['status'],'valid')
        for row,(gi,bg,ctx,r,i,o) in enumerate(product(range(g),range(b),range(2),range(2),range(2),range(2))):
            indices=built['group_need_indices'][gi];receiver=indices[ctx][r]*b+bg
            incoming=indices[1-ctx][r]*b+bg;outgoing=indices[ctx][1-r]*b+bg
            for key,value in (('group_index',gi),('background_index',bg),('context_index',ctx),('self_endpoint',r),
                              ('incoming',i),('outgoing',o),('receiver_state_indices',receiver),
                              ('incoming_donor_state_indices',incoming),('outgoing_donor_state_indices',outgoing),
                              ('effective_incoming_donor_state_indices',incoming if i else receiver),
                              ('effective_outgoing_donor_state_indices',outgoing if o else receiver)):
                self.assertEqual(meta[key][row],value)
            self.assertEqual(len({receiver,incoming,outgoing}),3)
            self.assertTrue(np.array_equal(states[receiver,3:],states[incoming,3:]))
            self.assertTrue(np.array_equal(states[receiver,3:],states[outgoing,3:]))
            self.assertNotEqual(meta['receiver_target_pair'][row],meta['incoming_donor_target_pair'][row])
            self.assertEqual(meta['incoming_donor_target_pair'][row],meta['outgoing_donor_target_pair'][row])
        self.assertTrue(all(len(v)==built['flow_cells'] for v in meta.values()))
        json.dumps(built,allow_nan=False)

    def test_official_PL_invariance_is_local_to_the_unchanged_need(self):
        built=c.build_cases(spec());meta=c.expand(built);states=native.pack_states(spec())
        views=[{a:native.env.observe(native.env.State(tuple(s[:3]),tuple(s[3:7]),tuple(s[7:])),a,information='PL')
                for a in native.env.AGENTS} for s in states]
        features=native.encode_observations(views)
        ids=np.flatnonzero((meta['incoming']==0)&(meta['outgoing']==0))
        for row in ids:
            receiver=meta['receiver_state_indices'][row];di=meta['incoming_donor_state_indices'][row];do=meta['outgoing_donor_state_indices'][row]
            focus=meta['focus_actor'][row];others=meta['recipient_agents'][row]
            self.assertTrue(np.array_equal(features[receiver,focus],features[di,focus]))
            self.assertTrue(np.array_equal(features[receiver,others],features[do,others]))
            self.assertFalse(np.array_equal(features[receiver,focus],features[do,focus]))
            for other in others:self.assertFalse(np.array_equal(features[receiver,other],features[di,other]))
        self.assertNotIn('primary',built)

    def test_reject_invalid_packing_and_retain_empty_strata_without_scoring(self):
        source=spec();built=c.build_cases(source);states=native.pack_states(source);ids=np.arange(len(states))
        for state,index in ((states[::-1],ids),(states,ids[::-1]),(states[:-1],ids[:-1]),(states.astype(float),ids)):
            with self.assertRaises(ValueError):c.validate_partition_arrays(built,state,index)
        broken=deepcopy(built);broken['flow_cells']-=1
        with self.assertRaises(ValueError):c.expand(broken)
        source['needs']=source['needs'][:1];source['world_count']=4;empty=c.build_cases(source)
        self.assertEqual(len(empty['empty_strata']),9);self.assertEqual(empty['flow_cells'],0)
        self.assertEqual(c.expand(empty)['receiver_state_indices'].shape,(0,))
        self.assertIsNone(c.metrics(empty,np.zeros(0))['I_given_O0'])

    def test_scalar_vector_factorial_contrasts_and_nine_equal_strata(self):
        built=c.build_cases(spec());g=built['group_count'];b=built['n_backgrounds']
        arms=np.array([[.2,.3],[.4,.8]])
        values=np.broadcast_to(arms,(g,b,2,2,2,2)).copy()
        result=c.metrics(built,values.ravel())
        expected=dict(S00=.2,S01=.3,S10=.4,S11=.8,I_given_O0=.2,I_given_O1=.5,
                      O_given_I0=.1,O_given_I1=.4,interaction=.3)
        for key,value in expected.items():self.assertAlmostEqual(result[key],value)
        vector=np.stack((values,values*2,values*0),axis=-1).reshape(-1,3)
        vr=c.metrics(built,vector)
        self.assertEqual(vr['response_shape'],[3]);self.assertTrue(np.allclose(vr['I_given_O0'],[.2,.4,0]))
        self.assertEqual(np.asarray(vr['per_group_background']['interaction']).shape,(g,b,3))
        # Endpoint/context pairing precedes means; one of four states receives
        # a response increase, never a different state's unpaired baseline.
        values[:]=.25;values[:,:,0,0,1,0]=.65
        self.assertAlmostEqual(c.metrics(built,values.ravel())['I_given_O0'],.1)
        from research_program.triadic_need_response_study.test_cases import NEED_EDGES
        uneven=c.build_cases(spec(NEED_EDGES+[((0,14,23),(3,14,23))]))
        values=np.zeros((uneven['group_count'],uneven['n_backgrounds'],2,2,2,2))
        values[uneven['strata'][0]['group_indices'],:,:,:,1,0]=1
        weighted=c.metrics(uneven,values.ravel())
        self.assertAlmostEqual(weighted['I_given_O0'],1/9)
        self.assertNotAlmostEqual(weighted['I_given_O0'],values[:,:,:,:,1,0].mean())
        json.dumps(vr,allow_nan=False)
        for invalid in (np.zeros((built['flow_cells']-1,)),np.ones(built['flow_cells'],bool),np.full(built['flow_cells'],np.nan)):
            with self.assertRaises(ValueError):c.metrics(built,invalid)


if __name__=='__main__':unittest.main()
