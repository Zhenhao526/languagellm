"""Bounded arithmetic and native-format synthetic transcript fixtures;0NN."""
from copy import deepcopy
from itertools import combinations,product
import json,math
import unittest

import numpy as np

from . import metrics as m


def runs_fixture():
    rows=[];time=np.asarray(m.UPDATES)/6000
    for index,seed in enumerate(m.SEEDS):
        slope=.02*(index+1)
        for rule,live in product(m.RULES,(False,True)):
            q=(np.full(6,.2+.02*live) if rule=='strict' else np.full(6,.25+.04*live)+live*slope*time)
            timeline=[]
            for update,value in zip(m.UPDATES,q):
                timeline.append(dict(update=update,evaluation=dict(need_response={
                    'native':dict(Q=float(value),Q_excess=float(value*.5-.01)),
                    'common_reciprocal':dict(Q=float(value+.01*(rule=='strict')),Q_excess=float(value*.5))})))
            rows.append(dict(seed=seed,rule=rule,live=live,condition='fixture',trajectory=timeline))
    return rows


def spec():
    return dict(world_count=12,needs=[[i,0,0] for i in range(6)],layouts=[[0,1,2,3],[3,1,0,2]],
                private_sites=[[1,2,3]],state_order=m.STATE_ORDER)


class MetricTests(unittest.TestCase):
    def test_primary_linear_trajectory_and_sixteen_society_statistics(self):
        source=runs_fixture();before=deepcopy(source);result=m.primary(source)
        self.assertEqual(source,before);self.assertAlmostEqual(result['mean_centered_AUC'],.085)
        self.assertEqual(len(result['by_seed']),16);self.assertEqual(result['statistics']['n'],16)
        self.assertEqual(result['statistics']['df'],15)
        expected_sd=.01*math.sqrt(16*17/12)
        self.assertAlmostEqual(result['statistics']['sample_sd'],expected_sd)
        self.assertAlmostEqual(result['statistics']['standard_error'],expected_sd/4)
        self.assertAlmostEqual(result['statistics']['ci95_lower'],.085-m.T15_975*expected_sd/4)
        self.assertAlmostEqual(result['statistics']['ci95_upper'],.085+m.T15_975*expected_sd/4)
        for index,row in enumerate(result['by_seed']):
            native=row['measures']['native_Q'];self.assertAlmostEqual(native['baseline_DiD'],.02)
            self.assertAlmostEqual(native['raw_AUC'],.02+.01*(index+1))
            self.assertAlmostEqual(native['centered_AUC'],.01*(index+1))
            self.assertAlmostEqual(native['endpoint_DiD'],.02+.02*(index+1))
            self.assertAlmostEqual(row['measures']['native_Q_excess']['centered_AUC'],.005*(index+1))
        self.assertEqual(set(result['auxiliary_statistics']),set(m.MEASURES));json.dumps(result,allow_nan=False)

    def test_irregular_time_weights_centered_range_and_no_clipping(self):
        low=np.array([0,1,1,1,1,1]);high=1-low
        cells={('strict',False):low,('strict',True):high,('reciprocal',False):high,('reciprocal',True):low}
        result=m.trajectory_contrasts(cells)
        self.assertEqual(result['DiD'],[-2,2,2,2,2,2]);self.assertEqual(result['centered_DiD'],[0,4,4,4,4,4])
        self.assertAlmostEqual(result['centered_AUC'],4*(6000-50)/6000)
        self.assertNotAlmostEqual(result['centered_AUC'],np.mean(result['centered_DiD']))
        constant=m.society_statistics(np.full(16,3.5));self.assertEqual(constant['sample_sd'],0)
        self.assertEqual(constant['ci95_lower'],3.5);self.assertEqual(constant['ci95_upper'],3.5)
        for bad in (np.zeros(15),np.full(16,np.nan)):
            with self.assertRaises(ValueError):m.society_statistics(bad)

    def test_t15_fixed_constant_and_primary_input_rejections(self):
        # Independent t15 CDF from x=sqrt(15)*tan(theta), reducing to integral
        # cos(theta)^14 by the elementary even-power recurrence.
        theta=math.atan(m.T15_975/math.sqrt(15));integral=theta
        for power in range(2,15,2):
            integral=math.sin(theta)*math.cos(theta)**(power-1)/power+(power-1)*integral/power
        cdf=.5+math.gamma(8)/(math.sqrt(math.pi)*math.gamma(7.5))*integral
        self.assertAlmostEqual(cdf,.975,places=13);self.assertEqual(round(m.T15_975,3),2.131)
        for mutate in (
            lambda r:r.pop(),lambda r:r.__setitem__(1,deepcopy(r[0])),
            lambda r:r[0]['trajectory'].reverse(),lambda r:r[0].update(live=0),
            lambda r:r[0]['trajectory'][0]['evaluation']['need_response']['native'].update(Q=1.1),
        ):
            bad=runs_fixture();mutate(bad)
            with self.assertRaises(ValueError):m.primary(bad)

    def test_sparse_ARI_boundaries_relabeling_and_direct_pair_reference(self):
        self.assertEqual(m.adjusted_rand_index([0,0,0],[1,1,1]),1)
        self.assertEqual(m.adjusted_rand_index([0,0,0],[0,1,1]),0)
        self.assertEqual(m.adjusted_rand_index([0,1,2],[9,4,7]),1)
        self.assertEqual(m.adjusted_rand_index([0],[8]),1)
        self.assertEqual(m.adjusted_rand_index([0,0,1,1],[0,1,0,1]),-.5)
        partitions=list(product(range(2),repeat=4));pairs=list(combinations(range(4),2));total=len(pairs)
        for left,right in product(partitions,repeat=2):
            aa=sum(left[i]==left[j] for i,j in pairs);bb=sum(right[i]==right[j] for i,j in pairs)
            common=sum(left[i]==left[j] and right[i]==right[j] for i,j in pairs)
            expected=aa*bb/total;den=(aa+bb)/2-expected
            reference=(common-expected)/den if den else 1.
            actual=m.adjusted_rand_index(left,right)
            self.assertAlmostEqual(actual,reference)
            self.assertEqual(actual,m.adjusted_rand_index(np.array(left)*17-3,np.array(right)*-5+11))
        for left,right in (([],[]),([0],[0,1]),([0.],[0.])):
            with self.assertRaises(ValueError):m.adjusted_rand_index(left,right)

    def test_messages_raw_change_partition_stability_and_constant_code_entropy(self):
        source=spec();a=np.zeros((12,2,3,4),dtype=np.int8);b=a.copy();needs=np.arange(12)//2
        a[:,0,0,3]=needs%2;b[:,0,0,3]=1-needs%2
        b[:,0,1,3]=1
        snapshot=m.message_snapshot(a,source);transition=m.message_transition(a,b,source)
        self.assertEqual(len(snapshot['panels']),6);self.assertEqual(len(transition['panels']),6)
        row=snapshot['panels'][0];self.assertEqual(row['conditional_entropy_bits'],1)
        self.assertEqual(row['conditional_effective_packet_count'],2)
        changed=transition['panels'][0]
        self.assertEqual(changed['packet_equal_rate'],0);self.assertEqual(changed['token_hamming_fraction'],.25)
        self.assertEqual(changed['global_ARI'],1);self.assertEqual(changed['mean_background_ARI'],1)
        constant=transition['panels'][2]
        self.assertEqual(constant['global_ARI'],1);self.assertEqual(constant['packet_equal_rate'],0)
        self.assertEqual(constant['conditional_entropy_bits_before'],0);self.assertEqual(constant['conditional_entropy_bits_after'],0)
        self.assertTrue(all(x['constant_before'] and x['constant_after'] and x['ARI']==1 for x in constant['backgrounds']))
        json.dumps(snapshot,allow_nan=False);json.dumps(transition,allow_nan=False)

    def test_background_entropy_effective_counts_and_global_ARI_are_distinct(self):
        source=spec();a=np.zeros((12,2,3,4),dtype=np.int8);b=a.copy();ids=np.arange(12)
        # Public layout alone determines the first code: conditional H=0.
        a[:,1,2,3]=ids%2;b[:,1,2,3]=(ids//2)%2
        snap=m.message_snapshot(a,source);row=snap['panels'][5]
        self.assertEqual(row['global_observed_packet_count'],2);self.assertEqual(row['conditional_entropy_bits'],0)
        self.assertEqual(row['conditional_effective_packet_count'],1)
        moved=m.message_transition(a,b,source)['panels'][5]
        self.assertEqual(moved['mean_background_ARI'],0);self.assertNotEqual(moved['global_ARI'],moved['mean_background_ARI'])
        a[ids%2==1,0,0,3]=(ids[ids%2==1]//2)%2
        mixed=m.message_snapshot(a,source)['panels'][0]
        self.assertEqual(mixed['conditional_entropy_bits'],.5)
        self.assertAlmostEqual(mixed['conditional_effective_packet_count'],math.sqrt(2))
        self.assertNotEqual(mixed['conditional_effective_packet_count'],np.mean([r['effective_packet_count'] for r in mixed['backgrounds']]))
        for bad in (a[:-1],a.astype(float),np.full(a.shape,8,dtype=np.int8)):
            with self.assertRaises(ValueError):m.message_snapshot(bad,source)
        wrong=deepcopy(source);wrong['state_order']='background-major'
        with self.assertRaises(ValueError):m.message_transition(a,b,wrong)


if __name__=='__main__':unittest.main()
