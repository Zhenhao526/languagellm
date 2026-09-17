"""Small native-domain and exact permutation fixtures; no policy files."""
from copy import deepcopy
from itertools import combinations,permutations,product
import json
import unittest

import numpy as np

from . import cases as c
from research_program.triadic_action_dependency_study import dataset as original


NEED_EDGES=[((0,14,20),(3,14,20)),((6,14,17),(9,14,17)),((0,6,7),(1,6,7)),
            ((0,14,19),(0,20,19)),((6,14,16),(6,17,16)),((0,6,1),(0,7,1)),
            ((0,19,14),(0,19,20)),((6,16,14),(6,16,17)),((0,1,6),(0,1,7))]


def spec(needs=None):
    rows=sorted({n for edge in NEED_EDGES for n in edge}) if needs is None else sorted(set(needs))
    layouts=[[0,1,2,3],[3,1,0,2]];owners=[[1,2,3],[3,2,1]]
    return dict(partition='fixture',needs=[list(n) for n in rows],layouts=layouts,private_sites=owners,
                world_count=len(rows)*4,state_order=c.STATE_ORDER)


class CasesTests(unittest.TestCase):
    def test_native_flips_unique_changed_need_axis_and_complete_edges(self):
        for need in range(24):self.assertEqual(list(c.flips(need)),list(original.flips(need)))
        source=spec();built=c.build_cases(source);self.assertEqual(built['empty_strata'],[])
        needs=list(map(tuple,source['needs']));expected=set()
        for i,j in combinations(range(len(needs)),2):
            changed=[a for a in range(3) if needs[i][a]!=needs[j][a]]
            if len(changed)!=1:continue
            actor=changed[0]
            for axis,value in original.flips(needs[i][actor]):
                if value==needs[j][actor] and built['need_target_pairs'][i]!=built['need_target_pairs'][j]:
                    expected.add((i,j,actor,c.AXES.index(axis)))
        actual={(a,b,p,axis) for (a,b),p,axis in zip(built['edge_need_indices'],built['changed_person'],built['axis_index'])}
        self.assertEqual(actual,expected);self.assertEqual(len(actual),built['need_edges'])
        for indices,targets in zip(built['edge_need_indices'],built['target_pairs']):
            self.assertEqual(targets,[built['need_target_pairs'][i] for i in indices]);self.assertNotEqual(*targets)
        self.assertEqual(sum(s['need_edges'] for s in built['strata']),built['need_edges'])
        self.assertEqual(built['state_edges'],built['need_edges']*4)
        json.dumps(built,allow_nan=False)

    def test_complete_state_order_expansion_and_rejections(self):
        source=spec();built=c.build_cases(source);states=original.pack_states(source);indices=np.arange(len(states))
        self.assertEqual(c.validate_partition_arrays(built,states,indices)['status'],'valid')
        all_edges=set()
        for background in range(built['n_backgrounds']):
            expanded=c.expand_background(built,background)
            for (left,right),person in zip(expanded['state_indices'],expanded['changed_person']):
                self.assertTrue(np.array_equal(states[left,3:],states[right,3:]))
                self.assertEqual(np.flatnonzero(states[left,:3]!=states[right,:3]).tolist(),[int(person)])
                all_edges.add((int(left),int(right)))
        self.assertEqual(len(all_edges),built['state_edges'])
        for bad_states,bad_indices in ((states[::-1],indices),(states,indices[::-1]),(states[:-1],indices[:-1]),
                                      (states.astype(float),indices),(states,indices.astype(float))):
            with self.assertRaises(ValueError):c.validate_partition_arrays(built,bad_states,bad_indices)
        duplicate=indices.copy();duplicate[1]=0
        with self.assertRaises(ValueError):c.validate_partition_arrays(built,states,duplicate)
        altered=states.copy();altered[1,7:10]=altered[0,7:10]
        with self.assertRaises(ValueError):c.validate_partition_arrays(built,altered,indices)
        with self.assertRaises(ValueError):c.expand_background(built,4)

    def test_oracle_constant_none_equal_strata_and_raw_counts(self):
        built=c.build_cases(spec());b=built['n_backgrounds'];targets=np.array(built['need_target_pairs'],dtype=np.int8)
        choices=np.repeat(targets,b);result=c.metrics(built,choices)
        self.assertEqual(result['Q'],1);self.assertEqual(result['both_correct_rate'],1)
        self.assertEqual(result['both_executed_pair_change_rate'],1);self.assertEqual(result['partner_change_rate'],1)
        self.assertEqual(result['raw_counts']['both_correct'],built['state_edges'])
        counts=np.bincount(targets,minlength=3);n=len(targets);by_stratum=[]
        for st in built['strata']:
            rates=[int(counts[a])*int(counts[b])/(n*(n-1)) for a,b in (built['target_pairs'][i] for i in st['edge_indices'])]
            by_stratum.append(sum(rates)/len(rates))
        self.assertAlmostEqual(result['Q_shuffle'],sum(by_stratum)/9)
        self.assertAlmostEqual(result['Q_excess'],1-result['Q_shuffle'])
        pattern=np.repeat(targets[:,None],b,axis=1);pattern[:,1]=-1
        mixed=c.metrics(built,pattern.ravel());self.assertEqual(mixed['Q'],.75)
        self.assertEqual(mixed['raw_counts']['both_correct']+mixed['raw_counts']['one_correct']+mixed['raw_counts']['neither_correct'],built['state_edges'])
        for fixed in (-1,0,1,2):
            answer=c.metrics(built,np.full(built['world_count'],fixed,dtype=np.int8))
            self.assertEqual(answer['Q'],0);self.assertEqual(answer['Q_shuffle'],0)
            self.assertEqual(answer['partner_change_rate'],0);self.assertEqual(answer['both_executed_pair_change_rate'],0)
        uneven=c.build_cases(spec([tuple(n) for n in built['needs']]+[(0,14,23),(3,14,23)]))
        choices=np.full(uneven['world_count'],-1,dtype=np.int8)
        for need in ([0,14,23],[3,14,23]):
            index=uneven['needs'].index(need);choices[index*4:(index+1)*4]=uneven['need_target_pairs'][index]
        weighted=c.metrics(uneven,choices)
        self.assertEqual([s['need_edges'] for s in uneven['strata']],[2,1,1,1,1,1,1,1,1])
        self.assertAlmostEqual(weighted['Q'],1/18);self.assertAlmostEqual(weighted['raw_pooled_Q'],1/10)
        self.assertNotEqual(weighted['Q'],weighted['raw_pooled_Q'])
        json.dumps(result,allow_nan=False)

    def test_exact_shuffle_uses_all_need_worlds_and_change_excludes_none(self):
        built=c.build_cases(spec([(0,1,6),(0,1,7),(0,14,20)]));self.assertEqual(built['need_edges'],1)
        pair=built['edge_need_indices'][0];truth=built['target_pairs'][0];n=built['need_worlds']
        labels=np.array([truth[0],truth[1],-1],dtype=np.int8)
        # One non-edge world remains in the permutation denominator.
        total=0
        for permuted in permutations(labels.tolist()):
            total+=int(permuted[pair[0]]==truth[0] and permuted[pair[1]]==truth[1])
        expected=total/6;self.assertEqual(expected,1/6)
        choice=np.repeat(labels,built['n_backgrounds']);result=c.metrics(built,choice)
        st=next(s for s in result['strata'] if not s['empty'])
        self.assertEqual(st['Q_shuffle'],expected)
        self.assertEqual(st['backgrounds'][0]['shuffle_numerator_sum'],1)
        self.assertEqual(st['backgrounds'][0]['shuffle_denominator'],n*(n-1))
        self.assertIsNone(result['Q']);self.assertIsNone(result['Q_excess']);self.assertFalse(result['complete_nine_strata'])
        labels[pair[1]]=-1;changed=c.metrics(built,np.repeat(labels,built['n_backgrounds']))
        st=next(s for s in changed['strata'] if not s['empty'])
        self.assertEqual(st['Q'],0);self.assertEqual(st['one_correct_rate'],1)
        self.assertEqual(st['partner_change_rate'],1);self.assertEqual(st['both_executed_pair_change_rate'],0)
        self.assertEqual(st['execution_status_change_rate'],1)

    def test_reject_invalid_specs_outputs_and_keep_all_empty_strata(self):
        base=spec()
        for key,value in (('world_count',base['world_count']-1),('state_order','layout-major'),('needs',[[24,0,0]]),
                          ('needs',base['needs']+[base['needs'][0]]),('layouts',[[0,0,1,2]]),('private_sites',[[1,1,3]])):
            bad=deepcopy(base);bad[key]=value
            with self.assertRaises(ValueError):c.build_cases(bad)
        unsupported=spec([(0,0,0)])
        with self.assertRaises(ValueError):c.build_cases(unsupported)
        built=c.build_cases(base)
        for bad in (np.zeros(built['world_count']-1,dtype=int),np.zeros(built['world_count'],float),
                    np.full(built['world_count'],3,dtype=int),np.full(built['world_count'],-2,dtype=int)):
            with self.assertRaises(ValueError):c.metrics(built,bad)
        empty=c.build_cases(spec([(0,1,6)]));self.assertEqual(empty['need_edges'],0);self.assertEqual(len(empty['empty_strata']),9)
        self.assertEqual(c.expand_background(empty,0)['state_indices'].shape,(0,2))
        out=c.metrics(empty,np.full(empty['world_count'],-1,dtype=int));self.assertIsNone(out['Q'])
        self.assertEqual(len(out['strata']),9);json.dumps(out,allow_nan=False)


if __name__=='__main__':unittest.main()
