import copy,itertools,unittest
from pathlib import Path
import numpy as np
from . import audit_response as a


def toy(extra=False,empty=False):
    edges=[[0,1] for _ in range(9)];targets=[[0,1] for _ in range(9)]
    if extra:edges.append([1,2]);targets.append([1,2])
    strata=[]
    for k in range(9):
        ids=[] if empty and k==8 else [k]+([9] if extra and k==0 else [])
        strata.append(dict(changed_person=k//3,axis=a.AXES[k%3],axis_index=k%3,edge_indices=ids,need_edges=len(ids),state_edges=len(ids),empty=not ids,
            target_transition_counts={f'{x}_{y}':sum(targets[i]==[x,y] for i in ids) for x,y in itertools.permutations(range(3),2)}))
    return dict(partition='fixture',need_worlds=3,n_backgrounds=1,world_count=3,background_layout_owner_indices=[[0,0]],layouts=[[0,1,2,3]],private_sites=[[1,2,3]],
        edge_need_indices=edges,target_pairs=targets,strata=strata,empty_strata=[dict(changed_person=s['changed_person'],axis=s['axis']) for s in strata if s['empty']])

class AuditFixture(unittest.TestCase):
    def test_exact_without_replacement(self):
        c=toy();r=a.metrics(c,np.array([0,1,2]));self.assertEqual(r['Q'],1)
        brute=np.mean([a.metrics(c,np.array(p))['Q'] for p in itertools.permutations((0,1,2))])
        self.assertAlmostEqual(r['Q_shuffle'],1/6);self.assertAlmostEqual(r['Q_shuffle'],brute)
        self.assertEqual(r['strata'][0]['backgrounds'][0]['shuffle_denominator'],6)
    def test_macro_not_pooled(self):
        r=a.metrics(toy(extra=True),np.array([0,1,0]));self.assertAlmostEqual(r['Q'],17/18);self.assertAlmostEqual(r['raw_pooled_Q'],9/10)
    def test_empty_is_not_dropped(self):
        r=a.metrics(toy(empty=True),np.array([0,1,2]));self.assertIsNone(r['Q']);self.assertIsNone(r['Q_shuffle']);self.assertFalse(r['complete_nine_strata'])
    def test_none_and_single_pair(self):
        for values in ([-1,-1,-1],[0,0,0]):
            r=a.metrics(toy(),np.array(values));self.assertEqual(r['Q'],0);self.assertEqual(r['Q_shuffle'],0);self.assertEqual(r['partner_change_rate'],0)
        r=a.metrics(toy(),np.array([-1,0,1]));self.assertEqual(r['partner_change_rate'],1);self.assertEqual(r['both_executed_pair_change_rate'],0);self.assertEqual(r['execution_status_change_rate'],1)
    def test_static_full_support_and_order(self):
        a.references();p=a.ROOT/'research_program/triadic_action_dependency_study/results/context_001/prepared.json'
        self.assertEqual(a.sha(p),a.ORIGINAL_SHA)
        for part,spec in a.read(p)['partitions'].items():
            c=a.build_cases(spec);held=part in ('new_needs','new_needs_and_layouts')
            self.assertEqual([s['need_edges'] for s in c['strata']],([40,40,32] if held else [152,152,168])*3)
            self.assertEqual(c['state_edges'],dict(train=152928,new_needs=36288,new_layouts=50976,new_needs_and_layouts=12096)[part])
            ids=np.arange(spec['world_count'],dtype=np.int64);states=a.env.packed(spec);a.validate_partition(c,states,ids)
            with self.assertRaises(Exception):a.validate_partition(c,states,ids[::-1])
    def test_mask_truth_and_state_errors(self):
        c=toy();c['target_pairs'][0]=[0,0]
        with self.assertRaises(Exception):a.metrics(c,np.array([0,1,2]))
        with self.assertRaises(Exception):a.metrics(toy(),np.array([0,1,3]))

if __name__=='__main__':unittest.main()
