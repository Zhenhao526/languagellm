"""Synthetic, finite v29 support checks; no images, models or training."""
import hashlib
import itertools
import json
import unittest
import numpy as np
import support

MAPS=np.asarray(list(itertools.permutations(range(6),2)),np.int64)


def table():
    # Nonconsecutive IDs prevent accidentally confusing IDs with photo ranks.
    photos=np.asarray(list(itertools.product((0,2,3,5,8,9),(1,7))),np.int64)
    ids=np.repeat(np.arange(30),12)
    return dict(map_id=np.tile(ids,2),positions=np.tile(MAPS[ids],(2,1)),
        photo_ids=np.tile(photos,(60,1)),shown=np.repeat(np.arange(2),360))


class SupportTests(unittest.TestCase):
    def test_initialization_identity_is_deterministic_and_separate(self):
        ids=[];old=[]
        for seed,p,d in itertools.product(range(34101,34105),(1,2,3),(0,1)):
            value=support.seed_value(seed,p,d,1);ids.append(value)
            expected=int(np.random.SeedSequence([29029,seed,p,d,1]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))
            self.assertEqual(value,expected);self.assertEqual(value,support.seed_value(seed,p,d,1))
            self.assertIsInstance(value,int);self.assertTrue(0<=value<2**63)
            old.append(int(np.random.SeedSequence([23023,seed,p,d,1]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1)))
        self.assertEqual(len(set(ids)),24);self.assertEqual(set(ids)&set(old),set())
        self.assertNotEqual(support.seed_value(34101,1,0,0),support.seed_value(34101,1,0,1))

    def test_frozen_selection_and_partition_sets(self):
        self.assertEqual(hashlib.sha256(support.SUPPORT_FILE.read_bytes()).hexdigest(),support.SUPPORT_SHA256)
        data=json.loads(support.SUPPORT_FILE.read_text());self.assertEqual(data['number_valid_q'],80)
        for p in (1,2,3):
            g,h=(support.groups(p,a) for a in support.ARMS)
            for x in (g,h):
                for values in x.values():
                    self.assertEqual(values.dtype,np.int64)
                    np.testing.assert_array_equal(values,np.unique(values))
                self.assertEqual({k:len(v) for k,v in x.items()},dict(train18=18,common_target6=6,shared_train13=13,held12=12,other_held6=6,common_unseen7=7,extra_common_unseen1=1,common30=30))
                np.testing.assert_array_equal(np.union1d(x['train18'],x['held12']),np.arange(30))
                self.assertEqual(len(np.intersect1d(x['train18'],x['common_target6'])),0)
            np.testing.assert_array_equal(g['common_target6'],h['common_target6'])
            np.testing.assert_array_equal(np.intersect1d(g['train18'],h['train18']),g['shared_train13'])
            np.testing.assert_array_equal(np.intersect1d(g['held12'],h['held12']),g['common_unseen7'])
            np.testing.assert_array_equal(np.setdiff1d(g['common_unseen7'],g['common_target6']),g['extra_common_unseen1'])

    def test_exact_graph_invariants_and_target_paths(self):
        for p,arm in itertools.product((1,2,3),support.ARMS):
            g=support.groups(p,arm);a=np.zeros((6,6),np.int64)
            a[MAPS[g['train18'],0],MAPS[g['train18'],1]]=1
            np.testing.assert_array_equal(a.sum(0),[3]*6);np.testing.assert_array_equal(a.sum(1),[3]*6)
            self.assertEqual(int(np.trace(a)),0);self.assertEqual(int((a*a.T).sum()),12)
            target=MAPS[g['common_target6']];paths=a@a.T@a
            np.testing.assert_array_equal(paths[target[:,0],target[:,1]],[3 if arm=='target_paths3' else 2]*6)
            overlap=a@a.T;cycles=sum(int(overlap[i,j]*(overlap[i,j]-1)//2) for i in range(6) for j in range(i))
            self.assertEqual(cycles,3 if arm=='target_paths3' else 6)
            reached=set();stack=[0]
            while stack:
                x=stack.pop()
                if x in reached:continue
                reached.add(x);stack.extend([6+j for j in range(6) if a[x,j]] if x<6 else [i for i in range(6) if a[i,x-6]])
            self.assertEqual(len(reached),12)

    def test_per_batch_layout_position_mask_balance(self):
        w=table()
        for p,arm,d,step in itertools.product((1,2,3),support.ARMS,(0,1),(0,1,2100,2399)):
            f=support.fixture(34101,p,d,step,arm,w);raw=support.subset(w,f['indices'])
            self.assertEqual(f['indices'].shape,(252,));self.assertEqual(f['indices'].dtype,np.int64)
            self.assertEqual(f['uniforms'].shape,(252,4));self.assertEqual(f['uniforms'].dtype,np.float32)
            self.assertTrue(np.logical_and(f['uniforms']>=0,f['uniforms']<1).all())
            np.testing.assert_array_equal(np.bincount(raw['map_id'],minlength=30)[support.groups(p,arm)['train18']],[14]*18)
            self.assertEqual(set(raw['map_id']),set(support.groups(p,arm)['train18']))
            for kind in (0,1):np.testing.assert_array_equal(np.bincount(raw['positions'][:,kind],minlength=6),[42]*6)
            np.testing.assert_array_equal(raw['shown'],np.repeat((0,1),126))
            for key in ('map_id','positions','photo_ids'):np.testing.assert_array_equal(raw[key][:126],raw[key][126:])
            self.assertEqual(set(f),{'indices','uniforms'})

    def test_cross_arm_exact_shared_worlds_and_policy_streams(self):
        w=table()
        for p,d,step in itertools.product((1,2,3),(0,1),(0,13,2100,2399)):
            f,h=(support.fixture(34104,p,d,step,a,w) for a in support.ARMS)
            x,y=(support.subset(w,v['indices']) for v in (f,h))
            np.testing.assert_array_equal(f['uniforms'],h['uniforms'])
            for key in ('photo_ids','shown'):np.testing.assert_array_equal(x[key],y[key])
            same=x['map_id']==y['map_id'];self.assertEqual(int(same.sum()),182)
            np.testing.assert_array_equal(x['positions'][same],y['positions'][same])
            self.assertEqual(set(x['map_id'][same]),set(support.groups(p,support.ARMS[0])['shared_train13']))
            self.assertEqual(set(x['map_id'][~same])&set(y['map_id'][~same]),set())

    def test_independent_index_and_rng_reconstruction(self):
        w=table();seed,p,d,step=34103,2,1,2100
        for arm in support.ARMS:
            f=support.fixture(seed,p,d,step,arm,w);g=support.groups(p,arm)
            pool=np.r_[g['shared_train13'],np.setdiff1d(g['train18'],g['shared_train13'])]
            sequence=np.asarray(list(pool)*7)
            order=np.random.default_rng(np.random.SeedSequence([29029,seed,p,d,step,0])).permutation(126)
            draws=np.random.default_rng(np.random.SeedSequence([29029,seed,p,d,step,1])).random((126,2))
            indices=[]
            for i in range(126):
                m=int(sequence[order[i]]);food=(0,2,3,5,8,9)[int(draws[i,0]*6)];water=(1,7)[int(draws[i,1]*2)]
                candidates=np.flatnonzero((w['map_id']==m)&(w['photo_ids'][:,0]==food)&(w['photo_ids'][:,1]==water)&(w['shown']==0))
                self.assertEqual(len(candidates),1);indices.append(candidates[0])
            np.testing.assert_array_equal(f['indices'],np.r_[indices,np.asarray(indices)+360])
            expected=np.random.default_rng(np.random.SeedSequence([29029,seed,p,d,step,2])).random((252,4)).astype(np.float32)
            np.testing.assert_array_equal(f['uniforms'],expected)

    def test_no_global_rng_side_effect_and_repeated_identity(self):
        w=table();np.random.seed(519);before=np.random.get_state()
        f=support.fixture(34101,1,0,0,support.ARMS[0],w);after=np.random.get_state()
        for a,b in zip(before,after):np.testing.assert_equal(a,b)
        again=support.fixture(34101,1,0,0,support.ARMS[0],w)
        for key in f:np.testing.assert_array_equal(f[key],again[key])
        other=support.fixture(34101,1,0,1,support.ARMS[0],w)
        self.assertFalse(np.array_equal(f['uniforms'],other['uniforms']))

    def test_invalid_support_or_table_rejected(self):
        for p,arm in ((0,support.ARMS[0]),(1,'unknown')):
            with self.assertRaises(ValueError):support.groups(p,arm)
        w=table()
        for d,step in ((2,0),(0,-1)):
            with self.assertRaises(ValueError):support.fixture(1,1,d,step,support.ARMS[0],w)
        with self.assertRaises(ValueError):support.fixture(1,1,0,0,support.ARMS[0],{k:v[:180] for k,v in w.items()})
        bad={k:v.copy() for k,v in w.items()};bad['photo_ids'][[0,1]]=bad['photo_ids'][[1,0]]
        with self.assertRaises(ValueError):support.fixture(1,1,0,0,support.ARMS[0],bad)


if __name__=='__main__':unittest.main(verbosity=2)
