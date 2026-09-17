"""Synthetic v0.30 graph and fixture checks; no model or image access."""
import itertools,unittest
import numpy as np
import support

class Tests(unittest.TestCase):
    def table(self):
        photos=np.asarray(list(itertools.product((0,2,3,5,8,9),(1,7))),np.int64)
        mids=np.repeat(np.arange(30),12)
        return dict(map_id=np.tile(mids,2),positions=np.tile(support.MAPS[mids],(2,1)),photo_ids=np.tile(photos,(60,1)),shown=np.repeat(np.arange(2),360))
    def test_design_digest_and_sets(self):
        self.assertEqual(support.DESIGN_SHA256,'4f641601f5cea1fd5261285416252f9e74339e90ffc7e24eea8d76e4accfa4f6')
        self.assertEqual(set(support.ARMS),{'aligned_paths1','sparse_paths'})
        for p in (1,2,3):
            a,b=(support.groups(p,x) for x in support.ARMS)
            for g in (a,b):
                self.assertEqual({k:len(v) for k,v in g.items()},{'train12':12,'target12':12,'held18':18,'common30':30})
                for k,v in g.items(): np.testing.assert_array_equal(v,np.unique(v))
                np.testing.assert_array_equal(np.intersect1d(g['train12'],g['target12']),np.empty(0,np.int64))
            np.testing.assert_array_equal(a['target12'],b['target12'])
    def test_degrees_paths_and_connectedness(self):
        for p,arm in itertools.product((1,2,3),support.ARMS):
            g=support.groups(p,arm);A=np.zeros((6,6),int);B=np.zeros((6,6),int)
            for i,j in support.MAPS[g['train12']]:A[i,j]=1
            for i,j in support.MAPS[g['target12']]:B[i,j]=1
            np.testing.assert_array_equal(A.sum(0),[2]*6);np.testing.assert_array_equal(A.sum(1),[2]*6)
            np.testing.assert_array_equal(B.sum(0),[2]*6);np.testing.assert_array_equal(B.sum(1),[2]*6)
            for M in (A,):
                seen={0};todo=[0]
                while todo:
                    x=todo.pop();nxt=[6+j for j in np.flatnonzero(M[x])] if x<6 else list(np.flatnonzero(M[:,x-6]))
                    for y in nxt:
                        if y not in seen:seen.add(y);todo.append(y)
                self.assertEqual(len(seen),12)
            paths=A@A.T@A;target=support.MAPS[g['target12']]
            expected=[1]*12 if arm=='aligned_paths1' else [0]*9+[1]*3
            np.testing.assert_array_equal(sorted(paths[i,j] for i,j in target),expected)
    def test_exact_balanced_fixture_and_pairing(self):
        w=self.table()
        for p,arm,d,step in itertools.product((1,2,3),support.ARMS,(0,1),(0,1,2100,2399)):
            f=support.fixture(34101,p,d,step,arm,w);idx=f['indices'];raw=support.subset(w,idx)
            self.assertEqual(idx.shape,(240,));self.assertEqual(f['uniforms'].shape,(240,4));self.assertEqual(f['uniforms'].dtype,np.float32)
            np.testing.assert_array_equal(np.bincount(raw['map_id'],minlength=30)[support.groups(p,arm)['train12']],[20]*12)
            for kind in (0,1):np.testing.assert_array_equal(np.bincount(raw['positions'][:,kind],minlength=6),[40]*6)
            np.testing.assert_array_equal(raw['shown'],np.repeat((0,1),120));np.testing.assert_array_equal(raw['map_id'][:120],raw['map_id'][120:]);np.testing.assert_array_equal(raw['photo_ids'][:120],raw['photo_ids'][120:])
    def test_cross_arm_streams_and_no_global_rng(self):
        w=self.table();before=np.random.get_state();a=support.fixture(34101,1,0,0,support.ARMS[0],w);after=np.random.get_state();np.testing.assert_equal(before,after)
        b=support.fixture(34101,1,0,0,support.ARMS[1],w);np.testing.assert_array_equal(a['uniforms'],b['uniforms'])
        shared=np.intersect1d(support.groups(1,support.ARMS[0])['train12'],support.groups(1,support.ARMS[1])['train12'])
        self.assertEqual(int(np.isin(w['map_id'][a['indices']],shared).sum()),140)
        paired=np.isin(w['map_id'][a['indices']],shared)
        np.testing.assert_array_equal(w['map_id'][a['indices']][paired],w['map_id'][b['indices']][paired])
        np.testing.assert_array_equal(w['positions'][a['indices']][paired],w['positions'][b['indices']][paired])
    def test_invalid_inputs(self):
        w=self.table()
        with self.assertRaises(ValueError):support.groups(0,support.ARMS[0])
        with self.assertRaises(ValueError):support.fixture(1,1,0,0,'bad',w)
        with self.assertRaises(ValueError):support.fixture(1,1,0,0,support.ARMS[0],{k:v[:180] for k,v in w.items()})

if __name__=='__main__':unittest.main(verbosity=2)
