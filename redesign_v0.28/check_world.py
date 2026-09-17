"""Synthetic, no-image tests for the rectangular v28 bank/world contract."""
import itertools,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import world

class WorldChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        self.rows=[dict(id=f'{i:02d}',stratum=world.STRATA[i//3],category='water' if i//3==3 else 'food',split='test' if i%3==2 else 'train') for i in range(12)]
        self.raw=torch.from_numpy(np.random.default_rng(527).normal(size=(12,1024)).astype(np.float32))
        self.save(self.rows,self.raw)
        self.bank=world.NonSquareImageBank(self.path/'feature_cache.pt',self.path/'selection.json')
    def tearDown(self):self.tmp.cleanup()
    def save(self,rows,features,ids=None):
        (self.path/'selection.json').write_text(json.dumps(dict(images=rows)))
        torch.save(dict(features=features,image_ids=ids if ids is not None else [r['id'] for r in rows]),self.path/'feature_cache.pt')
    def test_training_only_normalization_matches_legacy_order(self):
        train=np.asarray([r['split']=='train' for r in self.rows]);z=self.raw.numpy();center=z[train].mean(0);scale=float(np.sqrt(((z[train]-center)**2).mean()))
        self.assertTrue(np.array_equal(center,self.bank.center));self.assertEqual(scale,self.bank.scale)
        self.assertTrue(np.array_equal(self.bank.features.numpy(),(z-center)/max(scale,1e-6)))
        changed=self.raw.clone();changed[~torch.from_numpy(train)]=100000
        self.save(self.rows,changed);other=world.NonSquareImageBank(self.path/'feature_cache.pt',self.path/'selection.json')
        self.assertTrue(np.array_equal(other.center,self.bank.center));self.assertEqual(other.scale,self.bank.scale)
        self.assertTrue(torch.equal(other.features[train],self.bank.features[train]))
    def test_zero_scale_and_input_validation(self):
        self.save(self.rows,torch.zeros_like(self.raw));bank=world.NonSquareImageBank(self.path/'feature_cache.pt',self.path/'selection.json')
        self.assertEqual(bank.scale,0.);self.assertTrue(torch.isfinite(bank.features).all())
        self.save(self.rows,self.raw,list(reversed([r['id'] for r in self.rows])))
        with self.assertRaises(ValueError):world.NonSquareImageBank(self.path/'feature_cache.pt',self.path/'selection.json')
        bad=[dict(r) for r in self.rows];bad[0]['category']='water';self.save(bad,self.raw)
        with self.assertRaises(ValueError):world.NonSquareImageBank(self.path/'feature_cache.pt',self.path/'selection.json')
        bad=[dict(r) for r in self.rows];bad[0]['split']='test';self.save(bad,self.raw)
        with self.assertRaises(ValueError):world.NonSquareImageBank(self.path/'feature_cache.pt',self.path/'selection.json')
    def test_all_rectangular_worlds_exact_once(self):
        for split,count in [('train',720),('test',180)]:
            w=world.table(self.bank,split);self.assertEqual(len(w['map_id']),count)
            expected=[(m,int(f),int(v),shown) for shown in (0,1) for m in range(30) for f in self.bank.pools[split,0] for v in self.bank.pools[split,1]]
            actual=list(zip(w['map_id'].tolist(),w['photo_ids'][:,0].tolist(),w['photo_ids'][:,1].tolist(),w['shown'].tolist()))
            self.assertEqual(actual,expected);self.assertEqual(len(set(actual)),count)
            self.assertTrue(np.array_equal(w['positions'],world.MAPS[w['map_id']]))
            for p in (1,2,3):
                g=world.partition(p);self.assertEqual([len(g[k]) for k in ('old','added','sealed')],[18,6,6]);self.assertEqual(len(set(np.r_[g['old'],g['added'],g['sealed']].tolist())),30)
    def test_fixture_non_square_ids_and_matched_randomness(self):
        w=world.table(self.bank,'train')
        for phase,half in [('private',256),('social',128)]:
            for scope in ('old','all'):
                for step in (0,1,2100,2399):
                    f=world.fixture(99528,1,0,step,scope,phase,w);sample=world.subset(w,f['indices'])
                    self.assertEqual(len(f['indices']),half*2);self.assertEqual(f['uniforms'].shape,(half*2,1 if phase=='private' else 4))
                    self.assertTrue(np.array_equal(sample['photo_ids'][:half],sample['photo_ids'][half:]));self.assertTrue(np.array_equal(sample['positions'][:half],sample['positions'][half:]))
                    self.assertTrue(np.all(sample['shown'][:half]==0) and np.all(sample['shown'][half:]==1));self.assertTrue(np.array_equal(f['goals'][:half],f['goals'][half:]))
                    rng=np.random.default_rng(np.random.SeedSequence([23023,99528,1,0,0 if phase=='private' else 1,step,1]));u=rng.random((half,3))
                    map_pool=world.partition(1)['old'] if scope=='old' else np.arange(30)
                    expected_map=map_pool[(u[:,0]*len(map_pool)).astype(int)]
                    expected_photos=np.column_stack((self.bank.pools['train',0][(u[:,1]*6).astype(int)],self.bank.pools['train',1][(u[:,2]*2).astype(int)]))
                    self.assertTrue(np.array_equal(sample['map_id'][:half],expected_map));self.assertTrue(np.array_equal(sample['photo_ids'][:half],expected_photos))
                    if scope=='old':self.assertTrue(np.isin(sample['map_id'],world.partition(1)['old']).all())
            old=world.fixture(99528,1,0,0,'old',phase,w);all_=world.fixture(99528,1,0,0,'all',phase,w)
            self.assertTrue(np.array_equal(old['uniforms'],all_['uniforms']));self.assertTrue(np.array_equal(old['goals'],all_['goals']))
    def test_square_pool_legacy_fixture_compatibility(self):
        bank=SimpleNamespace(pools={('train',0):np.arange(22),('train',1):np.arange(22,44)})
        w=world.table(bank,'train');f=world.fixture(33101,2,1,100,'old','social',w)
        rng=np.random.default_rng(np.random.SeedSequence([23023,33101,2,1,1,100,1]));u=rng.random((128,3));maps=world.partition(2)['old'][(u[:,0]*18).astype(int)]
        base=maps*484+(u[:,1]*22).astype(int)*22+(u[:,2]*22).astype(int)
        self.assertTrue(np.array_equal(f['indices'],np.r_[base,base+30*484]))
        self.assertTrue(np.array_equal(f['goals'],np.tile(rng.integers(2,size=128),2)))
        ur=np.random.default_rng(np.random.SeedSequence([23023,33101,2,1,1,100,2]));self.assertTrue(np.array_equal(f['uniforms'],ur.random((256,4)).astype(np.float32)))
    def test_render_full_then_only_visible_resource(self):
        w=world.table(self.bank,'test');projected=torch.arange(12*64,dtype=torch.float32).reshape(12,64)+1
        frames,bits=world.frames(projected,w);self.assertEqual(frames.shape,(180,2,390));self.assertTrue(torch.equal(bits,torch.tensor([[1.,0.]]).repeat(180,1)))
        for row in range(len(w['map_id'])):
            for side in (0,1):
                slots=frames[row,side,:384].reshape(6,64);exists=frames[row,side,384:]
                visible=(0,1) if side==0 else (int(w['shown'][row]),)
                self.assertEqual(int(exists.sum()),len(visible))
                for site in range(6):
                    kinds=[k for k in visible if w['positions'][row,k]==site]
                    if kinds:self.assertTrue(torch.equal(slots[site],projected[w['photo_ids'][row,kinds[0]]]))
                    else:self.assertTrue(torch.equal(slots[site],torch.zeros(64)))
        # Map IDs are metadata only: changing IDs without visual state cannot alter frames.
        altered={k:v.copy() for k,v in w.items()};altered['map_id'][:]=0
        af,ab=world.frames(projected,altered);self.assertTrue(torch.equal(af,frames));self.assertTrue(torch.equal(ab,bits))
    def test_sample_one_water_test_row_and_no_cross_split(self):
        kinds=np.tile([0,1],(17,1));feat,ids=self.bank.sample(kinds,'test',np.random.default_rng(1))
        self.assertEqual(feat.shape,(17,2,1024));self.assertTrue(np.isin(ids[:,0],self.bank.pools['test',0]).all());self.assertTrue(np.all(ids[:,1]==self.bank.pools['test',1][0]));self.assertTrue(torch.equal(feat,self.bank.features[torch.from_numpy(ids)]))
        with self.assertRaises(ValueError):self.bank.sample(np.asarray([2]),'train',np.random.default_rng(0))

if __name__=='__main__':unittest.main(verbosity=2)
