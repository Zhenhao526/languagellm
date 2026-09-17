"""Frozen 12-image bank and unchanged-layout two-frame resource worlds.

Image metadata chooses simulator photographs only. Agents receive projected
visual slots and existence/view bits. No images, model weights or network are
opened by this module: the bank reads the locally prepared feature cache.
"""
from pathlib import Path
import itertools
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
MAPS=np.asarray(list(itertools.permutations(range(6),2)),np.int64)
MATCHINGS=(((0,1),(2,3),(4,5)),((0,2),(1,4),(3,5)),((0,3),(1,5),(2,4)))
STRATA=('apple','banana','orange','water')


class NonSquareImageBank:
    """Six food/two water train rows; three food/one water test rows.

    The float32 NumPy centering/global RMS calculation is the original v4
    ImageBank formula, fitted to precisely the eight training photographs.
    Per-feature standard deviations and test-fitted normalization are not used.
    """
    def __init__(self,feature_cache_path=None,selection_path=None):
        feature_cache_path=Path(feature_cache_path or ROOT/'data/feature_cache.pt')
        selection_path=Path(selection_path or ROOT/'data/selection.json')
        selection=json.loads(selection_path.read_text())
        self.entries=selection['images']
        if len(self.entries)!=12:raise ValueError('exactly12 selected images required')
        ids=[r['id'] for r in self.entries]
        if not all(isinstance(i,str) for i in ids) or len(set(ids))!=12 or ids!=sorted(ids):
            raise ValueError('selection IDs must be distinct and sorted')
        for r in self.entries:
            if r['stratum'] not in STRATA or r['split'] not in ('train','test'):
                raise ValueError('unknown resource stratum or split')
            category='water' if r['stratum']=='water' else 'food'
            if r['category']!=category:raise ValueError('stratum/category mismatch')
        for stratum in STRATA:
            if {split:sum(r['stratum']==stratum and r['split']==split for r in self.entries)
                for split in ('train','test')}!={'train':2,'test':1}:
                raise ValueError('each stratum must have2train/1test images')
        cache=torch.load(feature_cache_path,map_location='cpu',weights_only=True)
        if cache['image_ids']!=ids:raise ValueError('feature cache and selection ID order differ')
        raw=cache['features']
        if not isinstance(raw,torch.Tensor) or raw.shape!=(12,1024) or raw.dtype!=torch.float32 or raw.requires_grad or not torch.isfinite(raw).all():
            raise ValueError('raw features must be finite frozen float32[12,1024]')
        z=raw.detach().numpy().astype(np.float32)
        train=np.asarray([e['split']=='train' for e in self.entries])
        if int(train.sum())!=8:raise ValueError('normalization requires exactly8 training rows')
        # Keep expression/order identical to v4.run_pilot.ImageBank.
        self.center=z[train].mean(0)
        self.scale=float(np.sqrt(((z[train]-self.center)**2).mean()))
        self.features=torch.from_numpy((z-self.center)/max(self.scale,1e-6))
        self.pools={(split,kind):np.asarray([i for i,e in enumerate(self.entries)
                    if e['split']==split and e['category']==name],np.int64)
                    for split in ('train','test') for kind,name in enumerate(('food','water'))}
        if tuple(len(self.pools[k]) for k in (('train',0),('train',1),('test',0),('test',1)))!=(6,2,3,1):
            raise ValueError('unexpected rectangular resource pools')
        self.feature_cache_path=feature_cache_path.resolve()
        self.selection_path=selection_path.resolve()

    def sample(self,kinds,split,rng):
        kinds=np.asarray(kinds)
        if split not in ('train','test') or not np.issubdtype(kinds.dtype,np.integer) or not np.isin(kinds,[0,1]).all():
            raise ValueError('sample requires integer food0/water1 and train/test split')
        ids=np.empty(kinds.shape,dtype=np.int64)
        for kind in (0,1):
            selected=kinds==kind
            ids[selected]=rng.choice(self.pools[split,kind],int(selected.sum()))
        return self.features[torch.from_numpy(ids)],ids


def partition(p):
    if p not in (1,2,3):raise ValueError('partition must be1,2,3')
    def index(matching):
        pairs={pair for a,b in matching for pair in ((a,b),(b,a))}
        return np.asarray([i for i,row in enumerate(MAPS) if tuple(row) in pairs],np.int64)
    added=index(MATCHINGS[p-1]);sealed=index(MATCHINGS[p%3])
    return dict(old=np.setdiff1d(np.arange(30),np.r_[added,sealed]),added=added,sealed=sealed,common30=np.arange(30))


def table(bank,split):
    if split not in ('train','test'):raise ValueError('unknown split')
    food,water=(np.sort(bank.pools[split,k]) for k in (0,1))
    if not len(food) or not len(water):raise ValueError('both resource pools must be nonempty')
    photos=np.asarray(list(itertools.product(food,water)),np.int64)
    mids=np.repeat(np.arange(30),len(photos));ids=np.tile(photos,(30,1));n=len(mids)
    return dict(map_id=np.tile(mids,2),photo_ids=np.tile(ids,(2,1)),positions=np.tile(MAPS[mids],(2,1)),shown=np.repeat(np.arange(2),n))


def full_frame(projected,positions,photo_ids):
    n=len(photo_ids);slots=projected.new_zeros(n,6,64);exists=projected.new_zeros(n,6);rows=torch.arange(n)
    for kind in (0,1):
        sites=torch.from_numpy(positions[:,kind]);ids=torch.from_numpy(photo_ids[:,kind])
        slots[rows,sites]=projected[ids];exists[rows,sites]=1.
    return torch.cat((slots.flatten(1),exists),dim=1)


def frames(projected,w):
    n=len(w['map_id']);first=full_frame(projected,w['positions'],w['photo_ids'])
    slots=projected.new_zeros(n,6,64);exists=projected.new_zeros(n,6);r=np.arange(n)
    sites=torch.from_numpy(w['positions'][r,w['shown']]);ids=torch.from_numpy(w['photo_ids'][r,w['shown']])
    slots[torch.arange(n),sites]=projected[ids];exists[torch.arange(n),sites]=1.
    second=torch.cat((slots.flatten(1),exists),dim=1)
    bits=projected.new_ones(n,2);bits[:,1]=0
    return torch.stack((first,second),dim=1),bits


def seed_value(seed,p,d,kind):
    return int(np.random.SeedSequence([23023,seed,p,d,kind]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))


def fixture(seed,p,d,step,scope,phase,train_table):
    if scope not in ('old','all') or phase not in ('private','social'):
        raise ValueError('unknown training scope or phase')
    if step<0:raise ValueError('step must be zero-based nonnegative')
    phase_id=0 if phase=='private' else 1;n=256 if phase=='private' else 128
    pool=partition(p)['old'] if scope=='old' else np.arange(30)
    rng=np.random.default_rng(np.random.SeedSequence([23023,seed,p,d,phase_id,step,1]))
    draws=rng.random((n,3));maps=pool[np.minimum((draws[:,0]*len(pool)).astype(int),len(pool)-1)]
    # Map-major/food-major/water-minor table; dimensions need not be equal.
    per_map=len(train_table['map_id'])//60;first_pairs=train_table['photo_ids'][:per_map]
    food_count=len(np.unique(first_pairs[:,0]));water_count=len(np.unique(first_pairs[:,1]))
    if per_map!=food_count*water_count or len(train_table['map_id'])!=60*per_map:
        raise ValueError('training table is not a complete rectangular Cartesian pool')
    food=np.minimum((draws[:,1]*food_count).astype(int),food_count-1)
    water=np.minimum((draws[:,2]*water_count).astype(int),water_count-1)
    base=maps*per_map+food*water_count+water
    indices=np.r_[base,base+len(train_table['map_id'])//2]
    random=np.random.default_rng(np.random.SeedSequence([23023,seed,p,d,phase_id,step,2]))
    uniforms=random.random((n*2,1 if phase=='private' else 4)).astype(np.float32)
    goals=np.tile(rng.integers(2,size=n),2)
    return dict(indices=indices,uniforms=uniforms,goals=goals)


def subset(w,idx):return {k:v[idx] for k,v in w.items()}
