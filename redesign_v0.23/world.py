"""Two observations of an unchanged resource map; training support is explicit."""
import itertools,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'redesign_v0.20'))
from temporal_world import MAPS,partition,full_frame


def table(bank,split):
    pools=[np.sort(bank.pools[split,k]) for k in (0,1)]
    if split=='test':pools=[v[:4] for v in pools]
    photos=np.asarray(list(itertools.product(*pools)),np.int64)
    mids=np.repeat(np.arange(30),len(photos));ids=np.tile(photos,(30,1));n=len(mids)
    return dict(map_id=np.tile(mids,2),photo_ids=np.tile(ids,(2,1)),positions=np.tile(MAPS[mids],(2,1)),
                shown=np.repeat(np.arange(2),n))


def frames(projected,w):
    n=len(w['map_id']);first=full_frame(projected,w['positions'],w['photo_ids'])
    slots=projected.new_zeros(n,6,64);exists=projected.new_zeros(n,6);r=np.arange(n)
    where=torch.from_numpy(w['positions'][r,w['shown']]);photo=torch.from_numpy(w['photo_ids'][r,w['shown']])
    slots[torch.arange(n),where]=projected[photo];exists[torch.arange(n),where]=1
    second=torch.cat((slots.flatten(1),exists),-1)
    bits=projected.new_ones(n,2);bits[:,1]=0
    return torch.stack((first,second),1),bits


def seed_value(seed,p,d,kind):
    return int(np.random.SeedSequence([23023,seed,p,d,kind]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))


def fixture(seed,p,d,step,scope,phase,train_table):
    assert scope in ('old','all') and phase in ('private','social')
    phase_id=0 if phase=='private' else 1;n=256 if phase=='private' else 128
    pool=partition(p)['old'] if scope=='old' else np.arange(30)
    rng=np.random.default_rng(np.random.SeedSequence([23023,seed,p,d,phase_id,step,1]))
    draws=rng.random((n,3));maps=pool[np.minimum((draws[:,0]*len(pool)).astype(int),len(pool)-1)]
    per_map=len(train_table['map_id'])//60;side=int(np.sqrt(per_map));assert side*side==per_map
    photo0=np.minimum((draws[:,1]*side).astype(int),side-1);photo1=np.minimum((draws[:,2]*side).astype(int),side-1)
    base=maps*per_map+photo0*side+photo1;indices=np.r_[base,base+len(train_table['map_id'])//2]
    random=np.random.default_rng(np.random.SeedSequence([23023,seed,p,d,phase_id,step,2]))
    uniforms=random.random((n*2,1 if phase=='private' else 4)).astype(np.float32)
    goals=np.tile(rng.integers(2,size=n),2)
    return dict(indices=indices,uniforms=uniforms,goals=goals)


def subset(w,idx):return {k:v[idx] for k,v in w.items()}
