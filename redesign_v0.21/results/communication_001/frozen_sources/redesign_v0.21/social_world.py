"""Frozen two-frame resource world tables and explicit social RNG identities."""
import itertools,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'redesign_v0.20'))
import temporal_world as temporal

def train_worlds(bank,p):
    events=temporal.events(p,True)
    photos=np.asarray(list(itertools.product(*[np.sort(bank.pools['train',k]) for k in (0,1)])),np.int64)
    rows=np.repeat(events,len(photos),axis=0)
    return dict(source_map=rows[:,0],target_map=rows[:,1],mover=rows[:,2],positions=temporal.MAPS[rows[:,1]],photo_ids=np.tile(photos,(len(events),1)))

def identity(seed,p,who,purpose,step=0):
    return np.random.SeedSequence([21021,int(seed),int(p),int(who),int(purpose),int(step)])

def comm_seed(seed,p,who):
    return int(identity(seed,p,who,0).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))

def fixture(seed,p,who,step,nworlds,batch=256):
    idx=np.random.default_rng(identity(seed,p,who,1,step)).integers(nworlds,size=batch)
    uniform=np.random.default_rng(identity(seed,p,who,2,step)).random((batch,4)).astype(np.float32)
    return idx,uniform

def local_keys(w):
    n=len(w['mover']);return np.column_stack((w['mover'],w['positions'][np.arange(n),w['mover']],w['photo_ids']))

def group_indices(keys):
    _,inv=np.unique(keys,axis=0,return_inverse=True)
    return [np.flatnonzero(inv==i) for i in range(inv.max()+1)]
