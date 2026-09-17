"""Frozen v0.30 multi-partner target and balanced training fixtures.

The target contains two water partners for every food position and two food
partners for every water position. Two connected 2-regular training supports
are disjoint from that target. No model, image or token semantics enter here.
"""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
DESIGN_FILE=ROOT/'support_design.json'
DESIGN_SHA256='4f641601f5cea1fd5261285416252f9e74339e90ffc7e24eea8d76e4accfa4f6'
_design_bytes=DESIGN_FILE.read_bytes()
if hashlib.sha256(_design_bytes).hexdigest()!=DESIGN_SHA256:
    raise ValueError('the prespecified multi-partner design has changed')
_DESIGN=json.loads(_design_bytes)
ARMS=('aligned_paths1','sparse_paths')
NAMESPACE=int(_DESIGN['sampling']['namespace'])
BASE_BATCH=int(_DESIGN['sampling']['base_worlds_per_direction'])
MAPS=np.asarray([(i,j) for i in range(6) for j in range(6) if i!=j],np.int64)
_INDEX={tuple(x):i for i,x in enumerate(MAPS)}
_BASE={k:tuple(tuple(x) for x in _DESIGN[k+'_pairs']) for k in ('target12','aligned_paths1','sparse_paths')}
_PANELS=[tuple(x) for x in _DESIGN['panels']]

def _mapped(name,p):
    q=_PANELS[p-1]
    return np.sort(np.asarray([_INDEX[(q[i],q[j])] for i,j in _BASE[name]],np.int64))

def groups(p,arm):
    if p not in (1,2,3) or arm not in ARMS: raise ValueError('unknown panel or arm')
    target=_mapped('target12',p);train=_mapped(arm,p)
    return dict(train12=train,target12=target,held18=np.setdiff1d(np.arange(30,dtype=np.int64),train),common30=np.arange(30,dtype=np.int64))

def ordered_pool(p,arm): return groups(p,arm)['train12']

def seed_value(seed,p,d,kind):
    return int(np.random.SeedSequence([NAMESPACE,int(seed),int(p),int(d),int(kind)]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))

def fixture(seed,p,d,step,arm,train_table):
    if d not in (0,1) or not isinstance(step,(int,np.integer)) or step<0: raise ValueError('invalid direction or step')
    if len(train_table['map_id'])!=720: raise ValueError('requires frozen 720-row source training table')
    photo=np.asarray(train_table['photo_ids']); food=np.unique(photo[:,0]); water=np.unique(photo[:,1])
    if photo.shape!=(720,2) or len(food)!=6 or len(water)!=2 or not np.array_equal(photo[:12],np.asarray([(f,w) for f in food for w in water],np.int64)):
        raise ValueError('requires six-by-two food-major source photos')
    rng=lambda stream: np.random.default_rng(np.random.SeedSequence([NAMESPACE,int(seed),int(p),int(d),int(step),int(stream)]))
    pool=ordered_pool(p,arm); order=rng(0).permutation(BASE_BATCH); maps=np.tile(pool,10)[order]
    photos=rng(1).random((BASE_BATCH,2));base=maps*12+(photos[:,0]*6).astype(np.int64)*2+(photos[:,1]*2).astype(np.int64)
    indices=np.concatenate((base,base+360));uniforms=rng(2).random((2*BASE_BATCH,4)).astype(np.float32)
    return dict(indices=indices,uniforms=uniforms)

def subset(w,indices): return {k:v[indices] for k,v in w.items()}
