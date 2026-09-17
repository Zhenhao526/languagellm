"""Frozen v29 co-occurrence supports and exactly balanced social fixtures.

No model, image, reward, or token semantics enter this module. Graph selection
was completed by a separate finite enumeration before these fixtures existed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
SUPPORT_FILE=ROOT.parent/'paper_program/mechanism_pressure_20260916/support_enumeration.json'
SUPPORT_SHA256='46bc6633b7cdf00f77f2690800f5bf180c187e8438370b5218635e8006dee688'
ARMS=('target_paths3','target_paths2')
BASE_BATCH=126
NAMESPACE=29029
_bytes=SUPPORT_FILE.read_bytes()
if hashlib.sha256(_bytes).hexdigest()!=SUPPORT_SHA256:
    raise ValueError('the prespecified support enumeration has changed')
_PANELS=json.loads(_bytes)['panels']


def seed_value(seed,p,d,kind):
    """63-bit initialization identity; kind1 resets communication modules."""
    return int(np.random.SeedSequence([NAMESPACE,seed,p,d,kind]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))


def groups(p,arm):
    """Sorted int64 map identifiers; the shared primary target has six maps."""
    if p not in (1,2,3) or arm not in ARMS:raise ValueError('unknown panel or support arm')
    panel=_PANELS[p-1];condition=panel['conditions'][arm]
    train=np.sort(np.asarray(condition['training_map_ids'],np.int64))
    target=np.sort(np.asarray(condition['target_map_ids'],np.int64))
    return dict(train18=train,common_target6=target,
        shared_train13=np.asarray(panel['shared_training_map_ids'],np.int64),
        held12=np.setdiff1d(np.arange(30,dtype=np.int64),train),
        other_held6=np.sort(np.asarray(condition['other_held_map_ids'],np.int64)),
        common_unseen7=np.asarray(panel['all_common_unseen_map_ids'],np.int64),
        extra_common_unseen1=np.asarray(panel['extra_common_unseen_map_ids'],np.int64),
        common30=np.arange(30,dtype=np.int64))


def ordered_pool(p,arm):
    """Thirteen shared map slots followed by five arm-specific map slots."""
    g=groups(p,arm)
    return np.concatenate((g['shared_train13'],np.setdiff1d(g['train18'],g['shared_train13'])))


def fixture(seed,p,d,step,arm,train_table):
    """Return 252 indices and independent policy uniforms.

    Every map appears seven times in the 126-row base batch. One shared random
    permutation shuffles those rows; the second half repeats each world's other
    mask. Photo and policy streams are independent of the map permutation and
    arm. Social reception always performs both resource actions; there is no
    sampled private-goal field and no goal is supplied to the sender.
    """
    if d not in (0,1) or not isinstance(step,(int,np.integer)) or step<0:
        raise ValueError('invalid direction or zero-based update')
    pool=ordered_pool(p,arm)
    if len(train_table['map_id'])!=720:raise ValueError('requires the frozen720-row training table')
    photo=np.asarray(train_table['photo_ids'])
    food=np.unique(photo[:,0]);water=np.unique(photo[:,1])
    if photo.shape!=(720,2) or len(food)!=6 or len(water)!=2:
        raise ValueError('requires exactly six training food and two training water rows')
    expected=np.asarray([(f,w) for f in food for w in water],np.int64)
    if not np.array_equal(photo[:12],expected):raise ValueError('table must be food-major/water-minor')
    rng=lambda stream:np.random.default_rng(np.random.SeedSequence([NAMESPACE,int(seed),int(p),int(d),int(step),stream]))
    order=rng(0).permutation(BASE_BATCH)
    maps=np.tile(pool,7)[order]
    photos=rng(1).random((BASE_BATCH,2))
    food_index=(photos[:,0]*6).astype(np.int64);water_index=(photos[:,1]*2).astype(np.int64)
    base=maps*12+food_index*2+water_index
    indices=np.concatenate((base,base+360))
    uniforms=rng(2).random((2*BASE_BATCH,4)).astype(np.float32)
    return dict(indices=indices,uniforms=uniforms)


def subset(w,indices):return {k:v[indices] for k,v in w.items()}
