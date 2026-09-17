"""Fixed connected-versus-three-products supports and balanced fixtures.

Only finite map metadata and external random streams are handled here.
No images, models, learned predictions, action feedback, or token meanings.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
SUPPORT_FILE=ROOT.parent/'paper_program/nonmatching_design_20260916/nonmatching_candidates.json'
SUPPORT_SHA256='796a1e3ac0b48c81897bac539d010a74abb6b8ad6c7e47638df38ba35f6e636a'
ARMS=('connected_cycle12','three_products')
BASE_BATCH=120
NAMESPACE=30030
_data=SUPPORT_FILE.read_bytes()
if hashlib.sha256(_data).hexdigest()!=SUPPORT_SHA256:
    raise ValueError('the fixed nonmatching support selection has changed')
_PANELS=json.loads(_data)['panels']


def seed_value(seed,p,d,kind):
    return int(np.random.SeedSequence([NAMESPACE,seed,p,d,kind]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))


def groups(p,arm):
    """Sorted int64 IDs; target12 is identical between arms in each panel."""
    if p not in (1,2,3) or arm not in ARMS:raise ValueError('unknown panel or support arm')
    panel=_PANELS[p-1];condition=panel['conditions'][arm]
    train=np.sort(np.asarray(condition['training_map_ids'],np.int64))
    return dict(train12=train,target12=np.asarray(panel['common_test_map_ids'],np.int64),
        shared_train9=np.asarray(panel['shared_training_map_ids'],np.int64),
        held18=np.setdiff1d(np.arange(30,dtype=np.int64),train),
        other_held6=np.asarray(condition['remaining_untrained_map_ids'],np.int64),
        common_unseen15=np.asarray(panel['all_common_untrained_map_ids'],np.int64),
        extra_common_unseen3=np.asarray(panel['extra_common_untrained_map_ids'],np.int64),
        common30=np.arange(30,dtype=np.int64))


def ordered_pool(p,arm):
    g=groups(p,arm)
    return np.concatenate((g['shared_train9'],np.setdiff1d(g['train12'],g['shared_train9'])))


def fixture(seed,p,d,step,arm,train_table):
    """120 balanced base worlds, followed by the same worlds' other mask.

    Stream0 permutes ten copies of the12-map pool. Independent stream1 chooses
    photos and stream2 supplies two-token/two-action uniforms. Arm never enters
    a seed. Both resource actions are performed; there is no private-goal draw.
    """
    if d not in (0,1) or not isinstance(step,(int,np.integer)) or step<0:
        raise ValueError('invalid direction or zero-based update')
    pool=ordered_pool(p,arm)
    if len(train_table['map_id'])!=720:raise ValueError('requires the frozen720-row training table')
    photo=np.asarray(train_table['photo_ids']);food=np.unique(photo[:,0]);water=np.unique(photo[:,1])
    if photo.shape!=(720,2) or len(food)!=6 or len(water)!=2:
        raise ValueError('requires six training food and two training water images')
    expected=np.asarray([(f,w) for f in food for w in water],np.int64)
    if not np.array_equal(photo[:12],expected):raise ValueError('table must be food-major/water-minor')
    rng=lambda stream:np.random.default_rng(np.random.SeedSequence([NAMESPACE,int(seed),int(p),int(d),int(step),stream]))
    maps=np.tile(pool,10)[rng(0).permutation(BASE_BATCH)]
    photos=rng(1).random((BASE_BATCH,2))
    base=maps*12+(photos[:,0]*6).astype(np.int64)*2+(photos[:,1]*2).astype(np.int64)
    return dict(indices=np.concatenate((base,base+360)),
        uniforms=rng(2).random((2*BASE_BATCH,4)).astype(np.float32))


def subset(w,indices):return {k:v[indices] for k,v in w.items()}
