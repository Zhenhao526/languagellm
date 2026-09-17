"""Frozen v0.31 population schedule and balanced multi-partner fixtures."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
DESIGN_FILE=ROOT/'support_design.json'
DESIGN_SHA256='df69de1f994802ec6256af4b62e23d87c96663affc54bf343a8768d961c286da'
_design_bytes=DESIGN_FILE.read_bytes()
if hashlib.sha256(_design_bytes).hexdigest()!=DESIGN_SHA256:raise ValueError('prespecified v0.31 design changed')
_DESIGN=json.loads(_design_bytes)
CONDITIONS=tuple(_DESIGN['conditions'])
ARMS=CONDITIONS  # compatibility name used by the shared metric tests
NAMESPACE=int(_DESIGN['sampling']['namespace']);BASE_BATCH=int(_DESIGN['sampling']['base_worlds_per_agent_type_direction'])
MAPS=np.asarray([(i,j) for i in range(6) for j in range(6) if i!=j],np.int64);_INDEX={tuple(x):i for i,x in enumerate(MAPS)}
_PANELS=[tuple(x) for x in _DESIGN['panels']];_TRAIN=tuple(tuple(x) for x in _DESIGN['training12_pairs']);_TARGET=tuple(tuple(x) for x in _DESIGN['target12_pairs'])

def _mapped(p,pairs):
    q=_PANELS[p-1];return np.sort(np.asarray([_INDEX[(q[i],q[j])] for i,j in pairs],np.int64))

def groups(p,condition):
    if p not in (1,2,3) or condition not in CONDITIONS:raise ValueError('unknown panel or partner condition')
    train=_mapped(p,_TRAIN);target=_mapped(p,_TARGET)
    return dict(train12=train,target12=target,held18=np.setdiff1d(np.arange(30,dtype=np.int64),train),common30=np.arange(30,dtype=np.int64))

def ordered_pool(p,condition):return groups(p,condition)['train12']

def matching(condition,step):
    if condition not in CONDITIONS or step<0:raise ValueError('invalid partner condition or step')
    key='fixed_matchings' if condition=='fixed_partners' else 'rotating_matchings';sets=_DESIGN['population'][key]
    return tuple(tuple(pair) for pair in sets[0 if condition=='fixed_partners' else step%len(sets)])

def seed_value(seed,p,agent,kind):
    return int(np.random.SeedSequence([NAMESPACE,int(seed),int(p),int(agent),int(kind)]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1))

def fixture(seed,p,agent_type,step,condition,train_table):
    if agent_type not in (0,1) or not isinstance(step,(int,np.integer)) or step<0 or condition not in CONDITIONS:raise ValueError('invalid fixture identity')
    if len(train_table['map_id'])!=720:raise ValueError('requires frozen 720-row source training table')
    photo=np.asarray(train_table['photo_ids']);food=np.unique(photo[:,0]);water=np.unique(photo[:,1])
    if photo.shape!=(720,2) or len(food)!=6 or len(water)!=2 or not np.array_equal(photo[:12],np.asarray([(f,w) for f in food for w in water],np.int64)):raise ValueError('requires six-by-two food-major source photos')
    rng=lambda stream:np.random.default_rng(np.random.SeedSequence([NAMESPACE,int(seed),int(p),int(agent_type),int(step),int(stream)]))
    maps=np.tile(ordered_pool(p,condition),10)[rng(0).permutation(BASE_BATCH)]
    photos=rng(1).random((BASE_BATCH,2));base=maps*12+(photos[:,0]*6).astype(np.int64)*2+(photos[:,1]*2).astype(np.int64)
    return dict(indices=np.concatenate((base,base+360)),uniforms=rng(2).random((2*BASE_BATCH,4)).astype(np.float32))

def subset(w,indices):return {k:v[indices] for k,v in w.items()}
