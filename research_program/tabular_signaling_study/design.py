"""Frozen design for an exactly auditable tabular signaling control."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

HORIZON=6
ACTION_START=1
MESSAGE_ROUNDS=(0,)
ALPHABET_SIZE=8
NULL_MESSAGE=ALPHABET_SIZE
ACTION_COUNT=3
MEMORIES=("stateless","recurrent")
SCARCITIES=("scarce","abundant")
INFORMATIONS=("PI","FI")
CHANNELS=("silent","live")
TASKS=("persistent","switching")
SEEDS=(68101,68102,68103,68104,68105,68106,68107,68108)
UPDATES=3000
BATCH_SIZE=512
LEARNING_RATE=0.08
ENTROPY_INITIAL=0.01
ENTROPY_ZERO_AFTER=5000
CORRECT_REWARD=1.0
WRONG_REWARD=-0.25
EVAL_SEED_OFFSET=900000

# Sender message table is [context, sender-local-type, token].  The worker
# table is [memory token, time, local type, local inventory, visible goal,
# action].  The visible-goal slot is fixed to zero in PI.
CONTEXTS={"persistent":2,"switching":8}
WORKER_STATE_SHAPE=(ALPHABET_SIZE+1,HORIZON,2,HORIZON+1,2,ACTION_COUNT)


def require(ok,msg):
    if not ok: raise ValueError(msg)


def parse_condition(condition):
    parts=condition.split("_"); require(len(parts)==5,f"bad condition {condition}")
    m,s,i,c,t=parts
    require(m in MEMORIES and s in SCARCITIES and i in INFORMATIONS and c in CHANNELS and t in TASKS,'unknown condition')
    return m,s,i,c,t


CONDITIONS=tuple(f"{m}_{s}_{i}_{c}_{t}" for m in MEMORIES for s in SCARCITIES for i in INFORMATIONS for c in CHANNELS for t in TASKS)


def capacity(scarcity):
    require(scarcity in SCARCITIES,'unknown scarcity')
    return 2 if scarcity=='scarce' else HORIZON


def task_index(task):
    require(task in TASKS,'unknown task')
    return TASKS.index(task)


def entropy_coefficient(update):
    require(1<=update<=UPDATES,'invalid update')
    return ENTROPY_INITIAL*max(0.,1.-(update-1)/ENTROPY_ZERO_AFTER)


def source_hash(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_stream(seed,scarcity,task,count,*,evaluation=False,update=0):
    require(count>0 and task in TASKS and scarcity in SCARCITIES,'bad episode factor')
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),EVAL_SEED_OFFSET if evaluation else 0,task_index(task),int(update)]))
    site_type=rng.integers(0,2,size=(count,2),dtype=np.int8)
    if task=='persistent':
        bit=rng.integers(0,2,size=(count,2),dtype=np.int8)
        goal=np.repeat(bit[:,:,None],HORIZON,axis=2); context=bit[:,0].astype(np.int16)
    else:
        patterns=np.array([[0,0,0,1,1,1],[1,1,1,0,0,0],[0,0,1,1,0,0],[1,1,0,0,1,1],
                           [0,1,0,1,0,1],[1,0,1,0,1,0],[0,1,1,1,0,0],[1,0,0,0,1,1]],dtype=np.int8)
        choices=np.array([4,5,6,7] if evaluation else [0,1,2,3])
        context=rng.choice(choices,size=count).astype(np.int16)
        # Only the scout's request is operational in this role task; retain a
        # second latent sequence for pairing diagnostics.
        goal_a=patterns[context]
        goal_b=patterns[rng.choice(choices,size=count)]
        goal=np.stack([goal_a,goal_b],axis=1)
    return {'site_type':site_type,'site0_type':site_type[:,0].copy(),'sender':np.zeros(count,dtype=np.int8),
            'goal':goal,'context':context,'message_uniforms':rng.random((count,HORIZON,2)),
            'action_uniforms':rng.random((count,HORIZON,2)),
            'capacity':np.full(count,capacity(scarcity),dtype=np.int8),
            'seed':int(seed),'scarcity':scarcity,'task':task,'evaluation':bool(evaluation),'update':int(update)}


def prepare():
    return {'schema':'tabular_signaling_v1','horizon':HORIZON,'action_start':ACTION_START,
            'message_rounds':list(MESSAGE_ROUNDS),'blackout_rounds':[t for t in range(HORIZON) if t not in MESSAGE_ROUNDS],
            'alphabet_size':ALPHABET_SIZE,'null_message':NULL_MESSAGE,
            'actions':['wait','take_site0','take_site1'],'seeds':list(SEEDS),'conditions':list(CONDITIONS),
            'tasks':list(TASKS),'updates':UPDATES,'batch_size':BATCH_SIZE,'learning_rate':LEARNING_RATE,
            'entropy_initial':ENTROPY_INITIAL,'entropy_zero_after':ENTROPY_ZERO_AFTER,'contexts':CONTEXTS,
            'worker_state_shape':list(WORKER_STATE_SHAPE),'resource_capacity':{'scarce':2,'abundant':HORIZON},
            'task_pressure':'fixed scout privately observes the target; worker receives one token then must choose a site',
            'information':'PI hides target; FI exposes target to the worker','reward':'+1 correct available site; -0.25 wrong/depleted; wait 0',
            'pairing':'same worlds, requests, fixed role and score uniforms across channel and scarcity controls',
            'no_teacher_or_language_prior':True,'evaluation':'training-support and held-out switching contexts with natural/closed/permuted controls',
            'automatic_followon_experiment':False}
