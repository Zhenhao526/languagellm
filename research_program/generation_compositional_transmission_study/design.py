"""Frozen design for transmitting a staged dual-token protocol."""
from __future__ import annotations
import hashlib
import numpy as np
from research_program.action_dependent_signaling_study import design as base

HORIZON=base.HORIZON; ACTION_START=base.ACTION_START; SUBTASKS=base.SUBTASKS; STEPS_PER_SUBTASK=base.STEPS_PER_SUBTASK
FORM="dual2"; PARTNER_MODE="rotating"; VISIBILITY="hidden"; TASK="factorized"; PROTOCOL="staged"
WORKERS=base.WORKERS; CAPACITY=base.CAPACITY; MESSAGE_SLOTS=base.MESSAGE_SLOTS; NULL_MESSAGE=base.NULL_MESSAGE
ACTION_COUNT=base.ACTION_COUNT; UPDATES=3000; BATCH_SIZE=512; LEARNING_RATE=0.08
ENTROPY_INITIAL=base.ENTROPY_INITIAL; ENTROPY_ZERO_AFTER=base.ENTROPY_ZERO_AFTER
CORRECT_REWARD=base.CORRECT_REWARD; WRONG_REWARD=base.WRONG_REWARD
SEEDS=tuple(range(76101,76133)); ROLES=("worker","sender"); CHANNELS=("live","silent","permuted")
CONDITIONS=tuple(f"{r}_{c}" for r in ROLES for c in CHANNELS)
CHECKPOINTS=(0,500,1000,2000,3000); TARGET_WORKER=0
PARENT_CONDITION="dual2_rotating_hidden_live_factorized_staged"
EVAL_SEED_OFFSET=930000


def require(ok,msg):
    if not ok: raise ValueError(msg)

def parse_condition(c):
    x=c.split("_"); require(len(x)==2 and x[0] in ROLES and x[1] in CHANNELS,f"bad condition {c}"); return x[0],x[1]

def entropy_coefficient(update):
    require(1<=update<=UPDATES,"invalid update"); return ENTROPY_INITIAL*max(0.0,1.0-(update-1)/ENTROPY_ZERO_AFTER)

def array_sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def episode_stream(seed,count,*,evaluation=False,update=0):
    return base.episode_stream(int(seed),"rotating","hidden",TASK,count,evaluation=evaluation,update=update)

def prepare(parent_root,parent_hashes):
    return {"schema":"generation_compositional_transmission_v1","horizon":HORIZON,"action_start":ACTION_START,"subtasks":SUBTASKS,"steps_per_subtask":STEPS_PER_SUBTASK,"form":FORM,"partner_mode":PARTNER_MODE,"visibility":VISIBILITY,"task":TASK,"protocol":PROTOCOL,"workers":WORKERS,"capacity":CAPACITY,"target_worker":TARGET_WORKER,"seeds":list(SEEDS),"roles":list(ROLES),"channels":list(CHANNELS),"conditions":list(CONDITIONS),"updates":UPDATES,"batch_size":BATCH_SIZE,"learning_rate":LEARNING_RATE,"entropy_initial":ENTROPY_INITIAL,"entropy_zero_after":ENTROPY_ZERO_AFTER,"checkpoints":list(CHECKPOINTS),"parent_condition":PARENT_CONDITION,"parent_root":str(parent_root),"parent_checkpoint_sha256":parent_hashes,"pairing":"child channel arms share goals, worlds, partners, message uniforms and action uniforms","replacement":"replace worker 0 or hidden sender; complementary parent parameters remain frozen","controls":["live","silent","partner-permuted"],"composition_rule":"heldout recombined−natural and recovery of the parent staged slot map","no_teacher_or_language_prior":True}
