"""Frozen design for an alternating multi-generation protocol chain."""
from __future__ import annotations
import hashlib
import numpy as np
from research_program.heldout_composition_study import design as heldout
from research_program.action_dependent_signaling_study import design as action

HORIZON=action.HORIZON
ACTION_START=action.ACTION_START
SUBTASKS=action.SUBTASKS
STEPS_PER_SUBTASK=action.STEPS_PER_SUBTASK
FORM="dual2"
PARTNER_MODE="rotating"
VISIBILITY="hidden"
TASK="factorized"
PROTOCOL="staged"
WORKERS=action.WORKERS
CAPACITY=action.CAPACITY
MESSAGE_SLOTS=action.MESSAGE_SLOTS
NULL_MESSAGE=action.NULL_MESSAGE
ACTION_COUNT=action.ACTION_COUNT
UPDATES=3000
BATCH_SIZE=512
LEARNING_RATE=0.08
ENTROPY_INITIAL=action.ENTROPY_INITIAL
ENTROPY_ZERO_AFTER=action.ENTROPY_ZERO_AFTER
CORRECT_REWARD=action.CORRECT_REWARD
WRONG_REWARD=action.WRONG_REWARD
SEEDS=tuple(range(76101,76133))
REPRESENTATIONS=("joint_history","slot_local")
CHANNELS=("live","silent")
GENERATIONS=(1,2,3)
EVENT_ROLES={1:"worker",2:"sender",3:"worker"}
SUPPORT="leave_one_out"
CONDITIONS=tuple(f"{representation}_{channel}" for representation in REPRESENTATIONS for channel in CHANNELS)
CHECKPOINTS=(0,500,1000,2000,3000)
TARGET_WORKER=0
PARENT_CONDITION="dual2_rotating_hidden_live_factorized_staged"
EVAL_SEED_OFFSET=960000
TRAIN_GENERATION_OFFSET=100000
REPLACEMENT_OFFSETS={1:700000,2:710000,3:720000}


def require(ok,msg):
    if not ok: raise ValueError(msg)

def parse_condition(condition):
    representation,channel=condition.rsplit("_",1)
    require(representation in REPRESENTATIONS and channel in CHANNELS,f"bad condition {condition}")
    return representation,channel

def entropy_coefficient(update):
    require(1<=update<=UPDATES,"invalid update")
    return ENTROPY_INITIAL*max(0.0,1.0-(update-1)/ENTROPY_ZERO_AFTER)

def array_sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def heldout_goal(seed): return heldout.heldout_goal(seed)

def episode_stream(seed,count,*,generation,update,support=SUPPORT,heldout_goal_index=None):
    require(generation in GENERATIONS,"bad generation")
    if heldout_goal_index is None: heldout_goal_index=heldout_goal(seed)
    stream_update=int(generation)*TRAIN_GENERATION_OFFSET+int(update)
    return heldout.episode_stream(int(seed),count,evaluation=False,update=stream_update,support=support,heldout=int(heldout_goal_index))

def balanced_eval_stream(seed,count,goal_kind,heldout_goal_index):
    return heldout.balanced_eval_stream(int(seed),count,goal_kind,int(heldout_goal_index))

def prepare(parent_root,parent_hashes,parent_composable):
    return {
      "schema":"multi_generation_chain_study_v1","horizon":HORIZON,"action_start":ACTION_START,
      "subtasks":SUBTASKS,"steps_per_subtask":STEPS_PER_SUBTASK,"form":FORM,"partner_mode":PARTNER_MODE,
      "visibility":VISIBILITY,"task":TASK,"protocol":PROTOCOL,"workers":WORKERS,"capacity":CAPACITY,
      "target_worker":TARGET_WORKER,"seeds":list(SEEDS),"representations":list(REPRESENTATIONS),
      "channels":list(CHANNELS),"conditions":list(CONDITIONS),"generations":list(GENERATIONS),
      "event_roles":{str(k):v for k,v in EVENT_ROLES.items()},"support":SUPPORT,"updates":UPDATES,
      "batch_size":BATCH_SIZE,"learning_rate":LEARNING_RATE,"entropy_initial":ENTROPY_INITIAL,
      "entropy_zero_after":ENTROPY_ZERO_AFTER,"checkpoints":list(CHECKPOINTS),"parent_condition":PARENT_CONDITION,
      "parent_root":str(parent_root),"parent_checkpoint_sha256":parent_hashes,
      "parent_composable":{str(k):bool(v) for k,v in parent_composable.items()},
      "heldout_assignment":{str(s):heldout_goal(s) for s in SEEDS},
      "replacement_offsets":{str(k):v for k,v in REPLACEMENT_OFFSETS.items()},
      "training_generation_offset":TRAIN_GENERATION_OFFSET,
      "pairing":"live and silent chain branches share parent, generation streams, worlds, goals, partners, message uniforms and action uniforms at every event",
      "chain":"generation 1 replaces worker 0, generation 2 replaces hidden sender, generation 3 replaces worker 0 again; the resulting branch is passed to the next event",
      "representation_control":"joint_history indexes complete staged message history; slot_local indexes only the current staged slot for worker 0; workers 1-3 always use complete history",
      "controls":["live","from-scratch silent","whole-sequence permutation","slot recombination","parent composability stratum"],
      "zero_shot_rule":"heldout live natural team return >= 0.60 and absolute recombined-natural gap <= 0.02",
      "no_teacher_or_language_prior":True,
    }
