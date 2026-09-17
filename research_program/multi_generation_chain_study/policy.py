"""Policy and deterministic replacement helpers for the chain."""
from __future__ import annotations
import hashlib
import numpy as np
from research_program.action_dependent_signaling_study import policy as base
from . import design

softmax=base.softmax
sample=base.sample

def clone(p): return {k:np.asarray(v,dtype=np.float64).copy() for k,v in p.items()}
def combined_hash(p): return base.combined_parameter_hash(p)
def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k],dtype=np.float64).copy() for k in ("sender_logits_hidden","sender_logits_visible","worker_logits")}
def file_hash(path): return hashlib.sha256(open(path,"rb").read()).hexdigest()
def make_replacement(seed,generation,role):
    design.require(generation in design.GENERATIONS and role in ("worker","sender"),"bad replacement")
    return base.make_policy(int(seed)+design.REPLACEMENT_OFFSETS[generation],design.FORM)
def replace(p,seed,generation,role):
    out=clone(p); q=make_replacement(seed,generation,role)
    if role=="worker": out["worker_logits"][design.TARGET_WORKER]=q["worker_logits"][0]
    else:
        out["sender_logits_hidden"]=q["sender_logits_hidden"]
        out["sender_logits_visible"]=q["sender_logits_visible"]
    return out
def save(path,p,update):
    np.savez_compressed(path,update=np.array(update,dtype=np.int64),sender_logits_hidden=p["sender_logits_hidden"],sender_logits_visible=p["sender_logits_visible"],worker_logits=p["worker_logits"])
    return file_hash(path)
