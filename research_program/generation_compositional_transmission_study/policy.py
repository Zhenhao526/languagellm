"""Policy helpers shared with the frozen action-dependent tabular protocol."""
from __future__ import annotations
import hashlib
import numpy as np
from research_program.action_dependent_signaling_study import policy as base_policy
from . import design

softmax=base_policy.softmax

def clone(p): return {k:np.asarray(v,dtype=np.float64).copy() for k,v in p.items()}
def make_replacement(seed): return base_policy.make_policy(int(seed)+700000,design.FORM)
def combined_hash(p): return base_policy.combined_parameter_hash(p)
def load(path):
    with np.load(path) as z:
        return {"sender_logits_hidden":np.asarray(z["sender_logits_hidden"],dtype=np.float64),"sender_logits_visible":np.asarray(z["sender_logits_visible"],dtype=np.float64),"worker_logits":np.asarray(z["worker_logits"],dtype=np.float64)}
def file_hash(path): return hashlib.sha256(open(path,'rb').read()).hexdigest()
