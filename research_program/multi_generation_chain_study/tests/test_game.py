"""Fast invariants for the alternating chain."""
from __future__ import annotations
import numpy as np
from research_program.action_dependent_signaling_study import policy as base_policy
from research_program.action_dependent_signaling_study import environment as base_environment
from research_program.multi_generation_chain_study import design,environment,policy

def main():
    assert design.parse_condition("slot_local_live")==("slot_local","live")
    ep=design.episode_stream(76101,2048,generation=1,update=1)
    assert not np.any((ep["goal"][:,0]*2+ep["goal"][:,1])==design.heldout_goal(76101))
    p=base_policy.make_policy(12,design.FORM)
    q=policy.replace(p,12,1,"worker")
    tr=environment.rollout(q,ep,"slot_local","live",sample=False,partner_filter=design.TARGET_WORKER)
    assert tr["actions"].shape==(len(ep["goal"]),design.HORIZON)
    assert max(np.asarray(x).max() for x in tr["state"] if x is not None)<=2
    # The generalized environment must reproduce the frozen baseline exactly
    # when the representation is joint_history.
    a=environment.rollout(q,ep,"joint_history","live",sample=False,partner_filter=design.TARGET_WORKER)
    b=base_environment.rollout(q,ep,design.FORM,design.PARTNER_MODE,design.VISIBILITY,design.TASK,"live",design.PROTOCOL,sample=False,partner_filter=design.TARGET_WORKER)
    for key in ("messages","selected_messages","actions","rewards","team_return","active"):
        assert np.array_equal(a[key],b[key]),key
    q2=policy.replace(q,12,2,"sender")
    tr2=environment.rollout(q2,ep,"slot_local","live",sample=False)
    assert tr2["active"].all()
    # Replacement is component-local: worker replacement leaves the sender
    # untouched, while sender replacement leaves every worker table intact.
    q_worker=policy.replace(p,12,1,"worker")
    assert np.array_equal(q_worker["sender_logits_hidden"],p["sender_logits_hidden"])
    assert np.array_equal(q_worker["sender_logits_visible"],p["sender_logits_visible"])
    assert np.array_equal(q_worker["worker_logits"][1:],p["worker_logits"][1:])
    q_sender=policy.replace(p,12,2,"sender")
    assert np.array_equal(q_sender["worker_logits"],p["worker_logits"])
    print("multi_generation_chain_study tests passed")
if __name__=="__main__": main()
