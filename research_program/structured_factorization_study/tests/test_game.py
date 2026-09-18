from __future__ import annotations
import numpy as np
from research_program.structured_factorization_study import design, environment, policy

def main():
    assert design.parse_child_condition("factorized_conflict_hidden_heldout_combo") == ("factorized","conflict","hidden","heldout_combo")
    ep=design.episode_stream(design.SEEDS[0], design.BATCH_SIZE, support="full", update=1)
    for arch in design.ARCHITECTURES:
        p=policy.make_policy(9001,arch)
        probs,_=policy.sender_arrays(p,ep["goal_meaning"],ep["partner_id"],"hidden")
        assert probs.shape==(design.BATCH_SIZE,2,4) and np.allclose(probs.sum(-1),1.0)
        tr=environment.rollout([p,p],None,ep,arch,"aligned","hidden")
        assert tr["receiver_probs"].shape==(design.BATCH_SIZE,design.SCENE_SIZE)
        assert np.allclose(tr["receiver_probs"].sum(-1),1.0)
    p=policy.make_policy(9002,"factorized")
    state=design.message_index(np.array([[0,1],[-1,-1]],dtype=np.int64))
    assert np.array_equal(design.decode_message_state(state),np.array([[0,1],[-1,-1]]))
    assert environment.recombination_messages([p,p],p,ep,"factorized","aligned","hidden").shape==(design.BATCH_SIZE,2)
    print("structured_factorization_study tests passed")
if __name__=="__main__": main()
