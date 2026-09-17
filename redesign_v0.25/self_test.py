"""Finite flow/initial-function checks; no experimental training."""
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));import run_joint as run
import joint_model as model

def main():
    torch.set_num_threads(1);ref=ROOT.parent/'redesign_v0.24/results/smoke_001'
    agents=run.restore(99523,1,ref);states=torch.load(ref/'social/s99523_p1_mean/initial.pt',weights_only=True);oldagents=[]
    for d,new in enumerate(agents):
        a=copy.deepcopy(new);run.previous.communication.initialize(a,'mean',run.previous.seed_value(99523,1,d));a.load_state_dict(states[d]);oldagents.append(a)
        for name,v in new.state_dict().items():
            if name=='focus_sender.contrast.weight':assert torch.count_nonzero(v)==0
            else:assert torch.equal(v,states[d][name]),name
        rng=torch.random.get_rng_state().clone();model.JointSender(a.focus_sender);assert torch.equal(rng,torch.random.get_rng_state())
    cache=run.load_cache(99523,1,ref)
    for d in (0,1):
        for lo in range(0,960,256):
            x=cache['test',d][lo:lo+256]
            with torch.no_grad():
                assert torch.equal(model.message_log_probs(agents[d],x),run.previous.communication.message_log_probs(oldagents[d],x))
                assert torch.equal(model.greedy_messages(agents[d],x),run.previous.communication.greedy_messages(oldagents[d],x)[0])
        assert torch.equal(model.old.receiver_logits(agents[d]),model.old.receiver_logits(oldagents[d]))
    x=cache['test',0][:32];u=np.random.default_rng(25025).random((32,4),dtype=np.float32)
    w=dict(np.load(ref/'test_worlds.npz'));pos=w['positions'][:32]
    sl,rl,tr=model.direction_loss(*agents,x,u,pos,.02);_,_,original=run.previous.communication.direction_loss(*oldagents,x,u,pos,.02)
    for k,v in tr.items():assert np.array_equal(v,original[k]),k
    pars=model.trainable_groups(agents[0])['sender']+model.trainable_groups(agents[1])['receiver'];n=len(model.trainable_groups(agents[0])['sender'])
    sg=torch.autograd.grad(sl,pars,retain_graph=True,allow_unused=True);rg=torch.autograd.grad(rl,pars,allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in sg[:n]) and all(g is None for g in sg[n:])
    assert all(g is None for g in rg[:n]) and all(g is not None for g in rg[n:]);assert float(sg[7].norm())>0
    _,_,changed=model.direction_loss(*agents,x,u,(pos+2)%6,.02)
    for k in ('messages','first_logits','second_logits','actions','action_logits'):assert np.array_equal(tr[k],changed[k]),k
    out=dict(passed=True,experimental_training=False,initial_sender_worlds=1920,initial_receiver_tables=2,initial_trace_equal_to_mean=True,
        contrast_initial_gradient_norm=float(sg[7].norm()),source_hashes={str(ROOT/n):run.sha(ROOT/n) for n in ('joint_model.py','run_joint.py','self_test.py')},
        checks=['initial complete49 probabilities and greedy equal historical mean on all dev worlds','shared initial tensors exact and contrast zero',
                'initial sampled trace equal mean; RNG preserved on conversion','discrete cross-agent link only; disjoint gradients','gold positions affect rewards only',
                'contrast has nonzero first-step gradient; no claim of matched updates'])
    run.write(ROOT/'self_test_qa.json',out);print(json.dumps(out))
if __name__=='__main__':main()
