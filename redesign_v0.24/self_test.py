"""Information-flow and numerical checks, no experimental training."""
import copy,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));import run_attention as run
import focus_model as m

def main():
    torch.set_num_threads(1);source=ROOT.parent/'redesign_v0.23/results/smoke_001'
    a,ma=run.restore(99523,1,'mean',source);b,mb=run.restore(99523,1,'attention',source)
    assert all(torch.equal(v,b[d].state_dict()[k]) for d in (0,1) for k,v in a[d].state_dict().items())
    rng=np.random.default_rng(24024);p=torch.from_numpy(rng.normal(size=(32,2,6)).astype(np.float32)).softmax(-1)
    x=torch.cat((p,torch.eye(2)[None].expand(32,-1,-1)),-1);u=rng.random((32,4),dtype=np.float32)
    pos=np.array([(i%6,(i+1)%6) for i in range(32)],np.int64);errors=[]
    for agents in (a,b):
        sl,rl,tr=m.direction_loss(*agents,x,u,pos,.02)
        gl=m.trainable_groups(agents[0])['sender']+m.trainable_groups(agents[1])['receiver'];n=len(m.trainable_groups(agents[0])['sender'])
        sg=torch.autograd.grad(sl,gl,retain_graph=True,allow_unused=True);rg=torch.autograd.grad(rl,gl,allow_unused=True)
        assert all(g is not None and torch.isfinite(g).all() for g in sg[:n]) and all(g is None for g in sg[n:])
        assert all(g is None for g in rg[:n]) and all(g is not None for g in rg[n:])
        if agents[0].focus_sender.mode=='mean':assert torch.count_nonzero(sg[7])==0 # bilinear.weight
        _,_,other=m.direction_loss(*agents,x,u,(pos+2)%6,.02)
        for k in ('messages','first_logits','second_logits','actions','action_logits'):assert np.array_equal(tr[k],other[k]),k
        lp=m.message_log_probs(agents[0],x);errors.append(float((lp.exp().sum(-1)-1).abs().max().detach()));assert errors[-1]<1e-6
        code=7*tr['messages'][:,0]+tr['messages'][:,1]
        assert np.array_equal(lp.detach().numpy()[np.arange(len(x)),code],tr['sender_logp'])
        tok,aw=m.greedy_messages(agents[0],x);assert torch.allclose(aw.sum(-1),torch.ones(32,2))
        keys,state,first,_=agents[0].focus_sender.start(x)
        assert torch.equal(tok[:,0],first.argmax(-1))
        assert torch.equal(tok[:,1],agents[0].focus_sender.step(keys,tok[:,0],state)[1].argmax(-1))
        assert np.allclose(m.old.receiver_logits(agents[1]).detach().numpy()[code],tr['action_logits'],atol=2e-7,rtol=2e-6)
        assert all(v.grad is None for person in agents for v in person.parameters())
    out=dict(passed=True,experimental_training=False,counts=ma[0]['focus']['counts'],max_joint_errors=errors,
        source_hashes={str(ROOT/n):run.sha(ROOT/n) for n in ('focus_model.py','run_attention.py','self_test.py')},
        checks=['paired tensor identity across arms','only discrete cross-agent link and disjoint role gradients','gold positions change reward only',
                'mean bilinear zero gradient','49 sequence probabilities normalized and sampled prefix matched','sequential greedy and attention simplex',
                'receiver full table agrees with direct message forward'])
    run.write(ROOT/'self_test_qa.json',out);print(json.dumps(out))
if __name__=='__main__':main()
