"""Fixed-policy reward-score moments. No optimization or model-state mutation.

Float32 network logits are converted to float64 for categorical log_softmax.
Autograd differentiates that graph with respect to original float32 parameters.
The resulting full-message score vectors are accumulated in float64. This is
not a claim of exact finite-precision PRNG-bin probabilities.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
import numpy as np
import torch
from torch.nn import functional as F

POLICY_PREFIXES=('send_context.','send_embedding.','send_recur.','send_out.')
PROB_ATOL=5e-13
SCORE_ZERO_RTOL=2e-6
MOMENT_RTOL=1e-9
NEAR_ZERO_A=1e-12


def policy_parameters(sender):
    named=[(k,p) for k,p in sender.named_parameters() if k.startswith(POLICY_PREFIXES)]
    if len(named)!=9 or sum(p.numel() for _,p in named)!=43319:
        raise ValueError('expected all nine tensors / 43319 parameters of the fixed v15 sender policy')
    if any(p.dtype!=torch.float32 or p.device.type!='cpu' or not p.requires_grad for _,p in named):
        raise ValueError('sender-policy parameters must be differentiable CPU float32')
    return named


def _policy_graph(sender,h):
    local=torch.cat((h,h.new_zeros(1,2),h.new_zeros(1,2)),-1)
    state=sender.send_context(local)
    first_logits=sender.send_out(state).squeeze(0)
    rows=[]
    for first in range(7):
        token=torch.tensor([first],dtype=torch.int64,device=h.device)
        next_state=sender.send_recur(sender.send_embedding(token),state)
        rows.append(sender.send_out(next_state).squeeze(0))
    second_logits=torch.stack(rows)
    first_logp=F.log_softmax(first_logits.double(),dim=-1)
    second_logp=F.log_softmax(second_logits.double(),dim=-1)
    joint=(first_logp[:,None]+second_logp).reshape(49)
    return local,first_logits,second_logits,first_logp,second_logp,joint


def _softmax(v):
    v=np.asarray(v,np.float64);v=np.exp(v-v.max(-1,keepdims=True));return v/v.sum(-1,keepdims=True)


def target_moments(receiver_logits,positions):
    """Two conditional independent receiver draws; target T in {-1,-.75,0}."""
    if isinstance(receiver_logits,torch.Tensor):receiver_logits=receiver_logits.detach().cpu().numpy()
    logits=np.asarray(receiver_logits)
    if logits.shape!=(49,2,6) or not np.isfinite(logits).all():raise ValueError('expected finite receiver logits[49,2,6]')
    position=np.asarray(positions)
    if position.shape!=(2,) or not np.issubdtype(position.dtype,np.integer) or not (0<=position).all() or not (position<6).all() or position[0]==position[1]:
        raise ValueError('expected two distinct integer physical resource locations')
    q=_softmax(logits);pf=q[:,0,int(position[0])];pw=q[:,1,int(position[1])]
    none=(1-pf)*(1-pw);one=pf*(1-pw)+(1-pf)*pw
    et=-none-.75*one;et2=none+.5625*one
    assert np.all(et>=-1-PROB_ATOL) and np.all(et<=PROB_ATOL)
    assert np.all(et2>=et*et-PROB_ATOL) and np.all(et2<=1+PROB_ATOL)
    return et,et2,q


def _score_moments(probability,et,et2,scores,baseline):
    pi=np.asarray(probability,np.float64);et=np.asarray(et,np.float64);et2=np.asarray(et2,np.float64);z=np.asarray(scores,np.float64)
    assert pi.ndim==1 and et.shape==et2.shape==pi.shape and z.shape[0]==len(pi)
    assert np.isfinite(z).all() and np.isfinite(pi).all() and (pi>=0).all() and abs(pi.sum()-1)<=PROB_ATOL
    norm=np.einsum('mp,mp->m',z,z,dtype=np.float64)
    ez=np.sum(pi[:,None]*z,axis=0,dtype=np.float64);mu=np.sum((pi*et)[:,None]*z,axis=0,dtype=np.float64)
    A=float(pi@norm);B=float((pi*et)@norm);C=float((pi*et2)@norm)
    ez2=float(ez@ez);mu2=float(mu@mu);ez_l2=float(np.sqrt(ez2));limit=SCORE_ZERO_RTOL*max(1.,float(np.sqrt(A)))
    assert ez_l2<=limit,('score mean residual',ez_l2,limit)
    b=float(baseline)
    if not np.isfinite(b):raise ValueError('baseline must be finite')
    optimal=B/A if A>0 else None
    if optimal is not None:assert -1-MOMENT_RTOL<=optimal<=MOMENT_RTOL
    summary=dict(A=A,B=B,C=C,mu_norm_sq=mu2,expected_score_norm_sq=ez2,expected_score_l2=ez_l2,
        expected_score_max_abs=float(np.abs(ez).max(initial=0)),expected_score_relative=ez_l2/max(np.sqrt(A),np.finfo(np.float64).tiny),
        expected_score_gate_limit=limit,mu_dot_expected_score=float(mu@ez),
        mu_dot_mean_score=float(mu@ez),mean_score_norm_sq=ez2,baseline=b,
        optimal_baseline=optimal,zero_A=A==0.,near_zero_A=A<=NEAR_ZERO_A,
        expected_target=float(pi@et),expected_target_sq=float(pi@et2),probability_mass=float(pi.sum()),
        matched_second_moment=C-2*b*B+b*b*A,
        matched_mean_shift_l2=abs(b)*ez_l2,
        matched_mean_norm_sq_finite_precision=float((mu-b*ez)@(mu-b*ez)))
    return norm,mu,ez,summary


def measure_world(sender,h,receiver_logits,positions,return_scores=False):
    """Return {'arrays': NumPy arrays, 'summary': JSON-only metadata/moments}.

    h is the already-scaled original v15 observe output, shape[1,96]. No second
    scaling is performed. The hidden goal/inventory inputs are both zero, as in
    the source task. receiver_logits refer to physical locations in lexicographic
    full-message order 7*first+second. Optional scores/mu/Ez are development-only.
    """
    if not isinstance(return_scores,bool):raise TypeError('return_scores must be bool')
    if not isinstance(h,torch.Tensor) or h.shape!=(1,96) or h.dtype!=torch.float32 or h.device.type!='cpu' or not bool(torch.isfinite(h).all()):
        raise ValueError('expected finite effective h[1,96] on CPU in float32')
    if (sender.vocab,sender.length,sender.width)!=(7,2,96):raise ValueError('expected v15 7-symbol two-token sender')
    if not hasattr(sender,'visual_scale') or hasattr(sender.send_context,'h_scale'):
        raise ValueError('use original v15 ScaledCampAgent, not v16 branch wrapper')
    named=policy_parameters(sender);params=[p for _,p in named]
    et,et2,receiver_q=target_moments(receiver_logits,positions)
    with torch.enable_grad():
        local,first,second,first_lp,second_lp,joint=_policy_graph(sender,h.detach())
        pi=joint.detach().exp().cpu().numpy()
        scores=np.empty((49,sum(p.numel() for p in params)),dtype=np.float64)
        for message in range(49):
            # The complete log probability is differentiated once. Both token
            # scores contribute before squaring, including their cross term.
            gradients=torch.autograd.grad(joint[message],params,retain_graph=message<48,create_graph=False,allow_unused=False)
            offset=0
            for gradient in gradients:
                flat=gradient.detach().reshape(-1).cpu().numpy();scores[message,offset:offset+len(flat)]=flat;offset+=len(flat)
        with torch.no_grad():baseline=float(sender.send_value(local).squeeze())
    norm,mu,ez,summary=_score_moments(pi,et,et2,scores,baseline)
    summary.update(policy_parameter_count=scores.shape[1],policy_tensor_count=len(named),
        policy_parameter_layout=[dict(name=k,shape=list(p.shape),numel=p.numel()) for k,p in named],
        forward_dtype='float32',categorical_dtype='float64 log_softmax of float32 logits in same autograd graph',
        parameter_gradient_dtype='float32, then float64 accumulation',score_scope='all sender policy parameters; reward-score only, excluding value/frozen perception/entropy',
        prefix_batch_size=1,message_count=49,score_matrix_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
        tolerances=dict(probability_mass_atol=PROB_ATOL,score_zero_rtol=SCORE_ZERO_RTOL,moment_rtol=MOMENT_RTOL,near_zero_A=NEAR_ZERO_A))
    arrays=dict(message_ids=np.asarray([(a,b) for a in range(7) for b in range(7)],np.int64),probability=pi,
        expected_target=et,expected_target_sq=et2,score_norm_sq=norm,
        first_logits=first.detach().cpu().numpy().copy(),second_logits=second.detach().cpu().numpy().copy(),
        receiver_probabilities=receiver_q,positions=np.asarray(positions,np.int64).copy(),baseline=np.asarray(baseline,np.float64))
    if return_scores:arrays.update(score_matrix=scores,mu=mu,expected_score=ez)
    return dict(arrays=arrays,summary=summary)


def aggregate_support(world_summaries):
    """Analytic n-world conditional variances; never refit or change a baseline."""
    if not world_summaries:raise ValueError('at least one fixed world is required')
    A,B,C,mu2,b=[np.asarray([w[key] for w in world_summaries],np.float64) for key in ('A','B','C','mu_norm_sq','baseline')]
    if not all(np.isfinite(v).all() for v in (A,B,C,mu2,b)) or (A<0).any():raise ValueError('invalid moment summaries')
    n=len(A);bmean=float(b.mean());b2mean=float((b*b).mean());positive=A>0
    best=np.zeros(n);best[positive]=B[positive]/A[positive]
    assert ((A!=0)|(B==0)).all() and ((A!=0)|(C==0)).all() and ((A!=0)|(mu2==0)).all()
    terms_matched=C-2*b*B+b*b*A-mu2
    terms_permuted=C-2*bmean*B+b2mean*A-mu2
    terms_optimal=C-mu2;terms_optimal[positive]-=B[positive]**2/A[positive]
    Vmatch=float(terms_matched.sum()/n**2);Vperm=float(terms_permuted.sum()/n**2);Vbest=float(terms_optimal.sum()/n**2)
    excess=float(np.sum(A*(b-best)**2)/n**2)
    perm_excess=float(np.sum(A*((bmean-best)**2+max(0.,b2mean-bmean*bmean)))/n**2)
    scale=max(1.,float(np.sum(np.abs(C)+2*np.abs(b*B)+b*b*A+mu2)/n**2),abs(Vperm));tol=MOMENT_RTOL*scale
    # No negative variance is clipped to zero. Residuals are exposed and gated.
    assert min(Vmatch,Vperm,Vbest)>=-tol,('negative variance beyond tolerance',Vmatch,Vperm,Vbest,tol)
    assert abs((Vmatch-Vbest)-excess)<=tol
    assert abs((Vperm-Vbest)-perm_excess)<=tol
    assert Vmatch-Vbest>=-tol and Vperm-Vbest>=-tol
    return dict(world_count=n,support='fixed worlds, independent messages/receiver actions; uniform permutation expectation',
        baseline_mean=bmean,baseline_second_moment=b2mean,baseline_std=float(np.std(b)),
        variance_matched=Vmatch,variance_permuted=Vperm,variance_optimal=Vbest,
        permutation_minus_matched=Vperm-Vmatch,matched_excess_over_optimal=Vmatch-Vbest,
        permutation_excess_over_optimal=Vperm-Vbest,weighted_optimal_distance=excess,
        permutation_weighted_optimal_distance=perm_excess,optimal_identity_residual=(Vmatch-Vbest)-excess,
        permutation_optimal_identity_residual=(Vperm-Vbest)-perm_excess,
        role_weight=.5,role_variance_multiplier=.25,
        role_variance_matched=.25*Vmatch,role_variance_permuted=.25*Vperm,role_variance_optimal=.25*Vbest,
        role_permutation_minus_matched=.25*(Vperm-Vmatch),zero_A_count=int((A==0).sum()),near_zero_A_count=int((A<=NEAR_ZERO_A).sum()),
        optimal_baselines=[float(best[i]) if positive[i] else None for i in range(n)],
        max_expected_score_l2=max(w['expected_score_l2'] for w in world_summaries),
        max_expected_score_relative=max(w['expected_score_relative'] for w in world_summaries),
        numeric_tolerance=tol,variance_clamped=False,
        exclusions=['world/photo sampling','entropy gradients and covariance','value regression gradients','gradient clipping','Adam','longitudinal partner adaptation'])


def self_test():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'redesign_v0.15'))
    import scaled_interface as old
    from torch import nn
    from itertools import product,permutations
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99513)
        agent=old.CampAgent(nn.Sequential(nn.Linear(3,64)),7,2);old.attach_scale(agent,5.)
        old._social_flags(agent)
        h=torch.randn(1,96)*.5;logits=np.linspace(-2.,2.,49*2*6,dtype=np.float64).reshape(49,2,6)
        state={k:v.clone() for k,v in agent.state_dict().items()};rng=torch.random.get_rng_state().clone()
        result=measure_world(agent,h,logits,[1,4],True);repeat=measure_world(agent,h,logits,[1,4],False)
        assert result['summary']==repeat['summary'] and 'score_matrix' not in repeat['arrays']
        assert torch.equal(rng,torch.random.get_rng_state())
        assert all(torch.equal(v,agent.state_dict()[k]) for k,v in state.items())
        assert all(p.grad is None for p in agent.parameters())
        arr=result['arrays'];q=arr['receiver_probabilities']
        for m in range(49):
            expected=np.zeros(2)
            for f,w in product(range(6),repeat=2):
                F=float(f==1);W=float(w==4);T=.25*(F+W)+.5*F*W-1
                expected+=q[m,0,f]*q[m,1,w]*np.asarray([T,T*T])
            assert np.allclose(expected,[arr['expected_target'][m],arr['expected_target_sq'][m]],atol=2e-15,rtol=0)
        # Independent double-precision network copy is a sensitivity check, not
        # a claim that the original float32 network has a double trajectory.
        double=copy.deepcopy(agent).double();parameters=[p for k,p in double.named_parameters() if k.startswith(POLICY_PREFIXES)]
        _,_,_,lp1,lp2,joint=_policy_graph(double,h.double());score64=[]
        for m in range(49):score64.append(torch.cat([g.reshape(-1) for g in torch.autograd.grad(joint[m],parameters,retain_graph=True)]).detach().numpy())
        score64=np.asarray(score64);error=float(np.abs(score64-arr['score_matrix']).max())
        assert np.allclose(score64,arr['score_matrix'],atol=2e-6,rtol=2e-4),error
        # Token score cross terms: separately computed factors recover the full score.
        m=0;g1=torch.autograd.grad(lp1[0],parameters,retain_graph=True,allow_unused=True);g2=torch.autograd.grad(lp2[0,0],parameters,retain_graph=True)
        z1=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(g1,parameters)]).detach().numpy()
        z2=torch.cat([g.reshape(-1) for g in g2]).detach().numpy()
        assert np.allclose(z1+z2,score64[m],atol=1e-12,rtol=1e-10)
        assert abs(np.sum(score64[m]**2)-(z1@z1+z2@z2+2*z1@z2))<1e-10
        # Double finite differences across all four policy module families.
        picks=[('send_context.0.weight',0),('send_embedding.weight',0),('send_recur.weight_ih',0),('send_out.weight',0)]
        offset={};n=0
        for name,p in double.named_parameters():
            if name.startswith(POLICY_PREFIXES):offset[name]=n;n+=p.numel()
        fd_errors=[]
        for name,index in picks:
            p=dict(double.named_parameters())[name];view=p.reshape(-1);original=float(view[index].detach());eps=1e-5
            with torch.no_grad():
                view[index]=original+eps;plus=float(_policy_graph(double,h.double())[-1][0])
                view[index]=original-eps;minus=float(_policy_graph(double,h.double())[-1][0]);view[index]=original
            fd=(plus-minus)/(2*eps);err=abs(fd-score64[0,offset[name]+index]);fd_errors.append(err);assert err<1e-7
        # Exhaust all six permutations on synthetic centered two-parameter scores.
        pi=np.asarray([.2,.3,.5]);scores=np.asarray([[1.,0.],[0.,1.],[-.4,-.6]])
        moments=[]
        for et,b in [(np.asarray([-.8,-.4,-.2]),-.2),(np.asarray([-.2,-.7,-.4]),-.8),(np.asarray([-.5,-.6,-.1]),.1)]:
            _,_,_,s=_score_moments(pi,et,et*et+.01,scores,b);moments.append(s)
        agg=aggregate_support(moments);manual=[]
        for order in permutations(range(3)):
            terms=[w['C']-2*moments[order[i]]['baseline']*w['B']+moments[order[i]]['baseline']**2*w['A']-w['mu_norm_sq'] for i,w in enumerate(moments)]
            manual.append(sum(terms)/9)
        assert abs(np.mean(manual)-agg['variance_permuted'])<1e-14
        _,_,_,zero=_score_moments(np.ones(2)/2,np.zeros(2),np.zeros(2),np.zeros((2,2)),3.)
        z=aggregate_support([zero]);assert z['zero_A_count']==1 and z['optimal_baselines']==[None] and z['variance_matched']==0
    return dict(passed=True,policy_parameters=43319,score_float32_vs_double_max_abs=error,finite_difference_max_abs=max(fd_errors),
        checks=['All49 messages and36 receiver actions with ET2 distinct from squared ET',
        'Full9-tensor score includes both-token cross term','Float32-score/double accumulation compared to double network and finite differences',
        'Expected score gate; no model state/grad/RNG mutation','Optional score output leaves summary unchanged',
        'All six synthetic baseline permutations match analytic expectation','Weighted-optimal identity and zero-A case'],
        formal_inference_run=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true',required=True);p.parse_args()
    print(json.dumps(self_test(),ensure_ascii=False))
