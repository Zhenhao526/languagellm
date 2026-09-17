"""Exact strict/reciprocal reward objectives for the same 17-action policies.

Strict transport requires the third actor to wait. Reciprocal transport sums
over all 17 proposals of the uninvolved actor, so each of 24 disjoint physical
events has only two probability factors. A structural reward table retains
exactly one fully successful transport, though reciprocal joint actions have
17 full-success representatives. No model calls occur at import.
"""
import numbers
import numpy as np
from research_program.triadic_partial_payoff_study import utility as previous

core=previous.core
base=core.base
require,finite=base.require,base.finite
RULES=('strict','reciprocal')
JOINT=base.JOINT_ACTIONS
ACTIVE=JOINT!=0


def _rule(rule):
    require(isinstance(rule,str) and rule in RULES,'Unknown physical settlement rule')


def _event_probability(p,rule):
    factors=np.stack([p[:,actor,JOINT[:,actor]] for actor in range(3)],axis=-1)
    if rule=='reciprocal':factors=np.where(ACTIVE[None],factors,1.)
    return np.prod(factors,axis=-1)


def action_conditioned_rewards(p,rewards,rule):
    """E[R | actor i chooses a] for all 51 actor/action choices, without division."""
    _rule(rule)
    require(p.ndim==3 and p.shape[1:]==(3,17) and rewards.shape==(len(p),24),'Conditional-action shapes')
    value=np.zeros_like(p)
    for actor in range(3):
        remainder=np.ones(rewards.shape)
        for other in range(3):
            if other==actor:continue
            factor=p[:,other,JOINT[:,other]]
            remainder*=np.where(ACTIVE[None,:,other],factor,1.) if rule=='reciprocal' else factor
        weighted=remainder*rewards
        for event in range(24):
            if rule=='reciprocal' and not ACTIVE[event,actor]:
                value[:,actor,:]+=weighted[:,event,None]
            else:value[:,actor,JOINT[event,actor]]+=weighted[:,event]
    finite(value,'Action-conditioned native rewards')
    require(((value>=0)&(value<=1+1e-12)).all(),'Action-conditioned reward range')
    return value


def objective_terms(logits,rewards,rule):
    """Exact expected native reward, stable log reward and analytical gradients.

No epsilon, probability floor, reward rescaling or action masking is used.
Strict's original fields retain the frozen alpha=.5 implementation exactly.
    """
    _rule(rule)
    native=previous.utility_table(rewards,.5)
    logits=np.asarray(logits,dtype=np.float64)
    require(logits.shape==(len(native),3,17),'Incorrect action logit shape')
    finite(logits,'Action logits')
    if rule=='strict':
        terms=previous.objective_terms(logits,native,.5)
        event=_event_probability(terms['probabilities'],rule)
    else:
        p,lp=base.policy_distribution(logits)
        event=_event_probability(p,rule)
        weighted=event*native;J=weighted.sum(1)
        positive=native>0
        log_mass=np.full(native.shape,-np.inf)
        log_mass[positive]=np.log(native[positive])
        for actor in range(3):
            log_mass+=np.where(ACTIVE[None,:,actor],lp[:,actor,JOINT[:,actor]],0.)
        maximum=log_mass.max(1,keepdims=True);finite(maximum,'Largest rewarding-event log mass')
        shifted=np.exp(log_mass-maximum);normalizer=shifted.sum(1,keepdims=True)
        posterior=shifted/normalizer;log_J=(maximum+np.log(normalizer))[:,0]
        finite(log_J,'Stable reciprocal log reward');finite(posterior,'Reciprocal reward posterior')
        require((log_J<=1e-12).all() and (posterior[~positive]==0).all(),'Reciprocal log reward/posterior support')
        log_gradient=np.zeros_like(p);mean_gradient=np.zeros_like(p)
        for actor in range(3):
            participates=ACTIVE[:,actor]
            log_gradient[:,actor]=-p[:,actor]*posterior[:,participates].sum(1)[:,None]
            mean_gradient[:,actor]=-p[:,actor]*weighted[:,participates].sum(1)[:,None]
            for event_index in np.flatnonzero(participates):
                choice=JOINT[event_index,actor]
                log_gradient[:,actor,choice]+=posterior[:,event_index]
                mean_gradient[:,actor,choice]+=weighted[:,event_index]
        terms=dict(probabilities=p,log_probabilities=lp,J=J,log_J=log_J,
            mean_J_logit_gradient=mean_gradient,log_J_logit_gradient=log_gradient,posterior_weights=posterior,
            native_expected_reward=J.copy(),full_success_probability=(event*(native==1)).sum(1),
            partial_success_probability=(event*(native==.5)).sum(1),
            full_success_posterior_mass=(posterior*(native==1)).sum(1),
            partial_success_posterior_mass=(posterior*(native==.5)).sum(1),utility_values=native)
        for key in ('mean_J_logit_gradient','log_J_logit_gradient'):finite(terms[key],key)
    terms.update(rule=rule,execution_probability=event.sum(1),
        action_conditioned_expected_reward=action_conditioned_rewards(terms['probabilities'],native,rule))
    require((terms['execution_probability']<=1+1e-12).all(),'Disjoint physical-event probability exceeds one')
    return terms


def training_gradients(networks,observations,rewards,live,uniforms,update,rule):
    """Frozen discrete rollouts, two-trajectory LOO and action entropy schedule."""
    _rule(rule)
    x=np.asarray(observations,dtype=np.float64);native=previous.utility_table(rewards,.5)
    require(x.shape==(len(native),3,54),'Incorrect training observations');finite(x,'Training observations')
    require(isinstance(live,(bool,np.bool_)),'Live channel flag must be boolean')
    require(isinstance(update,numbers.Integral) and not isinstance(update,(bool,np.bool_))
            and 1<=update<=base.CONFIG['updates'],'Invalid training update')
    B=len(x);uniforms=np.asarray(uniforms,dtype=np.float64)
    require(uniforms.shape==(2,B,2,3,4) and np.isfinite(uniforms).all()
            and ((uniforms>=0)&(uniforms<1)).all(),'Two valid full trajectory uniforms required')
    trace=core.rollout(networks,np.concatenate((x,x)),live,uniforms.reshape(2*B,2,3,4))
    terms=objective_terms(trace['action_logits'],np.concatenate((native,native)),rule)
    receiver=core.coordination.loss_and_derivative(terms,'mean_log_J',update)
    entropy,_=base.entropy_and_logit_gradient(terms['probabilities'],terms['log_probabilities'])
    F=terms['log_J']+receiver['entropy_coefficient']*entropy
    sender=core.paired_sender_derivative(trace['sender_probabilities'].reshape(2,B,2,3,4,8),
        trace['sender_log_probabilities'].reshape(2,B,2,3,4,8),trace['messages'].reshape(2,B,2,3,4),F.reshape(2,B))
    ds=sender['derivative'].reshape(2*B,2,3,4,8);gradients=[]
    for actor in range(3):
        for module in range(3):
            derivative=receiver['derivative'][:,actor] if module==2 else ds[:,module,actor].reshape(2*B,32)
            gradients.append(base.actor_backward(networks[3*actor+module],trace['caches'][3*actor+module],derivative))
    row=dict(partial_success_utility=.5,mean_expected_utility=float(terms['J'].mean()),
        mean_log_expected_utility=float(terms['log_J'].mean()),min_log_expected_utility=float(terms['log_J'].min()),
        max_log_expected_utility=float(terms['log_J'].max()),zero_float_expected_utility_states=int((terms['J']==0).sum()),
        mean_native_expected_reward=float(terms['native_expected_reward'].mean()),
        mean_full_success_probability=float(terms['full_success_probability'].mean()),
        mean_partial_success_probability=float(terms['partial_success_probability'].mean()),
        mean_full_success_posterior_mass=float(terms['full_success_posterior_mass'].mean()),
        mean_partial_success_posterior_mass=float(terms['partial_success_posterior_mass'].mean()),
        mean_F=float(F.mean()),receiver_loss=receiver['loss'],mean_actor_entropy=receiver['mean_actor_entropy'],
        entropy_coefficient=receiver['entropy_coefficient'],sender_surrogate_loss=sender['surrogate_loss'],
        sender_advantage_mean=float(sender['advantage'].mean()),sender_advantage_abs_mean=float(np.abs(sender['advantage']).mean()),
        sender_advantage_squared_mean=float(np.square(sender['advantage']).mean()),
        sender_advantage_max_abs=float(np.abs(sender['advantage']).max()),
        sender_mean_complete_log_score=float(sender['log_scores'].mean()),sampled_messages_sha256=core.array_sha(trace['messages']),
        settlement_rule=rule,mean_execution_probability=float(terms['execution_probability'].mean()))
    return gradients,row
