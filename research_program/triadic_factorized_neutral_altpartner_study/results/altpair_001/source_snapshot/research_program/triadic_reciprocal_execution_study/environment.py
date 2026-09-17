"""Two execution rules over unchanged native states and all17 proposals.

This module contains no actor, network, observation builder, or model call.
Settlement never sees the correct plan or a reward table. Truth enters only
the separate researcher-side summary of completed proposals.
"""
from itertools import combinations,product

import numpy as np

from research_program.triadic_action_dependency_study import environment as original

RULES=('strict','reciprocal')
PAIRS=tuple(combinations(range(3),2))
PAIR_NAMES=('AB','AC','BC')
PROPOSAL_ROLES=np.full((3,17),-1,dtype=np.int8)
for _actor in range(3):
    _others=tuple(a for a in range(3) if a!=_actor)
    for _action in range(1,17):PROPOSAL_ROLES[_actor,_action]=_others[(_action-1)%2]
STRUCTURAL_PLANS=tuple((i,j,site,destination) for i,j in PAIRS for site,destination in product(range(4),range(2)))
STRUCTURAL_ACTIONS=np.zeros((24,3),dtype=np.int16)
for _index,(_i,_j,_site,_destination) in enumerate(STRUCTURAL_PLANS):
    for _actor,_partner in ((_i,_j),(_j,_i)):
        STRUCTURAL_ACTIONS[_index,_actor]=1+4*_site+2*_destination+tuple(a for a in range(3) if a!=_actor).index(_partner)
ACCEPTS=np.asarray([[[original.accepts(need,material,destination) for destination in range(2)]
                    for material in range(4)] for need in range(24)],dtype=bool)


def require(ok,message):
    if not ok:raise ValueError(message)


def _states(value):
    states=np.asarray(value)
    require(states.ndim==2 and states.shape[1:]==(10,) and len(states)>0 and states.dtype.kind in 'iu',
            'States must be nonempty integer [N,10]')
    require(np.all((states[:,:3]>=0)&(states[:,:3]<24)),'Invalid need ids')
    require(np.all(np.sort(states[:,3:7],axis=1)==np.arange(4)),'Each layout must contain every material once')
    require(np.all(np.sort(states[:,7:10],axis=1)==np.arange(1,4)),'Private-site owners must be a permutation')
    return states


def _actions(value,n):
    actions=np.asarray(value)
    require(actions.shape==(n,3) and actions.dtype.kind in 'iu' and np.all((actions>=0)&(actions<17)),
            'Proposals must be integer [N,3] with every action in0..16')
    return actions.astype(np.int64,copy=False)


def settle(states,actions,rule='strict'):
    """Execute at most one matching pair; never choose by needs or reward.

ignored_proposal denotes an unexecuted nonwait third proposal *while another
pair executes*. It is always false under strict. unexecuted_proposal additionally
includes nonwait proposals in completely failed worlds.
    """
    require(rule in RULES,'Unknown execution rule')
    states=_states(states);actions=_actions(actions,len(states));n=len(states)
    roles=PROPOSAL_ROLES[np.arange(3),actions]
    sites=np.where(actions>0,(actions-1)//4,-1)
    destinations=np.where(actions>0,((actions-1)//2)%2,-1)
    matches=np.zeros((n,3),dtype=bool)
    for pair,(i,j) in enumerate(PAIRS):
        matched=(roles[:,i]==j)&(roles[:,j]==i)&(sites[:,i]==sites[:,j])&(destinations[:,i]==destinations[:,j])
        if rule=='strict':matched &= actions[:,3-i-j]==0
        matches[:,pair]=matched
    require(np.all(matches.sum(axis=1)<=1),'More than one matching pair in a three-agent state')
    executes=matches.any(axis=1)
    pair_index=np.where(executes,matches.argmax(axis=1),-1).astype(np.int8)
    executed=np.zeros((n,3),dtype=bool)
    for pair,(i,j) in enumerate(PAIRS):
        executed[:,i] |= matches[:,pair];executed[:,j] |= matches[:,pair]
    material_per_proposal=np.take_along_axis(states[:,3:7],np.maximum(sites,0),axis=1)
    satisfied=executed & ACCEPTS[states[:,:3],material_per_proposal,np.maximum(destinations,0)]
    executed_roles=np.where(executed,roles,-1).astype(np.int8)
    first=np.asarray([p[0] for p in PAIRS])[np.maximum(pair_index,0)]
    rows=np.arange(n)
    executed_site=np.where(executes,sites[rows,first],-1).astype(np.int8)
    executed_material=np.where(executes,material_per_proposal[rows,first].astype(np.int8,copy=False),-1).astype(np.int8)
    executed_destination=np.where(executes,destinations[rows,first],-1).astype(np.int8)
    unexecuted=(actions>0)&~executed
    ignored=unexecuted & executes[:,None]
    require(np.all(executed.sum(axis=1)==2*executes),'Execution must contain exactly one pair or nobody')
    return dict(executed=executed,satisfied=satisfied,greedy_reward=satisfied.sum(axis=1)/2,
        executed_roles=executed_roles,actual_pair_index=pair_index,ignored_proposal=ignored,
        proposal_roles=roles,unexecuted_proposal=unexecuted,executed_site=executed_site,
        executed_material=executed_material,executed_destination=executed_destination)


def truth_from_rewards(states,rewards):
    """Unique correct executed pair/site/destination from old24 native rewards.

This is a researcher-only target. Under reciprocal, the full joint proposal
has17 alternatives because the third proposal is irrelevant to this execution.
    """
    states=_states(states);rewards=np.asarray(rewards)
    require(rewards.shape==(len(states),24) and rewards.dtype.kind in 'fiu' and
            np.isfinite(rewards).all() and np.isin(rewards,(0.,.5,1.)).all(),'Invalid original native reward table')
    require(np.all((rewards==1).sum(axis=1)==1),'Truth requires one original full-success structural plan')
    selected=(rewards==1).argmax(axis=1)
    plans=np.asarray(STRUCTURAL_PLANS,dtype=np.int8)[selected]
    rows=np.arange(len(states));material=states[rows,3+plans[:,2]]
    require(np.all(ACCEPTS[states[rows,plans[:,0]],material,plans[:,3]]) and
            np.all(ACCEPTS[states[rows,plans[:,1]],material,plans[:,3]]),'Supplied full-plan reward contradicts native needs')
    actions=STRUCTURAL_ACTIONS[selected]
    return dict(correct_actions=actions,correct_roles=PROPOSAL_ROLES[np.arange(3),actions],
                correct_pair_index=(selected//8).astype(np.int8),correct_site=plans[:,2],
                correct_material=material.astype(np.int8),correct_destination=plans[:,3])


def _rows_counts(values,key):
    unique,counts=np.unique(values,axis=0,return_counts=True)
    return [{key:row.tolist(),'worlds':int(count)} for row,count in zip(unique,counts)]


def metrics(data,truth,actions,rule):
    """Describe all worlds, with proposal roles distinct from actual roles.

Content metrics are global target-axis matches conditional only on an actual
execution. An incorrect executing partner pair may still get an axis right.
No execution is a miss, and all worlds remain in each denominator.
    """
    require(rule in RULES,'Unknown execution rule')
    n=len(data['greedy_reward']);actions=_actions(actions,n)
    physical=np.any(data['executed'],axis=1)
    target_material=truth['correct_material'];actual_material=data['executed_material']
    correct=dict(
        full_success_rate=data['greedy_reward']==1,
        physical_execution_rate=physical,
        executed_partner_correct_rate=physical & (data['actual_pair_index']==truth['correct_pair_index']),
        proposal_role_success_rate=np.all(data['proposal_roles']==truth['correct_roles'],axis=1),
        kind_correct_rate=physical & (actual_material//2==target_material//2),
        length_correct_rate=physical & (actual_material%2==target_material%2),
        destination_correct_rate=physical & (data['executed_destination']==truth['correct_destination']),
        material_identity_correct_rate=physical & (actual_material==target_material),
        ignored_proposal_world_rate=np.any(data['ignored_proposal'],axis=1))
    require(np.all(~correct['full_success_rate'] | correct['executed_partner_correct_rate']),'Full success needs correct executed pair')
    def summarize_mask(mask):
        count=int(mask.sum())
        result={'worlds':count,'reward_mean':float(data['greedy_reward'][mask].mean()) if count else None}
        result.update({key:float(values[mask].mean()) if count else None for key,values in correct.items()})
        result['ignored_proposal_agent_rate']=float(data['ignored_proposal'][mask].mean()) if count else None
        result['unexecuted_proposal_agent_rate']=float(data['unexecuted_proposal'][mask].mean()) if count else None
        return result
    result=summarize_mask(np.ones(n,dtype=bool));result['rule']=rule
    result.update(
        reward_counts={str(value):int((data['greedy_reward']==value).sum()) for value in (0.,.5,1.)},
        raw_joint_action_counts=_rows_counts(actions,'action_indices'),
        raw_proposal_role_counts=_rows_counts(data['proposal_roles'],'partner_indices'),
        raw_executed_role_counts=_rows_counts(data['executed_roles'],'partner_indices'),
        actual_pair_counts={name:int((data['actual_pair_index']==index).sum()) for index,name in ((-1,'none'),*enumerate(PAIR_NAMES))},
        ignored_proposal_counts_by_actor=data['ignored_proposal'].sum(axis=0).astype(int).tolist(),
        true_pair_strata={name:summarize_mask(truth['correct_pair_index']==pair) for pair,name in enumerate(PAIR_NAMES)},
        metric_scope={'content_axes':'Actual execution and equality to the unique global correct-plan axis; no correct-partner gate; denominator all worlds.',
                      'proposal_roles':'All three original partner/wait proposals equal the original unique-plan roles, regardless of physical execution.',
                      'executed_partner':'An actual mutual pair exists and equals the unique correct pair; location/destination need not be correct.',
                      'ignored':'Nonwait third proposal while another pair executes; failed unmatched proposals separately counted as unexecuted.',
                      'actor_rates':'Denominator3 times all worlds; never conditional on having proposed.'})
    return result


def summarize(states,actions,rule,rewards):
    states=_states(states);actions=_actions(actions,len(states))
    data=settle(states,actions,rule);truth=truth_from_rewards(states,rewards)
    return metrics(data,truth,actions,rule)
