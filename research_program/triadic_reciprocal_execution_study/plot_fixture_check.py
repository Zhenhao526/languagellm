"""In-memory arithmetic/schema fixture only; no measured results or figures."""
from copy import deepcopy
import numpy as np
from . import plot_results as p


def summary(worlds,rule,full_numerator):
    full=full_numerator/12;physical=.5 if rule=='strict' else 1.
    ignored=0. if rule=='strict' else .5
    rates=dict(full_success_rate=full,reward_mean=full,executed_partner_correct_rate=physical,
        proposal_role_success_rate=.5,physical_execution_rate=physical,kind_correct_rate=full,
        length_correct_rate=full,destination_correct_rate=full,material_identity_correct_rate=full,
        ignored_proposal_world_rate=ignored,ignored_proposal_agent_rate=ignored/3,
        unexecuted_proposal_agent_rate=.5 if rule=='strict' else 1/6)
    proposals=((1,1,0),(1,1,1),(2,0,1),(2,1,1),(0,2,2),(1,2,2))
    counter={}
    for actions in proposals:
        roles=tuple(-1 if a==0 else tuple(j for j in range(3) if j!=i)[(a-1)%2] for i,a in enumerate(actions))
        counter[roles]=counter.get(roles,0)+worlds//6
    return dict(worlds=worlds,rule=rule,**rates,reward_counts={'0.0':worlds-round(full*worlds),'0.5':0,'1.0':round(full*worlds)},
        actual_pair_counts={'none':worlds//2 if rule=='strict' else 0,**{key:worlds//6 if rule=='strict' else worlds//3 for key in ('AB','AC','BC')}},
        ignored_proposal_counts_by_actor=[0,0,0] if rule=='strict' else [worlds//6]*3,
        raw_joint_action_counts=[dict(action_indices=list(a),worlds=worlds//6) for a in proposals],
        raw_proposal_role_counts=[dict(partner_indices=list(r),worlds=n) for r,n in counter.items()],
        true_pair_strata={key:dict(worlds=worlds//3,**rates) for key in ('AB','AC','BC')})


def make_record(worlds,trained,matrix):
    values={settle:summary(worlds,settle,matrix[trained][settle]) for settle in p.RULES}
    result=deepcopy(values[trained]);result['cross_settlement']=values
    result.update(information='FI',live=False,path='synthetic-not-a-measurement.npz',data_sha256='a'*64,state_indices_sha256='b'*64,
        expected_reward_given_greedy_messages=.2,full_probability_given_greedy_messages=.1,
        execution_probability_given_greedy_messages=.6,full_posterior_mass_given_greedy_messages=.5)
    return result


def fixture():
    matrices=((4,6,2,3),(3,5,3,4),(2,4,4,8),(5,9,1,2))
    runs=[];paired=[]
    for seed,(ss,sr,rs,rr) in zip(p.SEEDS,matrices):
        matrix={'strict':{'strict':ss,'reciprocal':sr},'reciprocal':{'strict':rs,'reciprocal':rr}}
        cells={}
        for rule in p.RULES:
            final={part:make_record(p.WORLD_COUNTS[part],rule,matrix) for part in p.PARTS}
            monitor=[dict(update=step,monitor={part:make_record(p.MONITOR_COUNTS[part],rule,matrix) for part in p.PARTS}) for step in p.STEPS]
            runs.append(dict(seed=seed,rule=rule,condition='FI_silent',updates=6000,final=final,monitor=monitor))
            cells[rule]={key:final[p.TARGET][key] for key in p.METRICS}
        paired.append(dict(seed=seed,contrasts={key:cells['reciprocal'][key]-cells['strict'][key] for key in p.METRICS},
            cells=cells,full_success_cross_settlement={r:{q:matrix[r][q]/12 for q in p.RULES} for r in p.RULES},
            decomposition=dict(common_reciprocal_policy_difference=(rr-sr)/12,strict_policy_mechanical_release=(sr-ss)/12,
                common_strict_policy_difference=(rs-ss)/12,reciprocal_policy_mechanical_release=(rr-rs)/12)))
    return dict(status='completed',plan_sha256='c'*64,
        budget=dict(runs=8,training_updates=48000,training_world_samples=12288000,message_trajectories=24576000,
            categorical_symbol_samples=589824000,training_forward_module_samples=221184000,checkpoints=48,
            actual_monitor_files=192,actual_final_files=32,aliased_evaluations=0,actual_monitor_worlds=1032192,
            actual_final_worlds=6193152),runs=runs,
        primary=dict(metric='full_success_rate',partition=p.TARGET,paired_seeds=paired,
            mean_difference=float(np.mean([row['contrasts']['full_success_rate'] for row in paired]))))


def check():
    source=fixture();result=p.extract(source)
    differences=[r['contrasts']['full_success_rate'] for r in result['primary']['paired_seeds']]
    assert min(differences)<0<max(differences)
    assert abs(result['primary']['mean_difference']-1/16)<1e-12
    common=[r['decomposition']['common_reciprocal_policy_difference'] for r in result['primary']['paired_seeds']]
    assert min(common)<0<max(common)
    negative_checks=0
    for mutation in ('wrong_mean','missing_seed','changed_cross_actions','incomplete_status','wrong_monitor_denominator'):
        value=deepcopy(source)
        if mutation=='wrong_mean':value['primary']['mean_difference']+=.1
        elif mutation=='missing_seed':value['runs'].pop()
        elif mutation=='changed_cross_actions':value['runs'][0]['final'][p.TARGET]['cross_settlement']['reciprocal']['raw_joint_action_counts'][0]['action_indices'][0]=3
        elif mutation=='incomplete_status':value['status']='running'
        else:value['runs'][0]['monitor'][0]['monitor'][p.TARGET]['worlds']=53568
        try:p.extract(value)
        except ValueError:negative_checks+=1
        else:raise AssertionError('Failed to reject '+mutation)
    return dict(status='passed_synthetic_plot_extraction',positive_fixture=True,negative_checks=negative_checks,
        fixture_primary_mean=1/16,all_paired_signs_retained=True,figures_generated=0,formal_results_read=0,
        model_forward_samples=0,training_updates=0,scope='Synthetic schema/arithmetic fixture, not physically simulated policies or empirical data.')


if __name__=='__main__':
    import json
    print(json.dumps(check(),ensure_ascii=False))
