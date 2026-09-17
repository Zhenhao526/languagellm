"""In-memory schema/arithmetic fixture; no formal data, figures, or model calls."""
from copy import deepcopy
import json
from . import plot_results as p


def fixture():
    result={'status':'completed','plan_sha256':'0'*64,'budget':dict(runs=8,training_updates=48000,
        checkpoints=48,natural_monitor_files=192,natural_final_files=32,closed_final_files=16,aliased_evaluations=0),'runs':[]}
    silent=(.1,.2,.3,.4);live=(.05,.3,.15,.6)
    for i,seed in enumerate(p.SEEDS):
        for condition in p.CONDITIONS:
            is_live=condition=='PL_live';final={}
            for part in p.PARTS:
                modes={}
                for mode in (('natural','closed') if is_live else ('natural',)):
                    q=.02 if mode=='closed' else (live[i] if is_live else silent[i])
                    response=dict(partition=part,worlds=p.WORLD_COUNTS[part],complete_nine_strata=True,
                        empty_strata=[],Q=q,Q_shuffle=.01,Q_excess=q-.01,
                        strata=[dict(changed_person=a,axis=x,Q=q+(3*a+j-4)*.002) for a in range(3) for j,x in enumerate(p.AXES)])
                    modes[mode]=dict(worlds=p.WORLD_COUNTS[part],information='PL',rule='reciprocal',mode=mode,
                        live=is_live and mode=='natural',intervention='close_cross_agent_channel_from_window_1' if mode=='closed' else None,
                        need_response=response,full_success_rate=.25,executed_partner_correct_rate=.4,reward_mean=.3)
                final[part]=modes
            monitor=[dict(update=step,monitor={part:dict(worlds=p.MONITOR_COUNTS[part],information='PL',
                rule='reciprocal',mode='natural',live=is_live,full_success_rate=.1,executed_partner_correct_rate=.2)
                for part in p.PARTS}) for step in p.STEPS]
            result['runs'].append(dict(seed=seed,condition=condition,rule='reciprocal',updates=6000,final=final,monitor=monitor))
    result['primary']=dict(metric='Q',partition=p.TARGET,contrast='PL_live_minus_PL_silent',independent_paired_seed_blocks=4,
        mean_difference=sum(a-b for a,b in zip(live,silent))/4,
        paired_seeds=[dict(seed=seed,Q_live=live[i],Q_silent=silent[i],difference=live[i]-silent[i]) for i,seed in enumerate(p.SEEDS)])
    return result


def main():
    source=fixture();data=p.extract(source)
    assert abs(data['primary']['mean_difference']-.025)<1e-12
    assert len(data['rows'])==8 and len(data['primary']['paired_seeds'])==4
    differences=[r['Q_difference'] for r in data['primary']['paired_seeds']]
    assert min(differences)<0<max(differences)
    invalid=[]
    r=deepcopy(source);r['status']='running';invalid.append(r)
    r=deepcopy(source);r['runs'][-1]=r['runs'][0];invalid.append(r)
    r=deepcopy(source);r['primary']['mean_difference']=.9;invalid.append(r)
    r=deepcopy(source);r['runs'][0]['monitor'][0]['monitor'][p.TARGET]['need_response']={};invalid.append(r)
    r=deepcopy(source);r['runs'][0]['final'][p.TARGET]['closed']['live']=True;invalid.append(r)
    r=deepcopy(source);r['runs'][0]['final'][p.TARGET]['natural']['need_response']['strata'][0]['Q']=.9;invalid.append(r)
    for bad in invalid:
        try:p.extract(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid synthetic schema accepted')
    print(json.dumps(dict(status='passed',fixture_policy_fits=8,negative_checks=len(invalid),
        retained_positive_and_negative_differences=True,formal_result_reads=0,neural_forward_calls=0,figures_rendered=0)))


if __name__=='__main__':main()
