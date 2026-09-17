"""Descriptive report tables from a complete frozen run; no model evaluation."""
import argparse,json,hashlib
from pathlib import Path
from statistics import mean
STEPS=(0,100,500,1500,3000,6000)
AXES=('kind','length','destination')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(path):
    path=Path(path).resolve();result=read(path/'execution/results.json')
    assert result['status']=='completed' and read(path/'execution/status.json')['status']=='completed'
    sources={str(path/'execution/results.json'):sha(path/'execution/results.json')}
    def bound(p,digest):
        assert sha(p)==digest
        sources[str(p)]=digest
        return read(p)
    policies={}
    for record in result['policies']:
        summary=bound(record['path'],record['sha256'])
        points={int(t):bound(r['path'],r['sha256']) for t,r in summary['point_summaries'].items()}
        cross={(r['receiver_update'],r['donor_update']):bound(r['path'],r['sha256']) for r in summary['cross_summaries']}
        policies[record['seed'],record['condition']]=(summary,points,cross)
    measures=('target_apt','current_apt','unchanged_requirements_apt','native_reward','native_team_full',
              'counterfactual_reward','counterfactual_team_full','physical_pair_executed','packet_in_train_endpoint_greedy_set')
    rows=[]
    for (seed,condition),(s,points,cross) in policies.items():
        for t in STEPS:
            p=points[t];c=p['current'];z=p['discovery'];sel=c['selected']
            rows.append(dict(seed=seed,condition=condition,update=t,
                **{k:c[k] for k in ('diagonal_effect','offdiagonal_effect','selectivity','uniform_position_effect','diagonal_minus_uniform_position')},
                retrospective_S=p['retrospective_final']['selectivity'],
                by_axis={a:sel[a]['contrast']['by_axis'][a]['target_apt'] for a in AXES},
                diagonal_absolute={arm:{k:mean(sel[a][arm]['by_axis'][a][k] for a in AXES) for k in measures} for arm in ('same','opposite')},
                diagonal_common_outside={k:mean(sel[a]['contrast']['by_axis'][a][k] for a in AXES) for k in ('both_packets_outside_train_greedy_set','outside_both_target_apt_gain')},
                natural=p['natural']['macro'],whole_same_time=cross[t,t],discovery=z))
    groups={}
    def average(items):
        first=items[0]
        if isinstance(first,dict):return {k:average([x[k] for x in items]) for k in first}
        return mean(items)
    for condition in ('PL_live','LL_live','PL_silent','LL_silent'):
        group=[]
        for t in STEPS:
            cells=[r for r in rows if r['condition']==condition and r['update']==t]
            assert len(cells)==4
            names=('diagonal_effect','offdiagonal_effect','selectivity','uniform_position_effect','diagonal_minus_uniform_position','retrospective_S',
                'by_axis','diagonal_absolute','diagonal_common_outside','natural','whole_same_time')
            entry=dict(update=t,**{k:average([r[k] for r in cells]) for k in names})
            entry['sender_response']=mean(r['discovery']['mean_selected_response'] for r in cells)
            entry['score_counts']={k:sum(r['discovery'][k+'_selected_scores'] for r in cells) for k in ('positive','zero','negative')}
            group.append(entry)
        ps=[s for (seed,c),(s,_,_) in policies.items() if c==condition]
        groups[condition]=dict(points=group,mean_areas=average([s['normalized_areas'] for s in ps]),
            mean_retrospective_areas=average([s['retrospective_normalized_areas'] for s in ps]),
            mean_cross_time_target_effect=[[mean(s['cross_time_target_effect'][i][j] for s in ps) for j in range(6)] for i in range(6)],
            temporal_token_change=[mean(s['temporal_changes'][i]['train_token_change_rate'] for s in ps) for i in range(5)],
            temporal_position_change=[mean(s['temporal_changes'][i]['selected_position_change_rate'] for s in ps) for i in range(5)])
    out=dict(status='completed_descriptive_aggregation',source_sha256=sources,primary=result['primary'],groups=groups,all_policy_time_rows=rows,
        scope='Four paired existing developmental seeds; checkpoint rows and matrix cells are not independent populations.',neural_forward_samples=0)
    with (path/'descriptive_summary.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(primary=result['primary'],groups=groups),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);run(p.parse_args().run)
