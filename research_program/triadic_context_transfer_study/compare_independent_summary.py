"""Compare completed independent audit and official summary JSON only.

No model, NumPy, NPZ, producer aggregation, or additional policy forward.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import argparse
import json
import math
import time
import traceback

SEEDS=(51101,51102,51103,51104)
CONDS=('PL_silent','PL_live','LL_silent','LL_live')
PARTS=('train','new_needs','new_layouts','new_needs_and_layouts')
AXES=('kind','length','destination')
LAYERS=('eligible','all_other')
MODES=('remote_same_both','remote_opposite_both')
KEYS={'counterfactual_apt':'target_apt','current_apt':'current_apt',
    'counterfactual_probability':'target_probability','current_probability':'current_probability',
    'native_reward':'native_reward','native_full_success':'native_team_full',
    'counterfactual_reward':'counterfactual_reward','counterfactual_full_success':'counterfactual_team_full',
    'listener_action_changed':'listener_action_changed','natural_current_apt':'natural_current_apt',
    'C_gate_mass':'conservative_gate','C_counterfactual_apt':'conservative_target_apt',
    'copy_same_actual_donor':'action_equals_donor_same','copy_opposite_actual_donor':'action_equals_donor_opposite'}
CONTRAST_KEYS={key:KEYS[key] for key in ('counterfactual_apt','counterfactual_probability',
    'C_counterfactual_apt','current_apt','native_reward','native_full_success',
    'counterfactual_reward','counterfactual_full_success')}
TOL=2e-11

def require(ok,message):
    if not ok:raise AssertionError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def compare(audit,summary):
    audit=Path(audit);summary=Path(summary)
    verification=read(audit/'verification.json');official=read(summary)
    require(verification['status']=='passed','Independent audit not passed')
    require(official['status']=='completed_read_only_summary','Official summary not completed')
    sources={str(audit/'verification.json'):sha(audit/'verification.json'),str(summary):sha(summary)}
    policies={(p['seed'],p['condition']):p for p in official['policies']}
    require(len(official['policies'])==len(policies)==16 and set(policies)==set(product(SEEDS,CONDS)),'Complete policy matrix')
    workers={}
    for worker in verification['workers']:
        path=Path(worker['path']);require(sha(path)==worker['sha256'],'Independent worker SHA mismatch')
        data=read(path);require(data['status']=='passed' and data['seed']==worker['seed'],'Worker identity/status')
        workers[data['seed']]=data;sources[str(path)]=worker['sha256']
    require(set(workers)==set(SEEDS),'Four independent workers')
    counts={};maxima={};worst={}
    def check(x,y,family,label):
        require(isinstance(x,(int,float)) and isinstance(y,(int,float)) and math.isfinite(x) and math.isfinite(y),'Nonfinite scalar '+label)
        error=abs(x-y);counts[family]=counts.get(family,0)+1
        if error>maxima.get(family,-1):maxima[family]=error;worst[family]=label
        require(error<=TOL,'Scalar mismatch '+label+': '+repr((x,y,error)))
    def summary_values(ind_axes,official_cell,family,label,gain=False):
        keys=CONTRAST_KEYS if gain else KEYS
        for i,name in enumerate(AXES):
            if not gain:check(ind_axes[str(i)]['weight_mass'],official_cell['by_axis'][name]['weight_sum'],'weight_mass',label+'/'+name)
            for own,theirs in keys.items():
                key=theirs+'_gain' if gain else theirs
                check(ind_axes[str(i)][own],official_cell['by_axis'][name]['values'][key],family,label+'/'+name+'/'+own)
        for own,theirs in keys.items():
            key=theirs+'_gain' if gain else theirs
            check(sum(ind_axes[str(i)][own] for i in range(3))/3,official_cell['macro'][key],family,label+'/macro/'+own)
    for seed,condition,part in product(SEEDS,CONDS,PARTS):
        worker=workers[seed];ind=worker['condition_summaries'][condition][part];off=policies[seed,condition]['partitions'][part]
        for layer in LAYERS:
            for mode in MODES:summary_values(ind[layer]['arms'][mode],off[layer][mode],'remote_arm',f'{seed}/{condition}/{part}/{layer}/{mode}')
            summary_values(ind[layer]['contrast_by_axis'],off[layer]['contrast'],'remote_contrast',f'{seed}/{condition}/{part}/{layer}/contrast',True)
            for i,name in enumerate(AXES):
                check(ind[layer]['arms']['remote_same_both'][str(i)]['C_gate_mass'],off[layer]['contrast']['by_axis'][name]['values']['conservative_gate'],'contrast_common_gate',f'{seed}/{condition}/{part}/{layer}/{name}')
            check(sum(ind[layer]['arms']['remote_same_both'][str(i)]['C_gate_mass'] for i in range(3))/3,off[layer]['contrast']['macro']['conservative_gate'],'contrast_common_gate',f'{seed}/{condition}/{part}/{layer}/macro')
        for mode in ('sham_both','local_opposite_both'):
            chosen=[r for r in worker['records'] if r['condition']==condition and r['partition']==part and r['mode']==mode]
            require(len(chosen)==2 and {r['direction'] for r in chosen}=={0,1},'Complete control directions')
            axes={str(a):{k:sum(r['scalars']['control'][str(a)][k] for r in chosen) for k in ('weight_mass',*KEYS)} for a in range(3)}
            summary_values(axes,off['controls'][mode],'control',f'{seed}/{condition}/{part}/{mode}')
    official_pairs=official['primary_comparison']['all_partitions']
    for part,layer in product(PARTS,LAYERS):
        a=verification['paired_comparisons'][part][layer];b=official_pairs[part][layer]
        br={r['seed']:r for r in b['seed_pairs']};require(set(br)==set(SEEDS),'Official paired seeds')
        for row in a['seed_rows']:
            sr=br[row['seed']];label=f'{part}/{layer}/{row["seed"]}'
            check(row['PL_minus_LL'],sr['PL_minus_LL'],'paired',label+'/raw')
            check(row['C_PL_minus_LL'],sr['conservative_PL_minus_LL'],'paired',label+'/C')
            for cond in CONDS:
                for own,theirs in [('counterfactual_apt','target_apt_gain'),('C_counterfactual_apt','conservative_target_apt_gain')]:
                    check(row['cells'][cond][own],sr['conditions'][cond][theirs],'paired',label+'/'+cond+'/'+own)
        check(a['equal_seed_mean'],b['equal_seed_mean_PL_minus_LL'],'paired',f'{part}/{layer}/mean_raw')
        check(a['C_equal_seed_mean'],b['equal_seed_mean_conservative_PL_minus_LL'],'paired',f'{part}/{layer}/mean_C')
    for path,digest in sources.items():require(sha(path)==digest,'Compared source changed')
    return dict(status='passed_json_comparison',source_sha256=sources,comparison_source_sha256=sha(__file__),absolute_tolerance=TOL,
        comparisons_by_family=counts,total_scalar_comparisons=sum(counts.values()),max_absolute_error_by_family=maxima,
        max_absolute_error=max(maxima.values()),largest_error_labels=worst,
        coverage=dict(policies=16,partitions=4,remote_supports=2,remote_modes_and_contrast=3,control_modes=2,axes=3,macros=True,common_arm_metrics=KEYS,common_contrast_metrics=CONTRAST_KEYS,paired_seeds=list(SEEDS)),
        limits=['Only the common numerical intersection of the independently computed saved audit and official JSON summary is compared.',
            'No new neural forward, NPZ read, training, producer metrics call, or inference from private analysis.',
            'Official by_direction and by_sender_listener summaries, message-change metrics, donor-observation diagnostics and separate copy-correctness baselines are outside this JSON numerical comparison.'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--summary',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);require(not out.exists(),'No overwrite');out.mkdir(parents=True);start=time.perf_counter()
    write(out/'started.json',dict(at=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),automatic_retry=False))
    try:
        result=compare(args.audit,args.summary);result.update(completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        write(out/'verification.json',result);print(json.dumps({k:result[k] for k in ('status','total_scalar_comparisons','max_absolute_error')},ensure_ascii=False))
    except BaseException as e:
        write(out/'failure.json',dict(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),source_sha256=sha(__file__),automatic_retry=False));raise

if __name__=='__main__':main()
