"""Independent v0.8 paired-world recount, utility-pressure comparison and scientific figures.

No Torch or training imports. One saved social row is one world with two decisions.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from analysis_core import (audit_pair_arrays, paired_metrics, outcomes_metrics, verify_recorded_metrics,
                           describe, equal, near, require, split_audit)
ROOT=Path(__file__).resolve().parent
SEEDS=[27101,27102,27103,27104]
KINDS=("additive","mixed","joint")
WEIGHTS={"additive":0.,"mixed":.5,"joint":1.}
FULLKINDS=tuple(f"{prefix}_{k}" for prefix in ("full","blocked") for k in KINDS)
CONDITIONS=[f"split{s}_{k}" for s in (1,2,3) for k in KINDS]+list(FULLKINDS)
MODES=("normal","shuffle","blank","stochastic","erase_memory")
MATCHINGS={1:((0,1),(2,3),(4,5)),2:((0,2),(1,4),(3,5)),3:((0,3),(1,5),(2,4))}
SPLITS={s:[p for e in edges for p in (e,e[::-1])] for s,edges in MATCHINGS.items()}
LABELS={"additive":"加和 λ=0","mixed":"混合 λ=0.5","joint":"联合 λ=1"}
LABELS.update({f"{prefix}_{k}":("全图 " if prefix=="full" else "关通道 ")+LABELS[k] for prefix in ("full","blocked") for k in KINDS})
SCORE_FIELDS=("both_accuracy","single_accuracy","mean_reward","food_accuracy","water_accuracy","reward_variance","positive_rate",*("outcome_"+x for x in ("00","01","10","11")))


def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pct(x):return "不适用" if x is None else f"{100*x:.2f}%"
def pp(x):return "不适用" if x is None else f"{100*x:+.2f}"


def compare_world(a,b,context):
    for key in ('scout','episode','positions','photo_ids','goals','menu','inventory','history'):
        equal(a[key],b[key],f'{context}: {key}')


def validate_score(score,weight,context):
    """Check checkpoint aggregate arithmetic without claiming a saved trace replay."""
    c=score['outcome_counts'];n=score['n']
    require(set(c)=={'00','01','10','11'} and all(isinstance(x,int) and x>=0 for x in c.values()),context+': outcome count validity')
    require(sum(c.values())==n and score['decisions']==2*n,context+': world/decision counts')
    food=c['10']+c['11'];water=c['01']+c['11'];reward=c['11']+(1-weight)*(c['10']+c['01'])/2
    expected=dict(n=n,decisions=2*n,reward_sum=reward,mean_reward=reward/n if n else None,
        reward_variance=(c['11']+((1-weight)/2)**2*(c['10']+c['01']))/n-(reward/n)**2 if n else None,
        positive_rewards=c['11']+(c['10']+c['01'] if weight<1 else 0),
        single_correct=food+water,single_accuracy=(food+water)/(2*n) if n else None,
        both_correct=c['11'],both_accuracy=c['11']/n if n else None,
        food_correct=food,water_correct=water,outcome_counts=c)
    verify_recorded_metrics(score,expected,context)


def audit_trace(path,cfg,stated,photos):
    with np.load(path,allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
    plan=cfg['plan'];weight=plan['complementarity']
    require(stated['episodes']==cfg['eval_n'] and stated['horizon']==1,f'{path}: world count/horizon')
    success,audit=audit_pair_arrays(a,n_sites=6,vocab=7,length=2,episodes=stated['episodes'],
        mode=path.stem.removeprefix('final_'),blocked=plan['blocked'],weight=weight,photo_metadata=photos)
    metrics=paired_metrics(a,success,n_sites=6,heldout_maps=np.asarray(cfg['map_table'])[cfg['heldout_map_ids']],weight=weight)
    verify_recorded_metrics(stated,metrics,str(path))
    for d in (0,1):
        verify_recorded_metrics(stated['direction_metrics'][d],metrics['direction_metrics'][d],f'{path}: direction{d}')
        near(stated['direction_means'][d],metrics['direction_metrics'][d]['mean_reward'],'Direction reward compatibility')
    for source,target in (('seen','train_maps'),('unseen','heldout_maps')):
        verify_recorded_metrics(stated['map_groups'][source],metrics['map_subsets'][target],f'{path}: {source}')
    h=hashlib.sha256()
    for d in (0,1):
        ix=np.flatnonzero(a['scout']==d);ix=ix[np.argsort(a['episode'][ix])]
        for key in ('positions','photo_ids','goals','menu'):h.update(np.ascontiguousarray(a[key][ix]).tobytes())
    require(h.hexdigest()==stated['world_sha256'],f'{path}: saved world hash')
    audit.update(path=str(path),sha256=sha(path),world_hash_verified=True,balanced_evaluation=True)
    return metrics,a,audit


def read_run(path,photos,design):
    cfg,result=read_json(path/'config.json'),read_json(path/'result.json');plan=cfg['plan']
    for key in ('seed','condition','plan','updates','batch','initial_sha256'):
        require(cfg[key]==result[key],f'{path}: config/result {key}')
    require(cfg['trace_schema']=='paired_world_v1' and cfg['choices_per_world']==2,'Paired-world trace schema')
    require(cfg['sites']==6 and cfg['history_dim']==18,'Fixed site/history dimensions')
    require(plan['representation']=='identity' and plan['schedule']=='direct' and not plan['known'],'Fixed information and schedule')
    require(plan['vocab']==7 and plan['length']==2 and plan['reward_kind'] in KINDS,'Fixed message space')
    require(plan['complementarity']==WEIGHTS[plan['reward_kind']],'Reward condition lambda')
    equal(cfg['input_transforms'],np.tile(np.eye(6),(2,1,1)),'Identity visual input')
    equal(cfg['map_table'],design['all_maps'],'Map table')
    split=plan['split'];maps=np.asarray(cfg['map_table']);held=SPLITS[split] if split else []
    require(set(map(tuple,maps[cfg['heldout_map_ids']]))==set(held),'Heldout matching maps')
    require(set(cfg['train_map_ids'])==set(range(30))-set(cfg['heldout_map_ids']),'Training map complement')
    require(result['frozen_projection_verified'],'Frozen projection/buffer verification')
    require(sha(cfg['prepared_source']['path'])==cfg['prepared_source']['sha256'],'Preparation source changed')
    curve=read_json(path/'curve.json');equal([r['update'] for r in curve],cfg['checkpoints'],'Checkpoint coverage')
    require(curve[-1]['update']==cfg['updates'],'Final checkpoint missing')
    for row in curve:
        require((path/f"checkpoint_{row['update']:04d}.pt").is_file(),'Missing checkpoint weights')
        for mode in MODES:
            score=row['scores'][mode]
            require(score['episodes']==cfg['checkpoint_eval_n'] and score['horizon']==1,'Checkpoint world count')
            validate_score(score,plan['complementarity'],'Checkpoint whole evaluation')
            require(sum(g['n'] for g in score['map_groups'].values())==score['n'],'Checkpoint map group size')
            for g in score['map_groups'].values():validate_score(g,plan['complementarity'],'Checkpoint map group')
    scores={};audits={};baseline=None
    for mode in MODES:
        scores[mode],a,audits[mode]=audit_trace(path/f'final_{mode}.npz',cfg,result['scores'][mode],photos)
        if baseline is None:baseline=a
        else:compare_world(baseline,a,f'{path}: modes')
    training=[json.loads(line) for line in (path/'training.jsonl').read_text().splitlines() if line.strip()]
    equal([r['update'] for r in training],np.arange(1,cfg['updates']+1),'Update continuity')
    schedule=read_json(path/'training_schedule.json')
    equal([r['batch_identity'] for r in training],schedule['batch_identities'],'Batch order')
    equal([r['active_sites'] for r in training],schedule['levels'],'Site schedule')
    equal(sorted(schedule['batch_identities']),np.arange(cfg['updates']),'Each world batch once')
    for r in training:
        require(r['active_sites']==6 and set(r['allowed_sites'])==set(range(6)),'Direct six-site exposure')
        require(set(r['map_pool'])==set(cfg['train_map_ids']),'Legal training map support')
        near(r['entropy_weight'],.02 if r['update']<=cfg['entropy_off_after'] else 0.,'Entropy schedule')
        c=r['outcome_counts'];require(sum(c.values())==cfg['batch'],'Training world count')
        near(r['single_accuracy'],(2*c['11']+c['01']+c['10'])/(2*cfg['batch']),'Training single rate')
        near(r['both_accuracy'],c['11']/cfg['batch'],'Training both rate')
        near(r['reward'],(c['11']+(1-plan['complementarity'])*(c['01']+c['10'])/2)/cfg['batch'],'Training utility from outcome counts')
        near(r['reward_variance'],(c['11']+((1-plan['complementarity'])/2)**2*(c['01']+c['10']))/cfg['batch']-r['reward']**2,'Training utility variance')
        require(r['positive_rewards']==c['11']+(c['01']+c['10'] if plan['complementarity']<1 else 0),'Training positive utility count')
    return dict(seed=cfg['seed'],condition=cfg['condition'],split=split,kind=plan['reward_kind'] if split else cfg['condition'],
        directory=str(path),config=cfg,scores=scores,curve=curve,trace_audit=audits,seconds=result['seconds'],
        initial_sha256=result['initial_sha256'],final_state_sha256=result['final_sha256'],
        trainable_initial_sha256=cfg['trainable_initial_sha256'],receiver_initial_sha256=cfg['receiver_initial_sha256'],
        final_file_sha256=sha(path/'final.pt'),result_sha256=sha(path/'result.json'),training_sha256=sha(path/'training.jsonl'),
        training_record_count=len(training),
        batches_by_identity={str(r['batch_identity']):{k:r[k] for k in ('active_sites','allowed_sites','map_pool','world_sha256')} for r in training}),baseline


def provenance(runs):
    rows=[]
    for seed in sorted({r['seed'] for r in runs}):
        family=[r for r in runs if r['seed']==seed]
        for field in ('initial_sha256','trainable_initial_sha256','receiver_initial_sha256'):
            require(len({r[field] for r in family})==1,f'{seed}: unmatched {field}')
        require(len({tuple(r['config']['trainable_parameters']) for r in family})==1,'Parameter counts differ')
        for split in sorted({r['split'] for r in family}):
            selected=[r for r in family if r['split']==split]
            require(all(r['batches_by_identity']==selected[0]['batches_by_identity'] for r in selected),f'{seed}/{split}: unmatched world exposure')
            rows.append(dict(seed=seed,split=split,matched_conditions=len(selected),initialization_equal=True,training_batches_equal=True))
    return rows


def run_value(r,mode,group,field):
    def get(m):
        row=r['scores'][m] if group=='all' else r['scores'][m]['map_subsets'][group]
        if field.startswith('outcome_'):return row['outcome_counts'][field.removeprefix('outcome_')]/row['n'] if row['n'] else None
        return row[field]
    if mode=='gap':
        a,b=get('normal'),get('shuffle');return a-b if a is not None else None
    return get(mode)


def aggregate(runs):
    output={}
    for kind in (*KINDS,*FULLKINDS):
        selected=[r for r in runs if r['kind']==kind]
        if not selected:continue
        seeds=sorted({r['seed'] for r in selected});group={'seeds':seeds,'metrics':{}}
        for mode in (*MODES,'gap'):
            for subset in ('all','train_maps','heldout_maps'):
                for field in SCORE_FIELDS:
                    if any(run_value(r,mode,subset,field) is None for r in selected):continue
                    means=[]
                    for seed in seeds:
                        local=[r for r in selected if r['seed']==seed]
                        require(len(local)==(3 if kind in KINDS else 1),'Incomplete condition/seed family')
                        means.append(float(np.mean([run_value(r,mode,subset,field) for r in local])))
                    group['metrics']['/'.join((mode,subset,field))]=describe(means)
        output[kind]=group
    return output


def paired(runs):
    index={(r['seed'],r['split'],r['kind']):r for r in runs};rows=[]
    for left,right in (('joint','additive'),('mixed','additive'),('joint','mixed')):
        for mode,subset,field in (('normal','heldout_maps','both_accuracy'),('normal','heldout_maps','single_accuracy'),
                                  ('normal','train_maps','both_accuracy'),('gap','heldout_maps','both_accuracy')):
            local=[]
            for seed in SEEDS:
                for split in (1,2,3):
                    a,b=index[seed,split,left],index[seed,split,right]
                    local.append(dict(seed=seed,split=split,difference=run_value(a,mode,subset,field)-run_value(b,mode,subset,field)))
            values=[float(np.mean([r['difference'] for r in local if r['seed']==seed])) for seed in SEEDS]
            rows.append(dict(left=left,right=right,metric='/'.join((mode,subset,field)),seeds=SEEDS,
                seed_means_over_splits=describe(values),split_seed_differences=local))
    return rows

def protocol_summary(source, runs):
    path = source / "protocol_analysis.json"
    if not path.exists():
        return None
    data = read_json(path)
    require(data["status"] == "complete", "Protocol analysis is not complete")
    index = {(r["seed"], r["condition"]): r for r in runs}
    require({(r["seed"],r["condition"]) for r in data["runs"]} == set(index), "Protocol run scope differs from final analysis")
    ratios = 0
    def check_ratios(value):
        nonlocal ratios
        if isinstance(value, dict):
            if {"numerator", "denominator", "rate"}.issubset(value):
                n,d=value["numerator"],value["denominator"]
                require(0<=n<=d,"Protocol count bounds")
                if d:near(value["rate"],n/d,"Protocol integer ratio")
                else:require(value["rate"] is None,"Zero protocol denominator must remain undefined")
                ratios+=1
            for v in value.values():check_ratios(v)
        elif isinstance(value,list):
            for v in value:check_ratios(v)
    check_ratios(data)
    rows=[]
    for run in data["runs"]:
        base=index[run["seed"],run["condition"]]
        require(run["final_sha256"]==base["final_file_sha256"],"Protocol final checkpoint differs")
        require(sha(run["final_checkpoint"])==run["final_sha256"],"Protocol checkpoint file changed")
        equal(run["train_map_ids"],base["config"]["train_map_ids"],"Protocol training maps")
        equal(run["heldout_map_ids"],base["config"]["heldout_map_ids"],"Protocol heldout maps")
        require(len(run["directions"])==2,"Protocol direction coverage")
        for d in run["directions"]:
            menu=d["menu_audit"]
            require(menu["enumerated_messages"]==49 and menu["goals"]==2 and menu["menus"]==720 and menu["cases"]==70560,"Protocol full-menu coverage")
            require(menu["all_menu_permutations_physically_equivalent"] and menu["menus_used_after_check"]==1,"Unverified protocol menu collapse")
            row={"seed":base["seed"],"condition":base["condition"],"kind":base["kind"],"split":base["split"],
                 "scout":d["scout"],"collector":d["collector"],"counts":{},"menu_audit":d["menu_audit"]}
            phase=d["phases"]["validation"]
            for group in ("seen","unseen"):
                cross=phase["cross_goal_groups"][group]
                if cross is None:continue
                for short,key in (("native","native_goal"),("both","same_message_correct_for_both_goals")):
                    row[f"{group}_{short}"]=cross[key]["rate"];row["counts"][f"{group}_{short}"]=cross[key]
            row["all_native"]=phase["cross_goal"]["native_goal"]["rate"]
            row["all_both"]=phase["cross_goal"]["same_message_correct_for_both_goals"]["rate"]
            if "fragments" in d:
                frag=d["fragments"]
                equal(frag["selection_map_ids"],run["train_map_ids"],"Fragment calibration must use training maps")
                row["fragment"]=frag["validation"]["strict_transfer_all"]["rate"]
                row["counts"]["fragment"]=frag["validation"]["strict_transfer_all"]
                row["counts"]["conditional_fragment"]=frag["validation"]["strict_transfer_when_both_maps_both_goals_correct"]
            if "heldout_stitch_validation" in d:
                row["stitch"]=d["heldout_stitch_validation"]["both_goals_correct_all"]["rate"]
                row["counts"]["stitch"]=d["heldout_stitch_validation"]["both_goals_correct_all"]
            if "whole_message_recoding" in d:
                rec=d["whole_message_recoding"]
                require(rec["full_message_capacity"]==49 and rec["replicates"]==len(rec["records"])==100,"Protocol recoding count/capacity")
                require(rec["calibration_selection_repeated_for_every_recoding"],"Recoding calibration not repeated")
                for item in rec["records"]:
                    equal(sorted(item["old_to_new_code"]),np.arange(49),"Complete-message recoding is not bijective")
                for metric,statkey,recordkey in (("fragment","strict_transfer_all","validation_strict_transfer_all"),
                        ("stitch","heldout_stitch_both_goals_correct_all","validation_stitch_both_goals_correct_all")):
                    if statkey not in rec:continue
                    stat=rec[statkey];values=[r[recordkey]["rate"] for r in rec["records"]]
                    near(row[metric],stat["observed"],"Recoding observed score")
                    near(np.mean(values),stat["reference_mean"],"Recoding reference mean")
                    near(row[metric]-np.mean(values),stat["observed_minus_reference_mean"],"Recoding difference")
                    row[metric+"_null"]=float(np.mean(values));row[metric+"_delta"]=row[metric]-row[metric+"_null"]
            rows.append(row)
    metrics=("all_native","all_both","seen_native","seen_both","unseen_native","unseen_both",
             "fragment","fragment_null","fragment_delta","stitch","stitch_null","stitch_delta")
    groups={}
    for kind in (*KINDS,*FULLKINDS):
        selected=[r for r in rows if r["kind"]==kind]
        if not selected:continue
        seeds=sorted({r["seed"] for r in selected});g={"seeds":seeds,"metrics":{},"by_split":{}}
        for metric in metrics:
            if not all(metric in r for r in selected):continue
            per_seed=[float(np.mean([r[metric] for r in selected if r["seed"]==s])) for s in seeds]
            g["metrics"][metric]=describe(per_seed)
            for split in sorted({r["split"] for r in selected}):
                values=[float(np.mean([r[metric] for r in selected if r["seed"]==s and r["split"]==split])) for s in seeds]
                g["by_split"].setdefault(str(split),{})[metric]=describe(values)
        groups[kind]=g
    reference_seeds=[d['whole_message_recoding']['rng_seed'] for r in data['runs'] for d in r['directions']]
    main_reference_seeds=[d['whole_message_recoding']['rng_seed'] for r in data['runs'] if r['plan']['split'] for d in r['directions']]
    return {"source":str(path),"source_sha256":sha(path),"groups":groups,"direction_rows":rows,
            "reference_randomness":{"directions":len(reference_seeds),"distinct_rng_seeds":len(set(reference_seeds)),
                "main_directions":len(main_reference_seeds),"main_distinct_rng_seeds":len(set(main_reference_seeds)),
                "conditional_reference_scores":sum(d['whole_message_recoding']['replicates'] for r in data['runs'] for d in r['directions']),
                "note":"Full/blocked reference streams can repeat across condition/agent groups; reference scores are not independent experimental replicates."},
            "photo_splits":data["photo_splits"],"integer_ratios_checked":ratios,
            "audit":"passed: run scope, checkpoint hashes, calibration maps, integer ratios, 49-code bijections and reference means",
            "aggregation":"two directions and three coordinate splits averaged within seed, then four seeds equally weighted; full conditions occur only once per seed"}




def control_summary(source,runs,photos):
    path=source/"individual_controls/summary.json"
    data=read_json(path)
    require(data["complete"] and data["runs"]==4,"Personal controls incomplete")
    gate_path=source/"social_launch_gate.json"
    gate=read_json(gate_path)
    require(gate["controls_complete"]==4 and gate["social_runs_planned"]==60 and gate["all_passed"],"Social launch gate")
    require(gate["control_summary_sha256"]==sha(path) and not gate["diagnostic_weights_transferred"],"Launch gate control source")
    index={(r["seed"],r["config"]["plan"]["representation"]):r for r in runs}
    rows=[];total=0
    for seed in SEEDS:
        for kind in ("identity",):
            folder=source/"individual_controls"/f"s{seed}_{kind}"
            cfg,result=read_json(folder/"config.json"),read_json(folder/"result.json")
            require(result["complete"] and result["seed"]==seed and result["representation"]==kind,"Personal control metadata")
            require(cfg["updates"]==2400 and cfg["batch"]==512 and cfg["eval_n_per_agent"]==9600,"Personal control fixed budget")
            if runs:
                require(cfg["common_initial_sha256"]==index[seed,kind]["trainable_initial_sha256"],"Personal control not initialized from same pre-social trainable parameters")
                require(cfg["initial_sha256"]==index[seed,kind]["initial_sha256"],"Personal control initial agent state differs")
                equal(cfg["input_transforms"],index[seed,kind]["config"]["input_transforms"],"Personal control input transforms")
            require(result["frozen_unused_parameters_verified"],"Personal control unused weights were not verified frozen")
            checked={};baseline=None
            for mode in ("normal","erase_memory"):
                file=folder/f"final_{mode}.npz"
                with np.load(file,allow_pickle=False) as z:a={k:z[k] for k in z.files}
                require({"positions","goals","photo_ids","menu","agent","action","place","reward"}.issubset(a),"Missing personal trace fields")
                n=2*cfg["eval_n_per_agent"]
                require(all(len(v)==n for v in a.values()),"Personal trace size")
                equal(np.sort(a["menu"],axis=1),np.tile(np.arange(6),(n,1)),"Personal menu permutation")
                require(np.isin(a["agent"],[0,1]).all() and np.isin(a["goals"],[0,1]).all(),"Personal identity/goal range")
                require(((a["positions"]>=0)&(a["positions"]<6)).all() and (a["positions"][:,0]!=a["positions"][:,1]).all(),"Personal map validity")
                require(((a["action"]>=0)&(a["action"]<6)).all(),"Personal action range")
                equal(a["place"],a["menu"][np.arange(n),a["action"]],"Personal action routing")
                reward=(a["place"]==a["positions"][np.arange(n),a["goals"]]).astype(int)
                equal(reward,a["reward"],"Personal independently recounted reward")
                for resource,category in enumerate(("food","water")):
                    ids=a["photo_ids"][:,resource]
                    require(((ids>=0)&(ids<len(photos))).all(),"Personal photo range")
                    require(all(photos[int(i)]["category"]==category and photos[int(i)]["split"]=="test" for i in np.unique(ids)),"Personal photo source")
                counts=[]
                for who in (0,1):
                    ix=a["agent"]==who;count=int(ix.sum());correct=int(reward[ix].sum())
                    require(count==cfg["eval_n_per_agent"],"Personal agent count")
                    cells=np.column_stack((a["positions"][ix],a["goals"][ix]));unique,freq=np.unique(cells,axis=0,return_counts=True)
                    require(len(unique)==60,"Personal full map/goal support")
                    equal(freq,np.full(60,count//60),"Personal balanced map/goal cells")
                    recorded=result["scores"][mode]["per_agent"][who]
                    require(recorded["n"]==count and recorded["correct"]==correct,"Personal exact counts")
                    near(recorded["accuracy"],correct/count,"Personal reported accuracy")
                    counts.append({"agent":who,"n":count,"correct":correct,"accuracy":correct/count})
                near(result["scores"][mode]["mean_accuracy"],np.mean([r["accuracy"] for r in counts]),"Personal average")
                if baseline is None:baseline=a
                else:
                    for field in ("positions","goals","photo_ids","menu","agent"):equal(baseline[field],a[field],"Personal mode exogenous matching")
                checked[mode]={"per_agent":counts,"file_sha256":sha(file)};total+=n
            passed=all(r["correct"]*10>=r["n"]*9 for r in checked["normal"]["per_agent"])
            require(result["passed"]==passed,"Personal gate must use exact integer counts")
            training=[json.loads(line) for line in (folder/"training.jsonl").read_text().splitlines() if line.strip()]
            equal([r["update"] for r in training],np.arange(1,2401),"Personal update continuity")
            rows.append({"seed":seed,"representation":kind,"passed":passed,"scores":checked,
                         "directory":str(folder),"config":cfg,"result_sha256":sha(folder/"result.json"),
                         "training_sha256":sha(folder/"training.jsonl"),"training_record_count":len(training)})
    require(data["passed"]==all(r["passed"] for r in rows),"Personal summary gate")
    for seed in SEEDS:
        require(len({r["config"]["common_initial_sha256"] for r in rows if r["seed"]==seed})==1,"Personal control trainable starts differ across representations")
    groups={}
    for kind in ("identity",):
        selected=[r for r in rows if r["representation"]==kind]
        means=[float(np.mean([x["accuracy"] for x in r["scores"]["normal"]["per_agent"]])) for r in selected]
        individuals=[x["accuracy"] for r in selected for x in r["scores"]["normal"]["per_agent"]]
        erase=[float(np.mean([x["accuracy"] for x in r["scores"]["erase_memory"]["per_agent"]])) for r in selected]
        groups[kind]={"seeds":[r["seed"] for r in selected],"per_seed_agent_means":describe(means),
                      "individual_scores":describe(individuals),"erase_per_seed_agent_means":describe(erase),
                      "all_passed":all(r["passed"] for r in selected)}
    return {"source":str(path),"source_sha256":sha(path),"groups":groups,"runs":rows,"all_passed":data["passed"],
            "launch_gate":gate,"launch_gate_sha256":sha(gate_path),
            "trace_records_checked":total,"social_start_comparison_performed":bool(runs),
            "independent_recount":"reward, action routing, all60 map/goal cells per person, photo source, integer threshold, shared personal-control initial state; social state/matrices additionally checked when complete social runs supplied"}

def plots(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'font.size':11,
        'axes.unicode_minus':False,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    seed_colors=['#3B83BD','#D59331','#9171B1','#278D80'];palette=['#477AB2','#D19036','#238B79'];files=[]
    names=['加和 λ=0','混合 λ=0.5','联合 λ=1']
    def values(k,key):return summary['groups'][k]['metrics'][key]['values']
    def dots(ax,x,ys,color=None):
        ax.scatter(x+np.linspace(-.09,.09,len(ys)),100*np.array(ys),s=32,
            c=color if color else seed_colors,zorder=3)
        ax.plot([x-.18,x+.18],[100*np.mean(ys)]*2,c=color or '#263139',lw=2.2)
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=240)
        plt.close(fig);files.append(name+'.png')
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.7),constrained_layout=True)
    for ax,key,title in zip(axes.flat,
        ('normal/train_maps/both_accuracy','normal/heldout_maps/both_accuracy',
         'normal/heldout_maps/single_accuracy','gap/heldout_maps/both_accuracy'),
        ('训练24图：同消息双目标','留出6图：同消息双目标','留出6图：平均单目标','留出6图：双目标正常−打乱')):
        for i,k in enumerate(KINDS):dots(ax,i,values(k,key))
        ax.set(title=title,ylabel='百分点' if key.startswith('gap') else '成功率（%）',xticks=range(3),xticklabels=names)
        ax.grid(axis='y',alpha=.15)
        if key.startswith('gap'):ax.axhline(0,c='#8B9298',lw=1)
        elif 'heldout_maps/both' in key:
            ymax=max(max(values(k,key)) for k in KINDS)*100;ax.set_ylim(-1,max(15,min(104,ymax+6)))
        else:ax.set_ylim(-2,104)
    fig.suptitle('互补收益压力：点为种子跨三划分均值，短线为4种子均值',fontsize=13)
    save(fig,'complementarity_comparison')
    fig,axes=plt.subplots(2,2,figsize=(11,7.7),constrained_layout=True)
    for ax,(field,group,title) in zip(axes.flat,
        [('both_accuracy','seen','训练24图：双目标'),('both_accuracy','unseen','留出6图：双目标'),
         ('single_accuracy','seen','训练24图：平均单目标'),('single_accuracy','unseen','留出6图：平均单目标')]):
        for k,color in zip(KINDS,palette):
            family=[r for r in summary['runs'] if r['kind']==k]
            for mode,style in (('normal','-'),('shuffle','--')):
                seed_curves=[]
                for seed in SEEDS:
                    local=[r for r in family if r['seed']==seed];x=[c['update'] for c in local[0]['curve']]
                    require(all([c['update'] for c in r['curve']]==x for r in local),'Curve checkpoints not matched')
                    y=np.mean([[100*c['scores'][mode]['map_groups'][group][field] for c in r['curve']] for r in local],axis=0)
                    seed_curves.append(y)
                    if mode=='normal':ax.plot(x,y,c=color,lw=.6,alpha=.16)
                ax.plot(x,np.mean(seed_curves,axis=0),c=color,ls=style,lw=2 if mode=='normal' else 1.1,
                    marker='o' if mode=='normal' else None,ms=3,label=LABELS[k] if mode=='normal' else None)
        ax.set(title=title,xlabel='训练更新',ylabel='成功率（%）');ax.grid(axis='y',alpha=.15)
        if field=='both_accuracy' and group=='unseen':ax.set_ylim(bottom=-1)
        else:ax.set_ylim(-2,104)
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('形成过程：实线正常、虚线整条消息打乱；细线为种子均值',fontsize=13)
    save(fig,'complementarity_process')
    pg=summary['protocol']['groups']
    fig=plt.figure(figsize=(10.8,7.8),constrained_layout=True);grid=fig.add_gridspec(2,2)
    for ax,metrics,title in ((fig.add_subplot(grid[0,0]),('seen_both','unseen_both'),'自然消息：同一消息双目标均正确'),
            (fig.add_subplot(grid[0,1]),('fragment','fragment_null'),'严格局部替换：完整30图探针'),
            (fig.add_subplot(grid[1,:]),('stitch','stitch_null'),'训练供体拼接至留出6图：双目标均正确')):
        for i,k in enumerate(KINDS):
            for j,(metric,color) in enumerate(zip(metrics,('#397FB1','#D68B36'))):
                dots(ax,i+(-.16 if j==0 else .16),pg[k]['metrics'][metric]['values'],color)
                if i==0:ax.scatter([],[],color=color,label=('训练24图' if j==0 else '留出6图') if metrics[0]=='seen_both' else ('观察值' if j==0 else '整体重编码参考均值'))
        ax.set(title=title,ylabel='成功率（%）',ylim=(-3,104),xticks=range(3),xticklabels=names)
        ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('协议结构：照片平衡枚举；4种子跨划分和方向均值',fontsize=13)
    save(fig,'complementarity_protocol')
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.5),constrained_layout=True)
    control=summary['controls']['groups']['identity']
    dots(axes[0],0,control['per_seed_agent_means']['values']);dots(axes[0],1,control['erase_per_seed_agent_means']['values'])
    axes[0].set(title='个人 I 视觉行动正控',xticks=[0,1],xticklabels=['正常','清空现场表示'])
    axes[0].axhline(90,c='#8B9298',ls=':',lw=1)
    for ax,field,title in ((axes[1],'both_accuracy','全30图：双目标'),(axes[2],'single_accuracy','全30图：平均单目标')):
        for i,k in enumerate(KINDS):
            for j,prefix in enumerate(('full','blocked')):dots(ax,i+(-.14 if j==0 else .14),values(prefix+'_'+k,'normal/all/'+field),('#397FB1','#D68B36')[j])
        for color,label in (('#397FB1','正常通信'),('#D68B36','从头关闭通道')):ax.scatter([],[],color=color,label=label)
        ax.set(title=title,xticks=range(3),xticklabels=['λ=0','λ=0.5','λ=1'])
        ax.axhline(100/30 if field=='both_accuracy' else 100/6,c='#8B9298',ls=':',lw=1)
        ax.legend(frameon=False,fontsize=8.5)
    for ax in axes:ax.set_ylim(-2,104);ax.set_ylabel('成功率（%）');ax.grid(axis='y',alpha=.15)
    fig.suptitle('个人能力与完整地图基准；正控权重不回灌社会学习',fontsize=13)
    save(fig,'complementarity_controls')
    return files


def report(summary,out):
    g=summary['groups'];pg=summary['protocol']['groups']
    def metric(k,field='both_accuracy',group='heldout_maps',mode='normal'):
        return g[k]['metrics']['/'.join((mode,group,field))]
    lines=['# v0.8 资源互补收益与共同消息的形成','',
        f"生成时间：{summary['generated_utc']}。60个社会运行及4组个人诊断全部完成。",'',
        '## 主要发现','',
        '本轮固定视觉地点接口、信息、两个需求和消息容量，仅改变部分成功的效用。'
        '同一条消息必须支持互不观察对方行动的食物、水两次选择；联合成功率是跨条件共同主标准。','',
        '| 效用条件 | 训练24图双目标 | 留出6图双目标 | 留出平均单目标 | 留出双目标正常−打乱（百分点） |',
        '| --- | ---: | ---: | ---: | ---: |']
    for k in KINDS:
        lines.append(f"| {LABELS[k]} | {pct(metric(k,group='train_maps')['mean'])} | {pct(metric(k)['mean'])} | {pct(metric(k,'single_accuracy')['mean'])} | {pp(metric(k,mode='gap')['mean'])} |")
    lines+=['']
    for left,right,label in (('joint','additive','预先规定的主比较联合−加和'),('mixed','additive','中间剂量混合−加和')):
        diff=np.array(metric(left)['values'])-np.array(metric(right)['values'])
        single=np.array(metric(left,'single_accuracy')['values'])-np.array(metric(right,'single_accuracy')['values'])
        lines += [f"{label}：留出双目标均值差 {pp(diff.mean())} 个百分点，四种子依次为 "+'、'.join(pp(v) for v in diff)+
            f"；留出平均单目标均值差 {pp(single.mean())} 个百分点。独立重复为4个主体对种子，保留正负差异，不由大量评估世界扩大样本量。",'']
    trained=np.array(metric('joint',group='train_maps')['values'])-np.array(metric('additive',group='train_maps')['values'])
    held=np.array(metric('joint')['values'])-np.array(metric('additive')['values'])
    lines += [f"联合收益相对加和收益，在训练图上的双目标平均改善为 {pp(trained.mean())} 个百分点，"
        f"{int((trained>0).sum())}个种子为正；留出图改善为 {pp(held.mean())} 个百分点，"
        f"{int((held>0).sum())}个种子为正、{int((held<0).sum())}个为负。"
        '两类地图的均值方向一致，但熟悉图改善更大，未见组合的提升存在种子差异。'
        f"联合条件留出双目标仅为 {pct(metric('joint')['mean'])}，尚不能称为可靠的新组合表达。",'',
        '结构结果也没有随压力单调增强。按λ=0、0.5、1排列，严格片段成功率依次为 '+
        '、'.join(pct(pg[k]['metrics']['fragment']['mean']) for k in KINDS)+'，人工拼接为 '+
        '、'.join(pct(pg[k]['metrics']['stitch']['mean']) for k in KINDS)+'。'
        '混合条件在这两项上的均值最高；联合条件的拼接均值甚至低于加和条件。'
        '自然双目标终点确有平均提升，因此改善并非只出现在人工片段操作中，但功能兼得提升也不等于组合结构一致增强。', '']
    lines += ['这些比较检验效用压力下的学习与通信形成。三个条件都以两资源全部成功为满分；'
        '本轮不将λ的作用解释为从没有合作变成合作，也不将联合回报较高直接等同于组合语言。'
        '具体结构应结合自然新组合与片段干预结果判断。','',
        '## 实验问题、主体与已提供的能力','',
        'v0.7中视觉地点保留相对稠密混合的平均收益有限，地点置换条件未显示相同优势，自然未见组合双目标成功仍弱。'
        'v0.8改为效用压力实验：在同一消息已用于两个资源需求时，降低只完成一个需求的价值，是否改变形成过程及新组合可用性。'
        '没有沿用v0.7的社会权重，也不以跨版本分数差估计单目标改成双目标的因果作用。','',
        '使用冻结DINOv2 ViT-L/14旧照片缓存（44训练、16既往开发留出），新私人资源后果准备仅迁移并冻结1024→64投影。'
        '固定I地点槽：六地点各64视觉特征及存在量，经共享65→65线性层、Tanh及GRU96。'
        '两人具有独立参数与更新；社会接口从相同配对初值开始，λ不改变可训练参数数量、原始输入或经验世界。','',
        '采集者已有按食物/水需求选择的两组六地点动作分支、公共坐标和私有行动菜单。'
        '两次选择属于同一采集者，不是两名独立听者；没有自发职业、分工协商、长期库存、工具链或代际。'
        '发送者只看六地点照片表征，不看目标顺序或行动菜单，生成一次两个七选一编号，共49种完整消息。'
        '完整代码容量仍足以查表区分30图，因此高双目标成功不必依赖片段组合。','',
        '## 世界、效用与固定预算','',
        '六地点中一个食物和一个水且不共址，共30有序地图。每个世界发送一次消息，接收者分别查询两个需求；'
        '需求顺序均匀随机、两份菜单独立排列。两次查询共用同一条已投递消息、零库存和零18维历史，'
        '第一次行动、结果或菜单不进入第二次查询；两次完成后统一结算。','',
        'Rλ=(1−λ)(rF+rW)/2+λrFrW，rF和rW为各自是否选对的二值结果。','',
        '| 条件 | λ | 00收益 | 10/01收益 | 11收益 |','| --- | ---: | ---: | ---: | ---: |',
        '| additive | 0 | 0 | 0.50 | 1 |','| mixed | 0.5 | 0 | 0.25 | 1 |','| joint | 1 | 0 | 0 | 1 |','',
        '收益交互差R11−R10−R01+R00=λ。它改变部分成功的价值、正反馈密度和优化信号，不增加需要表达的地图区别。'
        '同一标量R供双方学习，没有个别资源成功辅助梯度。一次消息的log概率只计一次，接收策略score为两个动作log概率之和；'
        '两次动作独立取样、共享仅依赖合法输入的价值基线。λ条件也可能改变有限预算优化难度，不能只凭终点定位神经机制。','',
        '每个新主体对种子27101–27104有3留出划分×3λ、全30图3λ和从头关闭通道3λ，共15个运行，合计60。'
        '每运行2400次更新，每批512世界、每方向256世界，每世界两次采集选择。'
        f"实际社会更新 {sum(r['training_record_count'] for r in summary['runs']):,} 次，"
        '对应73,728,000训练世界、147,456,000采集选择；世界、方向、行动量分开计数。'
        '四组个人正控每组两人，每人2400×512次选择，共9,830,400个人选择，权重和优化器不回灌。','',
        '相同种子、划分各λ的初态和每个世界批次的地图、照片、需求顺序、菜单一致。'
        '均直接训练六地点，不根据中途表现续训、换种子或调整某组预算。私人REINFORCE与价值基线，Adam0.0007，'
        '梯度裁剪2；前2100更新熵系数0.02，末300更新为0，target=R−1。','',
        '| 划分 | 双向留出的无向地点对 |','| --- | --- |']
    for s,edges in MATCHINGS.items():lines.append(f"| {s} | "+'、'.join(f'({a},{b})' for a,b in edges)+' |')
    lines += ['', '每划分训练24图、留出6图，两资源地点边际均衡。三个同构坐标划分的留出互不相交，合计18/30图，'
        '先在每个种子内平均三划分、两方向，再等权平均四种子；它们不作为12个独立重复。全图/关闭通道各条件每种子仅一次。','',
        '## 功能终点、错误类型与形成过程','',
        '终点每运行每模式9600世界，30地图×2目标顺序×2方向每格80次照片抽样；过程为1200世界、每格10次。'
        '一个世界的两个动作共同定义联合成功，不能用不配对的单目标平均相乘。'
        '正常为贪心策略；打乱在同方向同目标顺序内重排整条消息，该消息同时用于两个需求。'
        '固定0消息、随机策略及清空发送者现场表示另存。','',
        f"![功能终点]({(out/'complementarity_comparison.png').resolve()})",'',
        '点为种子内三划分均值，短线为4种子均值；双目标图没有使用统一随机机会线。'
        '各面板纵轴范围不同，留出双目标面板作局部放大。','',
        '| 条件 | 地图组 | 食物正确 | 水正确 | 00 | 10仅食物 | 01仅水 | 11兼得 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for k in KINDS:
        for sub,label in (('train_maps','训练24图'),('heldout_maps','留出6图')):
            vals=[metric(k,f,sub)['mean'] for f in ('food_accuracy','water_accuracy','outcome_00','outcome_10','outcome_01','outcome_11')]
            lines.append(f"| {LABELS[k]} | {label} | "+' | '.join(pct(v) for v in vals)+' |')
    lines += ['', '四结果按资源F/W排序，与随机查询先后无关。这里只展示各种子先平均后的比例；逐运行、方向和整型计数保存在JSON/CSV。','',
        f"![形成过程]({(out/'complementarity_process.png').resolve()})",'',
        '实线正常、虚线打乱；细线为每种子跨划分均值。仅连接已保存的检查点，过程与终点样本量不同。'
        '各面板纵轴范围不同，右上留出双目标面板作局部放大。','',
        '| 比较 | 指标 | 均值差（百分点） | 四种子差（百分点） |','| --- | --- | ---: | --- |']
    ml={'normal/heldout_maps/both_accuracy':'留出双目标','normal/heldout_maps/single_accuracy':'留出单目标',
        'normal/train_maps/both_accuracy':'训练双目标','gap/heldout_maps/both_accuracy':'留出双目标干预落差'}
    for row in summary['paired_comparisons']:
        d=row['seed_means_over_splits']
        lines.append(f"| {LABELS[row['left']]} 减 {LABELS[row['right']]} | {ml[row['metric']]} | {pp(d['mean'])} | "+'、'.join(pp(x) for x in d['values'])+' |')
    lines += ['', '## 原生效用与反馈密度','',
        '下表辅助描述各条件优化的原生目标，R定义跨条件不同，不能把其高低用于共同能力排名。'
        '正回报比例与绝对方差分列；λ=1不必有更高的绝对回报方差，反馈稀疏性与方差不是同一量。'
        '表中方差先按各运行计算，再在种子内及种子间平均，不是混合所有世界后的总体方差。','',
        '| 条件 | 地图组 | R均值 | R方差 | 正回报比例 |','| --- | --- | ---: | ---: | ---: |']
    for k in KINDS:
        for sub,label in (('train_maps','训练24图'),('heldout_maps','留出6图')):
            lines.append(f"| {LABELS[k]} | {label} | {metric(k,'mean_reward',sub)['mean']:.4f} | {metric(k,'reward_variance',sub)['mean']:.4f} | {pct(metric(k,'positive_rate',sub)['mean'])} |")
    lines += ['', '## 自然协议、片段替换与人工拼接','',
        '协议探针使用旧test每类前4张形成16照片对作校准，后4张另16照片对作验证；'
        '这与终点随机照片平衡抽样不同。同一隐藏目标消息分别查询两个需求得到自然双目标标准；'
        'v0.8训练本身就做双目标选择，兼容字段native/switched不构成新增跨目标复用证据。','',
        '只在训练地图选择两个符号位置的资源分配；严格片段替换要求变化资源转到供体正确位置、另一资源仍正确。'
        '全30图分母包括自然失败，另存原本正确子集和训练图内结果，零分母不作0处理。'
        '人工拼接以实验者的地图信息选择训练消息供体，不是主体自主产生未见组合。','',
        '每方向100个完整49码双射，接收映射取逆，保留自然完整消息行为并重做校准；'
        '先验证49码×2需求×720菜单的物理地点等价再折叠菜单。'
        '这些参考不是新增主体、总体置信区间或组合语法证明。','',
        f"本轮合计 {summary['protocol']['reference_randomness']['conditional_reference_scores']:,} 次条件化参考评分。"
        f"预定随机种子公式在120方向中产生 {summary['protocol']['reference_randomness']['distinct_rng_seeds']} 个不同参考RNG种子；"
        '主36个留出运行的72方向无碰撞，复用发生在full/blocked与跨主体的其他条件之间。'
        '各方向100次参考仍按原方案保留，不能把12,000次评分称为独立抽样，也不增加4个主体对的独立重复数。'
        f"[参考随机源审计]({summary['reference_rng_audit']['source']})逐项检查已保存映射，未新增模型推理或重评分。",'',
        '| 条件 | 训练自然双目标 | 留出自然双目标 | 严格片段 | 片段参考 | 拼接 | 拼接参考 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for k in KINDS:
        p=pg[k]['metrics'];lines.append(f'| {LABELS[k]} | '+' | '.join(pct(p[f]['mean']) for f in ('seen_both','unseen_both','fragment','fragment_null','stitch','stitch_null'))+' |')
    lines += ['',f"![协议结构]({(out/'complementarity_protocol.png').resolve()})",'',
        '自然双目标、严格片段和拼接是不同证据。片段可解码不能替代主体自然产出，联合目标优化也不能保证结构会泛化。','',
        f"[固定消息实例]({summary['message_examples']['source']})展示种子27101、split1、方向0→1、"
        '首个验证照片对下，三种λ的全部6张留出地图，共18行。该切片来自已保存normal轨迹，'
        '用于查看实际消息与食物/水选择，未新增推理，也不按成功效果选择案例，不能提高统计证据强度。','',
        '## 完整地图、关闭通道与个人能力','',
        '| λ | 全图双目标 | 全图平均单目标 | 关通道双目标 | 关通道平均单目标 |','| --- | ---: | ---: | ---: | ---: |']
    for k in KINDS:
        vals=[metric(prefix+'_'+k,f,'all')['mean'] for prefix in ('full','blocked') for f in ('both_accuracy','single_accuracy')]
        lines.append(f'| {WEIGHTS[k]} | '+' | '.join(pct(v) for v in vals)+' |')
    control=summary['controls']['groups']['identity']
    lines += ['',f"个人I视觉行动正控8名副本的正常准确率范围为 {pct(control['individual_scores']['min'])}–{pct(control['individual_scores']['max'])}；"
        f"清空现场表示的种子均值为 {pct(control['erase_per_seed_agent_means']['mean'])}。"
        '均通过事先90%门槛，诊断在社会运行前完成，编码器、动作头和优化器完全隔离。'
        '它验证给定预算下视觉路径可学习行动，不代表社会主体在起点已经拥有该策略。','',
        f"![基准与个人正控]({(out/'complementarity_controls.png').resolve()})",'',
        '左图点为种子内两人平均，90%为工程门槛；中、右图点为4个主体对，分别比较全图通信及从头关闭通道。'
        '全30图双目标最优固定地点对上界为1/30，平均单目标为1/6；两条线适用于相应图的完整地图支持。','',
        '无信息基准依赖地图支持：训练24图最优固定合法对的双目标为1/24；留出6图固定对若正好是留出地图则为1/6，否则为0。'
        '各支持的最优固定对未必能由同一个训练后策略同时达到。独立均匀随机两个动作才是1/36，不能作为所有双目标图的共同机会线。'
        '全30图最优固定策略R为(1−λ)/6+λ/30。','',
        '## 已有研究与解释边界','',
        'Lee（2024）的One-to-Many Communication and Compositionality in Emergent Communication已研究平均与组内联合成功压力。'
        '第3节在印刷页20796（PDF第3页）定义组内全部正确奖励与发送者平均奖励；第6.3节及图4在20798–20799页（PDF第5–6页）讨论协调压力与结构。'
        '因此联合成功奖励本身不是本项目发现的研究空白。该研究有多个独立听者、属性真值输入和接收者交叉熵；'
        '本轮是固定视觉路径、同一接收者两需求、标量效用强度的有限探索，不能视作严格复现或普遍人类语言机制。','',
        '[Lee (2024) 本地全文](</Users/xia/Documents/ChatGPT/语言/文献调研_2026-09-14/论文/02_神经多智能体/2024_Lee_One-to-Many Communication and Compositionality in Emergent Communication.pdf>)。'
        '完整书目：Heeyoung Lee. 2024. Proceedings of EMNLP 2024, 20794–20811. DOI:10.18653/v1/2024.emnlp-main.1157。'
        '本轮仅复核现有本地来源，没有新增文献检索。','',
        '本轮四探索种子、旧照片及三个同构留出限制外推。没有分离回报稀疏性、梯度规模、有限预算优化与共享编码的中介作用；'
        '成功只能说明当前非语言能力和社会学习安排的组合可实现相应行为，不能识别最小充分能力集合或必要生存压力。'
        '后续需要预定的新种子、新照片确认及有针对性的学习信号对照，不能因本轮高分便宣称可靠组合语言已经形成。','',
        '## 保存记录与独立核查','',
        f"本分析独立重算社会 {summary['worlds_independently_checked']:,} 个双选择世界（{summary['decisions_independently_checked']:,} 个采集选择），"
        f"以及个人诊断 {summary['controls']['trace_records_checked']:,} 次选择。"
        '世界/行动计数、查询目标顺序、各自菜单、同一条消息、资源成功及四结果效用均核对；'
        '保持NumPy独立实现，不导入策略或训练器。训练日志逐更新核对四结果、效用均值/方差、正回报数、预算与跨λ世界哈希。','',
        f"协议120方向及12,000次条件化参考评分的范围、参数文件、校准地图和 {summary['protocol']['integer_ratios_checked']:,} 项整型比例已核对。",'']
    if summary.get('execution_audit'):
        a=summary['execution_audit'];lines += [f"独立执行审计 [audit_execution.json]({a['source']})：60/60运行通过，{a['checks_count']:,} 项检查，重建 {a['train_updates_verified']:,} 次社会更新。",'']
    lines += [f"[summary.json]({(out/'summary.json').resolve()})、[per_run_metrics.csv]({(out/'per_run_metrics.csv').resolve()})"
        f"及[完整协议分析]({summary['protocol']['source']})保留逐运行、方向、条件和全部五模式结果。",'',
        '复现汇总：`.venv/bin/python redesign_v0.8/analyze_complementarity.py`。','']
    (out/'资源互补收益实验报告.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/complementarity_001')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    parser.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=CONDITIONS)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();source=args.input.resolve();out=(args.output or source/'analysis').resolve()
    photos=read_json(ROOT.parent/'redesign_v0.4/data/manifest.json')['images'];design=split_audit(6,SPLITS)
    runs=[];missing=[];references={}
    for seed in args.seeds:
        for condition in args.conditions:
            path=source/f's{seed}_{condition}'
            if not (path/'result.json').is_file():missing.append(path.name);continue
            r,a=read_run(path,photos,design)
            if seed in references:compare_world(references[seed],a,'Across lambda/condition exogenous worlds')
            else:references[seed]=a
            runs.append(r);print(f"AUDITED {path.name}: both={r['scores']['normal']['both_accuracy']:.6f}, single={r['scores']['normal']['single_accuracy']:.6f}",flush=True)
    summary={'status':'complete' if not missing else 'incomplete','generated_utc':datetime.now(timezone.utc).isoformat(),
        'input':str(source),'expected_seeds':args.seeds,'expected_conditions':args.conditions,
        'expected_runs':len(args.seeds)*len(args.conditions),'completed_runs':len(runs),'missing_runs':missing,
        'analysis_source_hashes':{p.name:sha(p) for p in (Path(__file__),ROOT/'analysis_core.py')},
        'design_audit':design,'provenance_audit':provenance(runs),'runs':runs,
        'worlds_independently_checked':sum(a['worlds_checked'] for r in runs for a in r['trace_audit'].values()),
        'decisions_independently_checked':sum(a['decisions_checked'] for r in runs for a in r['trace_audit'].values())}
    if not args.check_only:
        require(not missing and args.seeds==SEEDS and args.conditions==CONDITIONS,'Complete fixed 60-run scope required')
        summary['protocol']=protocol_summary(source,runs);require(summary['protocol'] is not None,'Wait for protocol analysis')
        examples=source/'消息实例.md';require(examples.is_file(),'Wait for fixed saved-trace message examples')
        summary['message_examples']={'source':str(examples),'source_sha256':sha(examples),'scope':'fixed seed27101/split1/scout0/first validation photo pair; all three rewards and six heldout maps'}
        rng_audit=source/'reference_rng_audit.md';require(rng_audit.is_file(),'Wait for reference-randomness audit record')
        summary['reference_rng_audit']={'source':str(rng_audit),'source_sha256':sha(rng_audit),'json_source':str(source/'reference_rng_audit.json'),'json_sha256':sha(source/'reference_rng_audit.json')}
        summary['controls']=control_summary(source,runs,photos)
        audit_path=source/'audit_execution.json';require(audit_path.is_file(),'Wait for execution audit')
        audit=read_json(audit_path);require(audit['status']=='passed' and audit['completed_runs']==audit['planned_runs']==60,'Execution audit incomplete')
        summary['execution_audit']={'source':str(audit_path),'source_sha256':sha(audit_path),'completed_runs':60,'planned_runs':60,
            'checks_count':sum(audit['checks'].values()),'train_updates_verified':audit['train_updates_verified']}
        summary['groups']=aggregate(runs);summary['paired_comparisons']=paired(runs)
        out.mkdir(parents=True,exist_ok=True);summary['plots']=plots(summary,out)
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        with (out/'per_run_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['seed','condition','split','lambda','mode','scope','worlds','decisions','both_correct','single_correct','food_correct','water_correct','00','01','10','11','both_accuracy','single_accuracy','mean_reward','reward_variance','positive_rewards'])
            for r in runs:
                for mode in MODES:
                    score=r['scores'][mode]
                    for scope,row in [('all',score),*score['map_subsets'].items(),*[('scout'+str(i),d) for i,d in enumerate(score['direction_metrics'])]]:
                        w.writerow([r['seed'],r['condition'],r['split'],r['config']['plan']['complementarity'],mode,scope,row['n'],row['decisions'],row['both_correct'],row['single_correct'],row['food_correct'],row['water_correct'],*[row['outcome_counts'][x] for x in ('00','01','10','11')],row['both_accuracy'],row['single_accuracy'],row['mean_reward'],row['reward_variance'],row['positive_rewards']])
        report(summary,out)
    print(json.dumps({k:summary[k] for k in ('status','completed_runs','expected_runs','worlds_independently_checked','decisions_independently_checked')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
