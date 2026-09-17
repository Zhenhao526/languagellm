"""Read-only v0.7 static trajectory audit and representation-comparison report.

No training, Torch or policy imports. Run with the project .venv Python.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from analysis_core import audit_static_arrays, describe, equal, near, require, split_audit, static_metrics

ROOT=Path(__file__).resolve().parent
SEEDS=[26101,26102,26103,26104]
KINDS=("identity","permute","orthogonal")
FULLKINDS=tuple("full_"+k for k in KINDS)+("full_blocked",)
CONDITIONS=[f"split{s}_{k}" for s in (1,2,3) for k in KINDS]+list(FULLKINDS)
MODES=("normal","shuffle","blank","stochastic","erase_memory")
MATCHINGS={1:((0,1),(2,3),(4,5)),2:((0,2),(1,4),(3,5)),3:((0,3),(1,5),(2,4))}
SPLITS={s:[p for e in edges for p in (e,e[::-1])] for s,edges in MATCHINGS.items()}
LABELS={"identity":"I 保留地点槽","permute":"P 置换地点槽","orthogonal":"Q 正交混合地点",
        "full_identity":"全图 I","full_permute":"全图 P","full_orthogonal":"全图 Q","full_blocked":"全图 I 关闭通道"}



def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_world(a, b, context):
    for key in ("scout", "step", "episode", "positions", "inventory", "history", "goals", "menu", "photo_ids", "refill_uniform"):
        equal(a[key], b[key], f"{context}: exogenous {key}")


def audit_trace(path, cfg, stated, photos):
    with np.load(path, allow_pickle=False) as saved:
        a = {k: saved[k] for k in saved.files}
    plan = cfg["plan"]
    require(stated["episodes"] == cfg["eval_n"] and stated["horizon"] == 1, f"{path}: evaluation size/horizon")
    reward, audit = audit_static_arrays(a, n_sites=cfg["sites"], vocab=plan["vocab"], length=plan["length"],
        history_events=2, episodes=stated["episodes"], mode=path.stem.removeprefix("final_"),
        blocked=plan["blocked"], photo_metadata=photos)
    maps = np.asarray(cfg["map_table"])
    metrics = static_metrics(a, reward, n_sites=cfg["sites"], heldout_maps=maps[cfg["heldout_map_ids"]])
    near(metrics["mean_reward"], stated["mean_reward"], f"{path}: reported reward")
    for i, direction in enumerate(metrics["direction_means"]):
        near(direction["mean_reward"], stated["direction_means"][i], f"{path}: direction {i}")
    for recorded, calculated in (("seen", "train_maps"), ("unseen", "heldout_maps")):
        m, s = metrics["map_subsets"][calculated], stated["map_groups"][recorded]
        require(m["n"] == s["n"] and m["reward_sum"] == s["correct"], f"{path}: {recorded} count")
        if m["n"]:
            near(m["mean_reward"], s["mean_reward"], f"{path}: {recorded} mean")
        else:
            require(s["mean_reward"] is None, f"{path}: empty group must be undefined")
    h = hashlib.sha256()
    for direction in (0, 1):
        ix = np.flatnonzero(a["scout"] == direction)
        ix = ix[np.argsort(a["episode"][ix])]
        for column in ("positions", "photo_ids", "goals", "menu", "refill_uniform"):
            h.update(np.ascontiguousarray(a[column][ix]).tobytes())
    require(h.hexdigest() == stated["world_sha256"], f"{path}: world hash")
    audit.update(path=str(path), sha256=sha(path), balanced_evaluation=True, world_hash_verified=True)
    return metrics, a, audit


def read_run(path, photos, design):
    cfg, result = read_json(path / "config.json"), read_json(path / "result.json")
    for key in ("seed", "condition", "plan", "updates", "batch", "initial_sha256"):
        require(cfg[key] == result[key], f"{path}: config/result {key}")
    require(cfg["sites"] == 6 and cfg["history_dim"] == 18, f"{path}: unexpected environment shape")
    require(not cfg["plan"]["known"], f"{path}: expected hidden goal")
    require(cfg["plan"]["vocab"] ** cfg["plan"]["length"] == 49, f"{path}: message capacity")
    require(cfg["plan"]["schedule"] == "direct" and cfg["plan"]["representation"] in KINDS,
            f"{path}: representation/direct plan")
    equal(cfg["map_table"], design["all_maps"], f"{path}: map table")
    split = cfg["plan"]["split"]
    held = SPLITS[split] if split else []
    all_maps = np.asarray(cfg["map_table"])
    require(set(map(tuple, all_maps[cfg["heldout_map_ids"]])) == set(held), f"{path}: heldout maps")
    require(set(cfg["train_map_ids"]) == set(range(30)) - set(cfg["heldout_map_ids"]), f"{path}: training maps")
    require(result["frozen_projection_verified"], f"{path}: frozen projection check absent")
    require(sha(cfg["prepared_source"]["path"]) == cfg["prepared_source"]["sha256"], f"{path}: preparation file changed")
    curve = read_json(path / "curve.json")
    equal([row["update"] for row in curve], cfg["checkpoints"], f"{path}: checkpoint coverage")
    require(curve[-1]["update"] == cfg["updates"], f"{path}: final checkpoint missing")
    for row in curve:
        require((path / f"checkpoint_{row['update']:04d}.pt").is_file(), f"{path}: missing checkpoint weights")
        for mode in MODES:
            score = row["scores"][mode]
            require(score["episodes"] == cfg["checkpoint_eval_n"] and score["horizon"] == 1, f"{path}: checkpoint evaluation size")
            require(sum(g["n"] for g in score["map_groups"].values())==score["episodes"],f"{path}: checkpoint group sizes")
            near(score["mean_reward"],sum(g["correct"] for g in score["map_groups"].values())/score["episodes"],f"{path}: checkpoint overall ratio")
            for group in score["map_groups"].values():
                if group["n"]:
                    near(group["mean_reward"], group["correct"] / group["n"], f"{path}: checkpoint group ratio")
                else:
                    require(group["mean_reward"] is None, f"{path}: empty checkpoint group")
    scores, audits, baseline = {}, {}, None
    for mode in MODES:
        scores[mode], arrays, audits[mode] = audit_trace(path / f"final_{mode}.npz", cfg, result["scores"][mode], photos)
        if baseline is None:
            baseline = arrays
        else:
            compare_world(baseline, arrays, f"{path}: {mode}")
    training = [json.loads(line) for line in (path / "training.jsonl").read_text().splitlines() if line.strip()]
    equal([r["update"] for r in training], list(range(1, cfg["updates"] + 1)), f"{path}: update continuity")
    require(all(np.isfinite(r["reward"]) for r in training), f"{path}: nonfinite training reward")
    schedule = read_json(path / "training_schedule.json")
    equal([r["batch_identity"] for r in training], schedule["batch_identities"], f"{path}: batch identity order")
    equal([r["active_sites"] for r in training], schedule["levels"], f"{path}: difficulty order")
    equal(sorted(schedule["batch_identities"]), list(range(cfg["updates"])), f"{path}: each batch once")
    for row in training:
        allowed = set(row["allowed_sites"])
        require(len(allowed) == row["active_sites"] and allowed.issubset(range(6)), f"{path}: allowed site set")
        require(row["active_sites"]==6 and allowed==set(range(6)),f"{path}: all updates must use the direct six-site task")
        expected = {i for i in cfg["train_map_ids"] if set(all_maps[i]).issubset(allowed)}
        require(set(row["map_pool"]) == expected and expected, f"{path}: legal map support")
        near(row["entropy_weight"], .02 if row["update"] <= cfg["entropy_off_after"] else 0, f"{path}: entropy schedule")
    gap = {"all": scores["normal"]["mean_reward"] - scores["shuffle"]["mean_reward"]}
    for group in ("train_maps", "heldout_maps"):
        x, y = scores["normal"]["map_subsets"][group], scores["shuffle"]["map_subsets"][group]
        gap[group] = x["mean_reward"] - y["mean_reward"] if x["n"] else None
    return {"seed": cfg["seed"], "condition": cfg["condition"], "split": split,
            "kind": cfg["condition"].split("_", 1)[1] if split else cfg["condition"],
            "directory": str(path), "config": cfg, "scores": scores, "normal_minus_shuffle": gap,
            "curve": curve, "trace_audit": audits, "seconds": result["seconds"],
            "initial_sha256": result["initial_sha256"], "final_state_sha256": result["final_sha256"],
            "trainable_initial_sha256": cfg["trainable_initial_sha256"],
            "receiver_initial_sha256": cfg["receiver_initial_sha256"],
            "final_file_sha256": sha(path / "final.pt"), "result_sha256": sha(path / "result.json"),
            "training_sha256": sha(path / "training.jsonl"), "training_record_count": len(training),
            "batches_by_identity": {str(r["batch_identity"]): {k: r[k] for k in ("active_sites", "allowed_sites", "map_pool", "world_sha256")} for r in training}}, baseline




def provenance(runs):
    records=[]
    for r in runs:
        kind=r["config"]["plan"]["representation"]
        transforms=np.asarray(r["config"]["input_transforms"],dtype=np.float64)
        require(transforms.shape==(2,6,6),"Two private 6x6 transforms required")
        for who,matrix in enumerate(transforms):
            error=float(np.max(np.abs(matrix.T@matrix-np.eye(6))))
            require(error<1e-6,"Input transform is not orthogonal to float32 tolerance")
            if kind=="identity":equal(matrix,np.eye(6),"Identity buffer")
            elif kind=="permute":
                require(np.isin(matrix,[0,1]).all(),"Permutation has nonbinary entries")
                equal(matrix.sum(0),np.ones(6),"Permutation column counts")
                equal(matrix.sum(1),np.ones(6),"Permutation row counts")
                require(np.diag(matrix).sum()==0,"Specified permutation must move every site")
            records.append({"condition":r["condition"],"seed":r["seed"],"agent":who,
                            "representation":kind,"max_orthogonality_error":error})
    for seed in sorted({r["seed"] for r in runs}):
        family=[r for r in runs if r["seed"]==seed]
        for field in ("trainable_initial_sha256","receiver_initial_sha256"):
            require(len({r[field] for r in family})==1,f"Seed {seed}: {field} differs")
        require(len({tuple(r["config"]["trainable_parameters"]) for r in family})==1,"Trainable parameter counts differ")
        records.append({"seed":seed,"trainable_and_receiver_initialization_match":True,"trainable_parameter_counts_match":True})
        for split in sorted({r["split"] for r in family}):
            group=[r for r in family if r["split"]==split]
            require(all(r["batches_by_identity"]==group[0]["batches_by_identity"] for r in group),
                    f"{seed}/{split}: representation conditions see different training batches")
            records.append({"seed":seed,"split":split,"training_worlds_match_each_batch":True,"run_count":len(group)})
    return records


def paired(runs):
    index={(r["seed"],r["split"],r["kind"]):r for r in runs}
    output=[]
    for left,right in (("identity","orthogonal"),("permute","orthogonal"),("identity","permute")):
        for metric in (("normal","heldout_maps"),("normal","train_maps"),("gap","heldout_maps")):
            rows=[]
            for seed in SEEDS:
                for split in (1,2,3):
                    if (seed,split,left) not in index or (seed,split,right) not in index:continue
                    a,b=index[seed,split,left],index[seed,split,right]
                    require(a["config"]["updates"]==b["config"]["updates"] and a["config"]["batch"]==b["config"]["batch"],"Paired budget differs")
                    rows.append({"seed":seed,"split":split,"difference":run_value(a,metric)-run_value(b,metric)})
            if rows:
                seeds=sorted({r["seed"] for r in rows})
                means=[float(np.mean([r["difference"] for r in rows if r["seed"]==s])) for s in seeds]
                output.append({"left":left,"right":right,"metric":"/".join(metric),"seeds":seeds,
                               "seed_means_over_splits":describe(means),"split_seed_differences":rows})
    return output


def run_value(run, metric):
    mode, group = metric
    if mode == "gap":
        return run["normal_minus_shuffle"][group]
    return run["scores"][mode]["mean_reward"] if group == "all" else run["scores"][mode]["map_subsets"][group]["mean_reward"]


def aggregate(runs):
    metrics = [(mode, group) for mode in MODES for group in ("all", "train_maps", "heldout_maps")] + [("gap", g) for g in ("all", "train_maps", "heldout_maps")]
    output = {}
    for kind in (*KINDS, *FULLKINDS):
        selected = [r for r in runs if r["kind"] == kind]
        if not selected:
            continue
        seeds = sorted({r["seed"] for r in selected})
        group = {"seeds": seeds, "metrics": {}, "by_split": {}}
        for metric in metrics:
            if any(run_value(r, metric) is None for r in selected):
                continue
            per_seed = []
            for seed in seeds:
                row = [r for r in selected if r["seed"] == seed]
                require(len(row) == (3 if kind in KINDS else 1), f"{kind}/{seed}: incomplete split family")
                per_seed.append(float(np.mean([run_value(r, metric) for r in row])))
            group["metrics"]["/".join(metric)] = describe(per_seed)
            for split in sorted({r["split"] for r in selected}):
                local = sorted([r for r in selected if r["split"] == split], key=lambda r: r["seed"])
                group["by_split"].setdefault(str(split), {"seeds": [r["seed"] for r in local], "metrics": {}})["metrics"]["/".join(metric)] = describe([run_value(r, metric) for r in local])
        output[kind] = group
    return output




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
    return {"source":str(path),"source_sha256":sha(path),"groups":groups,"direction_rows":rows,
            "photo_splits":data["photo_splits"],"integer_ratios_checked":ratios,
            "audit":"passed: run scope, checkpoint hashes, calibration maps, integer ratios, 49-code bijections and reference means",
            "aggregation":"two directions and three coordinate splits averaged within seed, then four seeds equally weighted; full conditions occur only once per seed"}




def pct(x):
    return "不适用" if x is None else f"{100*x:.2f}%"


def pp(x):
    return "不适用" if x is None else f"{100*x:+.2f}"


def control_summary(source,runs,photos):
    path=source/"individual_controls/summary.json"
    data=read_json(path)
    require(data["complete"] and data["runs"]==12,"Personal controls incomplete")
    gate_path=source/"social_launch_gate.json"
    gate=read_json(gate_path)
    require(gate["controls_complete"]==12 and gate["social_runs_planned"]==52 and gate["all_passed"],"Social launch gate")
    require(gate["control_summary_sha256"]==sha(path) and not gate["diagnostic_weights_transferred"],"Launch gate control source")
    index={(r["seed"],r["config"]["plan"]["representation"]):r for r in runs}
    rows=[];total=0
    for seed in SEEDS:
        for kind in KINDS:
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
    for kind in KINDS:
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
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":["PingFang SC","DejaVu Sans"],"font.size":11,
                         "axes.unicode_minus":False,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    colors=["#3B83BD","#D59331","#9171B1","#278D80"];files=[]
    def save(fig,stem):
        for ext in ("png","pdf"):fig.savefig(out/f"{stem}.{ext}",dpi=240)
        plt.close(fig);files.append(stem+".png")
    def scatter(ax,i,values,color=None):
        ax.scatter(i+np.linspace(-.10,.10,len(values)),np.asarray(values)*100,
                   c=color if color else colors[:len(values)],s=34,zorder=3)
        ax.plot([i-.19,i+.19],[100*np.mean(values)]*2,color="#263139" if color is None else color,lw=2.3)
    fig=plt.figure(figsize=(10,7.3),constrained_layout=True);grid=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
    for ax,metric,title in zip(axes,("normal/train_maps","normal/heldout_maps","gap/heldout_maps"),
                              ("训练24图准确率（%）","留出6图准确率（%）","留出6图正常−打乱（百分点）")):
        for i,k in enumerate(KINDS):scatter(ax,i,summary["groups"][k]["metrics"][metric]["values"])
        ax.set_ylabel(title);ax.set_xticks(range(3),["I 保留槽","P 置换槽","Q 正交混合"]);ax.grid(axis="y",alpha=.15)
        ax.axhline(100/6 if metric.startswith("normal") else 0,ls=":" if metric.startswith("normal") else "-",color="#8B9298",lw=1)
        if metric.startswith("normal"):ax.set_ylim(-2,104)
    fig.suptitle("视觉输入方式：点为种子跨三划分均值，短线为4种子均值",fontsize=13)
    save(fig,"binding_comparison")
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.6),sharey=True,constrained_layout=True)
    for k,ax in zip(KINDS,axes):
        local=[r for r in summary["runs"] if r["kind"]==k]
        for mode,group,color,style,text in (("normal","seen","#5179B1","-","训练图正常"),
                ("normal","unseen","#148781","-","留出图正常"),("shuffle","unseen","#BF8035","--","留出图打乱")):
            seed_curves=[];x=None
            for seed in SEEDS:
                family=[r for r in local if r["seed"]==seed]
                x=[row["update"] for row in family[0]["curve"]]
                require(all([row["update"] for row in r["curve"]]==x for r in family),"Process checkpoints differ")
                y=np.mean([[100*row["scores"][mode]["map_groups"][group]["mean_reward"] for row in r["curve"]] for r in family],axis=0)
                seed_curves.append(y);ax.plot(x,y,color=color,ls=style,lw=.7,alpha=.18)
            ax.plot(x,np.mean(seed_curves,axis=0),color=color,ls=style,lw=2,marker="o",ms=3,label=text)
        ax.set(title=LABELS[k],xlabel="训练更新",ylim=(-2,104));ax.grid(axis="y",alpha=.15);ax.axhline(100/6,color="#8B9298",ls=":",lw=1)
    axes[0].set_ylabel("单目标准确率（%）");axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle("通信形成过程：实际检查点；细线为种子跨划分均值",fontsize=13)
    save(fig,"binding_process")
    pg=summary["protocol"]["groups"]
    fig=plt.figure(figsize=(10.8,7.8),constrained_layout=True);grid=fig.add_gridspec(2,2)
    for ax,metrics,title in ((fig.add_subplot(grid[0,0]),("seen_both","unseen_both"),"自然消息：同一消息双目标均正确"),
            (fig.add_subplot(grid[0,1]),("fragment","fragment_null"),"严格局部替换：完整30图探针"),
            (fig.add_subplot(grid[1,:]),("stitch","stitch_null"),"训练供体拼接至留出6图：双目标均正确")):
        for i,k in enumerate(KINDS):
            for j,(metric,color) in enumerate(zip(metrics,("#397FB1","#D68B36"))):
                values=pg[k]["metrics"][metric]["values"];x=i+(-.16 if j==0 else .16)
                scatter(ax,x,values,color)
                if i==0:ax.scatter([],[],color=color,label=("训练24图" if j==0 else "留出6图") if metrics[0]=="seen_both" else ("观察值" if j==0 else "整体重编码参考均值"))
        ax.set(title=title,ylabel="成功率（%）",ylim=(-3,104));ax.set_xticks(range(3),["I 保留槽","P 置换槽","Q 正交混合"])
        ax.grid(axis="y",alpha=.15);ax.legend(frameon=False,fontsize=9)
    fig.suptitle("协议结构：照片平衡枚举；4种子跨划分和方向均值",fontsize=13)
    save(fig,"binding_protocol")
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5),sharey=True,constrained_layout=True)
    for i,k in enumerate(KINDS):
        scatter(axes[0],i,summary["controls"]["groups"][k]["per_seed_agent_means"]["values"])
        scatter(axes[1],i,summary["groups"]["full_"+k]["metrics"]["normal/all"]["values"])
    scatter(axes[1],3,summary["groups"]["full_blocked"]["metrics"]["normal/all"]["values"])
    axes[0].set(title="个人全信息行动正控",ylabel="单目标准确率（%）",ylim=(-2,104));axes[0].axhline(90,ls=":",color="#8B9298",lw=1)
    axes[0].set_xticks(range(3),["I 保留槽","P 置换槽","Q 正交混合"])
    axes[1].set(title="全30图社会通信基准",ylim=(-2,104));axes[1].axhline(100/6,ls=":",color="#8B9298",lw=1)
    axes[1].set_xticks(range(4),["I","P","Q","I 关通道"])
    for ax in axes:ax.grid(axis="y",alpha=.15)
    fig.suptitle("区分个人视觉行动学习与社会通信；正控权重不回灌",fontsize=13)
    save(fig,"capability_controls")
    return files


def report(summary,out):
    groups=summary["groups"];pg=summary["protocol"]["groups"]
    lines=["# v0.7 视觉地点局部性与共同通信的形成", "",
           f"生成时间：{summary['generated_utc']}。主实验 {summary['completed_runs']}/{summary['expected_runs']} 个运行完成。", "",
           "## 主要发现", "",
           "以下比较相同原始信息、相同可训练参数初始化下的三种视觉输入方式：I保留地点槽、P置换地点槽、Q稠密正交混合地点。"
           "Q在原始输入上可逆，未移除视觉预训练或世界信息；实验检验的是局部处理偏好及学习过程。", "",
           "| 输入方式 | 训练24图单目标 | 留出6图单目标 | 留出自然双目标 | 人工拼接双目标 | 拼接重编码参考 |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for k in KINDS:
        g=groups[k]["metrics"];p=pg[k]["metrics"]
        lines.append(f"| {LABELS[k]} | {pct(g['normal/train_maps']['mean'])} | {pct(g['normal/heldout_maps']['mean'])} | {pct(p['unseen_both']['mean'])} | {pct(p['stitch']['mean'])} | {pct(p['stitch_null']['mean'])} |")
    lines += [""]
    for left,right,role in (("identity","orthogonal","预先规定的主比较I−Q"),("permute","orthogonal","辅助比较P−Q")):
        reward_diff=np.asarray(groups[left]["metrics"]["normal/heldout_maps"]["values"])-np.asarray(groups[right]["metrics"]["normal/heldout_maps"]["values"])
        dual_diff=np.asarray(pg[left]["metrics"]["unseen_both"]["values"])-np.asarray(pg[right]["metrics"]["unseen_both"]["values"])
        rm,dm=float(reward_diff.mean()),float(dual_diff.mean())
        relation="均值变化方向一致" if rm*dm>0 else "均值变化方向不同" if rm*dm<0 else "至少一项均值差为零"
        lines += [f"{role}：留出单目标平均差 {pp(rm)} 个百分点，四种子差为 "+"、".join(pp(x) for x in reward_diff)+
                  f"；自然留出双目标平均差 {pp(dm)} 个百分点，四种子差为 "+"、".join(pp(x) for x in dual_diff)+
                  f"。两项{relation}，仍需保留各种子的波动，不能把平均单目标变化自动解释为自主新组合表达能力提高。", ""]
    ip_diff=np.asarray(groups["identity"]["metrics"]["normal/heldout_maps"]["values"])-np.asarray(groups["permute"]["metrics"]["normal/heldout_maps"]["values"])
    lines += [f"I相对P的留出单目标平均差为 {pp(float(ip_diff.mean()))} 个百分点，四种子均为正；"
              "P同样保留一对一地点局部槽，却未在单目标均值上超过Q。全图基准中，"
              f"Q为 {pct(groups['full_orthogonal']['metrics']['normal/all']['mean'])}，I为 {pct(groups['full_identity']['metrics']['normal/all']['mean'])}。"
              "这些结果提示成绩对地点槽的安排敏感；结合I−Q中一个种子反向，本轮不支持稳健的一般视觉地点局部性优势。", "",
              "三种表示的自然留出双目标成功仍很低，人工拼接成功则明显高于各自整体重编码参考。"
              "这支持有限的片段解码与复用，尚未形成可靠的自主新组合产出。人工拼接与自然产出的差距"
              "不能单独定位为视觉编码、消息产出或优化中的某一个瓶颈。", "",
              "个人全信息视觉行动正控中，I/P/Q三类24个主体副本均为9600/9600=100%，"
              "清空现场表示后均为1600/9600=16.67%。这说明三种视觉路径在该正控任务与预算下都能支持行动学习；"
              "它不保证社会场景中的发信者会学到相同编码，也不把正控满分视为自然通信成功。", "",
              "单目标回报与同一消息双目标成功是不同标准。人工拼接使用实验者掌握的地图信息选取训练消息供体，"
              "不等于主体自然地产生了新组合表达。差异需同时结合个人正控、完整地图基准、种子差异和结构干预判断，"
              "不能将一种表示的成绩直接解释为某种一般能力必要或充分。", "",
              "## 主体、输入与能力对照", "",
              "使用官方冻结DINOv2 ViT-L/14的旧照片缓存：44张训练、16张既往开发留出，没有新的视觉确认集。"
              "四个新主体对种子各自进行个人资源后果准备，仅迁移并冻结私人1024→64视觉投影；社会接口新初始化，"
              "不继承上一轮成功消息。每个主体的六地点视觉向量与存在量整理为6×65数组X。", "",
              "三条件分别输入IX、PX、QX，再对每个槽使用共享的65→65线性层与Tanh，串接后输入GRU96。"
              "I保持原地点槽；P是每主体固定的私有地点置换，保持一对一局部槽；Q是每主体固定的私有6×6稠密正交矩阵，"
              "在地点轴上混合64维视觉和第65维存在量。三组具有相同可训练参数数目、相同初始化和接收者参数；"
              "条件特定矩阵是不可训练buffer，故整个状态文件的初始哈希无需相同。", "",
              "正交性仅说明原始输入上的可逆性与范数保持；经过局部非线性之后不保证可恢复性、可读性或有限预算下优化等价。"
              "P帮助区分局部槽保留与坐标重新编号，Q同时改变局部混合和非线性处理的输入分布。"
              "本轮不能把差异单独归因为移除世界知识，也不能据此确定抽象因果推理或一般语言能力。", "",
              "采集者仍获供二维自身需求，并按需求选择两组六地点行动分支；公共坐标、存在量及行动菜单都是给定的非语言接口。"
              "发送者不知道对方当轮需求和私有菜单，仅发送两个七选一编号，共49条完整消息；它们足以查表表示30图。"
              "没有预设资源词、位置词或组合语法奖励，各主体间只传递不可微整数消息。", "",
              "## 任务、预算和留出", "",
              "六地点中一个食物、一个水且不共址，共30张有序地图；单轮采集，没有跨轮库存、长期记忆、自发分工或新人。"
              "库存及18维个人历史恒零。三个留出划分各保留三条完美匹配边的双向地图，训练24图、留出6图；"
              "各资源地点边际平衡，三组留出互不相交，合计覆盖18/30图。它们是同构坐标划分，非三个独立总体。", "",
              "| 划分 | 双向留出的无向地点对 |", "| --- | --- |"]
    for split,edges in MATCHINGS.items():lines.append(f"| {split} | "+"、".join(f"({a},{b})" for a,b in edges)+" |")
    lines += ["", "4种子×3划分×3表示共36个主留出运行；每种子再做三表示的全30图通信和I关闭通道，共16个基准运行，总52。"
              "全部直接六地点、2400次更新×每批512段。同一种子同一划分的三表示匹配每个训练批次的世界、照片、需求和菜单。"
              "没有课程因素，不把本轮结果写成新的课程结论。", "",
              f"主社会训练实际 {sum(r['training_record_count'] for r in summary['runs']):,} 次更新。"
              "另有4种子×3表示的12个个人全信息行动正控，每个包含两人的独立视觉路径与私人行动头。"
              "个人正控为每人每更新512次选择、共2400更新；主社会运行每主体对每更新512段、每方向256段。"
              "两者更新数相同，总行动量的单位不同，不能声称个人与社会任务的全部经历量匹配。"
              "正控训练权重全部隔离，不回灌社会训练；它检验给定场景与自身需求时能否学习行动，不能证明社会通信已经学会。", "",
              "终点每运行每模式9600段，30地图×2需求×2方向每格80次照片抽样；过程1200段，每格10次。"
              "正常取最大概率消息与动作，打乱在同方向同需求内重排完整消息；blank固定0、策略采样和清空现场表示另存。"
              "旧test照片是开发留出，不作为新确认数据。统计单位为4个主体对种子，先在种子内跨三划分平均；"
              "方向、划分、照片和评估案例不增加独立重复数。", "",
              "## 功能通信与形成过程", "",
              "| 表示 | 训练图 | 留出图 | 留出四种子均值 | 留出正常−打乱（百分点） |", "| --- | ---: | ---: | --- | ---: |"]
    for k in KINDS:
        g=groups[k]["metrics"]
        lines.append(f"| {LABELS[k]} | {pct(g['normal/train_maps']['mean'])} | {pct(g['normal/heldout_maps']['mean'])} | "+"、".join(pct(x) for x in g['normal/heldout_maps']['values'])+f" | {pp(g['gap/heldout_maps']['mean'])} |")
    for stem,caption in (("binding_comparison","点为种子跨三划分均值，短线为4种子均值。1/6虚线只对应均衡单目标无通信期望。"),
                         ("binding_process","仅连接真实检查点；细线是每种子的跨划分均值。过程1200例与终点9600例不是同一抽样口径。")):
        lines += ["",f"![{stem}]({(out/(stem+'.png')).resolve()})","",caption]
    lines += ["", "## 同种子、同划分配对差", "", "| 比较 | 指标 | 均值差（百分点） | 四种子差值（百分点） |", "| --- | --- | ---: | --- |"]
    ml={"normal/heldout_maps":"留出单目标","normal/train_maps":"训练单目标","gap/heldout_maps":"留出消息干预落差"}
    for r in summary["paired_comparisons"]:
        g=r["seed_means_over_splits"]
        lines.append(f"| {LABELS[r['left']]} 减 {LABELS[r['right']]} | {ml[r['metric']]} | {pp(g['mean'])} | "+"、".join(pp(x) for x in g['values'])+" |")
    lines += ["", "## 个人能力与完整地图基准", "", "| 表示 | 个人正控两人种子均值 | 个人全部被试范围 | 全图社会正常 | 全图正常−打乱（百分点） |", "| --- | ---: | --- | ---: | ---: |"]
    for k in KINDS:
        c=summary["controls"]["groups"][k];g=groups["full_"+k]["metrics"]
        lines.append(f"| {LABELS[k]} | {pct(c['per_seed_agent_means']['mean'])} | {pct(c['individual_scores']['min'])}–{pct(c['individual_scores']['max'])} | {pct(g['normal/all']['mean'])} | {pp(g['gap/all']['mean'])} |")
    lines += ["",f"完整地图I关闭通道正常为 {pct(groups['full_blocked']['metrics']['normal/all']['mean'])}。"
              "该基准每种子只有一次，不复制为三个独立运行。个人全信息行动与双主体通信任务不同，不能直接用二者成绩差作单一因果效应。", "",
              f"![个人与完整地图基准]({(out/'capability_controls.png').resolve()})", "",
              "左图点为每种子两名主体的平均，90%线为事先工程门槛；右图点为主体对种子，1/6线为单目标无通信期望。"
              f"个人两种模式共 {summary['controls']['trace_records_checked']:,} 条保存记录已独立重算；"
              "逐人的正常及清空计数、共享起点检查与来源哈希保存在summary.json。", "",
              "## 自然双目标、片段替换与未见组合拼接", "",
              "协议探针采用旧test每类前4张的16照片组合作校准、后4张的16组合作验证，两个照片集合不重叠。"
              "这不同于随机照片终点评估。每个方向只在训练24图选择食物/水对应的两符号位置；full条件则使用全部30图，"
              "没有真正的未见图评价。", "",
              "自然双目标要求同一消息在两个需求下都选对地点。严格片段替换在另一资源位置不变时替换一个符号，"
              "要求变化资源从原正确位置转到供体位置，另一资源保持正确。全30图分母包含自然失败；另存自然双地图双需求均正确子集，"
              "零分母保持未定义。留出拼接只取训练地图消息的成分，由实验者按地图信息选择供体，要求组合消息双目标均正确。", "",
              "每方向事先固定100次49完整消息双射，接收映射取逆，保留自然行为与完整消息频次；每次重做同样校准再评替换和拼接。"
              "全部49码×2需求×720菜单先验证实际地点等价才折叠菜单；参考不是新增训练主体、总体置信区间或语法显著性检验。"
              "单目标1/6与1/36均不作为这些双目标指标的通用机会线。", "",
              "| 表示 | 训练自然双目标 | 留出自然双目标 | 严格片段 | 片段参考 | 片段差（百分点） | 拼接 | 拼接参考 | 拼接差（百分点） |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in KINDS:
        p=pg[k]["metrics"]
        lines.append(f"| {LABELS[k]} | "+" | ".join(pct(p[m]['mean']) for m in ("seen_both","unseen_both","fragment","fragment_null"))+f" | {pp(p['fragment_delta']['mean'])} | {pct(p['stitch']['mean'])} | {pct(p['stitch_null']['mean'])} | {pp(p['stitch_delta']['mean'])} |")
    lines += ["",f"![协议比较]({(out/'binding_protocol.png').resolve()})","",
              "主协议图点为4种子的跨三划分、两方向均值。人工拼接可解码而自然双目标弱时，只能说明两类任务之间有差距；"
              "视觉编码、消息产出、优化难度或局部处理方式都可能相关，本轮没有分别定位这些中介。", "",
              "## 逐运行记录", "", "| 种子 | 条件 | 训练图正常 | 留出图正常 | A→B全图 | B→A全图 |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for r in summary["runs"]:
        n=r["scores"]["normal"]
        lines.append(f"| {r['seed']} | {r['condition']} | {pct(n['map_subsets']['train_maps']['mean_reward'])} | {pct(n['map_subsets']['heldout_maps']['mean_reward'])} | {pct(n['direction_means'][0]['mean_reward'])} | {pct(n['direction_means'][1]['mean_reward'])} |")
    lines += ["", "## 独立核查与结论边界", "",
              f"独立NumPy汇总逐例重算 {summary['trace_steps_independently_checked']:,} 条五模式终点记录，"
              "检查地图、菜单、消息传递及条件频次、奖励、零库存/历史、照片类别与数据分割、平衡格数及世界哈希。"
              "可训练初始化和接收者初始化分别核对，不错误要求条件特定buffer相同；I/P/Q矩阵形状、置换性和正交误差单列。"
              "相同种子划分的训练批次逐一匹配，保存每种子的差值。", "",
              f"协议来源 [protocol_analysis.json]({summary['protocol']['source']}) 的完整范围、终点参数哈希、校准地图、"
              f"{summary['protocol']['integer_ratios_checked']:,} 项整数比例、49码双射及参考均值已核对。"
              "汇总不导入神经策略或训练器；神经策略干预与初始化张量审计由独立脚本完成，没有重新训练。", "",
              "本轮检验非语言输入与局部处理偏好的提供方式。四探索种子、旧照片和同构坐标划分限制了外推；"
              "相同参数量和原始信息量也不能排除有限预算优化、非线性可读性及主体差异。"
              "结果不能独立证明DINO的贡献、语言诞生的一般条件或人类社会分工机制。", "",
              "后续若出现稳定表示差异，应先以新种子和新照片确认，并通过预先规定的能力检查区分视觉—行动学习与社会编码。"
              "不依据当前成绩换种子、追加某组预算或只展示最佳方向。此处仅提出后续解释与确认需求，本轮不新增实验。", "",
              "复现汇总：`.venv/bin/python redesign_v0.7/analyze_binding.py`。"
              f"[summary.json]({(out/'summary.json').resolve()}) 保存完整机器记录，"
              f"[per_run_metrics.csv]({(out/'per_run_metrics.csv').resolve()}) 保存全部模式和分组结果。", ""]
    if summary.get("execution_audit"):
        a=summary["execution_audit"]
        lines += [f"独立执行审计 [audit_execution.json]({a['source']})：{a['completed_runs']}/{a['planned_runs']} 个运行，"
                  f"{a['checks_count']:,} 项检查通过，重建 {a['train_updates_verified']:,} 次训练更新输入。", ""]
    (out/"视觉地点局部性实验报告.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"results/binding_001")
    p.add_argument("--output",type=Path)
    p.add_argument("--seeds",type=int,nargs="+",default=SEEDS)
    p.add_argument("--conditions",nargs="+",choices=CONDITIONS,default=CONDITIONS)
    p.add_argument("--check-only",action="store_true")
    args=p.parse_args();source=args.input.resolve();out=(args.output or source/"analysis").resolve()
    photos=read_json(ROOT.parent/"redesign_v0.4/data/manifest.json")["images"]
    design=split_audit(6,SPLITS)
    runs=[];missing=[];references={}
    for seed in args.seeds:
        for name in args.conditions:
            path=source/f"s{seed}_{name}"
            if not (path/"result.json").is_file():missing.append(path.name);continue
            r,a=read_run(path,photos,design)
            if seed in references:compare_world(references[seed],a,f"seed{seed}: across conditions")
            else:references[seed]=a
            runs.append(r);print(f"AUDITED {path.name}: normal={r['scores']['normal']['mean_reward']:.6f}",flush=True)
    summary={"status":"complete" if not missing else "incomplete", "generated_utc":datetime.now(timezone.utc).isoformat(),
             "input":str(source),"expected_seeds":args.seeds,"expected_conditions":args.conditions,
             "expected_runs":len(args.seeds)*len(args.conditions),"completed_runs":len(runs),"missing_runs":missing,
             "analysis_source_hashes":{p.name:sha(p) for p in (Path(__file__),ROOT/"analysis_core.py")},
             "design_audit":design,"trace_steps_independently_checked":sum(a["steps_checked"] for r in runs for a in r["trace_audit"].values()),
             "provenance_audit":provenance(runs),"runs":runs}
    if not args.check_only:
        require(not missing,"Requested run scope incomplete; use --check-only until all planned runs complete")
        require(args.conditions==CONDITIONS,"Final report requires all planned condition families; use --check-only for smoke")
        require(args.seeds==SEEDS,"Final report requires all four fixed seeds; use --check-only for partial review")
        summary["protocol"]=protocol_summary(source,runs)
        require(summary["protocol"] is not None,"Final report waits for complete protocol analysis")
        summary["controls"]=control_summary(source,runs,photos)
        audit_path=source/"audit_execution.json"
        if audit_path.exists():
            audit=read_json(audit_path)
            require(audit["status"]=="passed" and audit["completed_runs"]==audit["planned_runs"]==52,"Execution audit not complete")
            summary["execution_audit"]={"source":str(audit_path),"source_sha256":sha(audit_path),
                "completed_runs":audit["completed_runs"],"planned_runs":audit["planned_runs"],
                "checks_count":sum(audit["checks"].values()),"train_updates_verified":audit["train_updates_verified"]}
        summary["groups"]=aggregate(runs);summary["paired_comparisons"]=paired(runs)
        out.mkdir(parents=True,exist_ok=True)
        summary["plots"]=plots(summary,out)
        (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        with (out/"per_run_metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.writer(f);w.writerow(["seed","condition","split","kind","mode","overall","seen_n","seen_correct","seen_mean","unseen_n","unseen_correct","unseen_mean","A_to_B","B_to_A"])
            for r in runs:
                for mode in MODES:
                    score=r["scores"][mode];x=score["map_subsets"]["train_maps"];y=score["map_subsets"]["heldout_maps"]
                    w.writerow([r["seed"],r["condition"],r["split"],r["kind"],mode,score["mean_reward"],x["n"],x["reward_sum"],x["mean_reward"],y["n"],y["reward_sum"],y["mean_reward"],*[d["mean_reward"] for d in score["direction_means"]]])
        report(summary,out)
    print(json.dumps({"status":summary["status"],"completed_runs":len(runs),"expected_runs":summary["expected_runs"],
                      "trace_steps_checked":summary["trace_steps_independently_checked"],"outputs_written":not args.check_only,"output":str(out)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
