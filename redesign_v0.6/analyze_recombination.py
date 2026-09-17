"""Independently audit saved v0.6 trajectories and build the Markdown report.

Run from the workspace root:
  .venv/bin/python redesign_v0.6/analyze_recombination.py
  .venv/bin/python redesign_v0.6/analyze_recombination.py --input PATH --seeds 25101 --conditions split1_course --check-only

No environment, training, policy or Torch imports are used. Protocol evaluations
are produced separately; this script integrates their saved results only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis_core import (audit_static_arrays, describe, equal, near, require,
                           split_audit, static_metrics)

ROOT = Path(__file__).resolve().parent
SEEDS = [25101, 25102, 25103, 25104]
KINDS = ("course", "mixed", "direct", "atomic_direct")
CONDITIONS = [f"split{s}_{k}" for s in (1, 2, 3) for k in KINDS] + ["full_direct", "full_blocked"]
MODES = ("normal", "shuffle", "blank", "stochastic", "erase_memory")
MATCHINGS = {1: ((0, 1), (2, 3), (4, 5)), 2: ((0, 2), (1, 4), (3, 5)), 3: ((0, 3), (1, 5), (2, 4))}
SPLITS = {s: [p for edge in edges for p in (edge, edge[::-1])] for s, edges in MATCHINGS.items()}
LABELS = {"course": "两→四→六地点课程", "mixed": "相同批次混排", "direct": "直接六地点",
          "atomic_direct": "49选1直接六地点", "full_direct": "全30图通信", "full_blocked": "全30图关闭通道"}


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
            "final_file_sha256": sha(path / "final.pt"), "result_sha256": sha(path / "result.json"),
            "training_sha256": sha(path / "training.jsonl"), "training_record_count": len(training),
            "batches_by_identity": {str(r["batch_identity"]): {k: r[k] for k in ("active_sites", "allowed_sites", "map_pool", "world_sha256")} for r in training}}, baseline


def provenance(runs):
    records = []
    index = {(r["seed"], r["condition"]): r for r in runs}
    for seed in sorted({r["seed"] for r in runs}):
        for length in (1, 2):
            family = [r for r in runs if r["seed"] == seed and r["config"]["plan"]["length"] == length]
            if len(family) > 1:
                require(len({r["initial_sha256"] for r in family}) == 1, f"Seed {seed}: unmatched same-shape initial state")
                records.append({"seed": seed, "length": length, "same_initial_state": True, "runs": len(family)})
        for split in (1, 2, 3):
            course, mixed = index.get((seed, f"split{split}_course")), index.get((seed, f"split{split}_mixed"))
            if course and mixed:
                require(course["batches_by_identity"] == mixed["batches_by_identity"], f"{seed}/{split}: course/mixed batch multiset")
                records.append({"seed": seed, "split": split, "course_mixed_world_multiset_equal": True})
    return records


def run_value(run, metric):
    mode, group = metric
    if mode == "gap":
        return run["normal_minus_shuffle"][group]
    return run["scores"][mode]["mean_reward"] if group == "all" else run["scores"][mode]["map_subsets"][group]["mean_reward"]


def aggregate(runs):
    metrics = [(mode, group) for mode in MODES for group in ("all", "train_maps", "heldout_maps")] + [("gap", g) for g in ("all", "train_maps", "heldout_maps")]
    output = {}
    for kind in (*KINDS, "full_direct", "full_blocked"):
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


def paired(runs):
    index = {(r["seed"], r["split"], r["kind"]): r for r in runs}
    output = []
    seeds = sorted({r["seed"] for r in runs})
    for left, right in (("course", "mixed"), ("course", "direct"), ("direct", "atomic_direct")):
        for metric in (("normal", "heldout_maps"), ("normal", "train_maps"), ("gap", "heldout_maps")):
            details = []
            for seed in seeds:
                for split in (1, 2, 3):
                    if (seed, split, left) not in index or (seed, split, right) not in index:
                        continue
                    a, b = index[seed, split, left], index[seed, split, right]
                    require(a["config"]["updates"] == b["config"]["updates"] and a["config"]["batch"] == b["config"]["batch"], "Paired budget mismatch")
                    details.append({"seed": seed, "split": split, "difference": run_value(a, metric) - run_value(b, metric)})
            if details:
                paired_seeds = sorted({r["seed"] for r in details})
                values = [float(np.mean([r["difference"] for r in details if r["seed"] == s])) for s in paired_seeds]
                output.append({"left": left, "right": right, "metric": "/".join(metric), "seeds": paired_seeds,
                               "seed_means_over_splits": describe(values), "split_seed_differences": details})
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
    for kind in (*KINDS,"full_direct","full_blocked"):
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


def plots(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["PingFang SC", "DejaVu Sans"], "font.size": 11,
                         "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#3B83BD", "#D59331", "#9171B1", "#278D80"]
    files = []
    def save(fig, stem):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{stem}.{ext}", dpi=240)
        plt.close(fig); files.append(stem + ".png")
    groups = summary["groups"]
    names = [k for k in KINDS if k in groups]
    if names:
        fig = plt.figure(figsize=(11, 7.4), constrained_layout=True)
        grid = fig.add_gridspec(2, 2)
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
        short_labels = {"course":"课程\n2→4→6", "mixed":"混排\n同批次", "direct":"直接\n六地点", "atomic_direct":"原子\n49选1"}
        for ax, metric, title in zip(axes, ("normal/train_maps", "normal/heldout_maps", "gap/heldout_maps"),
                                     ("训练二十四图准确率（%）", "留出六图准确率（%）", "留出六图正常−打乱（百分点）")):
            for i, name in enumerate(names):
                values = groups[name]["metrics"][metric]["values"]
                ax.scatter(i + np.linspace(-.12, .12, len(values)), np.asarray(values) * 100,
                           c=colors[:len(values)], s=38, zorder=3)
                ax.plot([i-.22,i+.22], [100*np.mean(values)]*2, color="#202A32", lw=2.4)
            ax.set_ylabel(title); ax.grid(axis="y", alpha=.16)
            if metric.startswith("normal"):
                ax.set_ylim(0, 104); ax.axhline(100/6, color="#8A9298", ls=":", lw=1)
            else:
                ax.axhline(0, color="#8A9298", lw=1)
            ax.set_xticks(range(len(names)), [short_labels[k] for k in names])
        fig.suptitle("主比较：先按种子平均三划分；点为4种子，短线为均值", fontsize=13)
        save(fig, "primary_comparison")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, constrained_layout=True)
        for split, ax in zip((1, 2, 3), axes):
            for i, name in enumerate(names):
                values = groups[name]["by_split"][str(split)]["metrics"]["normal/heldout_maps"]["values"]
                ax.scatter(i+np.linspace(-.12,.12,len(values)),np.asarray(values)*100,c=colors[:len(values)],s=32)
                ax.plot([i-.22,i+.22],[100*np.mean(values)]*2,color="#202A32",lw=2)
            ax.set(title=f"坐标划分 {split}",ylim=(0,104));ax.grid(axis="y",alpha=.16)
            ax.set_xticks(range(len(names)), ["课程", "混排", "直接", "原子"][:len(names)])
            ax.axhline(100/6,color="#8A9298",ls=":",lw=1)
        axes[0].set_ylabel("留出六图准确率（%）")
        fig.suptitle("三种同构坐标留出：各划分保留4种子", fontsize=13)
        save(fig, "split_comparison")
        fig, axes = plt.subplots(2, 2, figsize=(11,7.4),sharey=True,constrained_layout=True)
        for name, ax in zip(names, axes.flat):
            local = [r for r in summary["runs"] if r["kind"] == name]
            for mode, color, style in (("normal","#117D7C","-"),("shuffle","#B87536","--")):
                seed_curves=[];x=None
                for seed in sorted({r["seed"] for r in local}):
                    family=[r for r in local if r["seed"]==seed]
                    points=[[row["update"] for row in r["curve"]] for r in family]
                    require(all(p==points[0] for p in points), "Process checkpoints differ across splits")
                    x=points[0]
                    y=np.mean([[100*row["scores"][mode]["map_groups"]["unseen"]["mean_reward"] for row in r["curve"]] for r in family],axis=0)
                    seed_curves.append(y);ax.plot(x,y,color=color,ls=style,lw=.8,alpha=.3)
                ax.plot(x,np.mean(seed_curves,axis=0),color=color,ls=style,lw=2,marker="o",ms=3,label="正常" if mode=="normal" else "打乱")
            ax.set(title=LABELS[name],xlabel="训练更新",ylim=(0,104));ax.grid(axis="y",alpha=.16)
            ax.axhline(100/6,color="#8A9298",ls=":",lw=1)
        for ax in axes[:,0]:ax.set_ylabel("留出六图准确率（%）")
        axes[0,0].legend(frameon=False)
        fig.suptitle("留出地图过程：细线为种子跨划分均值，粗线为4种子均值",fontsize=13)
        save(fig,"heldout_process")
    if summary.get("protocol"):
        pg=summary["protocol"]["groups"]
        fig=plt.figure(figsize=(11,7.8),constrained_layout=True);grid=fig.add_gridspec(2,2)
        ax=fig.add_subplot(grid[0,0])
        for i,kind in enumerate(KINDS):
            for j,(metric,color,text) in enumerate((("seen_both","#397FB1","训练24图"),("unseen_both","#D68B36","留出6图"))):
                values=pg[kind]["metrics"][metric]["values"];x=i+(-.16 if j==0 else .16)
                ax.scatter(x+np.linspace(-.045,.045,len(values)),np.asarray(values)*100,color=color,s=20,alpha=.75,label=text if i==0 else None)
                ax.plot([x-.10,x+.10],[100*np.mean(values)]*2,color=color,lw=2.5)
        ax.set(title="自然消息：同一消息双目标均正确",ylabel="成功率（%）",ylim=(0,104))
        ax.set_xticks(range(4),["课程","混排","直接","原子"]);ax.legend(frameon=False,fontsize=9);ax.grid(axis="y",alpha=.16)
        for ax,metric,title in ((fig.add_subplot(grid[0,1]),"fragment","严格局部替换：完整30图探针"),
                                 (fig.add_subplot(grid[1,:]),"stitch","训练消息成分拼接至留出6图：双目标均正确")):
            for i,kind in enumerate(KINDS[:3]):
                for j,(key,color,text) in enumerate(((metric,"#397FB1","观察值"),(metric+"_null","#D68B36","整体重编码参考均值"))):
                    values=pg[kind]["metrics"][key]["values"];x=i+(-.14 if j==0 else .14)
                    ax.scatter(x+np.linspace(-.04,.04,len(values)),np.asarray(values)*100,color=color,s=25,alpha=.75,label=text if i==0 else None)
                    ax.plot([x-.10,x+.10],[100*np.mean(values)]*2,color=color,lw=2.5)
            ax.set(title=title,ylabel="成功率（%）",ylim=(0,104));ax.set_xticks(range(3),["课程","混排","直接"])
            ax.grid(axis="y",alpha=.16);ax.legend(frameon=False,fontsize=9)
        fig.suptitle("协议结构：照片平衡枚举；点为种子跨划分和方向均值",fontsize=13)
        save(fig,"protocol_comparison")
    return files


def pct(x):
    return "不适用" if x is None else f"{100*x:.2f}%"


def pp(x):
    return "不适用" if x is None else f"{100*x:+.2f}"


def report(summary, out):
    groups=summary["groups"]
    lines=["# v0.6 六地点组合留出实验报告", "",
           f"生成时间：{summary['generated_utc']}。完成并审计 {summary['completed_runs']}/{summary['expected_runs']} 个预期运行。", "",
           "## 问题与实验范围", "",
           "本轮检验共同消息能否支持训练未见的资源地点组合，以及课程顺序与消息接口如何影响这种迁移。"
           "环境为六个公共地点、两种不共址资源，共30张有序地图；每轮只有侦察和采集，角色在两个方向分别评价。"
           "没有跨轮库存、资源补充、自发分工或新人学习。", "",
           "冻结官方 DINOv2 ViT-L/14 及旧44张训练/16张开发留出照片。每个种子重新进行个人资源后果准备，"
           "再建立新的社会通信接口；不继承 v0.5 的社会权重。侦察者接收按公共地点排列的照片特征和存在掩码，"
           "看不到采集者的当前需求。采集者获供二维自身需求，按该需求选择两组六地点行动分支。"
           "这些是预先提供的非语言能力，消息位置没有预设资源词义；不能把结果视为从零获得全部世界知识或独立证明 DINO 的贡献。", "",
           "序列通道每次发送两个七选一编号，共49种完整消息；原子通道每次发送一个49选1编号。"
           "两者均足以查表表达30张地图。名义容量一致不等于架构、参数数量或优化难度一致，"
           "消息成功率本身不证明组合结构。", "",
           "主训练前另用开发种子99102给予固定随机地图代码，仅通过采集奖励学习接收行动。"
           "两个序列接收者为90.00%和90.00%，原子接收者为95.00%和91.67%，达到事先规定的≥90%门槛，"
           "没有达到全满分。初次序列判定因float32把8640/9600表示为0.899999976而误报失败，"
           "随后根据保存轨迹的整数计数修正；原判定及执行源码保留，没有重训或改变门槛。"
           "这只是有限预算下新接口可学习的工程诊断，不能把未达100%的部分忽略，也不能称为自然通信或视觉能力的独立贡献。"
           "开发代码和权重均未进入主社会训练；旧动作头的六地点准备验收另行保存。", "",
           "## 三种留出划分和训练安排", "",
           "每种划分留出三条无向完美匹配边的两个方向，共6图，训练其余24图。"
           "训练与留出均保持每种资源在各地点的边际平衡；三个留出集互不相交，覆盖18/30地图。"
           "它们是同构的坐标划分重复，不代表三种不同任务拓扑。", "",
           "| 划分 | 双向留出的无向地点对 | 训练图数 | 留出图数 |", "| --- | --- | ---: | ---: |"]
    for split,pairs in MATCHINGS.items():lines.append(f"| {split} | "+"、".join(f"({a},{b})" for a,b in pairs)+" | 24 | 6 |")
    lines += ["", "4个训练种子分别进行三划分×四条件，共48个运行；每种子另做全30图通信与全30图关闭通道，"
              "共8个基准运行，总56个。所有条件从该种子的个人准备出发，同形状接口初始化一致。"
              "完整地图基准没有三份独立副本，不能把它复制后扩大重复数。", "",
              "每个运行2400次更新、每次512个单步段；课程依次600次两地点、600次四地点、1200次六地点。"
              "允许地点集合公开限定合法动作；按集合内合法训练地图数加权选择集合，再均匀选地图，"
              "保证各难度的训练地图边际均匀。双向匹配留出避免两地点场景通过训练支持仅剩唯一资源配置。"
              "混排条件打乱前2100个同一批次身份，末300次与课程保持同序并关闭熵正则；完整训练世界/照片/目标多重集匹配。"
              "直接条件始终六地点，原子条件也是直接六地点；与课程相比，难度暴露不同。", "",
              f"实际核对 {sum(r['training_record_count'] for r in summary['runs']):,} 次社会训练更新。"
              "主结果先在同种子内跨三划分平均，再对四种子等权；各划分和各种子均保留。"
              "不把划分、两个方向或评估案例当作独立训练重复，不使用小样本 p 值声称创新。", "",
              "## 评估口径", "",
              "终点每运行、每模式9600个单步案例，30地图×2需求×2方向每格80次照片抽样；"
              "过程1200案例，每格10次。地图、需求和方向严格平衡，照片从旧开发留出集中随机抽取，"
              "不是照片组合完全枚举，也不是新的确认照片。", "",
              "正常使用最大概率消息与行动。打乱只在同方向、同需求内重排完整消息，保持条件频次；"
              "当前任务没有后续动态状态，因此正常减打乱是固定场景集合上的单步回报差。"
              "blank强制零消息，erase_memory清空侦察现场表示，策略采样按概率采样，两者与训练关闭通道不同。"
              "各均衡地图集合的静态无通信理想期望为1/6≈16.67%；完整通信码表可达100%。", "",
              "1/6只用于单目标回报，不能当作同消息双目标或拼接的通用机会水平。"
              "固定两个地点恰构成某张留出匹配地图时，留出六图双目标成功可为1/6，否则为0；"
              "完整30图上的固定合法地点对为1/30。协议图只使用各自整体重编码参照，不画统一机会线。", "",
              "## 主要发现", ""]
    course,mixed,direct,atomic=[groups[k]["metrics"] for k in KINDS]
    lines += [f"留出六图的四种子平均准确率：课程 {pct(course['normal/heldout_maps']['mean'])}、"
              f"混排 {pct(mixed['normal/heldout_maps']['mean'])}、直接 {pct(direct['normal/heldout_maps']['mean'])}、"
              f"原子直接 {pct(atomic['normal/heldout_maps']['mean'])}；相应训练二十四图为 "
              f"{pct(course['normal/train_maps']['mean'])}、{pct(mixed['normal/train_maps']['mean'])}、"
              f"{pct(direct['normal/train_maps']['mean'])}、{pct(atomic['normal/train_maps']['mean'])}。"
              "比较应同时看已训练地图的掌握程度与未训练组合的下降，不能用全图平均掩盖留出失败。", "",
              f"课程减混排的留出差为 {pp(course['normal/heldout_maps']['mean']-mixed['normal/heldout_maps']['mean'])} 个百分点，"
              f"直接减课程为 {pp(direct['normal/heldout_maps']['mean']-course['normal/heldout_maps']['mean'])} 个百分点，"
              f"直接减混排为 {pp(direct['normal/heldout_maps']['mean']-mixed['normal/heldout_maps']['mean'])} 个百分点。"
              "直接条件的留出准确率在四个种子的跨划分均值上均高于课程与混排；"
              "混排相对课程在一个种子下降、其余三个上升，不能只报告平均改善。"
              "本轮没有支持逐级课程优于直接复杂任务；观察结果只覆盖这套固定预算和训练安排。", "",
              f"直接序列条件的留出准确率减训练图准确率为 "
              f"{pp(direct['normal/heldout_maps']['mean']-direct['normal/train_maps']['mean'])} 个百分点；"
              f"完整三十图训练的通信基准为 {pct(groups['full_direct']['metrics']['normal/all']['mean'])}。"
              "完整地图基准的训练支持不同，不是留出泛化的直接因果对照；它帮助判断当前接口在完整训练支持下的学习水平。", "",
              ]
    structural=summary["protocol"]["groups"]["direct"]["metrics"]
    lines += [f"结构方面，直接条件自然产生的留出地图消息，双目标均正确只有 {pct(structural['unseen_both']['mean'])}；"
              f"从训练消息选择供体并人工拼接后为 {pct(structural['stitch']['mean'])}，整体重编码参考为 {pct(structural['stitch_null']['mean'])}。"
              f"全三十图严格局部替换为 {pct(structural['fragment']['mean'])}，相应参考为 {pct(structural['fragment_null']['mean'])}。"
              "这些结果支持有限的片段解码与复用，尚未形成可靠的自主新组合产出。"
              "人工拼接使用实验者的地图信息选择供体，不等于主体自己成功表达了新组合。"
              "消息产出或视觉到消息的映射可能构成瓶颈，但本轮没有因果定位，仍是候选解释。", "",
              "## 跨划分主结果", "",
              "| 条件 | 训练24图 | 留出6图 | 留出种子范围 | 留出正常−打乱（百分点） | 留出四种子均值 |",
              "| --- | ---: | ---: | --- | ---: | --- |"]
    for name in KINDS:
        if name not in groups:continue
        g=groups[name]["metrics"];unseen=g["normal/heldout_maps"]
        lines.append(f"| {LABELS[name]} | {pct(g['normal/train_maps']['mean'])} | {pct(unseen['mean'])} | {pct(unseen['min'])}–{pct(unseen['max'])} | "
                     f"{pp(g['gap/heldout_maps']['mean'])} | "+"、".join(pct(x) for x in unseen['values'])+" |")
    for stem,caption in (("primary_comparison","点为每种子跨三划分均值，短黑线为四种子均值。虚线为无通信期望。"),
                         ("split_comparison","每个坐标划分单独保留四种子，检查平均值是否掩盖划分差异。"),
                         ("heldout_process","仅连接实际记录检查点；过程1200例与终点9600例口径不同，不据此插值首次形成时间。")):
        if stem+".png" in summary["plots"]:lines += ["",f"![{stem}]({(out/(stem+'.png')).resolve()})","",caption]
    lines += ["", "## 同种子、同划分的条件差", "",
              "先在每个种子×划分内相减，再对每种子的三个划分取均值。课程减混排匹配训练批次多重集；"
              "课程减直接同时改变难度暴露；序列直接减原子直接同时改变架构。", "",
              "| 比较 | 指标 | 均值差（百分点） | 四种子差值（百分点） |", "| --- | --- | ---: | --- |"]
    metric_labels={"normal/heldout_maps":"留出6图准确率","normal/train_maps":"训练24图准确率","gap/heldout_maps":"留出正常−打乱落差"}
    for row in summary["paired_comparisons"]:
        stat=row["seed_means_over_splits"]
        lines.append(f"| {LABELS[row['left']]} 减 {LABELS[row['right']]} | {metric_labels[row['metric']]} | {pp(stat['mean'])} | "+"、".join(pp(x) for x in stat['values'])+" |")
    lines += ["", "## 完整地图基准", "", "| 条件 | 正常 | 打乱 | 正常−打乱（百分点） | 四种子正常 |", "| --- | ---: | ---: | ---: | --- |"]
    for name in ("full_direct","full_blocked"):
        if name not in groups:continue
        g=groups[name]["metrics"]
        lines.append(f"| {LABELS[name]} | {pct(g['normal/all']['mean'])} | {pct(g['shuffle/all']['mean'])} | {pp(g['gap/all']['mean'])} | "+"、".join(pct(x) for x in g['normal/all']['values'])+" |")
    if "full_map_baselines.png" in summary["plots"]:lines += ["",f"![完整地图基准]({(out/'full_map_baselines.png').resolve()})"]
    protocol=summary.get("protocol")
    if protocol:
        pg=protocol["groups"]
        lines += ["", "## 自然消息复用与组合干预", "",
                  "这一部分使用照片平衡枚举，区别于前文每格抽样80次的随机照片评估。"
                  "校准采用旧test中每类前4张照片的16个组合，验证采用每类后4张照片的另16个组合，两组照片不重叠。"
                  "每方向只在训练24图选择食物/水与两消息位置的对应，再固定该选择评价验证照片。"
                  "full基准可在全部30图校准，因而其任意地图子集成绩都不是未见组合泛化。", "",
                  "自然双目标成功要求同一条消息在食物需求和水需求下均选对地点。"
                  "局部替换保持另一资源位置不变，替换一个符号，严格要求变化资源从原正确地点转到供体地点、"
                  "另一资源的正确选择保持；全30图枚举分母保留自然消息失败，不只筛选原本正确情境。"
                  "JSON另存自然双地图双需求都正确子集，零分母保持未定义。"
                  "拼接则只从训练地图消息取两个成分，组成未训练地图的消息，要求两个需求均正确。", "",
                  "主训练前固定每方向100次49条完整消息的双射重编码，并逆向改写接收映射，保持自然行为与完整消息频次。"
                  "每次重新在训练图校准位置分配，再评估严格替换和留出拼接。"
                  "先验证全部49码×2需求×720菜单的实际地点等价，才折叠菜单重复；真实整数分母保存在来源JSON。"
                  "参考均值描述行为相同的整体码在不同位分解下的分数，不是新训练种子、置信区间或语法显著性检验。", "",
                  "| 条件 | 训练图自然双目标 | 留出图自然双目标 | 严格替换观察 | 替换重编码参考 | 观察减参考（百分点） |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for kind in KINDS:
            g=pg[kind]["metrics"]
            lines.append(f"| {LABELS[kind]} | {pct(g['seen_both']['mean'])} | {pct(g['unseen_both']['mean'])} | "
                         f"{pct(g.get('fragment',{}).get('mean'))} | {pct(g.get('fragment_null',{}).get('mean'))} | {pp(g.get('fragment_delta',{}).get('mean'))} |")
        lines += ["", "| 序列条件 | 留出拼接双目标 | 拼接重编码参考 | 观察减参考（百分点） | 四种子观察减参考（百分点） |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for kind in KINDS[:3]:
            g=pg[kind]["metrics"]
            lines.append(f"| {LABELS[kind]} | {pct(g['stitch']['mean'])} | {pct(g['stitch_null']['mean'])} | {pp(g['stitch_delta']['mean'])} | "+"、".join(pp(x) for x in g['stitch_delta']['values'])+" |")
        lines += ["", f"![协议结构比较]({(out/'protocol_comparison.png').resolve()})", "",
                  "各点先按种子平均三划分和两个方向。原子条件只有一个符号，不计算不存在的片段替换/拼接；"
                  "其自然消息的留出双目标表现仍保留。局部替换和未见组合拼接是不同指标，不能相互替代。", "",
                  f"协议来源：[protocol_analysis.json]({protocol['source']})。本汇总核对全部终点参数文件哈希、"
                  f"校准地图、{protocol['integer_ratios_checked']:,} 项整数计数比例、49码双射及100次重编码参考均值；"
                  "神经策略探针由独立机制脚本运行。", ""]
    lines += ["", "## 逐运行结果与方向", "",
              "| 种子 | 条件 | 正常全图 | 训练图 | 留出图 | 留出打乱 | A→B全图 | B→A全图 |",
              "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["runs"]:
        n=r["scores"]["normal"];sh=r["scores"]["shuffle"]
        lines.append(f"| {r['seed']} | {r['condition']} | {pct(n['mean_reward'])} | {pct(n['map_subsets']['train_maps']['mean_reward'])} | {pct(n['map_subsets']['heldout_maps']['mean_reward'])} | "
                     f"{pct(sh['map_subsets']['heldout_maps']['mean_reward'])} | {pct(n['direction_means'][0]['mean_reward'])} | {pct(n['direction_means'][1]['mean_reward'])} |")
    lines += ["", "## 审计与证据边界", "",
              f"独立 NumPy 审计重算 {summary['trace_steps_independently_checked']:,} 条终点评估记录，覆盖每个运行五种模式。"
              "逐例核对位置、菜单路由、采集、需求满足、消息投递与条件频次、零库存/零历史、照片类别和数据分割；"
              "核对每个地图×目标×方向评估单元的平衡计数，以及世界哈希、记录均值和seen/unseen整数计数。"
              "不导入训练器或神经策略，没有重新训练，也没有借助模型给出的奖励替代独立重算。", "",
              "参数来源指纹、冻结视觉投影验收、完整检查点、训练日志连续性、课程/混排按批次身份的世界多重集均检查。"
              "检查点的过程均值来自保存记录，其整数计数和样本数另行核对；最终成绩才有逐例NPZ重算。", "",
              "协议的双目标复用、严格片段替换、留出拼接与整体重编码基准由独立机制分析提供，"
              "已与保存终点参数来源核对；不能用功能奖励替代结构干预。", "",
              "四探索种子、旧照片、三种同构留出不足以确定语言诞生的一般条件。"
              "高留出回报不自动等于组合语法；低回报也要结合训练图能力与完整地图基准，区分尚未学会任务和组合迁移失败。", "",
              "## 下一轮优先的能力对照", "",
              "建议先固定六地点任务、消息容量和直接训练预算，再对照已提供的非语言接口："
              "例如显式对象—地点绑定与相同信息量的其他视觉整合方式，或当前需求分支与仍接收自身需求的共享行动分支。"
              "这些比较需要先用相同随机地图代码检验两组都能执行任务，再观察通信形成、自然双目标和留出拼接；"
              "否则接口不能行动造成的低分会被误归因为不能形成符号。架构改变与能力改变应分开解释。", "",
              "另一优先项是增加独立训练种子和未参与开发的新照片，用已固定的评价确认最稳定的结果。"
              "当前不宜同时加入工具链、长期记忆和新人，以免无法识别哪项非语言能力改变了结果。"
              "以上是后续建议，本轮没有执行这些新增对照，也没有因当前低分延长训练。", "",
              "复现汇总：`.venv/bin/python redesign_v0.6/analyze_recombination.py`。"
              f"完整机器记录见 [summary.json]({(out/'summary.json').resolve()})，"
              f"逐模式数据见 [per_run_metrics.csv]({(out/'per_run_metrics.csv').resolve()})。", ""]
    if summary.get("execution_audit"):
        audit=summary["execution_audit"]
        lines += [f"另见独立执行审计 [audit_execution.json]({audit['source']})："
                  f"{audit['completed_runs']}/{audit['planned_runs']} 个运行，{audit['checks_count']:,} 项核查通过，"
                  f"独立重建 {audit['train_updates_verified']:,} 次训练更新的世界输入，并检查新主体准备、"
                  "初始化、旧准备指纹重合和诊断整数判分。此审计独立于汇总脚本，未重新训练。", ""]
    first, last = lines.index("## 主要发现"), lines.index("## 跨划分主结果")
    findings = lines[first:last]
    lines = lines[:first] + lines[last:]
    lines[4:4] = findings
    (out/"六地点组合留出实验报告.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,default=ROOT/"results/recombination_001")
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
        diagnostic_path=ROOT/"results/receiver_control_99102/result.json"
        diagnostic=read_json(diagnostic_path)
        require(diagnostic["complete"] and diagnostic["passed"],"Receiver diagnostic must be completed")
        for condition in diagnostic["conditions"].values():
            for value,count in zip(condition["per_agent"],condition["counts"]):
                near(value,count["correct"]/count["n"],"Receiver diagnostic integer mean")
        summary["receiver_diagnostic"]={"source":str(diagnostic_path),"source_sha256":sha(diagnostic_path),"result":diagnostic}
        audit_path=source/"audit_execution.json"
        if audit_path.exists():
            audit=read_json(audit_path)
            require(audit["status"]=="passed" and audit["completed_runs"]==audit["planned_runs"]==56,"Execution audit not complete")
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
