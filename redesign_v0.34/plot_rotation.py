"""Render v0.34 partner-topology design and transfer outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "redesign_v0.9/.analysis_deps"))
import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle


CONDITIONS = ("single_full", "dual_same_full", "dual_complementary")
LABELS = {"single_full": "单发送者", "dual_same_full": "双发送者·冗余", "dual_complementary": "双发送者·互补"}
COLORS = {"single_full": "#527A9E", "dual_same_full": "#C47A3A", "dual_complementary": "#2A8C82"}
SCHEDULES = ("A", "B", "C")
SCHEDULE_LABELS = {"A": "A·native", "B": "B·训练见过", "C": "C·训练未见"}


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def setup():
    try:
        font = font_manager.findfont("PingFang SC", fallback_to_default=False)
    except Exception:
        font = font_manager.findfont("DejaVu Sans")
    plt.rcParams.update({"font.family": ["PingFang SC", "DejaVu Sans"], "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False})
    return font


def save(fig, directory: Path, stem: str):
    result = {}
    for ext in ("png", "pdf"):
        path = directory / f"{stem}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.14)
        result[path.name] = sha(path)
    plt.close(fig)
    return result


def design_figure(directory: Path):
    design = read(ROOT / "support_design.json")
    schedules = {"A": ((0,1,2),(1,2,3),(2,3,0),(3,0,1)), "B": ((0,2,3),(1,3,0),(2,0,1),(3,1,2)), "C": ((0,3,1),(1,0,2),(2,1,3),(3,2,0))}
    fig = plt.figure(figsize=(13.2, 7.0))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.15, 1.2), wspace=.30, left=.045, right=.98, top=.79, bottom=.23)
    ax = fig.add_subplot(gs[0, 0]); ax.set_title("A  训练期伙伴拓扑", loc="left", pad=9)
    positions = {0:(0,1.25), 1:(1.55,1.25), 2:(0,-.15), 3:(1.55,-.15)}
    for schedule, yy, color in (("A",1.85,"#527A9E"),("B",-.73,"#C47A3A")):
        for receiver, food, water in schedules[schedule]:
            x,y=positions[receiver]; xf,yf=positions[food]; xw,yw=positions[water]
            ax.add_patch(FancyArrowPatch((x,y),(xf,yf),arrowstyle="-|>",mutation_scale=9,linewidth=1.0,alpha=.38,color=color))
            ax.add_patch(FancyArrowPatch((x,y),(xw,yw),arrowstyle="-|>",mutation_scale=9,linewidth=1.0,alpha=.38,color=color))
        ax.text(.78,yy,"A：偶数步" if schedule=="A" else "B：奇数步",ha="center",va="center",color=color,weight="bold")
    for i,(x,y) in positions.items():
        ax.add_patch(plt.Circle((x,y),.18,facecolor="#4D5964",edgecolor="white",lw=1.3,zorder=4)); ax.text(x,y,str(i),ha="center",va="center",color="white",weight="bold",zorder=5)
    ax.text(.78,-1.25,"C 只在终点评估，训练中从未出现",ha="center",va="top",fontsize=9.3,color="#43505C")
    ax.set(xlim=(-.42,1.97),ylim=(-1.55,2.15)); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.set_title("B  条件保持不变", loc="left", pad=9)
    items=[("单 sender", "full → 2 tokens", "#527A9E"),("冗余双 sender", "full + full → 1+1", "#C47A3A"),("互补双 sender", "food-only + water-only → 1+1", "#2A8C82")]
    for i,(a,b,color) in enumerate(items):
        yy=1.65-i*.92; ax.add_patch(Rectangle((.05,yy-.22),2.6,.44,facecolor=color,edgecolor="white",lw=1.3)); ax.text(1.35,yy,a,ha="center",va="center",color="white",weight="bold"); ax.text(1.35,yy-.45,b,ha="center",va="top",fontsize=9.1,color="#43505C")
    ax.text(1.35,-1.22,"三种条件共享世界、输入、收益、\n模型初始化与 2400 更新预算",ha="center",va="top",fontsize=9.5,color="#43505C"); ax.set(xlim=(-.1,2.8),ylim=(-1.55,2.08)); ax.axis("off")
    ax = fig.add_subplot(gs[0, 2]); ax.set_title("C  训练图与留出图（p1）", loc="left", pad=9)
    target={tuple(x) for x in design["target12_pairs"]}; train={tuple(x) for x in design["training12_pairs"]}
    for food in range(6):
        for water in range(6):
            if food==water: face,label,color="#4A535B","×","white"
            elif (food,water) in target: face,label,color="#F0C56A","T","#5E3E00"
            elif (food,water) in train: face,label,color="#76A9C9","S","#12364A"
            else: face,label,color="#E7EAED","","#555"
            ax.add_patch(Rectangle((water-.5,food-.5),1,1,facecolor=face,edgecolor="white",lw=1.05));
            if label: ax.text(water,food,label,ha="center",va="center",color=color,weight="bold")
    ax.set(xlim=(-.5,5.5),ylim=(5.5,-.5),xticks=range(6),yticks=range(6),xlabel="水位置",ylabel="食物位置"); ax.set_aspect("equal"); ax.tick_params(length=0)
    ax.legend(handles=[Patch(facecolor="#F0C56A",label="T：12目标边"),Patch(facecolor="#76A9C9",label="S：12训练边"),Patch(facecolor="#E7EAED",label="留出布局"),Patch(facecolor="#4A535B",label="×：同址排除")],loc="upper center",bbox_to_anchor=(.5,-.14),ncol=2,frameon=False,fontsize=8.5)
    fig.suptitle("v0.34：伙伴轮换能否把配对特定协议变成可迁移协议？",fontsize=17,y=.98)
    fig.text(.5,.035,"互补条件在 A/B 之间轮换伙伴；C 只用于终点迁移检验。",ha="center",fontsize=10.5,color="#3F4B57")
    return save(fig,directory,"01_rotation_design"),{"schedules":{k:[list(x) for x in v] for k,v in schedules.items()},"target_pairs":sorted(map(list,target)),"training_pairs":sorted(map(list,train))}


def axis_percent(ax,title,ylabel="成功率（%）"):
    ax.set_title(title,loc="left",pad=8); ax.set_ylim(0,100); ax.set_ylabel(ylabel); ax.set_yticks(np.arange(0,101,20)); ax.grid(axis="y",color="#C9CED3",alpha=.58,lw=.7); ax.set_axisbelow(True)


def result_figure(directory: Path, analysis: dict, transfer: dict, topology: dict):
    agg=analysis["aggregate"]; seeds=analysis["seeds"]; rows={(r["seed"],r["condition"]):r for r in analysis["seed_rows"]}; times=[x["update"] for x in agg["single_full"]["curve"]]
    fig,axes=plt.subplots(2,3,figsize=(15.4,9.0)); fig.subplots_adjust(left=.06,right=.985,top=.82,bottom=.16,hspace=.48,wspace=.28); fig.suptitle("v0.34：伙伴轮换提高形式兼容性，但未解决未见拓扑迁移",fontsize=17,y=.985); fig.text(.5,.945,"细线为12个 seed×面板来源；粗线为来源均值。纵轴统一固定为0–100%。",ha="center",fontsize=10.5); fig.legend(handles=[Line2D([0],[0],color=COLORS[c],marker="o",lw=2.2,label=LABELS[c]) for c in CONDITIONS],loc="upper center",bbox_to_anchor=(.5,.915),ncol=3,frameon=False,fontsize=10)
    evidence={"times":times,"target_curve":{},"endpoint":{},"transfer":{},"topology":{},"replication":{},"agreement":{}}
    ax=axes[0,0]
    for c in CONDITIONS:
        source=np.asarray([[100*item["scores"]["target12"]["pooled"]["J"] for item in rows[(seed,c)]["curve"]] for seed in seeds]); mean=100*np.asarray([item["scores"]["target12"]["pooled"]["J"] for item in agg[c]["curve"]])
        for value in source: ax.plot(times,value,color=COLORS[c],alpha=.18,lw=.75)
        ax.plot(times,mean,color=COLORS[c],marker="o",ms=4,lw=2.2); evidence["target_curve"][c]={"source_percent":source.tolist(),"mean_percent":mean.tolist()}
    axis_percent(ax,"A  native A 的目标学习曲线","J（%）"); ax.set_xlabel("群体更新次数"); ax.set_xlim(0,2400); ax.set_xticks(times)
    ax=axes[0,1]; labels=["训练 J","目标 J","留出 J"]; x=np.arange(3); width=.24
    for j,c in enumerate(CONDITIONS):
        s=agg[c]["scores"]; values=[100*s["train12"]["pooled"]["J"],100*s["target12"]["pooled"]["J"],100*s["held18"]["pooled"]["J"]]; ax.bar(x+(j-1)*width,values,width=width,color=COLORS[c],alpha=.85,label=LABELS[c]); evidence["endpoint"][c]=values
    axis_percent(ax,"B  终点与留出表现"); ax.set_xticks(x,labels); ax.legend(frameon=False,fontsize=7.6,loc="upper left")
    ax=axes[0,2]; labels=[SCHEDULE_LABELS[s] for s in SCHEDULES]; x=np.arange(3); vals=[100*transfer["summary"][s]["J"]["mean"] for s in SCHEDULES]; colors=["#527A9E","#C47A3A","#8C6BB1"]; ax.bar(x,vals,color=colors,alpha=.88); axis_percent(ax,"C  rotating 条件的 A/B/C 迁移","目标 J（%）"); ax.set_xticks(x,labels,rotation=18,ha="right"); evidence["transfer"]={"labels":labels,"percent":vals}
    ax=axes[1,0]; labels=["within","same-type\nfood","same-type\nwater","all-cross"]; x=np.arange(4); width=.23
    for j,s in enumerate(("A","B","C")):
        values=[100*topology["summary"][s][m]["mean"] for m in ("within","same_type_food","same_type_water","all_cross")]; ax.bar(x+(j-1)*width,values,width=width,color=colors[j],alpha=.84,label=SCHEDULE_LABELS[s]); evidence["topology"][s]=values
    axis_percent(ax,"D  跨团队 token 兼容性","目标 J（%）"); ax.set_xticks(x,labels); ax.legend(frameon=False,fontsize=7.6,loc="upper right")
    ax=axes[1,1]; rep=transfer["replication_vs_v033"]; x=np.arange(len(rep)); vals=100*np.asarray([r["difference"] for r in rep]); colors_rep=["#B44C48" if v<0 else "#2A8C82" for v in vals]; ax.bar(x,vals,color=colors_rep,alpha=.88); ax.axhline(0,color="#4D5964",lw=.8); ax.set_title("E  native A − v0.33 固定拓扑",loc="left",pad=8); ax.set_ylabel("目标 J 差（百分点）"); ax.set_xticks(x,[str(r["seed"])[-2:] for r in rep]); ax.set_xlabel("seed（末两位）"); ax.grid(axis="y",color="#C9CED3",alpha=.58,lw=.7); ax.set_axisbelow(True); evidence["replication"]={"seed": [r["seed"] for r in rep],"difference_pp":vals.tolist()}
    ax=axes[1,2]
    for c in CONDITIONS:
        vals=100*np.asarray([item["agreement"]["full_message_agreement"] for item in agg[c]["agreement"]]); ax.plot(times,vals,color=COLORS[c],marker="o",ms=3.5,lw=2,label=LABELS[c]); evidence["agreement"][c]=vals.tolist()
    axis_percent(ax,"F  同类型完整消息一致率","一致率（%）"); ax.set_xlabel("群体更新次数"); ax.set_xlim(0,2400); ax.set_xticks(times); ax.legend(frameon=False,fontsize=7.6,loc="upper left")
    fig.text(.5,.085,"B schedule 被训练过，C schedule 从未被训练；C 迁移明显低于 A/B。",ha="center",fontsize=10.2); fig.text(.5,.05,"跨团队 all-cross 是固定接收者对16种 sender token 组合的平均，不等同于在线语言使用。",ha="center",fontsize=10.2,color="#43505C")
    return save(fig,directory,"02_rotation_outcomes"),evidence


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); args=parser.parse_args(); out=args.out.resolve(); font=setup()
    for name in ("analysis.json","rotation_transfer.json","cross_topology.json","raw_validation.json"):
        if not (out/name).is_file(): raise FileNotFoundError(out/name)
    analysis=read(out/"analysis.json"); transfer=read(out/"rotation_transfer.json"); topology=read(out/"cross_topology.json"); validation=read(out/"raw_validation.json")
    if not (analysis["formal"] and analysis["status"]=="complete" and transfer["formal"] and topology["formal"] and validation["passed"] and validation["analysis_sha256"]==sha(out/"analysis.json")): raise ValueError("formal result binding failed")
    directory=out/"figures"; directory.mkdir(exist_ok=True); first,cells=design_figure(directory); second,evidence=result_figure(directory,analysis,transfer,topology)
    record={"status":"rendered_pending_visual_QA","support_design_sha256":sha(ROOT/"support_design.json"),"plot_source_sha256":sha(__file__),"analysis_sha256":sha(out/"analysis.json"),"rotation_transfer_sha256":sha(out/"rotation_transfer.json"),"cross_topology_sha256":sha(out/"cross_topology.json"),"raw_validation_sha256":sha(out/"raw_validation.json"),"font_path":font,"figure_sha256":{**first,**second},"matrix_cells":cells,"plotted_values":evidence,"axes_fixed_before_results":True,"source_and_condition_selection":False}
    write(directory/"figure_source.json",record); print(json.dumps({"status":record["status"],"directory":str(directory),"files":list(record["figure_sha256"])},ensure_ascii=False))


if __name__=="__main__": main()
