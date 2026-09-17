"""Render v0.37 teacher-free origin outcomes."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent; PROJECT=ROOT.parent; sys.path.insert(0,str(PROJECT/"redesign_v0.9/.analysis_deps")); import matplotlib; matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
COLORS={"fixed_A":"#527A9E","rotating_AB":"#2A8C82","random_ABC":"#8C6BB1"}; LABELS={"fixed_A":"固定 A","rotating_AB":"轮换 A/B","random_ABC":"随机 A/B/C"}; SCHEDULES=("A","B","C"); CHECKPOINTS=(0,100,600,1200)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
def setup():
    try:font=font_manager.findfont("PingFang SC",fallback_to_default=False)
    except Exception:font=font_manager.findfont("DejaVu Sans")
    plt.rcParams.update({"font.family":["PingFang SC","DejaVu Sans"],"font.size":10,"axes.titlesize":12,"axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,"axes.unicode_minus":False,"pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white","axes.spines.top":False,"axes.spines.right":False}); return font
def save(fig,directory,stem):
    result={}
    for ext in ("png","pdf"):
        p=directory/f"{stem}.{ext}"; fig.savefig(p,dpi=220,bbox_inches="tight",pad_inches=.14); result[p.name]=sha(p)
    plt.close(fig); return result
def style(ax,title,ylabel="目标 J（%）"):
    ax.set_title(title,loc="left",pad=8); ax.set_ylabel(ylabel); ax.set_ylim(0,60); ax.set_yticks(np.arange(0,61,10)); ax.grid(axis="y",color="#C9CED3",alpha=.58,lw=.7); ax.set_axisbelow(True)
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); args=parser.parse_args(); out=args.out.resolve(); font=setup(); analysis=read(out/"origin_analysis.json"); validation=read(out/"origin_raw_validation.json"); audit=read(out/"origin_audit.json")
    if not(analysis["formal"] and analysis["status"]=="complete" and validation["passed"] and audit["passed"]):raise ValueError("formal origin outputs required")
    directory=out/"figures"; directory.mkdir(exist_ok=True); evidence={"target_curves":{},"agreement_curves":{},"final":{},"auc":{}}
    fig,axes=plt.subplots(2,3,figsize=(15.3,8.6)); fig.subplots_adjust(left=.06,right=.985,top=.82,bottom=.16,hspace=.50,wspace=.28); fig.suptitle("v0.37：无教师条件下的共同符号形成",fontsize=17,y=.98); fig.text(.5,.943,"所有通信模块从随机状态开始；只改变伙伴拓扑日程，私有视觉编码器和 grounded 任务保持不变。",ha="center",fontsize=10.5)
    for col,schedule in enumerate(SCHEDULES):
        ax=axes[0,col]
        for condition in analysis["conditions"]:
            x=analysis["summary"][condition][schedule]; mean=np.asarray([x[str(t)]["target_J"]["mean"] for t in CHECKPOINTS])*100; sd=np.asarray([x[str(t)]["target_J"]["sd"] for t in CHECKPOINTS])*100; ax.plot(CHECKPOINTS,mean,marker="o",ms=4,lw=2.1,color=COLORS[condition],label=LABELS[condition]); ax.fill_between(CHECKPOINTS,np.maximum(0,mean-sd),np.minimum(60,mean+sd),color=COLORS[condition],alpha=.10,linewidth=0); evidence["target_curves"].setdefault(schedule,{})[condition]={"mean_percent":mean.tolist(),"sd_percent":sd.tolist()}
        style(ax,f"{chr(65+col)}  评估拓扑 {schedule}"); ax.set_xlabel("训练更新次数"); ax.set_xlim(0,1200); ax.set_xticks(CHECKPOINTS)
        if col==0:ax.legend(frameon=False,fontsize=8,loc="upper left")
    ax=axes[1,0]
    for condition in analysis["conditions"]:
        x=analysis["summary"][condition]["A"]; mean=np.asarray([x[str(t)]["agreement_joint"]["mean"] for t in CHECKPOINTS])*100; sd=np.asarray([x[str(t)]["agreement_joint"]["sd"] for t in CHECKPOINTS])*100; ax.plot(CHECKPOINTS,mean,marker="o",ms=4,lw=2.1,color=COLORS[condition],label=LABELS[condition]); ax.fill_between(CHECKPOINTS,np.maximum(0,mean-sd),np.minimum(100,mean+sd),color=COLORS[condition],alpha=.10,linewidth=0); evidence["agreement_curves"][condition]={"mean_percent":mean.tolist(),"sd_percent":sd.tolist()}
    ax.set_title("D  A 评估下同类型双 token 一致率",loc="left",pad=8); ax.set_ylabel("一致率（%）"); ax.set_ylim(0,100); ax.set_yticks(np.arange(0,101,20)); ax.set_xlabel("训练更新次数"); ax.set_xlim(0,1200); ax.set_xticks(CHECKPOINTS); ax.grid(axis="y",color="#C9CED3",alpha=.58,lw=.7); ax.set_axisbelow(True); ax.legend(frameon=False,fontsize=8,loc="upper left")
    ax=axes[1,1]; x=np.arange(3); width=.25
    for j,condition in enumerate(analysis["conditions"]):
        vals=[100*analysis["summary"][condition]["A"]["1200"]["target_J"]["mean"],100*analysis["summary"][condition]["B"]["1200"]["target_J"]["mean"],100*analysis["summary"][condition]["C"]["1200"]["target_J"]["mean"]]; err=[100*analysis["summary"][condition][s]["1200"]["target_J"]["sd"] for s in SCHEDULES]; ax.bar(x+(j-1)*width,vals,width,yerr=err,capsize=3,color=COLORS[condition],alpha=.86,label=LABELS[condition]); evidence["final"][condition]={"mean_percent":vals,"sd_percent":err}
    style(ax,"E  无教师形成后的三拓扑终点"); ax.set_xticks(x,SCHEDULES); ax.legend(frameon=False,fontsize=8,loc="upper left")
    ax=axes[1,2]; x=np.arange(3); vals=[100*analysis["formation_auc"][c]["A"]["mean"] for c in analysis["conditions"]]; err=[100*analysis["formation_auc"][c]["A"]["sd"] for c in analysis["conditions"]]; ax.bar(x,vals,yerr=err,capsize=3,color=[COLORS[c] for c in analysis["conditions"]],alpha=.86); style(ax,"F  A 目标 J 的形成速度", "AUC（%·更新归一化）"); ax.set_xticks(x,[LABELS[c] for c in analysis["conditions"]],rotation=12,ha="right"); evidence["auc"]={"mean_percent":vals,"sd_percent":err}
    fig.text(.5,.075,"随机初始化下，grounded 成功率和形式一致率可以沿不同路径形成；一致字符串不是共同协议的充分条件。",ha="center",fontsize=10.2); fig.text(.5,.04,"均值 ± 1 SD；每个条件包含 12 个 seed×panel 运行。",ha="center",fontsize=10.2,color="#43505C")
    figures=save(fig,directory,"01_origin_outcomes"); source={"status":"rendered_pending_visual_QA","formal":True,"plot_source_sha256":sha(Path(__file__)),"analysis_sha256":sha(out/"origin_analysis.json"),"raw_validation_sha256":sha(out/"origin_raw_validation.json"),"audit_sha256":sha(out/"origin_audit.json"),"font_path":font,"figure_sha256":figures,"plotted_values":evidence,"axes_fixed_before_results":True,"source_and_condition_selection":False}; write(directory/"figure_source.json",source); print(json.dumps({"status":source["status"],"directory":str(directory),"files":list(figures)},ensure_ascii=False))
if __name__=="__main__":main()
