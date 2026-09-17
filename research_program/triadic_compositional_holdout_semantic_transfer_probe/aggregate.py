"""Aggregate actor-specific heldout aligned/placebo transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

AXES = ("kind", "length")
FIELDS = ("plan_transfer", "partner_transfer", "physical", "q", "conditional_q", "action_change")
T7 = 2.3646242510102993


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat(values):
    values=[float(x) for x in values]; mean=sum(values)/len(values); sd=math.sqrt(sum((x-mean)**2 for x in values)/(len(values)-1)) if len(values)>1 else 0.; half=T7*sd/math.sqrt(len(values)) if len(values)>1 else 0.
    return dict(seed_values=values,mean=mean,sd=sd,ci95_t7=[mean-half,mean+half],n=len(values))


def gvalue(group,prefix,field):
    if field in ("plan_transfer","partner_transfer"): return float(group[f"{prefix}_{field}"])
    if field=="physical": return float(group[f"{prefix}_physical"])
    if field=="q": return float(group[f"{prefix}_q"])
    if field=="conditional_q": return float(group[f"{prefix}_conditional_q"])
    if field=="action_change": return float(group[f"{prefix}_action_change"])
    raise KeyError(field)


def actor_summary(row, sender):
    axes=[]
    for axis in AXES:
        group=row["groups"][f"{axis}/{sender}"]; axes.append(group)
    out={}
    for prefix in ("aligned","placebo"):
        out[prefix]={field:sum(gvalue(group,prefix,field) for group in axes)/len(axes) for field in FIELDS}
    out["aligned_minus_placebo"]={field:out["aligned"][field]-out["placebo"][field] for field in FIELDS}
    return out


def analyze(probe,audit,out):
    probe=Path(probe).resolve(); audit=Path(audit).resolve(); out=Path(out).resolve(); require(not out.exists(),"Refuse to overwrite summary")
    result_path=probe/"execution/results.json"; status_path=probe/"execution/status.json"; receipt_path=probe/"execution/receipt.json"; verification_path=audit/"verification.json"; audit_receipt_path=audit/"receipt.json"
    for path in (result_path,status_path,receipt_path,verification_path,audit_receipt_path,probe/"prepared.json",probe/"plan.json",probe/"freeze.json"): require(path.is_file(),"Missing input "+str(path))
    result=json.loads(result_path.read_text()); status=json.loads(status_path.read_text()); receipt=json.loads(receipt_path.read_text()); verification=json.loads(verification_path.read_text())
    require(result.get("status")=="completed_compositional_holdout_semantic_transfer_probe" and len(result.get("rows",[]))==32,"Incomplete result")
    require(status.get("status")=="completed" and status.get("results_sha256")==sha(result_path)==receipt.get("results_sha256"),"Receipt mismatch")
    require(verification.get("status")=="passed" and verification.get("max_abs_error",1)>-1 and verification.get("max_abs_error",1)<=1e-12,"Audit failed")
    rows=result["rows"]; require(len({(r["seed"],r["schedule"],bool(r["live"])) for r in rows})==32,"Policy grid")
    for row in rows:
        row["actor_balanced"]={str(sender):actor_summary(row,sender) for sender in range(3)}
    cells={}
    for schedule in ("static","rematched"):
        for live in (True,False):
            selected=[r for r in rows if r["schedule"]==schedule and bool(r["live"])==live]
            cells[f"{schedule}/{'live' if live else 'silent'}"]={}
            for actor in ("0","parent_BC"):
                vals=[]
                for row in selected:
                    if actor=="0": vals.append(row["actor_balanced"]["0"])
                    else:
                        vals.append({prefix:{field:(row["actor_balanced"]["1"][prefix][field]+row["actor_balanced"]["2"][prefix][field])/2 for field in FIELDS} for prefix in ("aligned","placebo","aligned_minus_placebo")})
                cells[f"{schedule}/{'live' if live else 'silent'}"][actor]={prefix:{field:stat([v[prefix][field] for v in vals]) for field in FIELDS} for prefix in ("aligned","placebo","aligned_minus_placebo")}
    gaps={}
    for live in (True,False):
        for schedule in ("static","rematched"):
            key=f"{schedule}/{'live' if live else 'silent'}"; gaps[f"{schedule}/{'live' if live else 'silent'}"]={field:stat([cells[key]["0"]["aligned_minus_placebo"][field]["seed_values"][i]-cells[key]["parent_BC"]["aligned_minus_placebo"][field]["seed_values"][i] for i in range(8)]) for field in FIELDS}
    interaction={field:stat([gaps["rematched/live"][field]["seed_values"][i]-gaps["static/live"][field]["seed_values"][i] for i in range(8)]) for field in FIELDS}
    out.mkdir(parents=True); summary=dict(status="completed",created_at=datetime.now(timezone.utc).isoformat(),probe=str(probe),audit=dict(path=str(audit),verification_sha256=sha(verification_path),max_abs_error=verification["max_abs_error"]),contract=dict(seeds=list(range(66701,66709)),schedules=["static","rematched"],channels=["live","silent"],axes=list(AXES),case_count=3456,policy_blocks=32,worlds_per_policy=11232,optimizer_updates=0,model_forward_samples=result["model_forward_samples"]),cells=cells,gaps=gaps,interaction=interaction,interpretation_boundary="A-vs-parent aligned-minus-placebo is a post-hoc heldout-combination transfer diagnostic, not lexical or language-origin evidence.",input_sha256={str(p):sha(p) for p in (result_path,status_path,receipt_path,verification_path,audit_receipt_path,probe/"prepared.json",probe/"plan.json",probe/"freeze.json")})
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    lines=["# 联合组合留出 A vs B/C aligned/placebo 消息转移","","A 是只见联合组合训练的新接收者；B/C 是全组合训练的冻结父代。先在 kind、length 两个 heldout 轴内等权，再对8个源种子汇总。单位为百分点，区间为 t(7)。","","| 条件 | A aligned−placebo | B/C aligned−placebo | A−B/C |","|---|---:|---:|---:|"]
    for key in ("static/live","rematched/live","static/silent","rematched/silent"):
        a=cells[key]["0"]["aligned_minus_placebo"]["plan_transfer"]; b=cells[key]["parent_BC"]["aligned_minus_placebo"]["plan_transfer"]; gap=gaps[key]["plan_transfer"]; fmt=lambda x:f"{100*x['mean']:+.4f} [{100*x['ci95_t7'][0]:+.4f}, {100*x['ci95_t7'][1]:+.4f}]"; lines.append(f"| {key} | {fmt(a)} | {fmt(b)} | {fmt(gap)} |")
    lines += ["","## A−B/C live schedule interaction","","| 指标 | 均值 | t(7) 区间 |","|---|---:|---:|"]
    for field in FIELDS:
        x=interaction[field]; lines.append(f"| {field} | {100*x['mean']:+.4f} | [{100*x['ci95_t7'][0]:+.4f}, {100*x['ci95_t7'][1]:+.4f}] |")
    lines += ["","静默条件的 aligned 与 placebo 应逐行相等；A−B/C 的正值表示 A 在未见联合组合上的方向性内容转移高于已训练全组合的父代控制。","","[JSON](summary.json)；[独立审计](../../../triadic_compositional_holdout_semantic_transfer_probe/audit_transfer_001/verification.json)。"]
    (out/"汇总表.md").write_text("\n".join(lines)+"\n",encoding="utf8"); rec=dict(status="completed",script_sha256=sha(__file__),outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}); (out/"receipt.json").write_text(json.dumps(rec,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8"); return dict(status="completed",output=str(out),summary_sha256=sha(out/"summary.json"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--probe",required=True); parser.add_argument("--audit",required=True); parser.add_argument("--out",required=True); args=parser.parse_args(); print(json.dumps(analyze(args.probe,args.audit,args.out),ensure_ascii=False))
