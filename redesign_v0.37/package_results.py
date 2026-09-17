"""Seal the formal v0.37 teacher-free origin batch."""
from __future__ import annotations
import argparse, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def assert_hash(path,expected):
    if sha(path)!=expected:raise AssertionError(f"hash mismatch: {path}")
def linked(report):
    for m in re.finditer(r"\]\((<?)([^)>]+)>?\)",report.read_text()):
        p=Path(m.group(2))
        if p.is_absolute():yield p
def check(out):
    i=read(out/"invocation.json");t=read(out/"training_complete.json")
    if not(i.get("formal") is True and t.get("formal") is True and t.get("status")=="complete"):raise AssertionError("formal training completion required")
    for key in ("source_hashes","input_hashes"):
        if i[key]!=t[key]:raise AssertionError(f"{key} changed")
        for p,d in i[key].items():assert_hash(Path(p),d)
    for p,d in t.get("files",{}).items():assert_hash(out/p,d)
    return i,t
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);args=parser.parse_args();root=args.out.resolve();out=root/"results"/"origin_001";manifest=out/"completion_manifest.json"
    if manifest.exists():raise FileExistsError(manifest)
    invocation,complete=check(out);analysis=read(out/"origin_analysis.json");validation=read(out/"origin_raw_validation.json");audit=read(out/"origin_audit.json");visual=read(out/"figures"/"visual_qa.json");review=read(root/"结果审查.json");report=out/"无教师条件下的共同符号形成研究报告.md";report_links=list(linked(report));missing=[str(p) for p in report_links if not p.exists()]
    if not(analysis.get("formal") and analysis.get("status")=="complete" and validation.get("passed") and audit.get("passed") and visual.get("passed") and review.get("passed") and review.get("version")=="v0.37-teacher-free-origin" and not missing):raise AssertionError("formal analysis/audit/review required")
    files=set()
    for p in root.rglob("*"):
        if not p.is_file() or ".venv" in p.parts or "__pycache__" in p.parts or p.name=="completion_manifest.json":continue
        if "results" in p.parts and any(part.startswith("smoke_") for part in p.parts):continue
        files.add(p.resolve())
    files.update(Path(p).resolve() for p in invocation["source_hashes"]);files.update(Path(p).resolve() for p in invocation["input_hashes"]);files.update(p.resolve() for p in report_links)
    artifacts=[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(files) if p.is_file()]
    record={"status":"complete_development_batch","version":"v0.37-teacher-free-origin","created_utc":datetime.now(timezone.utc).isoformat(),"study":"teacher-free grounded protocol formation from random communication modules under topology schedules","overall_research_goal_complete":False,"formal_output":str(out),"counts":{"runs":complete["runs"],"updates_per_run":complete["updates_per_run"],"messages":complete["messages"],"actions":complete["actions"]},"quality":{"independent_checks":validation["checks"],"scalar_comparisons":validation["scalar_comparisons"],"maximum_metric_absolute_difference":validation["maximum_metric_absolute_difference"],"trace_audit_checks":audit["checks"],"visual_qa":visual["passed"],"scientific_review":review["status"]},"report":str(report),"review":str(root/"结果审查.json"),"artifacts":artifacts,"limitations":["Private visual encoders are inherited frozen perceptual interfaces; only communication modules start random.","Training uses a shared grounded reward and synchronized policy gradients rather than decentralized observation-only learning.","The four-agent two-resource task has one token per sender and no vocabulary growth, grammar or intention.","Topology schedules are experimenter specified and random-ABC is deterministic from seeds.","Agreement is a surface statistic, not semantic equivalence; the audit does not replay every optimizer state transition."],"next_step":"Feed each formed population into iterated replacement, then repeat origin with decentralized episode-level feedback and a third resource/multi-token protocol.","excluded":["results/origin_smoke_001","__pycache__",".venv","prior completion manifests"],"link_check":{"report_links":len(report_links),"missing":missing}}
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n");print(json.dumps({"status":record["status"],"version":record["version"],"artifacts":len(artifacts),"manifest_sha256":sha(manifest)},ensure_ascii=False))
if __name__=="__main__":main()
