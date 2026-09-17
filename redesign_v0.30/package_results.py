"""Seal a completed v0.30 batch without modifying earlier evidence."""
import argparse,hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve();manifest=out/'completion_manifest.json';assert not manifest.exists()
    inv=read(out/'invocation.json');done=read(out/'training_complete.json');assert inv['formal'] and done['status']=='complete'
    for key in ('source_hashes','input_hashes'):
        assert inv[key]==done[key]
        for p,h in inv[key].items():assert sha(p)==h,p
    for rel,h in done['files'].items():assert sha(out/rel)==h,rel
    assert read(out/'analysis.json')['formal'] and read(out/'raw_validation.json')['passed'] and read(out/'audit_execution.json')['passed']
    assert read(out/'figures/visual_qa.json')['passed'] and read(ROOT/'结果审查.json')['passed_with_stated_limits']
    report=out/'多伙伴共同目标与支持结构研究报告.md';assert report.is_file()
    files={p.resolve() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files.update({Path(p) for p in inv['source_hashes']});files.update({Path(p) for p in inv['input_hashes']})
    files.update({Path(p) for p in read(out/'audit_execution.json')['audit_dependencies'] if Path(p).exists()})
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if target.startswith('/'):
            p=Path(target);assert p.exists(),p;files.add(p)
    files.discard(manifest)
    artifacts=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)]
    record={'status':'complete_development_batch','version':'v0.30','created_utc':datetime.now(timezone.utc).isoformat(),
        'study':'multi-partner target and matched support-path contrast','overall_research_goal_complete':False,
        'terminal':read(out/'terminal_receipt.json'),'counts':{k:done[k] for k in ('social_runs','pair_updates','messages','actions','new_private_fits','new_dino_inferences')},
        'report':str(report),'artifacts':artifacts,'limitations':['Four inherited initialization sources; nested panels/directions/masks are not independent samples.','The two supports differ in concrete edge identity as well as path distribution, so path count is not isolated.','No full gradient/Adam replay; inherited visual and private inputs are hash-bound.','The target is a controlled symbolic resource-location task, not a reconstruction of human ecology.']}
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':record['status'],'artifacts':len(artifacts),'manifest_sha256':sha(manifest)},ensure_ascii=False))
if __name__=='__main__':main()
