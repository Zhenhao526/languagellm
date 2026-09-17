"""Seal the completed development batch without modifying earlier evidence."""
import argparse,hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
    manifest=out/'completion_manifest.json';assert not manifest.exists()
    inv=read(out/'invocation.json');done=read(out/'training_complete.json')
    for key in ('source_hashes','input_hashes'):
        assert inv[key]==done[key]
        for p,h in inv[key].items():assert sha(p)==h,p
    for rel,h in done['files'].items():assert sha(out/rel)==h,rel
    for name in ('raw_validation.json','audit_execution.json'):assert read(out/name)['passed'],name
    assert read(out/'terminal_receipt.json')['exit_code']==0
    assert read(out/'figures/visual_qa.json')['passed']
    assert read(ROOT/'结果审查.json')['passed_with_stated_limits']
    files=set(p.resolve() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    design=PROJECT/'paper_program/mechanism_pressure_20260916'
    files.update(p.resolve() for p in design.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(Path(p) for p in inv['source_hashes']);files.update(Path(p) for p in inv['input_hashes'])
    files.update(Path(p) for p in read(out/'audit_execution.json')['audit_dependencies'])
    files.update([PROJECT/'README.md',PROJECT/'paper_program/README.md',PROJECT/'paper_program/证据与投稿门槛.md'])
    for p in (Path(inv['source'])/'audit_execution.json',Path(inv['source'])/'completion_manifest.json'):files.add(p)
    report=out/'训练共现结构与共同符号迁移研究报告.md'
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if target.startswith('/'):
            p=Path(target);assert p.exists(),p;files.add(p);links.append(str(p))
    files.discard(manifest)
    artifacts=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)]
    record=dict(status='complete_development_batch',created_utc=datetime.now(timezone.utc).isoformat(),
        study='v29 matched support structure, all fixed24 new social runs complete',overall_research_goal_complete=False,
        terminal=read(out/'terminal_receipt.json'),counts={k:done[k] for k in ('social_runs','pair_updates','messages','actions','new_private_fits','new_dino_inferences')},
        report=str(report),verified_report_links=links,artifacts=artifacts,
        limitations=['Four reused initialization sources and one test-water photograph; no independent confirmation.',
            'Joint support topology changes, not isolated path-count mechanism; commonP6 is a perfect matching.',
            'No complete gradient/Adam replay; original frontend and DINO caches inherited by hashes.',
            'PNG visual QA; PDF same-figure exports without separate PDF rendering.'])
    manifest.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    for row in read(manifest)['artifacts']:assert sha(row['path'])==row['sha256']
    print(json.dumps(dict(status=record['status'],artifacts=len(artifacts),links=len(links),manifest_sha256=sha(manifest)),ensure_ascii=False))
if __name__=='__main__':main()
