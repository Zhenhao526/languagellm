"""Bind a completed temporal experiment; keep the broader research goal open."""
import hashlib,json,re,shutil
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results/temporal_001'
REPORT=OUT/'资源移动后的私人记忆与更新研究报告.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    d=read(OUT/'training_complete.json');assert d['status']=='complete' and d['formal']
    assert (d['head_fits'],d['private_updates'],d['selected_goal_actions'])==(48,115200,58982400)
    assert d['new_social_training']==d['new_dino_inferences']==0
    for p,h in {**d['source_hashes'],**d['input_hashes']}.items():assert sha(p)==h,p
    for p,h in d['files'].items():assert sha(OUT/p)==h,p
    for name in ('audit_execution.json','independent_recount.json','analysis_comparison.json','figure_qa.json','audit_report_review.json'):
        assert read(OUT/name)['passed'],name
    assert read(OUT/'audit_report_review.json')['report_sha256']==sha(REPORT)
    assert read(ROOT/'task_figure/figure_qa.json')['passed'] and read(ROOT/'prior_integrity_qa.json')['passed']
    snap=OUT/'supporting_materials';snap.mkdir()
    for name in ('README.md','paper_program/README.md','paper_program/证据与投稿门槛.md'):
        target=snap/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/name,target)
    links=[]
    for p in (REPORT,ROOT/'README.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            target=target.strip('<>')
            if re.match(r'^https?://',target):continue
            link=Path(unquote(target.split('#')[0]));link=link if link.is_absolute() else p.parent/link
            assert link.exists(),(p,target);links.append(dict(source=str(p.relative_to(PROJECT)),target=target))
    write(OUT/'delivery_qa.json',dict(passed=True,local_links_checked=len(links),links=links,report_sha256=sha(REPORT)))
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc' and p!=OUT/'completion_manifest.json']
    result=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),source_seeds=4,private_person_fits=48,
        private_updates=115200,private_actions=58982400,new_communication_training=0,new_dino_inferences=0,
        main_report_sha256=sha(REPORT),scope='v20 private temporal experiment complete; overall ICLR-level research goal remains active',
        capability_gate=read(OUT/'analysis.json')['capability_gate'],
        limitations=['Single relocation,2frames,old images and map framework','Detach preserves history and changes credit assignment,not a pure memory deletion',
            'Engineering capability gate is not significance or equivalence','No new communication protocol or independent visual confirmation in this matrix'],
        artifacts={str(p.relative_to(PROJECT)):sha(p) for p in sorted(files)})
    assert not (OUT/'completion_manifest.json').exists();write(OUT/'completion_manifest.json',result)
    print(json.dumps(dict(status='complete',artifacts=len(files),links=len(links),report_sha256=sha(REPORT))))
if __name__=='__main__':main()
