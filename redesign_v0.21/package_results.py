"""Seal the completed v21 experiment without completing the broader goal."""
import json,hashlib,re,shutil
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import unquote
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'results/communication_001'
REPORT=OUT/'资源可见性与共同通信形成研究报告.md'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
    d=read(OUT/'training_complete.json');assert d['status']=='complete' and d['formal']
    assert (d['social_runs'],d['pair_updates'],d['messages'],d['actions'])==(24,57600,29491200,58982400)
    assert d['new_dino_inferences']==d['new_private_training']==0
    for p,h in {**d['source_hashes'],**d['input_hashes']}.items():assert sha(p)==h,p
    for p,h in d['files'].items():assert sha(OUT/p)==h,p
    for name in ('audit_execution.json','comparison.json','figure_qa.json','audit_report_review.json'):
        assert read(OUT/name)['passed'],name
    assert read(OUT/'comparison.json')['analysis_sha256']==sha(OUT/'analysis.json')
    assert read(OUT/'audit_report_review.json')['report_sha256']==sha(REPORT)
    assert read(ROOT/'preflight_qa.json')['passed'] and read(ROOT/'prior_integrity_qa.json')['passed']
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
    manifest=OUT/'completion_manifest.json';assert not manifest.exists()
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc' and p!=manifest]
    write(manifest,dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),source_seeds=4,
        social_runs=24,pair_updates=57600,messages=29491200,actions=58982400,new_private_training=0,new_dino_inferences=0,
        main_report_sha256=sha(REPORT),scope='v21 observation-condition communication comparison completed; broader ICLR-level research goal remains active',
        limitations=['All v20 full sources retained; v20 full-detach capability gate remains failed',
            'Observation-condition total effect, not a pure memory intervention','Old images and task framework; no independent visual confirmation',
            '49 complete codes can encode all30 maps; natural success does not prove compositional language'],
        artifacts={str(p.relative_to(PROJECT)):sha(p) for p in sorted(files)}))
    print(json.dumps(dict(status='complete',artifacts=len(files),local_links=len(links),report_sha256=sha(REPORT))))
if __name__=='__main__':main()
