"""Seal the completed private-readout batch and its reviewed interpretation."""
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT/'results/readout_001'
REPORT = OUT/'冻结视觉接口的新私人行动学习研究报告.md'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    done = read(OUT/'training_complete.json')
    assert done['status']=='complete' and done['formal']
    assert (done['head_fits'],done['head_updates'],done['simulated_selected_goal_actions'],done['supervised_labels']) == (96,230400,235929600,117964800)
    assert done['no_interface_training'] and done['no_communication']
    for p,h in {**done['source_hashes'],**done['input_hashes']}.items(): assert sha(p)==h,p
    for p,h in done['files'].items(): assert sha(OUT/p)==h,p
    for name in ('audit_execution.json','independent_recount.json','analysis_comparison.json','audit_report_review.json','figure_qa.json'):
        assert read(OUT/name)['passed'],name
    assert read(OUT/'audit_report_review.json')['report_sha256']==sha(REPORT)
    assert read(ROOT/'prior_integrity_qa.json')['passed']
    snap = OUT/'supporting_materials'; snap.mkdir(exist_ok=False)
    supporting = [PROJECT/'README.md',PROJECT/'paper_program/README.md',PROJECT/'paper_program/证据与投稿门槛.md']
    literature = PROJECT/'paper_program/literature_capability_20260916'
    supporting += [p for p in literature.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    for p in supporting:
        target=snap/p.relative_to(PROJECT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    links=[]
    for file in [REPORT,ROOT/'README.md']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',file.read_text()):
            target=target.strip('<>')
            if re.match(r'^https?://',target):continue
            path=Path(unquote(target.split('#')[0]))
            if not path.is_absolute():path=file.parent/path
            assert path.exists(),(file,target)
            links.append(dict(from_file=str(file.relative_to(PROJECT)),target=target))
    write(OUT/'delivery_qa.json',dict(passed=True,local_links_checked=len(links),links=links,
        report_sha256=sha(REPORT),scope='local delivery links; external literature has separate provenance checks'))
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix!='.pyc'
        and p != OUT/'completion_manifest.json']
    manifest=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='v0.19 fresh private readout diagnostic completed; broader research goal remains active',
        source_seeds=4,head_fits=96,head_updates=230400,new_communication_training=0,new_dino_inferences=0,
        main_report_sha256=sha(REPORT),limitations=[
            'Four inherited development sources; no independent visual confirmation',
            'Specific private readout architecture and learning budget, not information-theoretic equivalence',
            'Reward-only primary comparison; supervised CE reference is separate',
            'No new common communication, grammar or language-origin mechanism established'],
        artifacts={str(p.relative_to(PROJECT)):sha(p) for p in sorted(files)})
    assert not (OUT/'completion_manifest.json').exists()
    write(OUT/'completion_manifest.json',manifest)
    print(json.dumps(dict(status='complete',artifacts=len(files),links=len(links),report_sha256=sha(REPORT))))


if __name__=='__main__':main()
