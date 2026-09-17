"""Validate and seal the completed v18 diagnostic without changing old batches."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT/'results/gradient_001'
REPORT = OUT/'固定策略下的通信学习梯度方差研究报告.md'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, obj): Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    done = read(OUT/'measurement_complete.json')
    assert done['status']=='complete' and done['formal']
    assert (done['policies'],done['worlds'],done['message_scores'],done['training_updates'])==(72,1296,63504,0)
    for p,h in {**done['source_hashes'],**done['input_hashes']}.items(): assert sha(p)==h, p
    for p,h in done['files'].items(): assert sha(OUT/p)==h, p
    for name in ('audit_execution.json','independent_recount.json','analysis_comparison.json','audit_report_review.json','figure_qa.json'):
        assert read(OUT/name)['passed'], name
    assert read(OUT/'audit_report_review.json')['report_sha256']==sha(REPORT)
    snap = OUT/'supporting_materials'
    snap.mkdir(exist_ok=False)
    supporting = [PROJECT/'README.md', PROJECT/'paper_program/README.md', PROJECT/'paper_program/证据与投稿门槛.md']
    water = PROJECT/'paper_program/visual_confirmation_v2_water_review'
    supporting += [p for p in water.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
    for source in supporting:
        dest = snap/source.relative_to(PROJECT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    links = []
    for file in [REPORT,ROOT/'README.md']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',file.read_text()):
            target=target.strip('<>')
            if re.match(r'^https?://',target): continue
            path=Path(unquote(target.split('#')[0]))
            if not path.is_absolute(): path=file.parent/path
            assert path.exists(), (file,target)
            links.append(dict(from_file=str(file.relative_to(PROJECT)),target=target))
    write(OUT/'delivery_qa.json',dict(passed=True,local_links_checked=len(links),links=links,
        report_sha256=sha(REPORT),scope='linked local artifacts exist; external links not re-fetched'))
    files = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in str(p)
        and p.name!='completion_manifest.json' and p.suffix!='.pyc']
    manifest = dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='v0.18 finite-support fixed-policy diagnostic completed; broader research goal remains active',
        policies=72,source_seeds=4,worlds=1296,message_score_vectors=63504,new_training_updates=0,
        main_report_sha256=sha(REPORT),
        limitations=['Inherited development sources and one fixed original training photo pair',
            'Reward-score conditional variance only, not total training gradient or mediation',
            'Classical estimator diagnostic does not itself establish novelty or language origin',
            '79 water candidates reviewed only as metadata; no new pixels or accepted images'],
        artifacts={str(p.relative_to(PROJECT)):sha(p) for p in sorted(files)})
    assert not (OUT/'completion_manifest.json').exists()
    write(OUT/'completion_manifest.json',manifest)
    print(json.dumps(dict(status='complete',artifacts=len(files),report_sha256=sha(REPORT),links=len(links))))


if __name__=='__main__': main()
