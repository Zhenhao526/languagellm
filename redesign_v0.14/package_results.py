"""Verify current delivery links and bind completed v14 artifacts; not an ICLR completion claim."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,shutil
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;BATCH=ROOT/'results/reset_001'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
report=BATCH/'私人非语言经验的迁移研究报告.md'
review=read(BATCH/'audit_report_review.json');assert review['passed']
report_hash=sha(report);assert review['report_sha256']==report_hash
for name in ('audit_execution.json','independent_recount.json','independent_recount_comparison.json'):
 assert read(BATCH/name)['passed'],name
assert read(BATCH/'training_complete.json')['status']=='complete'
assert read(BATCH/'independent_recount_comparison.json')['analysis_sha256']==sha(BATCH/'reset_analysis.json')
source=read(BATCH/'source_receipt.json')
for path,digest in source['files'].items():assert sha(Path(path))==digest,path
files=[report,ROOT/'README.md',PROJECT/'README.md',PROJECT/'paper_program/README.md',PROJECT/'paper_program/证据与投稿门槛.md',ROOT/'贡献边界预审.md',PROJECT/'paper_program/visual_confirmation_v1/pixel_workflow_001/README.md']
links=[]
for p in files:
 for href in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
  if href.startswith(('https:','http:','#','app:','codex:')):continue
  target=href.strip('<>').split('#')[0]
  q=Path(target) if target.startswith('/') else p.parent/target
  links.append(dict(source=str(p),target=href,exists=q.exists()))
assert all(x['exists'] for x in links),[x for x in links if not x['exists']]
write(BATCH/'delivery_qa.json',dict(passed=True,report_sha256=report_hash,local_links_checked=len(links),links=links,
 scope='Completion of fixed v14 batch, statistical audits, report review, consumed-source integrity and local delivery links'))
support=BATCH/'supporting_materials';support.mkdir(exist_ok=True)
for relative in ['README.md','paper_program/README.md','paper_program/证据与投稿门槛.md','paper_program/visual_confirmation_v1/README.md','paper_program/visual_confirmation_v1/pixel_workflow_001/README.md','paper_program/visual_confirmation_v1/pixel_workflow_001/identity_license_review_20260915.md','paper_program/visual_confirmation_v1/pixel_workflow_001/identity_license_review_20260915.json','paper_program/visual_confirmation_v1/pixel_workflow_001/curation_status_20260916.json']:
 p=PROJECT/relative;dest=support/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for name in ('manipulation_001','manipulation_002'):
 shutil.copytree(ROOT/name,support/name,dirs_exist_ok=True)
paths=[p for p in BATCH.rglob('*') if p.is_file() and p.name!='completion_manifest.json' and '__pycache__' not in p.parts]
paths += [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json')]
manifest=dict(status='complete',scope='v0.14 fixed reset batch only; broader ICLR research goal remains active',completed_utc=datetime.now(timezone.utc).isoformat(),
 previous_goal_turn_classification='progress: completed v0.13 experiment and report',current_goal_turn_classification='progress: executed paired reset and obtained new causal-development evidence',
 new_private_runs=0,new_social_runs=12,reused_control_runs=12,source_seeds=4,social_updates=2400,communications=14745600,actions=29491200,
 primary_retained_minus_reset=read(BATCH/'independent_recount.json')['primary_mean_difference'],main_report_sha256=report_hash,
 limitations=['Four inherited development sources, old photos; no independent confirmation','Intervention jointly changes content, amplitude and readout/optimization conditions','Next reset_scaled is a candidate only; no ICLR completion claim'],
 artifacts={str(p.relative_to(PROJECT)):sha(p) for p in paths})
write(BATCH/'completion_manifest.json',manifest)
print(json.dumps(dict(local_links=len(links),artifacts=len(paths),report_sha256=report_hash)))
