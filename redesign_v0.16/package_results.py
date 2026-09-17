"""Bind the completed fixed v16 batch, including failed development records."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BATCH = ROOT/'results/branches_001'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

report = BATCH/'策略与价值输入幅度的通信形成研究报告.md'
review = read(BATCH/'audit_report_review.json')
assert review['passed'] and review['report_sha256'] == sha(report)
for name in ['audit_execution.json', 'independent_recount.json', 'independent_recount_comparison.json', 'figure_qa.json', 'root_figure_qa.json', 'display_revision.json', 'clip_activity_independent_qa.json']:
    assert read(BATCH/name)['passed'], name
train = read(BATCH/'training_complete.json')
assert train['status'] == 'complete' and train['new_social_runs'] == 24
for path, digest in train['source_hashes'].items(): assert sha(path) == digest, path
comparison = read(BATCH/'independent_recount_comparison.json')
for k, name in [('analysis_sha256','branches_analysis.json'), ('independent_recount_sha256','independent_recount.json'), ('report_numbers_sha256','report_numbers.json')]:
    assert comparison[k] == sha(BATCH/name), name
source = read(BATCH/'source_receipt.json')
for arm in ['retained','reset','scaled']:
    for path, digest in source[arm]['files'].items(): assert sha(path) == digest, path
calibration = read(BATCH/'calibration_receipt.json')
assert sha(calibration['path']) == calibration['sha256']
for relative, digest in calibration['files'].items():
    assert sha(Path(calibration['path']).parent/relative) == digest, relative

delivery = [report, ROOT/'README.md', ROOT/'文献与贡献判断.md', ROOT/'裁剪核查后的机制解释与候选.md', PROJECT/'README.md', PROJECT/'paper_program/README.md', PROJECT/'paper_program/证据与投稿门槛.md', PROJECT/'paper_program/visual_confirmation_v1/README.md', PROJECT/'paper_program/visual_confirmation_v1/pixel_workflow_003/README.md']
links = []
for p in delivery:
    for href in re.findall(r'\[[^\]]*\]\(([^)]+)\)', p.read_text()):
        if href.startswith(('https:', 'http:', '#', 'app:', 'codex:')): continue
        target = href.strip('<>').split('#')[0]
        line_match = re.search(r':([1-9][0-9]*)$', target)
        line = int(line_match.group(1)) if line_match else None
        if line_match: target = target[:line_match.start()]
        q = Path(target) if target.startswith('/') else p.parent/target
        exists = q.exists()
        line_valid = line is None or (q.is_file() and line <= len(q.read_text().splitlines()))
        links.append(dict(source=str(p), target=href, exists=exists, line=line, line_valid=line_valid))
assert all(x['exists'] for x in links), [x for x in links if not x['exists']]
assert all(x['line_valid'] for x in links), [x for x in links if not x['line_valid']]
write(BATCH/'delivery_qa.json', dict(passed=True, report_sha256=sha(report), local_links_checked=len(links), links=links))

support = BATCH/'supporting_materials'
for relative in ['README.md','paper_program/README.md','paper_program/证据与投稿门槛.md','paper_program/visual_confirmation_v1/README.md']:
    dest = support/relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT/relative, dest)
data = PROJECT/'paper_program/visual_confirmation_v1/pixel_workflow_003'
data_paths = [p for p in data.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.py') and '__pycache__' not in p.parts]
for p in data_paths:
    dest = support/p.relative_to(PROJECT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)

paths = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p != BATCH/'completion_manifest.json']
rec = read(BATCH/'independent_recount.json')
manifest = dict(status='complete', scope='Fixed v0.16 policy/value input factorial batch only; broader ICLR research goal remains active',
    completed_utc=datetime.now(timezone.utc).isoformat(),
    previous_goal_turn_classification='progress: completed v0.15 fixed amplitude experiment and report',
    current_goal_turn_classification='progress: completed 2x2 policy/value input intervention and bounded third image workflow',
    new_private_runs=0, new_social_runs=24, reused_social_runs=24, source_seeds=4,
    social_updates=2400, communications=29491200, actions=58982400,
    primary_route_difference=rec['primary_mean_difference'], factorial_contrast_means=rec['contrast_means'],
    main_report_sha256=sha(report),
    exploratory_clip_activity_sha256=sha(BATCH/'clip_activity_exploratory.json'),
    exploratory_clip_independent_qa_sha256=sha(BATCH/'clip_activity_independent_qa.json'),
    limitations=['Four inherited development seeds and original photos; no independent confirmation',
                'Branch intervention identifies effects within this finite-budget algorithm, not a mediated proportion or a unique knowledge mechanism',
                'Four inherited sources and visual workflow are not new independent model confirmation',
                'Image workflow review is separate from model confirmation; no claim of ICLR readiness'],
    artifacts={str(p.relative_to(PROJECT)):sha(p) for p in sorted(paths)})
write(BATCH/'completion_manifest.json', manifest)
print(json.dumps(dict(local_links=len(links), artifacts=len(paths), report_sha256=sha(report))))
