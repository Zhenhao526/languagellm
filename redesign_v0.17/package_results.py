"""Bind the completed fixed v17 batch, including failed development records."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, shutil

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
BATCH = ROOT/'results/baseline_001'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

report = BATCH/'发送基线与场景对应关系的通信形成研究报告.md'
review = read(BATCH/'audit_report_review.json')
assert review['passed'] and review['report_sha256'] == sha(report)
for name in ['audit_execution.json', 'independent_recount.json', 'independent_recount_comparison.json', 'figure_qa.json', 'root_figure_qa.json', 'display_revision.json', 'independent_diagnostics_comparison.json', 'independent_baseline_diagnostics.json', 'clip_activity.json']:
    assert read(BATCH/name)['passed'], name
train = read(BATCH/'training_complete.json')
assert train['status'] == 'complete' and train['new_social_runs'] == 12
for path, digest in train['source_hashes'].items(): assert sha(path) == digest, path
comparison = read(BATCH/'independent_recount_comparison.json')
for k, name in [('analysis_sha256','baseline_analysis.json'), ('independent_recount_sha256','independent_recount.json'), ('report_numbers_sha256','report_numbers.json')]:
    assert comparison[k] == sha(BATCH/name), name
source = read(BATCH/'source_receipt.json')
for arm in ['retained','reset','scaled']:
    for path, digest in source[arm]['files'].items(): assert sha(path) == digest, path
calibration = read(BATCH/'calibration_receipt.json')
assert sha(calibration['path']) == calibration['sha256']
for relative, digest in calibration['files'].items():
    assert sha(Path(calibration['path']).parent/relative) == digest, relative

delivery = [report, ROOT/'README.md', ROOT/'文献定位与后续.md', PROJECT/'README.md', PROJECT/'paper_program/README.md', PROJECT/'paper_program/证据与投稿门槛.md', PROJECT/'paper_program/visual_confirmation_v2_water/README.md', PROJECT/'paper_program/visual_confirmation_v2_water/元数据可行性报告.md']
links = []
for p in delivery:
    for href in re.findall(r'\[[^\]]*\]\(([^)]+)\)', p.read_text()):
        clean = href.strip('<>')
        if clean.startswith(('https:', 'http:', '#', 'app:', 'codex:')): continue
        target = clean.split('#')[0]
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
data = PROJECT/'paper_program/visual_confirmation_v2_water'
data_paths = [p for p in data.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.md','.py') and '__pycache__' not in p.parts]
for p in data_paths:
    dest = support/p.relative_to(PROJECT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)

paths = [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p != BATCH/'completion_manifest.json']
rec = read(BATCH/'independent_recount.json')
manifest = dict(status='complete', scope='Fixed v0.17 sender baseline correspondence batch only; broader ICLR research goal remains active',
    completed_utc=datetime.now(timezone.utc).isoformat(),
    previous_goal_turn_classification='progress: completed v0.16 2x2 policy/value input experiment and third image workflow',
    current_goal_turn_classification='progress: completed 12 baseline-shuffle runs with paired analysis and bounded new-water metadata study',
    new_private_runs=0, new_social_runs=12, reused_social_runs=12, source_seeds=4,
    social_updates=2400, communications=14745600, actions=29491200,
    primary_state_correspondence=rec['primary_mean_difference'], contrast_means=rec['contrast_means'],
    main_report_sha256=sha(report),
    prespecified_clip_activity_sha256=sha(BATCH/'clip_activity.json'),
    prespecified_diagnostics_comparison_sha256=sha(BATCH/'independent_diagnostics_comparison.json'),
    limitations=['Four inherited development seeds and original photos; no independent confirmation',
                'State-baseline correspondence intervention does not identify gradient variance, a mediated proportion, or a unique knowledge mechanism',
                'Water metadata has 79 automatic candidate clusters, no validated pixels and no independent model confirmation',
                'Image workflow review is separate from model confirmation; no claim of ICLR readiness'],
    artifacts={str(p.relative_to(PROJECT)):sha(p) for p in sorted(paths)})
write(BATCH/'completion_manifest.json', manifest)
print(json.dumps(dict(local_links=len(links), artifacts=len(paths), report_sha256=sha(report))))
