"""Verify and seal completed v23 evidence after final human report review.

This checks bytes, receipt bindings, counts and local links. It does not run
models, recompute scientific results, require a positive capability outcome,
or retroactively upgrade the limited preflight into a numerical audit.
Run only after all result, review, figure and living-index files are final.
"""
import hashlib
import itertools
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / 'results/experience_001'
REPORT = OUT / '私人状态经验与共同符号表达迁移研究报告.md'
SEEDS = [33101, 33102, 33103, 33104]
PARTITIONS = [1, 2, 3]
TIMES = [0, 100, 600, 1200, 2100, 2400]
ARMS = ['old_old', 'all_old', 'all_all']
LIVING_INDICES = ('README.md', 'paper_program/README.md', 'paper_program/证据与投稿门槛.md')
RECEIPTS = ('audit_execution.json', 'comparison.json', 'figure_qa.json', 'audit_report_review.json')
EXPECTED_COUNTS = dict(private_fits=48, social_runs=36, private_updates=115200,
    private_actions=58982400, pair_updates=86400, messages=44236800, actions=88473600,
    initial_preparation_updates=1600, initial_preparation_actions=102400)
EXPECTED_AUDIT = dict(private_fits=48, social_pairs=36,
    private_updates_stream_checked=115200, social_updates_stream_checked=86400,
    private_evaluation_tables=288, social_evaluation_tables=432,
    private_endpoint_worlds_replayed=46080, social_endpoint_sender_worlds_replayed=69120)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def require(value, label):
    if not value:
        raise ValueError(label)


def bound(path, expected):
    require(sha(path) == expected, f'hash mismatch: {path}')


def output_path(path):
    path = Path(path)
    return path if path.is_absolute() else OUT / path


def local_links(document):
    """Inline local Markdown links/images only; no external or anchor checks."""
    result = []
    for raw in re.findall(r'!?\[[^\]\n]*\]\((<[^>]+>|[^)\n]+)\)', document.read_text()):
        target = raw.strip()
        if target.startswith('<'):
            target = target[1:target.index('>')]
        else:
            target = re.sub(r'\s+(?:"[^"]*"|\x27[^\x27]*\x27)\s*$', '', target)
        if re.match(r'^(?:https?|mailto|data|app|codex):', target, re.I):
            continue
        decoded = re.sub(r':\d+$', '', unquote(target.split('#', 1)[0]))
        path = Path(decoded) if decoded else document
        if not path.is_absolute():
            path = document.parent / path
        path = path.resolve()
        require(path.exists(), f'broken local link in {document}: {target}')
        result.append(dict(source=str(document.relative_to(PROJECT)), target=target, resolved=str(path)))
    return result


def main():
    manifest = OUT / 'completion_manifest.json'
    require(not manifest.exists(), 'v23 already sealed; refusing to overwrite completion manifest')
    complete = read(OUT / 'training_complete.json')
    inv = read(OUT / 'invocation.json')
    require(complete['status'] == 'complete' and complete['formal'] is True and inv['formal'] is True,
            'formal training is incomplete')
    require((inv['seeds'], inv['partitions'], inv['updates']) == (SEEDS, PARTITIONS, 2400),
            'unexpected formal source scope or budget')
    require(inv['private_scopes'] == ['old', 'all'] and inv['social_arms'] == ARMS,
            'unexpected private/social matrix')
    for key, value in EXPECTED_COUNTS.items():
        require(complete[key] == value, f'wrong completed count: {key}')
    require(complete['new_dino_inferences'] == inv['new_dino_inferences'] == 0, 'unexpected DINO inference')
    for key in ('source_hashes', 'input_hashes'):
        require(complete[key] == inv[key], f'invocation/completion {key} differ')
        for path, digest in complete[key].items():
            bound(path, digest)
    for path, digest in complete['source_hashes'].items():
        bound(OUT / 'frozen_sources' / Path(path).relative_to(PROJECT), digest)
    for path, digest in complete['files'].items():
        bound(OUT / path, digest)

    private_names = {f's{s}_p{p}_d{d}_{scope}' for s, p, d, scope in
                     itertools.product(SEEDS, PARTITIONS, (0, 1), ('old', 'all'))}
    social_names = {f's{s}_p{p}_{arm}' for s, p, arm in itertools.product(SEEDS, PARTITIONS, ARMS)}
    for phase, names in (('private', private_names), ('social', social_names)):
        require({p.name for p in (OUT / phase).iterdir() if p.is_dir()} == names,
                f'incomplete or additional {phase} run directory')
        for name in names:
            folder = OUT / phase / name
            cfg, result = read(folder / 'config.json'), read(folder / 'result.json')
            require(cfg['updates'] == result['updates'] == 2400 and result['status'] == 'complete',
                    f'incomplete run: {folder}')
            require(cfg['checkpoints'] == TIMES and result['frozen_verified'] is True,
                    f'checkpoint/frozen-state contract failed: {folder}')
            expected_raw = ({f'evaluation_{t:04d}.npz' for t in TIMES} if phase == 'private' else
                            {f'protocol_{t:04d}_d{d}.npz' for t, d in itertools.product(TIMES, (0, 1))})
            pattern = 'evaluation_*.npz' if phase == 'private' else 'protocol_*.npz'
            require({p.name for p in folder.glob(pattern)} == expected_raw, f'raw evaluation inventory: {folder}')

    evidence = {name: read(OUT / name) for name in RECEIPTS}
    for name, receipt in evidence.items():
        require(receipt['passed'] is True, f'unsuccessful final receipt: {name}')
    analysis_sha = sha(OUT / 'analysis.json')
    analyzer_sha = sha(ROOT / 'analyze_experience.py')
    audit, comparison, figures, review = (evidence[name] for name in RECEIPTS)
    require(audit['formal'] is True and audit['failures'] == [], 'final formal execution audit required')
    require(audit['production_metrics_called'] is False and comparison['production_metrics_called'] is False,
            'independent numerical scope must be retained')
    bound(OUT / 'training_complete.json', audit['training_complete_sha256'])
    for receipt in (audit, comparison, figures):
        require(receipt['analysis_sha256'] == analysis_sha, 'final receipt binds a different analysis')
    require(audit['analysis_source_sha256'] == analyzer_sha, 'audit binds a different analyzer')
    bound(OUT / 'analyze_experience_source.py', analyzer_sha)
    for path, digest in audit['replay_dependency_sha256'].items():
        bound(path, digest)
    for key, value in EXPECTED_AUDIT.items():
        require(audit['counts'][key] == value, f'wrong formal audit coverage count: {key}')
    require(audit['counts']['private_key_updates_replayed'] >= 4 and
            audit['counts']['social_key_pair_updates_replayed'] >= 6, 'required first/2101 replay sample missing')
    expected_figures = {OUT / 'figures' / f'{name}.{ext}' for name, ext in itertools.product(
        ('01_private_and_communication', '02_learning_curves'), ('png', 'pdf'))}
    require({output_path(p).resolve() for p in figures['files']} == expected_figures,
            'two PNG and two PDF figure bindings required')
    for path, digest in figures['files'].items():
        bound(output_path(path), digest)
    # Figure/report receipt schemas may carry extra source bindings; retain and
    # verify them when present rather than guessing fields absent from v23.
    if 'analysis_source_sha256' in figures:
        require(figures['analysis_source_sha256'] == analyzer_sha, 'figure analyzer binding differs')
    if 'plot_source_sha256' in figures:
        bound(ROOT / 'plot_results.py', figures['plot_source_sha256'])
    bound(REPORT, review['report_sha256'])
    require(review['analysis_sha256'] == analysis_sha, 'report review binds a different analysis')
    for path, digest in review.get('bound_files', {}).items():
        bound(output_path(path), digest)

    analysis = read(OUT / 'analysis.json')
    require(analysis['status'] == 'complete' and analysis['formal'] is True, 'incomplete final analysis')
    require((analysis['seeds'], analysis['partitions'], analysis['updates'], analysis['times']) ==
            (SEEDS, PARTITIONS, 2400, TIMES), 'analysis scope differs')
    require(analysis['analysis_source_sha256'] == analyzer_sha, 'analysis source mismatch')
    bound(OUT / 'training_complete.json', analysis['training_complete_sha256'])
    require(analysis['replay_dependency_sha256'] == audit['replay_dependency_sha256'], 'analysis/audit dependency mismatch')
    require(len(analysis['rows']) == 84 and len(analysis['seed_rows']) == 20 and len(analysis['primary']) == 4,
            'analysis must retain48 private +36 social runs and four primary source summaries')
    require(sorted(r['seed'] for r in analysis['primary']) == SEEDS, 'primary source identities differ')
    applicability = read(OUT / 'private_applicability.json')
    require(applicability == analysis['capability'], 'private applicability differs from independent analysis')
    require(applicability['threshold'] == .8 and applicability['filter_applied'] is False and
            applicability['social_matrix_proceeds_regardless'] is True, 'capability selection contract differs')
    require(sorted(r['seed'] for r in applicability['source_rows']) == SEEDS, 'capability source coverage differs')
    # Deliberately do NOT require applicability['passed']: both outcomes must be sealed.

    gate, prior = read(ROOT / 'preflight_qa.json'), read(ROOT / 'prior_integrity_qa.json')
    require(gate['passed'] is True and prior['passed'] is True, 'preflight or prior integrity failed')
    require(gate['source_hashes'] == complete['source_hashes'], 'production differs from successful preflight')
    bound(inv['preflight']['path'], inv['preflight']['sha256'])
    bound(ROOT / 'world_selftest.json', gate['world_selftest_sha256'])
    bound(ROOT / '实现前审查.json', gate['implementation_review_sha256'])
    bound(ROOT / 'prior_integrity_qa.json', gate['previous_integrity_sha256'])
    bound(Path(gate['development_path']) / 'training_complete.json', gate['evaluation_receipt_sha256'])
    # Formal start explicitly preceded independent numerical replay. Bind that
    # history honestly; the successful FINAL receipts above provide completion.
    prior_manifest = PROJECT / 'redesign_v0.22/results/visibility_001/completion_manifest.json'
    bound(prior_manifest, prior['manifest_sha256'])
    for path, digest in read(prior_manifest)['artifacts'].items():
        bound(PROJECT / path, digest)
    links = local_links(REPORT) + local_links(ROOT / 'README.md')

    # Files are written only after all required evidence and links pass.
    supporting = OUT / 'supporting_materials'
    snapshots = []
    for name in LIVING_INDICES:
        source, target = PROJECT / name, supporting / name
        digest = sha(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            bound(target, digest)
        else:
            shutil.copy2(source, target)
        bound(target, digest)
        snapshots.append(dict(source=str(source), source_sha256=digest,
                              snapshot=str(target.relative_to(PROJECT)), snapshot_sha256=digest))
    binding_paths = [OUT / name for name in RECEIPTS] + [REPORT, ROOT / 'README.md', ROOT / 'analyze_experience.py',
        ROOT / 'plot_results.py', ROOT / 'preflight_qa.json', ROOT / 'prior_integrity_qa.json',
        OUT / 'analysis.json', OUT / 'training_complete.json', OUT / 'private_applicability.json']
    bindings = {str(p.relative_to(PROJECT)): sha(p) for p in binding_paths}
    write(OUT / 'delivery_qa.json', dict(passed=True, report_sha256=sha(REPORT), analysis_sha256=analysis_sha,
        local_links_checked=len(links), links=links, bound_evidence=bindings, living_index_snapshots=snapshots,
        capability_passed=applicability['passed'], capability_used_to_exclude=False,
        preflight_scope=gate['scope'], final_independent_audit_passed=True,
        scope='Stored hashes, completed counts, receipt bindings and inline local-link existence; no new model or numerical analysis'))
    artifacts = {str(p.relative_to(PROJECT)): sha(p) for p in sorted(ROOT.rglob('*'))
                 if p.is_file() and p != manifest and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    for path, digest in artifacts.items():
        bound(PROJECT / path, digest)
    for path, digest in bindings.items():
        bound(PROJECT / path, digest)
    write(manifest, dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(), source_seeds=4,
        **EXPECTED_COUNTS, raw_evaluation_files=720, new_dino_inferences=0,
        main_report_sha256=sha(REPORT), analysis_sha256=analysis_sha, analysis_source_sha256=analyzer_sha,
        source_hashes=complete['source_hashes'], input_hashes=complete['input_hashes'], evidence_hashes=bindings,
        capability_passed=applicability['passed'], capability_used_to_exclude=False,
        final_audit_coverage=audit['coverage'], preflight_scope=gate['scope'],
        living_index_snapshots=snapshots, artifacts=artifacts,
        scope='v23 matched private-state experience and new communication batch complete; broader paper goal remains active',
        limitations=['New initialization sources with reused images/task family; not independent visual confirmation',
            'PrimaryB-A is private experience-distribution transfer at fixed budget, not pure knowledge or memory',
            'Target12 is private-exposed/communication-unseen in B and trained in both stages in C',
            'Private capability is a descriptive source-average check; no source or run is excluded',
            'All30 communication success can use holistic49-code conventions; no compositional-language proof']))
    print(json.dumps(dict(status='complete', artifacts=len(artifacts), local_links=len(links), report_sha256=sha(REPORT))))


if __name__ == '__main__':
    main()
