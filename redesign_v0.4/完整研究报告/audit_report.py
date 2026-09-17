"""Read-only verification of primary summaries and the assembled report."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(6000914).integers(len(values), size=(10000, len(values)))
    return float(values.mean()), np.quantile(values[indices].mean(1), [.025, .975])


def audit(with_docx=False):
    completion = {name: read(RESULTS / name / 'completed.json') for name in (
        'formation_confirm_001', 'mechanism_origins_001', 'mechanism_001')}
    for name, expected in [('formation_confirm_001', 40), ('mechanism_origins_001', 7), ('mechanism_001', 47)]:
        assert completion[name]['status'] == 'completed' and completion[name]['runs'] == expected
    audit_paths = [RESULTS / 'formation_confirm_001/formation_confirm_audit.json',
                   RESULTS / 'mechanism_origins_001/独立新起点通信核查.json',
                   RESULTS / 'mechanism_001/机制训练独立核查.json',
                   RESULTS / 'mechanism_001/独立机制通信核查.json',
                   RESULTS / 'mechanism_001/固定函数上界独立核查.json']
    assert all(read(path)['status'] == 'passed' for path in audit_paths)
    source = HERE / '研究报告.md'
    text = source.read_text()
    assert '{{' not in text and '}}' not in text, 'Unfilled report placeholder'
    images = [Path(path) for path in re.findall(r'!\[[^\]]*\]\(([^\n]+)\)', text)]
    assert images and all(path.is_absolute() and path.is_file() for path in images)
    captions = [int(x) for x in re.findall(r'^图(\d+)\s', text, re.M)]
    assert captions == list(range(1, len(images) + 1)), captions
    bibliography = re.findall(r'^\[R(\d+)\]\s', text, re.M)
    citations = set(re.findall(r'\[R(\d+)\]\(', text))
    assert bibliography == [str(i) for i in range(1, 13)] and citations == set(bibliography)
    raw = read(RESULTS / 'formation_confirm_001/results.json')
    runs = {(r['condition'], r['seed']): r for r in raw}
    summary = read(RESULTS / 'formation_confirm_001/formation_confirm_summary.json')
    primary = {}
    for other, label in [('direct_communication', 'course_minus_direct_full_success'),
                         ('course_blocked', 'course_minus_blocked_native_full_success')]:
        delta = [runs['course_communication', s]['native_task']['full']['greedy_success'] -
                 runs[other, s]['native_task']['full']['greedy_success'] for s in range(1201, 1211)]
        mean, ci = paired(delta)
        reported = summary['primary_comparisons'][label]
        assert np.isclose(mean, reported['mean'], atol=1e-12, rtol=0)
        assert np.allclose(ci, reported['paired_bootstrap_percentile_95_interval'], atol=1e-12, rtol=0)
        assert f'{100*mean:.2f}个百分点' in text
        primary[label] = {'mean_pp': mean*100, 'ci95_pp': (ci*100).tolist()}
    conditions = ['behavior_frozen', 'sender_only', 'listener_only', 'both_learn', 'all_plastic']
    seeds = [404, 505, 606, 707, 808, 909, 1001]
    mechanism = read(RESULTS / 'mechanism_001/mechanism_summary.json')['cohorts']['confirmatory']
    values = {}
    for condition in conditions:
        values[condition] = np.array([read(RESULTS / 'mechanism_001' / f'{condition}_s{s}' / 'result.json')
                                     ['evaluation']['aggregates']['cross']['tasks']['full']['normal']['mean_reward_per_step']
                                     for s in seeds])
        assert np.allclose(values[condition], mechanism['conditions'][condition]['cross']['normal']['population_values'], atol=1e-12, rtol=0)
    formulas = {'listener_minus_frozen': {'listener_only': 1, 'behavior_frozen': -1},
                'sender_minus_frozen': {'sender_only': 1, 'behavior_frozen': -1},
                'both_minus_listener': {'both_learn': 1, 'listener_only': -1},
                'both_minus_sender': {'both_learn': 1, 'sender_only': -1},
                'interaction': {'both_learn': 1, 'sender_only': -1, 'listener_only': -1, 'behavior_frozen': 1},
                'projection_plasticity': {'all_plastic': 1, 'both_learn': -1}}
    contrasts = {}
    for name, formula in formulas.items():
        mean, ci = paired(sum(values[c]*w for c, w in formula.items()))
        reported = mechanism['paired_contrasts'][name]['cross']['normal']
        assert np.isclose(mean, reported['mean'], atol=1e-12, rtol=0)
        assert np.allclose(ci, reported['paired_seed_bootstrap_95_percentile_interval'], atol=1e-12, rtol=0)
        assert f'{100*mean:+.2f}' in text, f'{name} absent from complete contrast table'
        contrasts[name] = {'mean_pp': mean*100, 'ci95_pp': (ci*100).tolist()}
    record = {'status': 'passed', 'new_formal_runs': 94, 'completion': completion,
              'passed_independent_audits': {str(path): sha(path) for path in audit_paths},
              'formation_primary_independently_recomputed': primary,
              'mechanism_confirmatory_cross_contrasts_independently_recomputed': contrasts,
              'source_sha256': sha(source), 'figures': {str(p): sha(p) for p in images},
              'reference_count': len(bibliography),
              'scope': 'Summary arithmetic, report completeness and artifact correspondence; does not replace the independent trajectory audits or scientific peer review.'}
    if with_docx:
        docx = HERE / '预训练视觉主体中符号约定的形成与适应.docx'
        build = read(docx.with_suffix('.build.json'))
        assert build['source_sha256'] == sha(source)
        assert build['output_sha256'] == sha(docx)
        record['docx_sha256'] = sha(docx)
    (HERE / '报告数据与交付核查.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': record['status'], 'runs': 94, 'figures': len(images), 'with_docx': with_docx}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-docx', action='store_true')
    audit(parser.parse_args().with_docx)
