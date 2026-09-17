"""Compact exact values from a completed summary; no recomputation or filtering."""
from pathlib import Path
import argparse
import hashlib
import json

KEYS = ('target_apt', 'current_apt', 'natural_current_apt', 'current_minus_natural_apt',
 'conservative_gate', 'conservative_target_apt', 'target_probability',
 'donor_same_correct', 'donor_opposite_correct', 'copy_donor_same_target_apt',
 'copy_donor_opposite_target_apt', 'action_equals_donor_same', 'action_equals_donor_opposite',
 'observation_equal_donor', 'input_equal_donor_same', 'input_equal_donor_opposite',
 'native_reward', 'native_team_full', 'counterfactual_reward', 'counterfactual_team_full',
 'listener_action_changed', 'any_generated_message_changed', 'listener_own_second_message_changed')

def extract(path, out):
    path, out = Path(path).resolve(), Path(out).resolve()
    assert not out.exists()
    summary = json.loads(path.read_text())
    assert summary['status'] == 'completed_read_only_summary'
    rows = []
    for policy in summary['policies']:
        for part, cell in policy['partitions'].items():
            for support in ('eligible', 'all_other'):
                value = cell[support]
                rows.append(dict(seed=policy['seed'], condition=policy['condition'], partition=part,
                    support=support, T=value['contrast']['macro']['target_apt_gain'],
                    C_difference=value['contrast']['macro']['conservative_target_apt_gain'],
                    T_by_axis={a: r['values']['target_apt_gain'] for a, r in value['contrast']['by_axis'].items()},
                    same={k: value['remote_same_both']['macro'][k] for k in KEYS},
                    opposite={k: value['remote_opposite_both']['macro'][k] for k in KEYS}))
    conditions = ('PL_silent', 'PL_live', 'LL_silent', 'LL_live')
    parts = ('train', 'new_needs', 'new_layouts', 'new_needs_and_layouts')
    means = {}
    for part in parts:
        means[part] = {}
        for support in ('eligible', 'all_other'):
            means[part][support] = {}
            for condition in conditions:
                selected = [r for r in rows if (r['partition'], r['support'], r['condition']) == (part, support, condition)]
                assert len(selected) == 4
                means[part][support][condition] = dict(
                    T=sum(r['T'] for r in selected)/4,
                    C_difference=sum(r['C_difference'] for r in selected)/4,
                    T_by_axis={a: sum(r['T_by_axis'][a] for r in selected)/4 for a in ('kind', 'length', 'destination')},
                    **{arm: {k: sum(r[arm][k] for r in selected)/4 for k in KEYS} for arm in ('same','opposite')})
    result = dict(primary=summary['primary_comparison']['primary'],
        all_paired_comparisons=summary['primary_comparison']['all_partitions'],
        rows=rows, equal_seed_means=means, source_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        note='Exact unrounded reported rates; four seeds equal. This extracts finished official summaries and is not an independent numerical audit.')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['primary'], ensure_ascii=False))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summary',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();extract(args.summary,args.out)
