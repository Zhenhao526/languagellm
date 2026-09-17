"""Audit saved coordinate-recoding references without inference or rescoring.

Reads protocol_analysis.json only as experimental input. Never imports a model,
draws random numbers, edits protocol caches, or evaluates a policy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(row):
    return f"s{row['training_seed']}_{row['condition']} {row['scout']}→{row['collector']}"


def audit(path):
    source = json.loads(path.read_text())
    assert source['status'] == 'complete' and not source['missing_runs']
    rows, groups, sequence_by_seed = [], defaultdict(list), {}
    counts = defaultdict(int)
    for run in source['runs']:
        assert len(run['directions']) == 2
        counts[run['seed']] += 1
        for direction in run['directions']:
            assert direction['collector'] == 1 - direction['scout']
            reference = direction['whole_message_recoding']
            records = reference['records']
            assert reference['replicates'] == len(records) == 100
            assert reference['full_message_capacity'] == 49
            assert [item['replicate'] for item in records] == list(range(100))
            permutations = [item['old_to_new_code'] for item in records]
            assert all(sorted(p) == list(range(49)) for p in permutations)
            assert len({tuple(p) for p in permutations}) == 100
            sequence = json.dumps(permutations, separators=(',', ':'))
            sequence_sha = hashlib.sha256(sequence.encode()).hexdigest()
            rng_seed = reference['rng_seed']
            if rng_seed in sequence_by_seed:
                assert sequence_by_seed[rng_seed] == sequence
            else:
                sequence_by_seed[rng_seed] = sequence
            row = dict(training_seed=run['seed'], condition=run['condition'],
                       split=run['plan']['split'], scout=direction['scout'],
                       collector=direction['collector'], reference_rng_seed=rng_seed,
                       saved_reference_scores=len(records), distinct_permutations_within_direction=100,
                       permutation_sequence_sha256=sequence_sha)
            rows.append(row)
            groups[rng_seed].append(row)
    assert len(source['runs']) == source['expected_run_count'] == 60
    assert sorted(counts) == [27101, 27102, 27103, 27104]
    assert all(n == 15 for n in counts.values())
    assert len(rows) == 120
    main = [row for row in rows if row['split'] > 0]
    assert len(main) == 72 and len({row['reference_rng_seed'] for row in main}) == 72
    shared = [dict(reference_rng_seed=seed, member_count=len(members), members=members,
                   identical_saved_permutation_sequence=True)
              for seed, members in sorted(groups.items()) if len(members) > 1]
    assert len(groups) == 90 and len(shared) == 30
    assert all(group['member_count'] == 2 for group in shared)
    assert all(sum(member['split'] > 0 for member in group['members']) == 1 for group in shared)
    assert all(len({member['training_seed'] for member in group['members']}) == 2 for group in shared)
    return dict(
        status='passed_with_documented_reference_sequence_reuse',
        audit_scope='Saved reference RNG identifiers and permutations only; no model inference, rescoring, training, or random draws',
        source_protocol=str(path.resolve()), source_protocol_sha256=digest(path),
        audit_script=str(Path(__file__).resolve()), audit_script_sha256=digest(__file__),
        summary=dict(training_runs=60, training_seeds=sorted(counts),
                     protocol_directions=120, saved_conditional_reference_scores=12000,
                     distinct_reference_rng_seeds=90, reference_scores_per_direction=100,
                     main_split_directions=72, main_split_distinct_reference_rng_seeds=72,
                     shared_reference_seed_groups=30, directions_in_shared_groups=60,
                     remaining_unshared_directions=60,
                     shared_groups_have_identical_saved_permutations=True),
        interpretation=[
            'Each direction retains 100 valid distinct 49-code bijections and its original protocol-specific scores.',
            'Some full or blocked runs and a neighboring training seed main run use the same coordinate-recoding sequence.',
            'No two main split directions share a reference RNG seed. A main direction can still share with a full or blocked direction.',
            'Repeated reference coordinates do not make separately trained agents identical or invalidate natural-message results.',
            'Reference score counts are not counts of independent agents or independent inferential samples.',
            'The four training seeds are the replication units; splits and directions are within-seed repeated measurements.',
            'The original fixed RNG formula and all stored scores and caches are unchanged.',
        ],
        directions=rows, shared_reference_groups=shared)


def markdown(result):
    s = result['summary']
    lines = ['# 完整码重编码参考RNG审计', '',
        '本记录在主训练和原协议分析完成后，对已保存的参考种子、完整码双射及记录数作只读复核。没有加载模型、进行推理、生成新随机数或重算参考评分；原冻结RNG规则、协议计算和缓存均保持原样。', '',
        '| 项目 | 数量 |', '| --- | ---: |',
        f"| 正式训练运行 | {s['training_runs']} |",
        f"| 协议方向 | {s['protocol_directions']} |",
        f"| 已保存的条件化参考评分 | {s['saved_conditional_reference_scores']} |",
        f"| 每方向有效双射及评分 | {s['reference_scores_per_direction']} |",
        f"| 不同参考RNG种子 | {s['distinct_reference_rng_seeds']} |",
        f"| 主划分方向 / 主划分内不同参考种子 | {s['main_split_directions']} / {s['main_split_distinct_reference_rng_seeds']} |",
        f"| 共享参考种子组 | {s['shared_reference_seed_groups']} |", '',
        '原规则采用 `910000000 + training_seed*100 + condition_index*10 + scout`。条件数为15时，部分全图或阻断条件的索引偏移超过相邻训练种子的间隔，因此会与下一训练种子的部分主条件复用参考随机序列。下表列出全部30组，每组恰两个方向；逐条核验100个已存49码双射序列完全一致。', '',
        '**主划分72方向彼此没有参考种子碰撞**；其中部分仍与另一个训练种子的全图或阻断方向共享序列。每个协议方向内部保留100个不同且有效的完整码双射，原条件化评分仍然有效。相同重编码作用在不同的已学协议上，并不要求其评分相同。', '',
        '这项复用影响的是参考重命名序列之间的依赖关系，不表示主训练主体使用了同一个训练随机种子，也不改变自然消息成绩。报告应称“12,000次条件化参考评分”，不能称“12,000个独立实验样本”。四个训练种子27101–27104仍是重复单位；划分、方向、菜单和参考重编码都不能增加训练主体样本量。', '',
        '| 参考RNG种子 | 第一个协议方向 | 第二个协议方向 | 已存序列核对 |',
        '| ---: | --- | --- | --- |']
    for group in result['shared_reference_groups']:
        first, second = group['members']
        lines.append(f"| {group['reference_rng_seed']} | {label(first)} | {label(second)} | 100/100一致 |")
    lines += ['', f"来源：[protocol_analysis.json]({result['source_protocol']})；SHA-256：`{result['source_protocol_sha256']}`。", '',
        f"复现脚本：[audit_reference_rng.py]({result['audit_script']})；SHA-256：`{result['audit_script_sha256']}`。", '',
        'JSON保留全部120个方向的参考种子、序列哈希和共享组成员，便于逐项复核。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', type=Path, nargs='?',
                        default=ROOT/'results/complementarity_001/protocol_analysis.json')
    args = parser.parse_args()
    path = args.protocol.resolve()
    before = digest(path)
    result = audit(path)
    assert digest(path) == before
    (path.parent/'reference_rng_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (path.parent/'reference_rng_audit.md').write_text(markdown(result))
    print(json.dumps(dict(status=result['status'], **result['summary'],
                          output=str(path.parent/'reference_rng_audit.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
