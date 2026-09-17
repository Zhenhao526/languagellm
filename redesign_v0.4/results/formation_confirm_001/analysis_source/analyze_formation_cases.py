"""Post-run, read-only descriptions of a retained regression and individual codes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ['两份食物', '食物与水', '两份水']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', type=Path, default=ROOT/'results/formation_confirm_001')
    args = parser.parse_args()
    folder = args.directory
    run = folder/'course_communication_s1209'
    curve = read(run/'learning_curve.json')
    final = read(run/'result.json')
    result = {'analysis_status': 'descriptive follow-up selected after endpoint review; no new training or primary hypothesis',
              'regression_seed': 1209, 'curve': [], 'failure_contexts': {}, 'raw_code_example_seed': 1202,
              'source_sha256': {str(run/'learning_curve.json'): sha(run/'learning_curve.json')}}
    for r in curve:
        result['curve'].append({'update': r['update'],
                                'course_normal': r['curriculum']['normal']['mean_reward_per_step'],
                                'full_normal': r['full']['normal']['mean_reward_per_step'],
                                'full_stochastic': r['full']['stochastic']['mean_reward_per_step']})
    for task in ('curriculum', 'full'):
        trace = run/f'final_{task}_normal_trace.jsonl'
        total, failures, examples = Counter(), Counter(), {}
        one_restricted_failures = both_mixed_failures = 0
        for line in trace.open():
            r = json.loads(line)
            local = tuple(sum(kinds) for kinds in r['kinds'])
            total[local] += 1
            if not r['success']:
                failures[local] += 1
                key = f'{local[0]},{local[1]}'
                examples.setdefault(key, [])
                if len(examples[key]) < 3:
                    examples[key].append(r)
                if sum(v == 1 for v in local) == 1:
                    one_restricted_failures += 1
                elif local == (1, 1):
                    both_mixed_failures += 1
        result['source_sha256'][str(trace)] = sha(trace)
        result['failure_contexts'][task] = {
            'cases': sum(total.values()), 'failures': sum(failures.values()),
            'one_restricted_failures': one_restricted_failures,
            'both_mixed_failures': both_mixed_failures,
            'contexts': [{'agent0_private_category': a, 'agent1_private_category': b,
                           'cases': total[a,b], 'failures': failures[a,b],
                           'failure_rate': failures[a,b]/total[a,b]} for a,b in sorted(total)],
            'representative_failed_rows': examples,
        }
    epath = folder/'course_communication_s1202/result.json'
    example = read(epath)
    result['source_sha256'][str(epath)] = sha(epath)
    counts = example['full']['normal']['symbols_by_local_resource_set']
    result['raw_code_example'] = {'full_success': example['full']['normal']['mean_reward_per_step'],
                                  'raw_sender_counts_by_local_category': counts,
                                  'interpretation': 'different sender-specific raw IDs with successful reciprocal action; no forced semantic translation or claim of one shared lexicon'}
    (folder/'formation_confirm_cases.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 形成确认：过程后退与个人编码案例', '',
             '这是查看终点后选择的描述性案例补充，不是新增主检验；未更新任何参数。', '',
             '## 1209：完整任务提高，课程表现后退', '',
             '| 检查点 | 课程正常 | 完整任务正常 | 完整任务按策略采样 |',
             '| --- | ---: | ---: | ---: |']
    for r in result['curve']:
        if r['update'] >= 600:
            lines.append(f"| {r['update']} | {r['course_normal']:.2%} | {r['full_normal']:.2%} | {r['full_stochastic']:.2%} |")
    lines += ['', '1209在第650次检查点首次通过整套工程门槛；到750次，课程成功率由98.34%降至87.50%，'
              '随后各记录保持87.50%，完整任务同时从86.47%升至91.06%并继续小幅提高。'
              '这些曲线点使用相同外部案例，变化不是更换评估样本造成的；不能只报终点9/10通过而遗漏先过线再后退。', '',
              f"独立终点评估每任务8,192个案例。课程正常为{final['curriculum']['normal']['mean_reward_per_step']:.2%}，"
              f"完整正常为{final['full']['normal']['mean_reward_per_step']:.2%}。"
              '完整任务两个受限发送方向仍分别有36.46和37.27个百分点的正常−打乱落差。'
              '因此它不是未建立有用通信，未过的条目是课程90%工程标准。', '',
              '| 终点课程失败情境 | 该情境案例数 | 失败数 |', '| --- | ---: | ---: |']
    for r in result['failure_contexts']['curriculum']['contexts']:
        if r['failures']:
            lines.append(f"| A：{LABELS[r['agent0_private_category']]}；B：{LABELS[r['agent1_private_category']]} | {r['cases']} | {r['failures']} |")
    fs = result['failure_contexts']['full']
    lines += ['', f"课程共996次失败，全部发生于上面两个情境；其余两类一方受限情境无失败。"
              f"完整任务共{fs['failures']}次失败，其中{fs['one_restricted_failures']}次发生于一方受限、另一方可选，"
              f"{fs['both_mixed_failures']}次发生于双方都可选。"
              '这显示任务分布切换后错误重新分布，但单条轨迹不能证明特定认知机制或不可避免的协调代价。', '',
              '## 1202：成功互相配合，原始编号并不一致', '',
              '终点完整任务正常成功率为100%。下表来自同一终点的全部8,192个完整任务案例；'
              '每个主体在各私人资源集合下都只有一个最大概率编号，未进行编号置换。', '',
              '| 主体 | 两份食物 | 食物与水 | 两份水 |', '| --- | --- | --- | --- |']
    for i, table in enumerate(counts):
        cells = []
        for row in table:
            nonzero = [(symbol, n) for symbol,n in enumerate(row) if n]
            assert len(nonzero) == 1
            symbol,n = nonzero[0]
            cells.append(f'{symbol}（{n}次）')
        lines.append(f"| {'AB'[i]} | " + ' | '.join(cells) + ' |')
    lines += ['', 'A在只见食物时发4，B在只见食物时发1；A只见水时发3，B只见水时发0。'
              '双方可通过各自的发送和接收规则完成任务，不要求对应私人资源时发送相同编号。'
              '这支持区分“功能性互相理解”与“统一原始编号”；尚不能给每个编号指定唯一词义，'
              '混合资源下的编号也可能与行动安排有关。', '',
              '1202准备权重与旧伙伴批次C/D重复的来源限制及九种子敏感性已另行披露；'
              '这里是已保留主结果中的行为例子，不把它计作额外独立群体。', '',
              '[过程、全部失败情境、代表性原始案例及源文件哈希](' + str(folder.resolve()/'formation_confirm_cases.json') + ')', '']
    (folder/'形成确认_过程与编号案例.md').write_text('\n'.join(lines))
    (folder/'analysis_source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    provenance = read(folder/'analysis_source/source_hashes.json')
    provenance[Path(__file__).name] = sha(__file__)
    (folder/'analysis_source/source_hashes.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'failures': result['failure_contexts'], 'code': result['raw_code_example']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
