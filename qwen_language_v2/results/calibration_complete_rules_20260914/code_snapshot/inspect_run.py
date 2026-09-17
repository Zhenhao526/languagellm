"""Small factual action timeline, including failures; does not label semantics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def inspect(run_dir):
    path = run_dir/'steps.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    counts = Counter()
    lines = ['# 实际行动时间线', '', '只列程序记录的行为和目标完成度；不以主体广播中的自述判断动作已经发生。', '',
             '|条件|群体/段/步|A|B|C|目标完成度|', '|---|---|---|---|---|---:|']
    for row in rows:
        for a in 'ABC':
            counts[(row['condition'],row['actions'][a]['kind'])] += 1
        descriptions = [row['selections'][a]['description'].replace('|','／') +
            ('〔未成功〕' if not row['feedback'][a]['action_succeeded'] else '') for a in 'ABC']
        lines.append(f"|{row['condition']}|{row['group']}/{row['episode']}/{row['step']}|"+'|'.join(descriptions)+f"|{row['score']:.1%}|")
    lines += ['', '## 动作计数', '', '|条件|动作|次数|','|---|---|---:|']
    for (condition, kind), n in sorted(counts.items()):
        lines.append(f'|{condition}|{kind}|{n}|')
    (run_dir/'action_timeline.md').write_text('\n'.join(lines)+'\n')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run_dir',type=Path)
    args=p.parse_args()
    inspect(args.run_dir)
