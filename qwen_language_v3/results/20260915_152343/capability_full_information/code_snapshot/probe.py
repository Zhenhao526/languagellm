"""Frozen current-step message ablation; not a full no-communication baseline."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .agents import build_prompt
from .backend import Backend
from .environment import AGENTS, RULES, World
from .run import inference_seed, write_json


def replay_action_step(backend, case, arm):
    if arm not in ('original', 'empty_text'):
        raise ValueError(arm)
    world = World.from_state_dict(case['state_before'])
    transcript = deepcopy(case['messages'])
    if arm == 'empty_text':
        for message in transcript:
            message['text'] = ''
    chosen, actions = {}, {}
    for agent in AGENTS:
        menu = case['menus'][agent]
        prompt = build_prompt(agent, rules=RULES, observation=case['observations'][agent],
            history=case['private_histories_before_step'][agent], transcript=transcript,
            condition=case['condition'], menu=menu)
        choice = int(backend.decide(prompt, mode='action', choices=[x['id'] for x in menu],
            seed=inference_seed(case['group'],case['episode'],case['step'],0,agent,'action'),
            label={'phase':'frozen_probe','condition':case['condition'],'group':case['group'],
                   'episode':case['episode'],'step':case['step'],'agent':agent,'arm':arm}))
        selected = next(x for x in menu if x['id']==choice)
        chosen[agent] = selected
        actions[agent] = selected['action']
    score_before = world.score
    feedback = world.step(actions)
    return {'arm':arm, 'actions':deepcopy(actions), 'selections':deepcopy(chosen),
        'feedback':feedback, 'score_before':score_before, 'score_after_one_step':world.score,
        'score_delta':world.score-score_before,
        'changed_from_recorded_agents':[a for a in AGENTS if actions[a]!=case['actions'][a]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--step',type=int,default=4)
    parser.add_argument('--episode',type=int,default=1)
    args = parser.parse_args()
    files = sorted(args.run_dir.glob(f'probe_pilot_*_{args.episode}_{args.step}.json'))
    if not files:
        parser.error('No matching frozen inputs; no intervention was run')
    hashes = {}
    for filename in ('agents.py','backend.py','environment.py','protocol.py','run.py','probe.py'):
        snapshot = args.run_dir/'code_snapshot'/filename
        current = Path(__file__).parent/filename
        expected = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        actual = hashlib.sha256(current.read_bytes()).hexdigest()
        if expected != actual:
            parser.error(f'Source differs from recorded run: {filename}; use that run snapshot to replay')
        hashes[filename] = actual
    run_manifest = json.loads((args.run_dir/'manifest.json').read_text())
    out_dir = args.run_dir/f'frozen_probe_ep{args.episode}_step{args.step}'
    out_dir.mkdir(exist_ok=False)
    write_json(out_dir/'manifest.json', {'case_files':[str(p) for p in files], 'arms':['original','empty_text'], 'source_sha256':hashes,
        'selection_rule':f'all pilot groups, episode {args.episode}, physical step {args.step}',
        'scope':'current-step text availability and one-step action effect; past messages and nonverbal observations retained; not a language-structure test'})
    backend = Backend(Path(run_manifest['model']),out_dir/'inference.jsonl')
    results = []
    for path in files:
        case = json.loads(path.read_text())
        row = {key:case[key] for key in ('condition','group','episode','step')}
        row['arms'] = [replay_action_step(backend,case,arm) for arm in ('original','empty_text')]
        row['original_reproduced_exactly'] = not row['arms'][0]['changed_from_recorded_agents']
        row['changed_agents'] = [a for a in AGENTS if row['arms'][0]['actions'][a]!=row['arms'][1]['actions'][a]]
        results.append(row)
        write_json(out_dir/'results.json',results)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    write_json(out_dir/'backend_stats.json',backend.stats())
    lines = ['# 冻结状态下的本步消息置空检查', '',
        '选取规则在重放前固定：各符号群体第%d段第%d步。保留身份、窗口、过去记忆和当前观察；只将本步消息文字置空，重新生成三人的动作并结算一步。测试不写回群体记忆。'%(args.episode,args.step), '',
        '|条件|群体|原始动作重现|置空后改变动作人数|原消息下一步完成度|置空下一步完成度|',
        '|---|---:|---|---:|---:|---:|']
    for row in results:
        a,b=row['arms']
        lines.append(f"|{row['condition']}|{row['group']}|{row['original_reproduced_exactly']}|{len(row['changed_agents'])}|{a['score_after_one_step']:.3f}|{b['score_after_one_step']:.3f}|")
    lines += ['', '该检查最多表明本步对话文字是否改变给定状态下的选择及一步结算。文字置空也改变长度，不能把影响归为语义。它没有移除过去消息，不能视作整段无通信基线；没有针对候选片段或新组合确认语义、组合性或修复。']
    (out_dir/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
