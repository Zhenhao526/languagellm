"""Read-only message interventions at recorded, legal recipient histories.

Unlike a closed-loop rollout, inventory, world and personal history remain at
the normal trajectory. A donor is drawn only within goal/inventory/history/
step/direction strata. Private menu order remains fixed for the recipient;
menu equivariance is checked explicitly. No future consequences are simulated.
"""
import hashlib
import json
from pathlib import Path
import argparse
import numpy as np
import torch
from camp import remake_agents, write_json

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def score(z, ix, places):
    gathered = (z['positions'][ix] == places[:, None]).astype(np.int64)
    stored = np.minimum(z['inventory'][ix] + gathered, 2)
    return (stored[np.arange(len(ix)), z['goals'][ix]] > 0).astype(np.float64)


@torch.no_grad()
def probe(folder):
    cfg = json.loads((folder / 'config.json').read_text())
    expected = cfg['source_hashes']['camp.py']
    assert sha(ROOT/'camp.py') == expected, 'training architecture changed'
    seed, plan = cfg['seed'], cfg['plan']
    prepared = torch.load(folder.parent.parent / f'prepared_{seed}.pt', weights_only=True)
    agents = remake_agents(seed, prepared, plan['vocab'], plan['length'])
    states = torch.load(folder/'final.pt', weights_only=True)
    for a, s in zip(agents, states):
        a.load_state_dict(s)
    z = np.load(folder/'final_normal.npz')
    reports = []
    for scout in range(2):
        a = agents[1 - scout]
        for step in np.unique(z['step']):
            ix = np.flatnonzero((z['scout'] == scout) & (z['step'] == step))
            goals = torch.from_numpy(np.eye(2, dtype=np.float32)[z['goals'][ix]])
            inventory = torch.from_numpy(z['inventory'][ix].astype(np.float32))
            history = torch.from_numpy(z['history'][ix].astype(np.float32))
            menu = torch.from_numpy(z['menu'][ix])
            sent = z['delivered'][ix].copy()
            logits, _ = a.receive(torch.from_numpy(sent), goals, inventory, history, menu)
            native_places = menu.numpy()[np.arange(len(ix)), logits.argmax(-1).numpy()]
            assert np.array_equal(native_places, z['place'][ix])
            assert np.array_equal(score(z, ix, native_places), z['reward'][ix])
            identity = torch.arange(4).repeat(len(ix), 1)
            canonical, _ = a.receive(torch.from_numpy(sent), goals, inventory, history, identity)
            assert torch.allclose(logits, canonical.gather(1, menu), atol=1e-7, rtol=0)
            keys = np.column_stack((z['goals'][ix], z['inventory'][ix], z['history'][ix]))
            _, inverse = np.unique(keys, axis=0, return_inverse=True)
            donor = np.arange(len(ix))
            eligible = np.zeros(len(ix), dtype=bool)
            rng = np.random.default_rng(seed + 651000 + scout * 37 + int(step))
            for group in np.unique(inverse):
                rows = np.flatnonzero(inverse == group)
                eligible[rows] = len(rows) > 1
                donor[rows] = rng.permutation(rows)
            assert np.array_equal(keys, keys[donor])
            altered = sent[donor]
            changed, _ = a.receive(torch.from_numpy(altered), goals, inventory, history, menu)
            places = menu.numpy()[np.arange(len(ix)), changed.argmax(-1).numpy()]
            native_reward, other_reward = score(z, ix, native_places), score(z, ix, places)
            reports.append(dict(scout=scout, step=int(step), cases=len(ix),
                donor_pool_at_least_two=int(eligible.sum()),
                message_changed=int((altered != sent).any(1).sum()),
                action_changed=int((places != native_places).sum()),
                native_reward_sum=float(native_reward.sum()),
                intervened_reward_sum=float(other_reward.sum()),
                one_step_reward_drop=float((native_reward-other_reward).mean()),
                native_mean=float(native_reward.mean()), intervened_mean=float(other_reward.mean()),
                changed_message_action_changes=int(((altered != sent).any(1) & (places != native_places)).sum())))
    n = sum(r['cases'] for r in reports)
    return dict(seed=seed, condition=cfg['condition'], stage=folder.parent.name,
        cases=n, rows=reports,
        normal_mean=sum(r['native_reward_sum'] for r in reports)/n,
        intervened_mean=sum(r['intervened_reward_sum'] for r in reports)/n,
        current_message_reward_drop=sum(r['cases']*r['one_step_reward_drop'] for r in reports)/n,
        checkpoint_sha256=sha(folder/'final.pt'), trace_sha256=sha(folder/'final_normal.npz'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=ROOT/'results/progression_001')
    args=parser.parse_args()
    torch.set_num_threads(1)
    paths=sorted(list(args.input.glob('B/*/result.json'))+list(args.input.glob('C/*/result.json')))
    rows=[probe(p.parent) for p in paths]
    result=dict(status='complete' if len(rows)==27 else 'incomplete',
        runs=len(rows), cases_checked=sum(r['cases'] for r in rows),
        method='conditional donor-message permutation at fixed legal normal-trace recipient context; immediate reward only',
        limitations='Singleton strata remain unchanged; subgroup permutation may draw same message. Context is supplied state plus two actual collection events. Not a linguistic structure score or full-return treatment.',
        source_sha256=sha(Path(__file__)), rows=rows)
    write_json(args.input/'fixed_context_probe.json', result)
    lines=['# 固定当前情境的消息干预', '',
        '在正常轨迹上保持世界、当前需求、库存、自己的两次采集历史和菜单不变。供体消息只在相同方向、步数、需求、库存、历史内重排，不模拟后续环境。单例层及重复消息保留，所以该干预不保证完全移除信息。', '',
        '| 阶段 | 条件 | 种子 | 正常即时收益 | 替换后即时收益 | 差（百分点） |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['stage']} | {r['condition']} | {r['seed']} | {100*r['normal_mean']:.2f}% | {100*r['intervened_mean']:.2f}% | {100*r['current_message_reward_drop']:.2f} |")
    lines += ['',f"已检查{result['cases_checked']:,}条正常轨迹的模型动作与独立重算收益；运行数{len(rows)}/27。",
              '这项检验支持当前消息的行为作用，不能单独证明词义、组合性或长期总收益的因果效应。']
    (args.input/'fixed_context_probe.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('status','runs','cases_checked')}))


if __name__=='__main__':
    main()
