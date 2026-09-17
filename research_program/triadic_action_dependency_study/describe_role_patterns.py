"""Post-hoc description of complete natural endpoint joint-action histograms.

Reads only execution/results.json as experimental data. No checkpoints, NPZs,
monitors, scores, trained networks or task performance fields are inspected.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
from pathlib import Path
import argparse
import json
import time
from . import environment as env

PARTITIONS = ('train', 'new_needs', 'new_layouts', 'new_needs_and_layouts')
CONDITIONS = ('FI_silent', 'FI_live', 'PL_silent', 'PL_live', 'LL_silent', 'LL_live')
SEEDS = (51101, 51102, 51103, 51104)
MENUS = [env.all_actions(a) for a in env.AGENTS]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def signature(indices):
    # Read complete public action semantics, not any policy success score.
    actions = [MENUS[a][int(k)] for a, k in enumerate(indices)]
    return tuple(-1 if action['kind'] == 'wait' else env.AGENTS.index(action['partner'])
                 for action in actions)


def describe(counter):
    roles = Counter()
    agents = {a: dict(wait=0, partners=Counter(), sites=Counter(), destinations=Counter(), actions=Counter())
              for a in env.AGENTS}
    for indices, count in counter.items():
        roles[signature(indices)] += count
        for who, key in enumerate(env.AGENTS):
            action = MENUS[who][indices[who]]; row = agents[key]
            row['actions'][str(indices[who])] += count
            if action['kind'] == 'wait':
                row['wait'] += count
            else:
                row['partners'][action['partner']] += count
                row['sites'][action['site']] += count
                row['destinations'][action['destination']] += count
    n = sum(counter.values())
    for a, row in agents.items():
        row['all_worlds_wait'] = row['wait'] == n
        row['always_transport'] = row['wait'] == 0
        row['one_partner_whenever_transporting'] = len(row['partners']) == 1
        row['distinct_action_count'] = len(row['actions'])
        require(row['wait'] + sum(row['partners'].values()) == n, 'Actor proposal counts lost worlds')
    pattern_rows = []
    for sig, count in sorted(roles.items()):
        proposal = {a: 'wait' if p == -1 else 'partner_' + env.AGENTS[p]
                    for a, p in zip(env.AGENTS, sig)}
        active = [i for i, p in enumerate(sig) if p != -1]
        reciprocal = len(active) == 2 and sig[active[0]] == active[1] and sig[active[1]] == active[0]
        pattern_rows.append(dict(signature=list(sig), agent_proposals=proposal, worlds=count,
            active_agents=[env.AGENTS[i] for i in active], mutually_proposed_pair=[env.AGENTS[i] for i in active] if reciprocal else None))
    return dict(worlds=n, distinct_joint_action_count=len(counter), distinct_role_pattern_count=len(roles),
        globally_constant_role_pattern=len(roles) == 1, role_patterns=pattern_rows,
        agent_counts=agents,
        joint_action_counts=[dict(action_indices=list(a), worlds=c) for a, c in sorted(counter.items())])


def extract(results):
    require(results['status'] == 'completed' and results['completed_run_count'] == 24,
            'All 24 runs must have completed')
    runs=results['runs']
    require(len(runs) == 24 and [(r['seed'],r['condition']) for r in runs] == list(product(SEEDS,CONDITIONS)),
            'Wrong or incomplete seed-condition grid')
    rows=[]; consumed=0
    for run in runs:
        require(run['updates'] == 6000 and set(run['final']) == set(PARTITIONS), 'Wrong endpoint scope')
        merged=Counter(); by_part={}; raw_hashes={}
        for part in PARTITIONS:
            cell=run['final'][part]['natural']
            source_rows=cell['raw_joint_action_counts']
            counter=Counter()
            for row in source_rows:
                actions=row['action_indices']; count=row['worlds']
                require(len(actions) == 3 and all(type(k) is int and 0 <= k < 17 for k in actions), 'Invalid action')
                require(type(count) is int and count > 0, 'Invalid histogram count')
                key=tuple(actions)
                require(key not in counter, 'Duplicate raw joint-action bin')
                counter[key]=count
            require(sum(counter.values()) == cell['worlds'] and len(counter) > 0, 'Histogram denominator differs')
            by_part[part]=describe(counter)
            merged.update(counter); consumed+=1
            raw_hashes[part]=sha256(json.dumps(source_rows,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        require(sum(merged.values()) == 774144, 'Four-part global world count differs')
        combined=describe(merged)
        rows.append(dict(seed=run['seed'],condition=run['condition'],updates=6000,
                         natural_endpoint_only=True,raw_histogram_sha256=raw_hashes,
                         full_domain=combined,partitions=by_part))
    require(consumed == 96, 'Must use all 4×24 natural endpoint histograms')
    return dict(policy_rows=rows,natural_endpoint_histograms=consumed,
        policies_with_one_global_role_pattern=sum(r['full_domain']['globally_constant_role_pattern'] for r in rows),
        policies_with_multiple_global_role_patterns=sum(not r['full_domain']['globally_constant_role_pattern'] for r in rows),
        all_policy_world_occurrences=sum(r['full_domain']['worlds'] for r in rows),
        interpretation='Observed complete-domain action proposals, not roles inferred from a role-success rate. Reciprocal partner proposals do not guarantee matched sites/destinations, native execution or need satisfaction. Constant roles do not imply constant full actions or no state-dependent internal probabilities.')


def execute(results_path, out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        source=Path(results_path).resolve(); source_sha=sha(source)
        value=extract(json.loads(source.read_text()))
        require(sha(source) == source_sha,'Input changed during description')
        report=dict(schema='action_dependency_global_role_description_v1',status='completed_read_only_description',
            recorded_at=datetime.now(timezone.utc).isoformat(),
            source_results_path=str(source),source_results_sha256=source_sha,
            source_script_sha256=sha(__file__),action_semantics_source_sha256=sha(env.__file__),
            elapsed_seconds=time.perf_counter()-started,model_forwards=0,parameter_loads=0,training_updates=0,
            post_hoc=True,source_scope='Only all24completed execution/results.json, natural final raw_joint_action_counts, and grid/world-count metadata; no monitor or closed results, role scores, trained arrays or weights.',**value)
        p=out/'role_patterns.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        lines=['# 完整自然终点的运输角色与搭档描述','',
            '这是全部24组完成后增加的只读行为描述，依据原始终点联合动作频数，不依据角色成功率判断。读取四分区×24政策的全部96张自然终点表；没有前向、训练或新增策略。','',
            f"全域每政策 774,144 个状态；{report['policies_with_one_global_role_pattern']}/24 个政策只出现一种运输／等待及伙伴提议模式，{report['policies_with_multiple_global_role_patterns']}/24 个出现多种。详细每分区与逐主体频数保存在 [role_patterns.json](role_patterns.json)。",'',
            '| 种子 | 条件 | 全域实际角色模式 | 该模式计数 | 不同完整联合动作数 |',
            '|---|---|---|---:|---:|']
        for r in report['policy_rows']:
            f=r['full_domain']
            pattern='；'.join(', '.join(a+('等待' if p=='wait' else '→'+p.removeprefix('partner_')) for a,p in q['agent_proposals'].items()) for q in f['role_patterns'])
            counts='/'.join(str(q['worlds']) for q in f['role_patterns'])
            lines.append(f"| {r['seed']} | {r['condition']} | {pattern} | {counts} | {f['distinct_joint_action_count']} |")
        lines += ['',
            '“A→B”表示 A 在正式动作中提议 B 为搭档，而非只统计成功执行者。等待计数、所有伙伴提议计数、运输位置／目的地计数与完整联合动作频数均按全部状态累计，因此未被失败筛选。','',
            '固定互选伙伴不代表两人选中了相同位置或目的地，更不代表实际运输和需求满足；若完整联合动作数大于1，也不能称整套部署动作恒定。反过来，角色提议恒定仍允许位置、目的地以及隐藏表示或动作概率随情境变化。','',
            '不同种子或条件采用不同搭档，不能合并成一个社群具备切换搭档的能力。本记录没有检验消息语义、组合性或导致固定角色的内部机制。','',
            f"来源 results.json SHA：`{source_sha}`。描述 JSON SHA：`{sha(p)}`。"]
        (out/'说明.md').write_text('\n'.join(lines)+'\n')
        (out/'receipt.json').write_text(json.dumps(dict(status='passed',source_results_sha256=source_sha,
            output_sha256={k:sha(out/k) for k in ('role_patterns.json','说明.md')},model_forwards=0,training_updates=0),ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(status='completed',output=str(out),policies_constant=report['policies_with_one_global_role_pattern'],
            total=24,counts=[dict(seed=r['seed'],condition=r['condition'],patterns=r['full_domain']['role_patterns'],
                distinct_joint_actions=r['full_domain']['distinct_joint_action_count']) for r in report['policy_rows']],
            json_sha256=sha(p)),ensure_ascii=False))
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps(dict(status='failed',error=repr(error)),ensure_ascii=False,indent=2)+'\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();execute(args.results,args.out)
