"""Read-only audit of a completed 144-call Qwen shared-plan diagnostic.

Only local tokenizer assets are loaded. Frozen experiment inputs, histories and
model weights are not modified. A failed exact replay remains an auditable
completed diagnostic, but its paired intervention effect is not interpretable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations, product
import json
import math
from pathlib import Path
import re
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK))
from research_program.triadic_task.audit_qwen_capability_results import check, read, lines, sha, new_json, stage_requests

ARMS = ('original_replay', 'shared_plan_added')
REPRO_FIELDS = ('messages', 'output', 'prompt_sha256', 'seed', 'temperature', 'mode',
                'valid', 'fresh_cache', 'native_thinking', 'finish_reason', 'prompt_tokens', 'generation_tokens')
PRIOR = Path(__file__).resolve().parent / 'EXEC_01_prior_state_receipt.json'


def no_model_runtime():
    return all(name not in sys.modules for name in ('mlx', 'mlx.core', 'mlx_lm', 'torch'))


def independent_plan(env, state):
    # Fixed lexical enumeration, independent of the environment witness helper.
    for i, j in combinations(range(3), 2):
        for site, destination in product(env.SITES, env.DESTINATIONS):
            actions = {a: {'kind': 'wait'} for a in env.AGENTS}
            actions[env.AGENTS[i]] = {'kind': 'transport', 'partner': env.AGENTS[j],
                                     'site': site, 'destination': destination}
            actions[env.AGENTS[j]] = {'kind': 'transport', 'partner': env.AGENTS[i],
                                     'site': site, 'destination': destination}
            if env.settle(state, actions, require_match=True)['reward'] == 1:
                return actions
    raise AssertionError('No full-success reference plan')


def audit(run_dir, prior_receipt=PRIOR):
    run_dir, prior_receipt = Path(run_dir).resolve(), Path(prior_receipt).resolve()
    execution = run_dir / 'execution'
    check(no_model_runtime(), 'Audit must start without a model runtime')
    status = read(execution / 'status.json')
    check(status['status'] == 'completed', 'Audit requires a complete real execution')
    plan, freeze, prepared = (read(run_dir / f) for f in ('plan.json', 'freeze.json', 'prepared_cases.json'))
    check(sha(run_dir / 'plan.json') == freeze['plan_sha256'] and
          sha(run_dir / 'prepared_cases.json') == plan['prepared_cases_sha256'], 'Frozen pack hashes differ')
    for path, expected in plan['sources_sha256'].items():
        check(sha(path) == expected, 'Frozen source or CAP input changed: ' + path)
    for name in ('qwen_execution_diagnostic.py', '共享计划执行诊断_方案.md'):
        current = Path(__file__).resolve().parent / name
        check(sha(run_dir / name) == plan['sources_sha256'][str(current)], 'Local snapshot differs: ' + name)

    from research_program.triadic_task import environment as env
    from research_program.triadic_task import qwen_capability as cap
    from research_program.triadic_task import qwen_execution_diagnostic as diag
    # This also verifies CAP's frozen tokenizer/configuration and weight sizes;
    # no Backend initializer or model loading is invoked.
    check(diag.verify(run_dir) == (plan, prepared), 'Prepared source reconstruction differs')
    original_prepared = deepcopy(prepared)
    source = Path(plan['source_run'])
    source_calls = {c['call']: c for c in lines(source / 'execution/inference.jsonl')}
    source_trials = {r['case_id']: r for r in lines(source / 'execution/trials.jsonl')}
    check(len(source_calls) == 216 and len(source_trials) == 12, 'CAP source scope differs')
    check(plan['arms'] == list(ARMS) and plan['max_backend_calls'] == 144 and
          plan['source_decisions'] == plan['unique_decision_seeds'] == 36 and plan['calls_per_arm'] == 72,
          'Frozen arm/seed/budget scope differs')
    for key in ('new_probe_history', 'probe_actions_visible_to_partners', 'modifies_original_CAP_gate',
                'modifies_original_v3_gate', 'starts_symbolic_experiment'):
        check(plan[key] is False, 'Unexpected intervention scope: ' + key)

    prior = read(prior_receipt)
    check(Path(prior['planned_execution']) == run_dir and not prior['execution_started_exists'] and
          not prior['execution_status_exists'], 'Prior state receipt is not pre-execution for this pack')
    for path, digest in prior['sha256'].items():
        check(sha(path) == digest, 'Prior CAP/v3 gate or CAP history changed: ' + path)
    started = read(execution / 'started.json')
    check(datetime.fromisoformat(prior['recorded_at']) < datetime.fromisoformat(started['started_at']),
          'Prior-state receipt does not precede actual start')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True, trust_remote_code=False)
    calls = lines(execution / 'inference.jsonl')
    decisions, trials = lines(execution / 'decisions.jsonl'), lines(execution / 'trials.jsonl')
    result, backend_stats = read(execution / 'results.json'), read(execution / 'backend_stats.json')
    check(result['status'] == 'completed' and result['calls'] == status['calls'] == backend_stats['calls'] == len(calls) == 144,
          'Expected 144 completed real calls')
    check([c['call'] for c in calls] == list(range(1, 145)) and len(decisions) == 72 and len(trials) == 24,
          'Missing, duplicate or reordered execution records')
    check(Counter(c['mode'] for c in calls) == {'private_analysis': 72, 'action': 72}, 'Stage counts differ')
    check(result['plan_sha256'] == started['plan_sha256'] == freeze['plan_sha256'], 'Execution source plan differs')
    check(result['modifies_old_gates'] is False and result['symbolic_started'] is False, 'Result scope differs')

    case_ids, case_seeds, original_ids, common_additions = [], [], [], {}
    for i, case in enumerate(prepared['cases']):
        case_ids.append(case['case_id']); case_seeds.append(case['seed'])
        row = source_trials[case['source_trial_id']]
        agent = case['agent']
        check(case['group'] == (301, 302, 303)[i // 12] and case['episode'] == (i // 3) % 4 + 1 and
              agent == env.AGENTS[i % 3], 'Source case order differs')
        check(case['state'] == row['state'] and case['menu'] == row['menus'][agent], 'Source state or menu changed')
        check(case['choices'] == list(range(17)) and [m['id'] for m in case['menu']] == list(range(17)) and
              {json.dumps(m['action'], sort_keys=True) for m in case['menu']} ==
              {json.dumps(a, sort_keys=True) for a in env.all_actions(agent)}, 'Menu has missing or extra actions')
        base = cap.build_prompt(agent, row['observations'][agent], row['private_histories_before'][agent],
                                row['messages'], menu=row['menus'][agent])
        check(base == case['original_base_messages'], 'Original legal base prompt differs')
        data = json.loads(base[1]['content'])
        check(set(data) == {'你自己的历史', '本轮完整信息观察', '本轮已公开广播', '当前决策', '本次可选动作'} and
              len(data['本轮已公开广播']) == 6 and len(data['你自己的历史']) == case['episode'] - 1,
              'Unexpected history/transcript field or scope')
        for source_name, call_id in zip(('original_analysis', 'original_formal'), case['source_call_ids']):
            check(case[source_name] == source_calls[call_id], 'Embedded original call differs from CAP')
            original_ids.append(call_id)
        check(case['seed'] == cap.stable_seed('generation', case['group'], case['episode'], 0, agent, 'action') and
              case['temperature'] == 0, 'Original action seed or temperature differs')
        reference = independent_plan(env, env.State(**case['state']))
        check(case['shared_plan'] == reference and case['expected_plan_action'] == reference[agent],
              'Reference was not the first fixed semantic witness')
        addition = diag.shared_plan_message(reference)
        check(case['shared_plan_message'] == addition and case['plan_base_messages'] == base + [addition],
              'Plan arm does not add exactly one unchanged shared-plan message')
        check(not re.search(r'(?:编号|ID)\s*[:：=]?\s*\d+', addition['content']), 'Plan leaked a local correct ID')
        if case['source_trial_id'] in common_additions:
            check(common_additions[case['source_trial_id']] == addition, 'Agents received different reference plans')
        common_additions[case['source_trial_id']] = addition
    check(len(case_ids) == len(set(case_ids)) == len(case_seeds) == len(set(case_seeds)) == 36,
          'Source case/seed uniqueness differs')
    check(original_ids == [18 * i + offset for i in range(12) for offset in range(13, 19)],
          'Not all original action-stage call identities were included')
    check(Counter(c['seed'] for c in calls) == Counter({s: 4 for s in case_seeds}), 'Fourfold original-seed reuse differs')

    stage_checks, decision_checks, rebuilt_trials, reproduction_checks = [], [], [], []
    request, formal_request = stage_requests('action')
    for arm_index, arm in enumerate(ARMS):
        for i, case in enumerate(prepared['cases']):
            index = arm_index * 36 + i
            pair = calls[2 * index:2 * index + 2]
            base = case['original_base_messages'] if arm_index == 0 else case['plan_base_messages']
            expected_prompts = (base + [{'role': 'user', 'content': request}],
                base + [{'role': 'assistant', 'content': pair[0]['output']}, {'role': 'user', 'content': formal_request}])
            label = {'phase': 'triadic_execution_diagnostic_v1', 'arm': arm,
                     **{k: case[k] for k in ('case_id', 'source_trial_id', 'group', 'episode', 'agent', 'source_call_ids')}}
            for j, (entry, expected) in enumerate(zip(pair, expected_prompts)):
                stage, mode, limit = ('analysis', 'private_analysis', 256) if j == 0 else ('formal', 'action', 3)
                check(entry['label'] == {**label, 'stage': stage} and entry['mode'] == mode,
                      'Actual call label/stage differs')
                check(entry['messages'] == expected and entry['seed'] == case['seed'] and entry['temperature'] == 0,
                      'Actual prompt or generation setting differs')
                check(entry['valid'] is True and entry['fresh_cache'] is True and entry['native_thinking'] is False and
                      entry['symbol_limit'] is None, 'Backend flags differ')
                check(isinstance(entry['generation_tokens'], int) and 1 <= entry['generation_tokens'] <= limit and
                      entry['finish_reason'] in ('stop', 'length') and
                      (entry['finish_reason'] != 'length' or entry['generation_tokens'] == limit),
                      'Invalid generation budget or finish reason')
                check(math.isfinite(entry['seconds']) and entry['seconds'] >= 0, 'Invalid inference timing')
                rendered = tokenizer.apply_chat_template(expected, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                import hashlib
                check(hashlib.sha256(rendered.encode()).hexdigest() == entry['prompt_sha256'], 'Actual rendered prompt hash differs')
                add_special = tokenizer.bos_token is None or not rendered.startswith(tokenizer.bos_token)
                check(len(tokenizer.encode(rendered, add_special_tokens=add_special)) == entry['prompt_tokens'],
                      'Prompt token count differs')
                stage_checks.append(dict(call=entry['call'], arm=arm, case_id=case['case_id'], stage=stage,
                    source_call=case['source_call_ids'][j], seed=case['seed'], messages_exact=True,
                    rendered_sha_exact=True, prompt_tokens_exact=True, finish_reason=entry['finish_reason']))
                if arm_index == 0:
                    source_entry = source_calls[case['source_call_ids'][j]]
                    reproduction_checks.append(dict(new_call=entry['call'], source_call=source_entry['call'],
                        case_id=case['case_id'], equal={field: entry[field] == source_entry[field] for field in REPRO_FIELDS}))
            answer = pair[1]['output']
            check(answer in tuple(map(str, range(17))), 'Formal choice is outside complete menu')
            selected = next(m for m in case['menu'] if str(m['id']) == answer)
            expected_decision = {'arm': arm, 'case_id': case['case_id'], 'source_trial_id': case['source_trial_id'],
                'agent': case['agent'], 'group': case['group'], 'episode': case['episode'], 'seed': case['seed'],
                'source_call_ids': case['source_call_ids'], 'call_range': [2*index+1, 2*index+2], 'output': answer,
                'selected': deepcopy(selected), 'reference_plan_action': deepcopy(case['expected_plan_action']),
                'plan_was_shown': arm_index == 1, 'matches_reference_plan': selected['action'] == case['expected_plan_action']}
            check(decisions[index] == expected_decision, 'Decision record does not follow its own formal menu choice')
            decision_checks.append(expected_decision)
            if i % 3 == 2:
                bundle = decision_checks[-3:]
                actions = {d['agent']: d['selected']['action'] for d in bundle}
                outcome = env.settle(env.State(**case['state']), actions, require_match=True)
                same = actions == case['shared_plan']
                expected_trial = {'arm': arm, 'source_trial_id': case['source_trial_id'], 'group': case['group'],
                    'episode': case['episode'], 'actions': actions, 'outcome': outcome, 'plan_was_shown': arm_index == 1,
                    'all_actions_match_reference_plan': same, 'full_success_via_different_plan': outcome['reward'] == 1 and not same}
                check(trials[len(rebuilt_trials)] == expected_trial, 'Joint settlement or adherence label differs')
                rebuilt_trials.append(expected_trial)

    exact = all(all(c['equal'].values()) for c in reproduction_checks)
    reproduction = read(execution / 'reproduction.json')
    check(reproduction['checks'] == reproduction_checks and reproduction['source_calls'] == 72 and
          reproduction['complete_exact_reproduction'] is exact and reproduction['effect_interpretable'] is exact and
          reproduction['effect_status'] == ('paired_posthoc_diagnostic' if exact else 'effect_uninterpretable'),
          'Recorded exact-replay interpretation differs')
    check(result['reproduction'] == reproduction and status['effect_status'] == reproduction['effect_status'],
          'Result/status replay status differs')
    arm_summaries = []
    for arm in ARMS:
        ds, ts = [d for d in decision_checks if d['arm'] == arm], [t for t in rebuilt_trials if t['arm'] == arm]
        arm_summaries.append(dict(arm=arm, decisions=36, calls=72, trials=12,
            rewards=[t['outcome']['reward'] for t in ts], full_success_trials=sum(t['outcome']['full_success'] for t in ts),
            individual_actions_matching_reference_plan=sum(d['matches_reference_plan'] for d in ds),
            joint_actions_matching_reference_plan=sum(t['all_actions_match_reference_plan'] for t in ts),
            full_success_via_different_plan=sum(t['full_success_via_different_plan'] for t in ts),
            reference_plan_was_shown=arm == 'shared_plan_added'))
    check(result['arms'] == arm_summaries, 'Arm-level means or adherence summary differs')
    check(result['plan_distribution'] == plan['plan_distribution'] == diag.plan_distribution(prepared), 'Role distribution differs')
    check(prepared == original_prepared and sha(run_dir / 'prepared_cases.json') == plan['prepared_cases_sha256'],
          'Prepared histories were mutated')
    for path, digest in prior['sha256'].items():
        check(sha(path) == digest, 'Prior histories or gates changed during audit')
    seconds = sum(c['seconds'] for c in calls)
    tokens = {mode: sum(c['generation_tokens'] for c in calls if c['mode'] == mode) for mode in ('private_analysis', 'action')}
    check(tokens == backend_stats['generation_tokens_by_mode'] and
          max(c['prompt_tokens'] for c in calls) == backend_stats['max_prompt_tokens'] and
          math.isclose(seconds, backend_stats['inference_seconds'], rel_tol=1e-10, abs_tol=1e-6), 'Backend accounting differs')
    check(result['elapsed_seconds'] + .001 >= seconds + backend_stats['load_seconds'] and
          backend_stats['peak_mlx_memory_gb'] + 1e-6 >= max(c['peak_memory_gb'] for c in calls), 'Timing/memory accounting differs')
    check(no_model_runtime(), 'Auditor imported a model runtime')
    files = [run_dir / f for f in ('plan.json', 'freeze.json', 'prepared_cases.json')]
    files += [execution / f for f in ('started.json', 'status.json', 'results.json', 'decisions.jsonl', 'trials.jsonl',
                                     'inference.jsonl', 'reproduction.json', 'backend_stats.json')]
    truncations = [dict(call=c['call'], mode=c['mode'], label=c['label'], generation_tokens=c['generation_tokens'])
                   for c in calls if c['finish_reason'] == 'length']
    return dict(status='passed_real_execution_audit', audited_at=datetime.now(timezone.utc).isoformat(), run_dir=str(run_dir),
        audit_script_sha256=sha(__file__), source_artifact_sha256={str(p): sha(p) for p in files},
        prior_state_receipt_sha256=sha(prior_receipt), old_gate_and_CAP_history_sha_verified=prior['sha256'],
        calls_checked=144, stage_prompt_hash_and_token_checks=144, decisions_checked=72, menus_checked=72,
        independent_original_seed_count=36, reuse_per_seed=4, replay_calls_compared_to_CAP=72,
        settlements_replayed=24, reference_plans_lexically_verified=12, all_stage_checks=stage_checks,
        all_decisions=decision_checks, all_settlements=rebuilt_trials, arms=arm_summaries,
        reproduction=reproduction, plan_distribution=plan['plan_distribution'], length_truncated_calls=truncations,
        mode_statistics={mode: dict(calls=sum(c['mode'] == mode for c in calls), generation_tokens=tokens[mode],
                                   finish_reasons=dict(Counter(c['finish_reason'] for c in calls if c['mode'] == mode))) for mode in tokens},
        timing=dict(load_seconds=backend_stats['load_seconds'], inference_seconds=seconds,
                    elapsed_seconds=result['elapsed_seconds'], peak_mlx_memory_gb=backend_stats['peak_mlx_memory_gb']),
        prior_histories_and_old_gates_unchanged=True, new_history_in_probe_inputs=False,
        other_agents_actual_probe_actions_in_inputs=False, private_analysis_only_in_own_immediate_formal_input=True,
        tokenizer_only_no_model_runtime=True, new_model_calls=0,
        limitations=['All twelve source snapshots are included, but they are three histories over four shared semantic draws, not twelve independent worlds.',
            'Exact replay is checked, not counted as another independent model sample. All source seeds are reused.',
            'The intervention jointly supplies a correct shared semantic plan, says everyone received it, and instructs adherence; it does not isolate communication, clarity or deliberation.',
            'Plan compliance and physical success through a different valid plan are separate outcomes.',
            'Reference-plan roles follow a fixed first-witness order; role frequencies do not establish all-role capability.',
            'Inputs retain all old broadcasts and old personal histories, but no newly generated probe result enters a later probe.',
            'The audit checks logs, source/tokenizer hashes and weight sizes via CAP provenance; it does not reload model weights or independently attest hardware/KV-cache traces.',
            'Generation token counts and finish reasons are checked against recorded limits; no generation token-ID trace is available.',
            'Private analysis is generated text, not direct evidence of internal reasoning; truncation alone does not explain failure.',
            'No new symbols are learned or tested. Given-plan success does not pass the CAP or original v3 gate.'])


def markdown(r):
    reproduced = r['reproduction']['complete_exact_reproduction']
    out = ['# 共享计划执行诊断：真实记录独立核验', '',
        f"核验时间：{r['audited_at']}。全部144调用、72决定、24次结算核验通过；本次新增模型调用0。", '',
        f"原样重放的72个阶段与CAP原记录逐字段{'完全一致' if reproduced else '存在差异'}。"
        + ('可按冻结口径描述配对事后诊断。' if reproduced else '**配对干预效应不可解释**，差异与全部既定执行仍保留。'), '',
        '逐调用重建基础观察、本人旧历史及六条广播；共享计划臂只增加一条三人相同的研究者计划和执行指令。每个分析阶段及正式阶段的完整messages、渲染SHA、prompt token数均吻合。私有分析只注入本人紧随的正式输入，伙伴本次动作与新反馈未进入任何探针。', '',
        '72个菜单均保留17项；36个原决定种子各在两臂的两阶段复用，共四次。原始阶段顺序、CAP来源映射及指定计划的固定选取顺序均已核查。', '',
        '| 臂 | 满分场次 | 本人动作符合参考计划 | 三人共同符合参考计划 | 其他方案满分 |',
        '|---|---:|---:|---:|---:|']
    names = {'original_replay': '原样重放', 'shared_plan_added': '添加共享计划'}
    out += [f"| {names[a['arm']]} | {a['full_success_trials']}/12 | {a['individual_actions_matching_reference_plan']}/36 | {a['joint_actions_matching_reference_plan']}/12 | {a['full_success_via_different_plan']} |" for a in r['arms']]
    out += ['', '原样臂没有收到参考计划，其“符合计划”列只表示事后动作重合，不称为遵从指令。选择其他有效方案得到满分，也不等同于遵从指定计划。', '',
            '| 组 | 场景 | 原样R | 计划R | 计划臂三人遵从 | 计划臂其他解满分 |', '|---|---:|---:|---:|---|---|']
    for old, planned in zip(r['all_settlements'][:12], r['all_settlements'][12:]):
        out.append(f"| {old['group']} | {old['episode']} | {old['outcome']['reward']} | {planned['outcome']['reward']} | {'是' if planned['all_actions_match_reference_plan'] else '否'} | {'是' if planned['full_success_via_different_plan'] else '否'} |")
    out += ['', '计划分布（固定选择器取首个充分信息解）：', '',
            '```json', json.dumps(r['plan_distribution'], ensure_ascii=False, indent=2), '```', '',
            '| 阶段 | 调用数 | 生成token | 停止原因 |', '|---|---:|---:|---|']
    out += [f"| {mode} | {v['calls']} | {v['generation_tokens']} | {json.dumps(v['finish_reasons'], ensure_ascii=False)} |" for mode, v in r['mode_statistics'].items()]
    out += ['', f"达到长度上限共{len(r['length_truncated_calls'])}次；身份见JSON。截断不单独证明失败原因。推理累计{r['timing']['inference_seconds']:.2f}秒，模型报告峰值{r['timing']['peak_mlx_memory_gb']:.3f} GB。", '',
        '执行前保存的旧v3门槛与状态、CAP门槛与历史SHA在执行后均未改变；两臂全部基础历史还与CAP原快照相同。没有把诊断结果写成新历史、通过旧门槛或启动符号阶段。', '',
        '**解释范围：**这是3个上下文组共享4个语义场景的12个动作快照，不是12个独立世界。添加计划同时提供正确语义方案、共同收到的声明和遵从指令；即使改善，也不能归为某一种内部协商机制，更不能称为自然符号约定形成通过。', '',
        '只读tokenizer及冻结记录，未加载权重、MLX或Torch。模型权重仅沿用CAP来源链及字节大小核验；无新的完整权重哈希、硬件证明或KV-cache轨迹。', '',
        '[独立核验JSON](独立核验.json)包含全部调用身份、逐字段重放比较、动作、结算、截断和源哈希。']
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New prefix; creates PREFIX.json and PREFIX.md')
    parser.add_argument('--prior-receipt', type=Path, default=PRIOR)
    args = parser.parse_args()
    json_path, md_path = args.out.with_suffix('.json'), args.out.with_suffix('.md')
    check(not json_path.exists() and not md_path.exists(), 'Preserve prior audit output')
    result = audit(args.run, args.prior_receipt)
    new_json(json_path, result)
    with md_path.open('x', encoding='utf-8') as stream:
        stream.write(markdown(result))
    print(json.dumps({'status': result['status'], 'reproduction': result['reproduction']['complete_exact_reproduction'],
                      'arms': result['arms'], 'new_model_calls': 0}, ensure_ascii=False))
