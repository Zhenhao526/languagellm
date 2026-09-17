"""Audit a completed real Qwen capability control without loading model weights.

Rebuild all 216 stage prompts, render/tokenize them with the frozen local
tokenizer, reconstruct isolated histories/barriers, and replay 12 settlements.
Only run after execution is complete. This file is outside the frozen CAP pack.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


WORK = Path(__file__).resolve().parents[2]
GROUPS = (301, 302, 303)
AGENTS = ('A', 'B', 'C')


def check(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def stage_requests(mode):
    analysis = ('先用不超过120个汉字作私有分析：根据可见观察和已收到的互动记录，考虑本次决策。'
                '他人看不到此分析，分析不保存为后续记忆。不要假定未给定的共享符号含义已获伙伴理解。'
                + ('核对可选动作编号与动作描述；只依据自己的可见信息选择。' if mode == 'action'
                   else '考虑此次要表达什么、是否需回应伙伴，或是否沉默。'))
    formal = ('只输出你选择的一个本地动作编号，不要解释。' if mode == 'action' else
              '现在输出正式广播消息；最多约50个汉字，也可以沉默。不输出私有分析。')
    return analysis, formal


def audit(run_dir, *, tokenize=True):
    run_dir = Path(run_dir).resolve()
    execution = run_dir / 'execution'
    check((execution / 'status.json').is_file(), 'Execution has not produced a terminal status')
    status = read(execution / 'status.json')
    check(status['status'] == 'completed', 'Audit requires completed execution; do not audit a partial run as final')
    check(all(name not in sys.modules for name in ('mlx', 'mlx.core', 'mlx_lm', 'torch')),
          'Run audit in a clean process without a model runtime')
    plan, freeze = read(run_dir / 'plan.json'), read(run_dir / 'freeze.json')
    check(sha(run_dir / 'plan.json') == freeze['plan_sha256'], 'Frozen plan hash differs')
    for path, expected in plan['sources_sha256'].items():
        check(sha(path) == expected == sha(run_dir / 'source_snapshot' / Path(path).relative_to(WORK)),
              f'Frozen source or snapshot differs: {path}')
    check(sha(run_dir / 'prepared_cases.json') == plan['prepared_cases_sha256'], 'Prepared cases differ')
    check(sha(run_dir / 'fake_backend_audit.json') == plan['fake_audit_sha256'], 'Preparation audit differs')
    for entry in plan['weight_files_size_checked']:
        check(Path(entry['file']).stat().st_size == entry['bytes'], 'Weight byte size differs')

    sys.path.insert(0, str(WORK))
    from research_program.triadic_task import environment as env
    from research_program.triadic_task import qwen_capability as cap
    prepared = read(run_dir / 'prepared_cases.json')
    check(prepared == cap.build_cases(), 'Prepared scenarios/menus do not reconstruct from fixed seeds')
    check(plan['groups'] == list(GROUPS) and plan['scene_seeds'] == list(cap.SCENE_SEEDS)
          and plan['max_backend_calls'] == 216 and plan['unique_decision_seeds'] == 108,
          'Plan grid/budget differs')
    check(len(prepared['scenarios']) == 4 and len(prepared['trials']) == 12,
          'Expected four shared semantic draws and twelve context trials')

    tokenizer = None
    if tokenize:
        # Tokenizer assets/configuration only. AutoTokenizer does not initialize
        # a model, and MLX/PyTorch imports are explicitly checked at the end.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(plan['model'], local_files_only=True, trust_remote_code=False)

    calls = lines(execution / 'inference.jsonl')
    trials = lines(execution / 'trials.jsonl')
    result = read(execution / 'results.json')
    started, backend_stats = read(execution / 'started.json'), read(execution / 'backend_stats.json')
    check(result['status'] == 'completed' and result['plan_sha256'] == started['plan_sha256'] == freeze['plan_sha256'],
          'Execution result/plan differs')
    check(len(calls) == result['model_calls'] == status['calls'] == backend_stats['calls'] == 216,
          'Expected exactly 216 recorded real two-stage calls')
    check([row['call'] for row in calls] == list(range(1, 217)), 'Call IDs are missing, duplicated, or reordered')
    check(len(trials) == status['trials'] == 12 and sha(execution / 'trials.jsonl') == result['trials_sha256'],
          'Completed trial log differs')
    check(Counter(c['mode'] for c in calls) == {'private_analysis': 108, 'natural_message': 72, 'action': 36},
          'Mode counts differ')
    memories = {str(g): {a: [] for a in AGENTS} for g in GROUPS}
    seeds, decision_checks, settlement_checks = [], [], []
    menu_seeds = []
    cursor = 0

    def inspect_decision(base_prompt, mode, label, allowed_choices=None):
        nonlocal cursor
        analysis, formal = calls[cursor:cursor + 2]
        seed = cap.stable_seed('generation', label['group'], label['episode'], label['window'] or 0,
                               label['agent'], mode)
        seeds.append(seed)
        analysis_request, formal_request = stage_requests(mode)
        expected_prompts = (
            base_prompt + [{'role': 'user', 'content': analysis_request}],
            base_prompt + [{'role': 'assistant', 'content': analysis['output']},
                           {'role': 'user', 'content': formal_request}])
        for entry, expected, stage in zip((analysis, formal), expected_prompts, ('analysis', 'formal')):
            expected_mode = 'private_analysis' if stage == 'analysis' else mode
            check(entry['call'] == cursor + (1 if stage == 'analysis' else 2), 'Stage call identity differs')
            check(entry['label'] == {**label, 'stage': stage} and entry['mode'] == expected_mode,
                  'Stage label or mode differs')
            check(entry['seed'] == seed and entry['temperature'] == (0 if stage == 'analysis' or mode == 'action' else .7),
                  'Seed reuse or temperature differs')
            check(entry['messages'] == expected, 'Actual complete prompt differs from reconstructed legal history/transcript')
            check(entry['fresh_cache'] is True and entry['native_thinking'] is False and entry['valid'] is True
                  and entry['symbol_limit'] is None, 'Backend mode/cache/validity flags differ')
            check(isinstance(entry['output'], str), 'Output must be raw text')
            limit = 256 if stage == 'analysis' else 3 if mode == 'action' else 96
            check(isinstance(entry['generation_tokens'], int) and 1 <= entry['generation_tokens'] <= limit,
                  'Generated token count exceeds the frozen mode limit')
            check(entry['finish_reason'] in ('stop', 'length'), 'Unexpected terminal finish reason')
            check(entry['finish_reason'] != 'length' or entry['generation_tokens'] == limit,
                  'Length finish disagrees with token budget')
            check(math.isfinite(entry['seconds']) and entry['seconds'] >= 0 and entry['prompt_tokens'] > 0,
                  'Invalid timing or prompt-token record')
            if tokenizer is not None:
                rendered = tokenizer.apply_chat_template(expected, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                check(hashlib.sha256(rendered.encode()).hexdigest() == entry['prompt_sha256'],
                      'Rendered actual chat-template prompt hash differs')
                add_special = tokenizer.bos_token is None or not rendered.startswith(tokenizer.bos_token)
                prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=add_special))
                check(prompt_tokens == entry['prompt_tokens'], 'Prompt token count differs from frozen tokenizer rendering')
        if mode == 'action':
            check(formal['output'] in tuple(map(str, allowed_choices)), 'Formal action is outside the full menu')
        data = json.loads(base_prompt[1]['content'])
        check(set(data) == {'你自己的历史', '本轮完整信息观察', '本轮已公开广播', '当前决策'}
              | ({'本次可选动作'} if mode == 'action' else set()), 'Unexpected base-prompt fields')
        decision_checks.append({'analysis_call': analysis['call'], 'formal_call': formal['call'],
            'group': label['group'], 'episode': label['episode'], 'agent': label['agent'],
            'window': label['window'], 'mode': mode, 'seed': seed,
            'history_entries': len(data['你自己的历史']), 'visible_current_broadcasts': len(data['本轮已公开广播']),
            'base_and_both_stage_prompts_exact': True,
            'rendered_prompt_sha_and_token_counts_verified': tokenizer is not None,
            'analysis_finish_reason': analysis['finish_reason'], 'formal_finish_reason': formal['finish_reason']})
        cursor += 2
        return formal['output']

    for trial_index, (case, row) in enumerate(zip(prepared['trials'], trials)):
        group, episode = case['group'], case['episode']
        check(group == GROUPS[trial_index // 4] and episode == trial_index % 4 + 1, 'Trial order differs')
        for field in ('case_id', 'group', 'episode', 'scene_seed', 'state', 'observations', 'menus'):
            check(row[field] == case[field], f'Trial input/identity differs: {field}')
        check(row['call_range'] == [18 * trial_index + 1, 18 * (trial_index + 1)], 'Trial call range differs')
        history = memories[str(group)]
        check(row['private_histories_before'] == history, 'Group/agent histories or analysis boundary differs')
        for agent in AGENTS:
            check(len(history[agent]) == episode - 1, 'Within-group history length differs')
            check(case['observations'][agent] == env.observe(env.State(**case['state']), agent,
                      shared_needs=True, full_information=True), 'Full-information observation differs')
            menu = case['menus'][agent]
            check([entry['id'] for entry in menu] == list(range(17)) and
                  {json.dumps(m['action'], sort_keys=True) for m in menu} ==
                  {json.dumps(a, sort_keys=True) for a in env.all_actions(agent)}, 'Menu is pruned or duplicated')
            menu_seeds.append(case['menu_seeds'][agent])
        label = {'phase': cap.ROUTE, 'condition': 'full_information_natural', 'dependency': 'D1',
                 'group': group, 'episode': episode, 'scene_seed': case['scene_seed'], 'step': 1}
        check(all(row[key] == value for key, value in label.items()), 'Trial route/control label differs')
        transcript = []
        for window in (1, 2):
            pending = []
            for agent in AGENTS:
                base = cap.build_prompt(agent, case['observations'][agent], history[agent], transcript, window=window)
                if episode == 1 and window == 1:
                    check(base == prepared['first_prompts'][str(group)][agent], 'Frozen first prompt differs')
                message = inspect_decision(base, 'natural_message', {**label, 'agent': agent, 'window': window})
                pending.append({'agent': agent, 'window': window, 'text': message})
            # Delivery happens only after all three expected inputs were checked.
            transcript.extend(pending)
        check(transcript == row['messages'], 'Broadcast record differs from six actual formal outputs')
        actions, selections = {}, {}
        for agent in AGENTS:
            base = cap.build_prompt(agent, case['observations'][agent], history[agent], transcript, menu=case['menus'][agent])
            choice = inspect_decision(base, 'action', {**label, 'agent': agent, 'window': None}, list(range(17)))
            selected = next(m for m in case['menus'][agent] if m['id'] == int(choice))
            selections[agent], actions[agent] = deepcopy(selected), deepcopy(selected['action'])
        check(actions == row['actions'] and selections == row['selections'], 'Formal action/menu resolution differs')
        outcome = env.settle(env.State(**case['state']), actions, require_match=True)
        check(outcome == row['outcome'], 'Independent settlement replay differs')
        for agent in AGENTS:
            history[agent].append({'本轮观察': deepcopy(case['observations'][agent]),
                '公开广播': deepcopy(transcript), '自己提交的动作': deepcopy(actions[agent]),
                '自己可见结果': deepcopy(outcome['individual_feedback'][agent])})
        check(row['history_lengths_after'] == {a: episode for a in AGENTS}, 'Final history length differs')
        settlement_checks.append({'case_id': case['case_id'], 'group': group, 'episode': episode,
            'reward': outcome['reward'], 'full_success': outcome['full_success'],
            'active_agents': outcome['researcher']['active_agents'], 'overload': outcome['researcher']['overload'],
            'replayed_exactly': True})

    check(cursor == 216 and len(seeds) == len(set(seeds)) == 108, 'Decision coverage or seed uniqueness differs')
    check(Counter(c['seed'] for c in calls) == Counter({seed: 2 for seed in seeds}), 'Seeds not reused exactly within stage pairs')
    check(len(menu_seeds) == len(set(menu_seeds)) == 36 and not set(menu_seeds) & set(seeds),
          'Menu and generation seed identities collide')
    check(read(execution / 'histories_after.json') == memories, 'Final histories contain unexpected information')
    check(prepared['first_prompts']['301'] == prepared['first_prompts']['302'] == prepared['first_prompts']['303'],
          'First scene differs across context groups')
    for episode in range(1, 5):
        states = [c['state'] for c in prepared['trials'] if c['episode'] == episode]
        check(len(states) == 3 and states[0] == states[1] == states[2], 'Groups did not share the same four semantic scenes')
    expected_gate = {'rule': 'all_twelve_full_success', 'passed': all(r['full_success'] for r in settlement_checks),
        'full_success_trials': sum(r['full_success'] for r in settlement_checks), 'trials': 12,
        'new_task_only': True, 'modifies_original_v3_gate': False, 'starts_symbolic_experiment': False}
    check(result['gate'] == status['gate'] == expected_gate, 'Capability gate differs from all twelve actual settlements')
    check(result['by_group'] == [{'group': g, 'rewards': [r['reward'] for r in settlement_checks if r['group'] == g]}
                                for g in GROUPS], 'Group summary differs')
    check(result['symbolic_started'] is False and result['original_v3_gate_modified'] is False,
          'New capability result was misreported as original gate or symbolic experiment')
    mode_tokens = {mode: sum(c['generation_tokens'] for c in calls if c['mode'] == mode)
                   for mode in ('private_analysis', 'natural_message', 'action')}
    check(mode_tokens == backend_stats['generation_tokens_by_mode'], 'Backend token accounting differs')
    check(max(c['prompt_tokens'] for c in calls) == backend_stats['max_prompt_tokens'], 'Prompt token peak differs')
    seconds = sum(c['seconds'] for c in calls)
    check(math.isclose(seconds, backend_stats['inference_seconds'], rel_tol=1e-10, abs_tol=1e-6), 'Inference seconds differ')
    check(result['elapsed_seconds'] + 1e-3 >= seconds + backend_stats['load_seconds'], 'Elapsed time is smaller than recorded model work')
    check(backend_stats['peak_mlx_memory_gb'] + 1e-6 >= max(c['peak_memory_gb'] for c in calls), 'Memory peak accounting differs')
    check(all(name not in sys.modules for name in ('mlx', 'mlx.core', 'mlx_lm', 'torch')), 'Auditor imported a model runtime')
    truncations = [{'call': c['call'], 'label': c['label'], 'mode': c['mode'],
                    'generation_tokens': c['generation_tokens']} for c in calls if c['finish_reason'] == 'length']
    per_mode = {mode: {'calls': sum(c['mode'] == mode for c in calls), 'generation_tokens': mode_tokens[mode],
                      'finish_reasons': dict(Counter(c['finish_reason'] for c in calls if c['mode'] == mode)),
                      'max_prompt_tokens': max(c['prompt_tokens'] for c in calls if c['mode'] == mode)} for mode in mode_tokens}
    artifact_paths = [run_dir / name for name in ('plan.json', 'freeze.json', 'prepared_cases.json', 'fake_backend_audit.json')]
    artifact_paths += [execution / name for name in ('started.json', 'results.json', 'status.json', 'trials.jsonl',
                                                    'inference.jsonl', 'histories_after.json', 'backend_stats.json')]
    return {'status': 'passed_real_execution_audit', 'audited_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'run_dir': str(run_dir), 'audit_script_sha256': sha(__file__),
        'source_artifact_sha256': {str(path): sha(path) for path in artifact_paths},
        'real_recorded_model_calls_checked': 216, 'two_stage_decisions_checked': 108,
        'unique_decision_seeds': 108, 'analysis_and_formal_share_same_seed': True,
        'distinct_menu_seeds': 36, 'full_menus_checked': 36, 'actions_reconstructed': 36,
        'group_contexts': 3, 'semantic_draws': 4, 'context_trials': 12,
        'same_four_scenes_across_groups': True, 'within_group_history_lengths_before': [0, 1, 2, 3],
        'all_stage_messages_reconstructed_exactly': True, 'current_window_broadcast_counts': [0, 3, 6],
        'private_analysis_directly_injected_only_into_own_immediate_formal_call': True,
        'all_group_and_agent_histories_reconstructed_exactly': True,
        'rendered_prompt_hash_and_token_count_checks': 216 if tokenizer is not None else 0,
        'tokenizer_class': type(tokenizer).__name__ if tokenizer is not None else None,
        'tokenizer_only_no_weights_loaded': tokenizer is not None,
        'settlements': settlement_checks, 'gate': expected_gate, 'decision_checks': decision_checks,
        'mode_statistics': per_mode, 'length_truncated_calls': truncations,
        'natural_message_lengths': {'max_unicode_characters': max(len(c['output']) for c in calls if c['mode'] == 'natural_message'),
            'over_50_unicode_character_calls': [c['call'] for c in calls if c['mode'] == 'natural_message' and len(c['output']) > 50],
            'note': 'The prompt says approximately 50 Chinese characters; actual hard budget is 96 tokens. Unicode length is descriptive, not a strict failure rule.'},
        'timing': {'load_seconds': backend_stats['load_seconds'], 'summed_inference_seconds': seconds,
                   'runner_elapsed_seconds': result['elapsed_seconds'], 'peak_mlx_memory_gb': backend_stats['peak_mlx_memory_gb']},
        'new_model_calls': 0, 'model_weights_loaded_by_audit': 0, 'model_runtime_imported_by_audit': False,
        'limitations': ['This verifies a recorded real execution; it does not rerun Qwen or independently attest hardware/model weights.',
            'Weight files are checked by frozen sizes and provenance receipt, not freshly hashed by this audit.',
            'Full information and natural language are explicitly permitted in this control; this is not a communication-necessity or symbol-emergence test.',
            'Three context groups share four IID semantic draws and the same model weights; twelve trials are not twelve independent semantic samples.',
            'Within-group raw histories accumulate; these are not twelve history-reset trials.',
            'Private-analysis isolation is a structural injection/history claim. Formal public text may repeat analysis content by the model choice; substring overlap is not treated as an input leak.',
            'Fresh cache/native-thinking checks compare frozen code and recorded flags, not an independent KV-cache trace.',
            'Finish reason length indicates the generation cap, not proof that truncation caused an action failure.',
            'Passing this limited new-task gate does not amend the original v3 gate or establish robust planning, communication causality or language formation.']}


def markdown(result):
    rows = ['# Qwen 三人一步能力控制：真实运行独立审计', '',
        f"审计时间：{result['audited_at']}。核对的是已完成真实运行的216条调用记录；本审计新模型调用为0。", '',
        f"**记录核验通过；能力门槛{'通过' if result['gate']['passed'] else '未通过'}：{result['gate']['full_success_trials']}/12场满分。** 两者必须分开理解。", '',
        '216次调用组成108个决定：分析和正式阶段各用一次同一决定种子，共108个不同种子。逐条重建完整提示词、群体/主体历史和公开广播，再用冻结本地tokenizer核对渲染后SHA与prompt token数。没有加载模型权重、MLX或Torch。', '',
        '第一窗看到0条当轮广播，第二窗看到第一窗的3条，动作阶段看到全部6条；同窗消息和伙伴未执行动作未进入输入。私有分析只直接注入本人紧随的正式阶段，后续历史只保留合法观察、本人动作、本人反馈与正式公共广播。结构核验不把模型主动公开的同义内容判为历史泄漏。', '',
        '这是完整信息、自然语言控制。三上下文组共享相同四个预设场景，每组按场景顺序累计本人的历史；12场不是12次独立语义抽样，也不是12次清空历史的重复。', '',
        '| 组 | 场景 | 平均团队收益R | 满分 |', '|---|---:|---:|---|']
    rows += [f"| {r['group']} | {r['episode']} | {r['reward']} | {'是' if r['full_success'] else '否'} |" for r in result['settlements']]
    rows += ['', '所有36个菜单均保留17项；正式编号逐一映射到本人菜单，再独立重放12次同步结算。失败场仍保留，不按结果删选。', '',
        '| 调用类型 | 调用数 | 生成token | 停止原因 |', '|---|---:|---:|---|']
    rows += [f"| {mode} | {r['calls']} | {r['generation_tokens']} | {json.dumps(r['finish_reasons'], ensure_ascii=False)} |" for mode, r in result['mode_statistics'].items()]
    rows += ['', f"达到生成长度上限的调用共{len(result['length_truncated_calls'])}次；逐条身份见JSON。它不单独证明截断造成了失败。广播约50汉字是软提示，硬预算为96 token，二者分开记录。", '',
        f"推理累计{result['timing']['summed_inference_seconds']:.2f}秒；模型加载{result['timing']['load_seconds']:.2f}秒；MLX报告峰值{result['timing']['peak_mlx_memory_gb']:.3f} GB。", '',
        '本审计核对冻结来源、记录和tokenizer，不重新计算模型输出或完整模型权重哈希。通过新任务能力控制也不修改旧v3门槛、不自动启动符号实验，不能据此认定稳定语言能力或通信的因果作用。', '',
        '[结构化核验](独立核验.json)记录全部调用身份、提示词检查、种子、截断、结算和源哈希。']
    return '\n'.join(rows) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output prefix; writes PREFIX.json and PREFIX.md')
    args = parser.parse_args()
    json_path, md_path = args.out.with_suffix('.json'), args.out.with_suffix('.md')
    check(not json_path.exists() and not md_path.exists(), 'Audit output already exists; preserve it')
    result = audit(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_json(json_path, result)
    with md_path.open('x', encoding='utf-8') as stream:
        stream.write(markdown(result))
    print(json.dumps({'status': result['status'], 'gate': result['gate'], 'out': str(args.out),
                      'new_model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
