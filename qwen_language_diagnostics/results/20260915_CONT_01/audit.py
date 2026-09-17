"""Independent completed-run audit; loads a tokenizer, never model weights.

Run with the project's Python from any directory after status is completed.
Only verification.json and 独立核验.md in this result directory are written.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys


OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
ARMS = ('original_history', 'reset_history')
NAMES = {'original_history': '保留历史', 'reset_history': '清空起始历史'}
SOURCE_FIELDS = ('step', 'observations', 'menus', 'messages', 'channel_usage',
                 'selections', 'actions', 'feedback', 'score', 'state_before', 'state_after')
ANALYSIS_PREFIX = (
    '先用不超过120个汉字作私有分析：根据可见观察和已收到的互动记录，考虑本次决策。'
    '他人看不到此分析，分析不保存为后续记忆。不要假定未给定的共享符号含义已获伙伴理解。'
)
ANALYSIS_SUFFIX = {
    'action': '核对可选动作编号与动作描述；只依据自己的可见信息选择。',
    'natural_message': '考虑此次要表达什么、是否需回应伙伴，或是否沉默。',
}
FINAL_REQUEST = {
    'action': '只输出你选择的一个本地动作编号，不要解释。',
    'natural_message': '现在输出正式广播消息；最多约50个汉字，也可以沉默。不输出私有分析。',
}


def read(name):
    return json.loads((OUT / name).read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_outcomes(row):
    """Count reciprocal successful events once, using verified engine feedback."""
    before, after = row['state_before'], row['state_after']
    goals = []
    for index, goal in enumerate(after['goals']):
        goals.append({'goal_index': index,
                      **{k: goal[k] for k in ('kind', 'length', 'condition', 'destination', 'quantity')},
                      'delivered_before': before['goals'][index]['delivered'],
                      'delivered_after': goal['delivered'],
                      'delivered_this_step': goal['delivered'] - before['goals'][index]['delivered']})

    def identity(owner, action):
        return next((oid for oid, handle in before['handles'].get(owner, {}).items()
                     if handle == action.get('item')), None)

    proposals = []
    events = {'carry_together': set(), 'deliver_together': set()}
    for actor in ('A', 'B', 'C'):
        action = row['actions'][actor]
        if action['kind'] not in ('carry_together', 'deliver_together', 'drop_together'):
            continue
        partner = action.get('partner')
        other = row['actions'].get(partner, {})
        oid = identity(actor, action)
        reciprocal = (oid is not None and identity(partner, other) == oid
                      and other.get('partner') == actor and other.get('kind') == action['kind']
                      and (action['kind'] != 'carry_together'
                           or action.get('destination') == other.get('destination')))
        succeeded = row['feedback'][actor]['action_succeeded']
        proposals.append({'agent': actor, 'kind': action['kind'], 'researcher_object_id': oid,
                          'partner': partner, 'destination': action.get('destination'),
                          'proposals_match': reciprocal, 'action_succeeded': succeeded})
        if (action['kind'] in events and succeeded and reciprocal
                and row['feedback'][partner]['action_succeeded']):
            events[action['kind']].add((oid, tuple(sorted((actor, partner)))))
    return {'goals': goals, 'joint_proposals': proposals,
            'successful_joint_object_actions': {kind: len(values) for kind, values in events.items()},
            'delivered_this_step': sum(g['delivered_this_step'] for g in goals)}


def main():
    status = read('status.json')
    if status.get('status') != 'completed':
        print('Audit not run: continuation status is not completed.')
        return 2
    manifest, prepared = read('manifest.json'), read('prepared_cases.json')
    result, stats = read('results.json'), read('backend_stats.json')
    steps, calls = read_lines(OUT / 'steps.jsonl'), read_lines(OUT / 'inference.jsonl')
    arm_rows = read_lines(OUT / 'arms.jsonl')
    errors = []

    def check(value, label):
        if not value:
            errors.append(label)
        return bool(value)

    sources = {name: {'current': digest(WORK / name) == expected,
                      'snapshot': digest(OUT / 'code_snapshot' / name) == expected}
               for name, expected in manifest['source_sha256'].items()}
    check(len(sources) == 16, 'Expected sixteen frozen source files')
    for name, flags in sources.items():
        check(all(flags.values()), 'Frozen source mismatch: ' + name)
    source_run = Path(manifest['source_run'])
    source_records = {name: digest(source_run / name) == expected
                      for name, expected in manifest['source_records_sha256'].items()}
    prior_chain = {name: digest(Path(name)) == expected
                   for name, expected in manifest['prior_result_files_sha256'].items()}
    check(all(source_records.values()), 'Original v3 records changed')
    check(all(prior_chain.values()), 'Prior source chain changed')
    check(digest(OUT / 'prepared_cases.json') == manifest['cases_sha256'], 'Prepared cases changed')
    if errors:
        raise RuntimeError('Source audit failed before importing frozen pure modules: ' + '; '.join(errors))

    sys.path.insert(0, str(WORK))
    from qwen_language_diagnostics.continuation_cases import build_continuation_cases
    from qwen_language_v3.agents import build_prompt
    from qwen_language_v3.environment import AGENTS, RULES, World
    from qwen_language_v3.run import inference_seed

    check(build_continuation_cases(source_run) == prepared, 'Prepared reconstruction differs')
    check([a['arm_id'] for a in prepared['arms']] == list(ARMS), 'Initial arm order differs')
    check([a['arm_id'] for a in arm_rows] == list(ARMS), 'Both arms did not complete in planned order')
    check(arm_rows == result['arms'] == status['arms'], 'Arm summaries disagree')
    check([c['call'] for c in calls] == list(range(1, len(calls) + 1)), 'Call ids are missing or duplicated')
    check(len(calls) == result['model_calls'] == status['model_calls'] == stats['calls'], 'Call totals disagree')
    check(len(calls) <= manifest['max_backend_calls'] == result['max_model_calls'] == 180, 'Call cap differs')
    check(len(steps) == result['new_physical_steps'] and len(calls) == 30 * len(steps), 'Step/call accounting differs')
    check(result['symbolic_started'] is False and status['symbolic_started'] is False
          and manifest['unlocks_symbolic_experiment'] is False, 'Unexpected symbolic unlock')
    check(manifest['source_steps'] == [9, 10, 11] and manifest['source_call_range'] == [241, 330]
          and manifest['original_deadline'] == 12 and manifest['observer_stop_t'] == 11
          and manifest['modifies_model_deadline'] is False, 'Frozen continuation budget differs')

    cursor = 0
    pair_checks, step_checks, arm_checks, ordered_steps = [], [], [], []
    arm_details = []

    def verify_pair(prompt, label, mode, expected_output, choices=None):
        nonlocal cursor
        if cursor + 2 > len(calls):
            raise RuntimeError('Missing complete analysis/formal pair')
        analysis, formal = calls[cursor:cursor + 2]
        cursor += 2
        expected_seed = inference_seed(17, 1, label['step'], label['window'] or 0,
                                       label['agent'], 'action' if mode == 'action' else 'message')
        detail = {
            'analysis_messages': analysis['messages'] == prompt + [
                {'role': 'user', 'content': ANALYSIS_PREFIX + ANALYSIS_SUFFIX[mode]}],
            'formal_messages_same_decision_analysis': formal['messages'] == prompt + [
                {'role': 'assistant', 'content': analysis['output']},
                {'role': 'user', 'content': FINAL_REQUEST[mode]}],
            'labels_modes': analysis['label'] == {**label, 'stage': 'analysis'}
                            and formal['label'] == {**label, 'stage': 'formal'}
                            and analysis['mode'] == 'private_analysis' and formal['mode'] == mode,
            'source_seed': analysis['seed'] == formal['seed'] == expected_seed,
            'temperature': analysis['temperature'] == 0
                           and formal['temperature'] == (.7 if mode == 'natural_message' else 0),
            'backend_flags': all(c['fresh_cache'] is True and c['native_thinking'] is False
                                 and c['valid'] is True and c['symbol_limit'] is None
                                 for c in (analysis, formal)),
            'formal_output_equals_record': formal['output'] == expected_output,
            'within_generation_limits': analysis['generation_tokens'] <= 256
                                        and formal['generation_tokens'] <= (
                                            96 if mode == 'natural_message' else max(len(str(x)) for x in choices) + 1),
        }
        if choices is not None:
            detail['formal_choice_is_legal'] = formal['output'] in list(map(str, choices))
        for name, value in detail.items():
            check(value, f'call {analysis["call"]}/{formal["call"]}: {name}')
        pair_checks.append({'arm_id': label['arm_id'], 'step': label['step'],
                            'window': label['window'], 'agent': label['agent'],
                            'call_ids': [analysis['call'], formal['call']], 'equal': detail})

    for arm_id in ARMS:
        initial = next(a for a in prepared['arms'] if a['arm_id'] == arm_id)
        histories = deepcopy(initial['histories'])
        expected_initial = 8 if arm_id == 'original_history' else 0
        check(all(len(histories[a]) == expected_initial for a in AGENTS), 'Initial owner history length differs')
        world = World.from_state_dict(prepared['state'])
        trajectory = [r for r in steps if r['arm_id'] == arm_id]
        ordered_steps.extend(trajectory)
        check(1 <= len(trajectory) <= 3, 'Arm step count outside one to three')
        check([r['step'] for r in trajectory] == list(range(9, 9 + len(trajectory))), 'Arm steps are not source-contiguous')
        first_cursor = cursor
        for row in trajectory:
            check(not world.done, 'A new step was generated after environment completion')
            check(row['state_before'] == world.state_dict(), 'Before state does not replay')
            step = row['step']
            base_label = {'phase': 'history_continuation', 'arm_id': arm_id,
                          'condition': 'full_information', 'group': 17, 'episode': 1,
                          'seed': 92015, 'step': step}
            check(all(row.get(k) == v for k, v in base_label.items()), 'Step label mismatch')
            check(row['private_histories_before_step'] == histories, 'Owner histories differ before step')
            observations = {a: world.full_information_observe(a) for a in AGENTS}
            menus = {a: world.action_menu(a) for a in AGENTS}
            check(observations == row['observations'] and menus == row['menus'], 'Observation or local menu differs')
            messages = row['messages']
            check([(x['window'], x['sender']) for x in messages]
                  == [(w, a) for w in range(1, 5) for a in AGENTS], 'Broadcast window/order differs')
            check(all(set(x) == {'sender', 'window', 'text', 'step'} and x['step'] == step
                      and isinstance(x['text'], str) for x in messages), 'Unexpected public broadcast fields')
            usage = [{'sender': x['sender'], 'window': x['window'], 'characters': len(x['text'])} for x in messages]
            check(usage == row['channel_usage'], 'Natural message usage differs')
            by_message = {(x['window'], x['sender']): x['text'] for x in messages}
            before_cursor = cursor
            for window in range(1, 5):
                visible = [x for x in messages if x['window'] < window]
                for agent in AGENTS:
                    prompt = build_prompt(agent, rules=RULES, observation=observations[agent],
                                          history=histories[agent], transcript=visible,
                                          condition='full_information', window=window, remaining=None)
                    if step == 9 and window == 1:
                        check(prompt == prepared['first_prompts'][arm_id][agent], 'First prompt differs')
                    verify_pair(prompt, {**base_label, 'window': window, 'agent': agent},
                                'natural_message', by_message[(window, agent)])
            for agent in AGENTS:
                prompt = build_prompt(agent, rules=RULES, observation=observations[agent],
                                      history=histories[agent], transcript=messages,
                                      condition='full_information', menu=menus[agent])
                selection = row['selections'][agent]
                check(selection in menus[agent] and selection['action'] == row['actions'][agent],
                      'Recorded selection is not the executed local-menu action')
                verify_pair(prompt, {**base_label, 'window': None, 'agent': agent}, 'action',
                            str(selection['id']), choices=[x['id'] for x in menus[agent]])
            check(cursor - before_cursor == 30 and row['call_range'] == [before_cursor + 1, cursor],
                  'Complete step call range differs')
            feedback = world.step(deepcopy(row['actions']))
            replay = (feedback == row['feedback'] and world.state_dict() == row['state_after']
                      and math.isfinite(row['score']) and world.score == row['score'])
            check(replay, 'World action settlement/score differs')
            outcomes = derive_outcomes(row)
            check(outcomes == row['outcomes'], 'Goal or reciprocal joint-event accounting differs')
            for agent in AGENTS:
                entry = {'观察': deepcopy(observations[agent]), '公开广播': deepcopy(messages),
                         '自己执行的动作': deepcopy(row['actions'][agent]),
                         '自己可见结果': deepcopy(feedback[agent])}
                if world.done:
                    entry['任务段结束'] = {'目标完成比例': world.score, '已用动作步': step}
                histories[agent].append(entry)
            check(row['history_lengths_after'] == {a: len(histories[a]) for a in AGENTS}, 'History growth differs')
            step_checks.append({'arm_id': arm_id, 'step': step, 'calls': row['call_range'],
                                'settlement_equal': replay, 'outcomes_equal': outcomes == row['outcomes'],
                                'outcomes': outcomes, 'score': world.score,
                                'action_descriptions': {a: row['selections'][a]['description'] for a in AGENTS}})
        summary = next(a for a in arm_rows if a['arm_id'] == arm_id)
        expected_summary = {
            'arm_id': arm_id, 'start_t': 8, 'end_t': world.t, 'new_steps': len(trajectory),
            'engine_status': world.status,
            'end_reason': 'environment_terminal' if world.done else 'observation_limit',
            'success': world.status == 'success', 'score': world.score,
            'delivered_units': sum(g['delivered'] for g in world.goals),
            'required_units': sum(g['quantity'] for g in world.goals), 'goals': deepcopy(world.goals),
            'model_calls': cursor - first_cursor,
            'history_lengths_after': {a: len(histories[a]) for a in AGENTS},
        }
        flags = {key: summary.get(key) == value for key, value in expected_summary.items()}
        for key, value in flags.items():
            check(value, arm_id + ': summary ' + key)
        history_file = read(arm_id + '_histories_after.json')
        check(history_file == histories, 'Final private histories differ: ' + arm_id)
        if not world.done:
            check(world.t == 11 and len(trajectory) == 3 and world.max_steps == 12
                  and world.status == 'running', 'Observation cutoff became a deadline/timeout')
            check(all('任务段结束' not in entry for memory in histories.values()
                      for entry in memory), 'Observation cutoff was stored as episode termination')
        arm_calls = calls[first_cursor:cursor]
        stop_reasons = {mode: dict(Counter(c['finish_reason'] for c in arm_calls if c['mode'] == mode))
                        for mode in ('private_analysis', 'natural_message', 'action')}
        arm_checks.append({'arm_id': arm_id, 'summary_equal': flags, 'final_history_exact': history_file == histories})
        arm_details.append({**expected_summary, 'finish_reasons': stop_reasons,
                            'max_prompt_tokens': max(c['prompt_tokens'] for c in arm_calls),
                            'joint_carry_events': sum(r['outcomes']['successful_joint_object_actions']['carry_together'] for r in trajectory),
                            'joint_delivery_events': sum(r['outcomes']['successful_joint_object_actions']['deliver_together'] for r in trajectory)})

    check(ordered_steps == steps, 'Arm trajectories interleave or contain extra records')
    check(cursor == len(calls), 'Extra calls occur outside complete protocol decisions')
    expected_generation = dict(Counter())
    for mode in {c['mode'] for c in calls}:
        expected_generation[mode] = sum(c['generation_tokens'] for c in calls if c['mode'] == mode)
    check(expected_generation == stats['generation_tokens_by_mode'], 'Generation token accounting differs')
    check(abs(sum(c['seconds'] for c in calls) - stats['inference_seconds']) < 1e-6, 'Inference timing sum differs')
    check(max(c['prompt_tokens'] for c in calls) == stats['max_prompt_tokens'], 'Maximum prompt length differs')

    # Exact reproduction is measured, not assumed or required for audit validity.
    old_index = {(c['label']['step'], c['label']['window'], c['label']['agent'], c['label']['stage']): c
                 for c in prepared['source_calls']}
    reproduced = []
    fields = ('messages', 'prompt_sha256', 'output', 'seed', 'temperature', 'mode',
              'fresh_cache', 'native_thinking', 'valid', 'finish_reason')
    for call in calls:
        label = call['label']
        if label['arm_id'] != 'original_history':
            continue
        old = old_index[(label['step'], label['window'], label['agent'], label['stage'])]
        reproduced.append({'new_call': call['call'], 'source_call': old['call'],
                           'equal': {key: call[key] == old[key] for key in fields}})
    old_steps = {r['step']: r for r in prepared['source_records']}
    reproduced_steps = [{'step': row['step'], 'equal': {
        field: row[field] == old_steps[row['step']][field] for field in SOURCE_FIELDS}}
        for row in steps if row['arm_id'] == 'original_history']
    reproduction = {'calls_compared': len(reproduced), 'expected_source_calls': 90,
                    'calls': reproduced, 'steps': reproduced_steps,
                    'complete_90_call_reproduction': len(reproduced) == 90
                    and all(all(row['equal'].values()) for row in reproduced + reproduced_steps)}
    check(reproduction == result['reproduction'], 'Reported original-arm reproduction differs')

    # This loads tokenizer files only. There is no model load or inference call.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True, trust_remote_code=False)
    rendered = {str(c['call']): hashlib.sha256(tokenizer.apply_chat_template(
        c['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False
    ).encode()).hexdigest() == c['prompt_sha256'] for c in calls}
    check(all(rendered.values()), 'Actual template rendering hash differs')

    truncated_calls = [{key: call[key] for key in ('call', 'label', 'mode', 'generation_tokens')}
                       for call in calls if call['finish_reason'] == 'length']
    reset_steps = {row['step']: row for row in steps if row['arm_id'] == 'reset_history'}
    initial_fiber = next(item for item in reset_steps[9]['observations']['B']['carried_items']
                         if item['handle'] == '物155')
    selected_fiber = next(item for item in reset_steps[11]['observations']['B']['ground_items']
                          if item['handle'] == '物143')
    fiber_goal = next(goal for goal in reset_steps[9]['observations']['B']['task_board']
                      if goal['kind'] == '纤维')
    attribute_keys = ('kind', 'length', 'condition')
    behavior_evidence = {
        'reset_B_initial_carried_item': initial_fiber,
        'reset_B_step11_selected_ground_item': selected_fiber,
        'fiber_goal': fiber_goal,
        'both_items_match_required_attributes': all(
            all(item[key] == fiber_goal[key] for key in attribute_keys)
            for item in (initial_fiber, selected_fiber)),
        'reset_B_actions_by_step': {str(step): row['actions']['B'] for step, row in reset_steps.items()},
        'reset_B_private_action_analysis': {
            str(call['call']): call['output'] for call in calls
            if call['label']['arm_id'] == 'reset_history' and call['label']['agent'] == 'B'
            and call['label']['window'] is None and call['label']['stage'] == 'analysis'},
        'interpretation': 'Both fibers meet the goal. These records show replacement and action-stage planning problems, not selection of a fiber with wrong attributes. Step12 was not run.',
    }

    evidence_files = ['manifest.json', 'prepared_cases.json', 'status.json', 'results.json',
                      'steps.jsonl', 'arms.jsonl', 'inference.jsonl', 'backend_stats.json',
                      'original_history_histories_after.json', 'reset_history_histories_after.json']
    audit = {
        'audited_at': datetime.now().astimezone().isoformat(), 'status': 'verified' if not errors else 'discrepancies',
        'errors': errors, 'scope': {'model_calls': len(calls), 'formal_decisions': len(pair_checks),
                                  'physical_steps': len(steps), 'arms': 2, 'selected_source_states': 1},
        'checks': {'source_current_and_snapshot': sources, 'original_v3_records': source_records,
                   'prior_source_chain': prior_chain, 'decision_pairs': pair_checks,
                   'template_render_sha256': rendered, 'steps': step_checks, 'arms': arm_checks},
        'arm_results': arm_details, 'reproduction': reproduction, 'backend_stats': stats,
        'generation_truncated_calls': truncated_calls, 'behavior_evidence': behavior_evidence,
        'evidence_file_sha256': {name: digest(OUT / name) for name in evidence_files},
        'audit_script_sha256': digest(Path(__file__)), 'audit_loaded_model_weights': False,
        'audit_new_model_calls': 0, 'model_weight_files_fully_hashed': False,
        'interpretation_limits': [
            'Only initial raw histories differ; each arm generates its new broadcasts, actions and subsequent owner history.',
            'An ending at t11 with a running world is observation truncation, not timeout at the unchanged t12 deadline.',
            'Exact original-arm replay is a reproducibility control, not an independent cooperative sample.',
            'One selected failure state and changed history length do not identify a particular old message or internal mechanism.',
            'Current observations retain prior own feedback even in the reset-history arm.',
            'This full-information natural-language continuation does not demonstrate symbolic language formation.',
        ],
    }
    (OUT / 'verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# 三人续跑独立核验', '',
             f"核验时间：{audit['audited_at']}。覆盖已完成运行的全部 {len(calls)} 次调用、{len(pair_checks)} 次正式决定及 {len(steps)} 个物理步。逐项结果见 [verification.json](verification.json)，可复核脚本为 [audit.py](audit.py)。仅使用纯环境重放和本地 tokenizer 文本渲染，没有加载模型权重或追加推理。", '',
             '**' + ('全部输入、来源及结算核对通过。' if not errors else '核验发现不一致，详见下列错误及 verification.json。') + '**', '',
             '|条件|新步数|终态 t|交付单位|分数|成功共同搬运事件|成功共同交付事件|调用|停止方式|',
             '|---|---:|---:|---|---:|---:|---:|---:|---|']
    for arm in arm_details:
        ending = '环境任务结束' if arm['end_reason'] == 'environment_terminal' else '观察片段结束；环境仍运行'
        lines.append(f"|{NAMES[arm['arm_id']]}|{arm['new_steps']}|{arm['end_t']}|{arm['delivered_units']}/{arm['required_units']}|{arm['score']:.2f}|{arm['joint_carry_events']}|{arm['joint_delivery_events']}|{arm['model_calls']}|{ending}|")
    lines += ['', '## 核验事实', '',
              f"- 16 份源码当前文件及快照、预备案例摘要、旧 v3 记录与 R/S、M、MAP 来源链全部复核。所有 {len(calls)} 份实际提示均由本臂真实经历重新构造，并核对了模板 SHA、源种子、温度和缓存标记。权重文件未在本审计中重新全量计算摘要。",
              '- 起点均为同一 S0、t=8、原期限 12。初始个人历史分别为每人 8 条和 0 条；首次窗口没有旧第 9 步广播。每窗只看到本臂此前已结束窗口的广播，同窗输出不会提前公开；三人的物理动作统一使用该步开始时的世界结算。',
              '- 清空仅发生在该臂起点，后续正常累积本臂每个人的原始观察、自身动作、自身反馈与新公开广播。私有分析只用于同次正式生成，没有进入持续历史；研究者源记录、参考动作和其他人未执行动作没有加入提示。',
              '- 每步的三份完整信息观察、本地菜单、正式广播、动作、反馈、目标进度及世界终态均独立重算。双人成功事件按同一物件和同一搭档对计一次；搬运和交付分别计数。',
              '- 三步是研究者观察上限，没有作为新期限写入主体提示。若 t=11 时仍未完成，环境状态应为 running，未写入任务结束记忆；这不是第 12 步超时，也不预测剩余一步结果。', '',
              '## 原轨迹复现与停止原因', '']
    if reproduction['complete_90_call_reproduction']:
        lines += ['保留历史臂的新 call 1—90 对应旧 call 241—330，90 次输入、模板摘要、生成全文／输出、种子、温度、模式、缓存标记及结束原因全部精确一致。三个新物理步的观察、菜单、广播、动作、反馈、得分及世界状态也与旧第 9—11 步相同。这是复现控制，不能当作新的独立群体证据。']
    else:
        differing = sum(not all(r['equal'].values()) for r in reproduced)
        lines += [f"保留历史臂比较了 {len(reproduced)} 个源调用，其中 {differing} 个存在至少一项差异；未达到完整 90 调用复现。各字段差异已保留在 verification.json，不能静默当作原轨迹已重现。"]
    lines += ['', '|条件|私有分析 stop／length|自然广播 stop／length|正式动作 stop／length|最大提示 token|',
              '|---|---|---|---|---:|']
    for arm in arm_details:
        counts = [f"{arm['finish_reasons'][mode].get('stop', 0)}/{arm['finish_reasons'][mode].get('length', 0)}"
                  for mode in ('private_analysis', 'natural_message', 'action')]
        lines.append(f"|{NAMES[arm['arm_id']]}|{counts[0]}|{counts[1]}|{counts[2]}|{arm['max_prompt_tokens']}|")
    lines += ['', '触及 256 token 上限的私有分析为新 call 55（原臂步10 A 动作）、89（原臂步11 C 动作）、93（清空臂步9 首窗 B 广播）、119（清空臂步9 C 动作）、147（清空臂步10 B 动作）。所有正式广播和动作正常停止。截断次数是日志事实，不能据此直接判定某次错误的原因。', '', '## 实际动作', '',
              '|条件／源步|A|B|C|本步新增交付单位|', '|---|---|---|---|---:|']
    for row in step_checks:
        desc = row['action_descriptions']
        lines.append(f"|{NAMES[row['arm_id']]}／{row['step']}|{desc['A']}|{desc['B']}|{desc['C']}|{row['outcomes']['delivered_this_step']}|")
    lines += ['', '清空臂 B 在起点已携带物155（干长纤维），地面的物143也为干长纤维；两件都符合需求。第9步放下155、第10步等待、第11步拿143，不能记为“拿错属性”。call117明确计划先放155再换143；call147又把长纤维误说为需要双人，并在多个动作间摇摆后截断；call177正确识别物143，但把之后的移动与交付合写成“下步直接移至营地交付”。环境将移动与普通交付作为不同物理动作。这些是生成文本与实际记录的对应证据，不足以将失败归因于某个内部机制；第12步没有执行。']
    lines += ['', '两臂来自同一已知失败起点；删除历史同时改变过去信息和输入长度，后续消息、动作与状态也可能分化。因此不能将差异定位到某一句旧消息或某个内部机制。清空历史臂仍可见当前观察中的上步自身反馈。该完整信息、自然语言短续跑没有检验新符号约定，也不改写旧 v3 能力门槛或第二场实际合作成功的证据。']
    if errors:
        lines += ['', '## 核验错误', ''] + ['- ' + e for e in errors]
    (OUT / '独立核验.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': audit['status'], 'errors': errors, 'scope': audit['scope'],
                      'arms': arm_details, 'original_complete_reproduction': reproduction['complete_90_call_reproduction']},
                     ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
