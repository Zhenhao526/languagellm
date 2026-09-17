"""Paired three-agent continuations from one frozen physical state.

The three-step cap is researcher-side observation truncation, never a new
deadline in a model prompt. Each arm generates all broadcasts and actions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

from qwen_language_v3.agents import build_prompt, remember
from qwen_language_v3.backend import Backend
from qwen_language_v3.environment import AGENTS, RULES, World
from qwen_language_v3.protocol import run_natural_communication
from qwen_language_v3.run import append_json, inference_seed, write_json
from .experiment import ROOT, WORK, SOURCE_RUN, digest
from .map_presentation import map_sources, verify_map_prepared
from .continuation_cases import build_continuation_cases

PRIOR_MAP = ROOT / 'results/20260915_MAP_01'
DESIGN = PRIOR_MAP / '下一步设计判断.md'
ARMS = ('original_history', 'reset_history')
MAX_NEW_STEPS = 3
MAX_CALLS = 180
SOURCE_FIELDS = ('step', 'observations', 'menus', 'messages', 'channel_usage',
                 'selections', 'actions', 'feedback', 'score', 'state_before', 'state_after')


def validate_prepared(prepared):
    if [arm['arm_id'] for arm in prepared['arms']] != list(ARMS):
        raise ValueError('Unexpected continuation arm order')
    world = World.from_state_dict(prepared['state'])
    if world.t != 8 or world.max_steps != 12 or world.done:
        raise ValueError('Continuation must start at original t8 with deadline12')
    if [row['step'] for row in prepared['source_records']] != [9, 10, 11]:
        raise ValueError('Missing fixed source steps')
    for arm, expected in zip(prepared['arms'], (8, 0)):
        if set(arm['histories']) != set(AGENTS) or any(len(v) != expected for v in arm['histories'].values()):
            raise ValueError('Incorrect initial history lengths')


def _decide(backend, prompt, **kwargs):
    if backend.calls + 2 > MAX_CALLS:
        raise RuntimeError('Predeclared model-call cap exceeded')
    start = backend.calls
    answer = backend.decide(deepcopy(prompt), **kwargs)
    if backend.calls != start + 2:
        raise RuntimeError('Expected exactly two calls per decision')
    return answer


def step_outcomes(row):
    before, after = row['state_before'], row['state_after']
    goals = [
        {'goal_index': i, **{k: g[k] for k in ('kind', 'length', 'condition', 'destination', 'quantity')},
         'delivered_before': before['goals'][i]['delivered'], 'delivered_after': g['delivered'],
         'delivered_this_step': g['delivered'] - before['goals'][i]['delivered']}
        for i, g in enumerate(after['goals'])
    ]
    joint = []
    for agent in AGENTS:
        action = row['actions'][agent]
        if action['kind'] not in ('carry_together', 'deliver_together', 'drop_together'):
            continue
        oid = next((oid for oid, handle in before['handles'][agent].items()
                    if handle == action.get('item')), None)
        partner = action.get('partner')
        other = row['actions'].get(partner, {})
        other_oid = next((oid for oid, handle in before['handles'].get(partner, {}).items()
                          if handle == other.get('item')), None)
        matching = (oid is not None and oid == other_oid and other.get('partner') == agent
                    and action['kind'] == other.get('kind')
                    and (action['kind'] != 'carry_together' or action['destination'] == other.get('destination')))
        joint.append({'agent': agent, 'kind': action['kind'], 'researcher_object_id': oid,
                      'partner': partner, 'destination': action.get('destination'),
                      'proposals_match': matching,
                      'action_succeeded': row['feedback'][agent]['action_succeeded']})
    successes = {kind: len({(j['researcher_object_id'], tuple(sorted((j['agent'], j['partner']))))
                            for j in joint if j['kind'] == kind and j['action_succeeded']})
                 for kind in ('carry_together', 'deliver_together')}
    return {'goals': goals, 'joint_proposals': joint,
            'successful_joint_object_actions': successes,
            'delivered_this_step': sum(g['delivered_this_step'] for g in goals)}


def run_arm(backend, prepared, arm_id, on_step=lambda row: None):
    arm = next(arm for arm in prepared['arms'] if arm['arm_id'] == arm_id)
    world = World.from_state_dict(deepcopy(prepared['state']))
    histories = deepcopy(arm['histories'])
    start_calls, start_t = backend.calls, world.t
    started = time.perf_counter()
    steps = []
    while not world.done and len(steps) < MAX_NEW_STEPS:
        step = world.t + 1
        observations = {a: world.full_information_observe(a) for a in AGENTS}
        menus = {a: world.action_menu(a) for a in AGENTS}
        before = world.state_dict()
        history_before = deepcopy(histories)
        call_before = backend.calls
        label_base = dict(phase='history_continuation', arm_id=arm_id,
                          condition='full_information', group=17, episode=1, seed=92015, step=step)

        def decide_message(agent, window, remaining, visible):
            prompt = build_prompt(agent, rules=RULES, observation=observations[agent],
                                  history=histories[agent], transcript=visible,
                                  condition='full_information', window=window, remaining=remaining)
            if step == 9 and window == 1 and prompt != prepared['first_prompts'][arm_id][agent]:
                raise RuntimeError('Initial broadcast prompt differs from frozen input')
            return _decide(backend, prompt, mode='natural_message', limit=32, temperature=.7,
                           seed=inference_seed(17, 1, step, window, agent, 'message'),
                           label={**label_base, 'window': window, 'agent': agent})

        communication = run_natural_communication(decide_message, step=step)
        actions, selections = {}, {}
        for agent in AGENTS:
            prompt = build_prompt(agent, rules=RULES, observation=observations[agent],
                                  history=histories[agent], transcript=communication['transcripts'][agent],
                                  condition='full_information', menu=menus[agent])
            choice = int(_decide(backend, prompt, mode='action', choices=[m['id'] for m in menus[agent]],
                                 seed=inference_seed(17, 1, step, 0, agent, 'action'),
                                 label={**label_base, 'window': None, 'agent': agent}))
            selected = next(m for m in menus[agent] if m['id'] == choice)
            selections[agent] = deepcopy(selected)
            actions[agent] = deepcopy(selected['action'])
        feedback = world.step(actions)
        end = {'目标完成比例': world.score, '已用动作步': step} if world.done else None
        remember(histories, observations=observations, messages=communication['messages'],
                 actions=actions, feedback=feedback, episode_end=end)
        row = {**label_base, 'observations': observations, 'menus': menus,
               'messages': communication['messages'], 'channel_usage': communication.get('usage'),
               'actions': actions, 'selections': selections, 'feedback': feedback, 'score': world.score,
               'state_before': before, 'state_after': world.state_dict(),
               'private_histories_before_step': history_before,
               'history_lengths_after': {a: len(histories[a]) for a in AGENTS},
               'call_range': [call_before + 1, backend.calls]}
        row['outcomes'] = step_outcomes(row)
        if backend.calls - call_before != 30:
            raise RuntimeError('Each completed step must use30 calls')
        steps.append(row)
        on_step(deepcopy(row))
    summary = {'arm_id': arm_id, 'start_t': start_t, 'end_t': world.t, 'new_steps': len(steps),
               'engine_status': world.status, 'end_reason': 'environment_terminal' if world.done else 'observation_limit',
               'success': world.status == 'success', 'score': world.score,
               'delivered_units': sum(g['delivered'] for g in world.goals),
               'required_units': sum(g['quantity'] for g in world.goals), 'goals': deepcopy(world.goals),
               'model_calls': backend.calls - start_calls, 'seconds': time.perf_counter() - started,
               'history_lengths_after': {a: len(histories[a]) for a in AGENTS}}
    return {'summary': summary, 'steps': steps, 'histories_after': histories}


def run_cases(backend, prepared, on_step=lambda row: None, on_arm=lambda result: None):
    validate_prepared(prepared)
    if backend.calls != 0:
        raise RuntimeError('A new continuation batch must use a fresh backend log')
    results = []
    for arm_id in ARMS:
        result = run_arm(backend, prepared, arm_id, on_step)
        results.append(result)
        on_arm(deepcopy(result))
    return results


def reproduction(results, prepared, calls):
    actual = [r for r in calls if r['label']['arm_id'] == 'original_history']
    old_by_key = {(r['label']['step'], r['label']['window'], r['label']['agent'], r['label']['stage']): r
                  for r in prepared['source_calls']}
    checks = []
    fields = ('messages', 'prompt_sha256', 'output', 'seed', 'temperature', 'mode',
              'fresh_cache', 'native_thinking', 'valid', 'finish_reason')
    for row in actual:
        label = row['label']
        old = old_by_key[(label['step'], label['window'], label['agent'], label['stage'])]
        checks.append({'new_call': row['call'], 'source_call': old['call'],
                       'equal': {field: row[field] == old[field] for field in fields}})
    old_steps = {r['step']: r for r in prepared['source_records']}
    new_steps = next(r['steps'] for r in results if r['summary']['arm_id'] == 'original_history')
    step_checks = [{'step': r['step'], 'equal': {field: r[field] == old_steps[r['step']][field]
                                               for field in SOURCE_FIELDS}} for r in new_steps]
    return {'calls_compared': len(checks), 'expected_source_calls': 90, 'calls': checks, 'steps': step_checks,
            'complete_90_call_reproduction': len(checks) == 90
            and all(all(row['equal'].values()) for row in checks + step_checks)}


def continuation_sources():
    paths = map_sources()
    for name in ('continuation.py', 'continuation_cases.py'):
        paths[f'qwen_language_diagnostics/{name}'] = ROOT / name
    return paths


def prepare(out):
    old = verify_map_prepared(PRIOR_MAP)
    prepared = build_continuation_cases()
    validate_prepared(prepared)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'prepared_cases.json', prepared)
    previous = deepcopy(old['prior_result_files_sha256'])
    previous.update({str(PRIOR_MAP / name): digest(PRIOR_MAP / name)
                     for name in ('manifest.json', 'prepared_cases.json', 'results.json', 'inference.jsonl')})
    previous.update(prepared['source_files_sha256'])
    previous[str(DESIGN)] = digest(DESIGN)
    manifest = {'prepared_at': datetime.now().astimezone().isoformat(),
                'source_run': str(SOURCE_RUN), 'model': old['model'], 'model_commit': old['model_commit'],
                'cases_sha256': digest(out / 'prepared_cases.json'),
                'source_sha256': {name: digest(path) for name, path in continuation_sources().items()},
                'source_records_sha256': deepcopy(old['source_records_sha256']),
                'prior_result_files_sha256': previous, 'arm_order': list(ARMS),
                'max_new_steps_per_arm': MAX_NEW_STEPS, 'max_backend_calls': MAX_CALLS,
                'calls_per_complete_step': 30, 'same_source_inference_seeds_in_both_arms': True,
                'source_group': 17, 'source_episode': 1, 'source_world_seed': 92015,
                'source_steps': [9, 10, 11], 'source_call_range': [241, 330],
                'original_deadline': 12, 'observer_stop_t': 11, 'modifies_model_deadline': False,
                'history_intervention': 'Only clear all three initial histories in reset_history; accumulate new own histories normally.',
                'model_parameters': {'private_analysis_max_tokens': 256, 'private_analysis_temperature': 0,
                                     'natural_message_max_tokens': 96, 'natural_message_temperature': .7,
                                     'action_temperature': 0, 'fresh_cache_every_call': True, 'native_thinking': False},
                'no_retries': True, 'run_second_arm_regardless_of_first_outcome': True,
                'unlocks_symbolic_experiment': False, 'updates_source_histories': False,
                'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                            'packages': {name: importlib.metadata.version(name)
                                         for name in ('mlx', 'mlx-lm', 'transformers')}},
                'interpretation_limits': [
                    'One selected failure state; no independent population-level estimate.',
                    'Initial history removal changes past information and prompt length; subsequent interaction is generated anew.',
                    'Only A/C can immediately form the required pair at this start; partner-choice generalization is untested.',
                    'Observation ending at t11 is not a timeout at the unchanged t12 deadline.',
                    'Full-information natural-language coordination does not demonstrate new symbolic convention formation.']}
    for name, path in continuation_sources().items():
        target = out / 'code_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'status.json', {'status': 'prepared', 'model_calls': 0})
    return manifest


def verify_continuation_prepared(out):
    return verify_map_prepared(out)


def execute(out):
    manifest = verify_continuation_prepared(out)
    if json.loads((out / 'status.json').read_text())['status'] != 'prepared' or (out / 'inference.jsonl').exists():
        raise RuntimeError('Refuse to rerun or overwrite a started continuation')
    prepared = json.loads((out / 'prepared_cases.json').read_text())
    validate_prepared(prepared)
    write_json(out / 'status.json', {'status': 'running', 'started': datetime.now().astimezone().isoformat()})
    backend = None
    try:
        backend = Backend(Path(manifest['model']), out / 'inference.jsonl')

        def on_step(row):
            append_json(out / 'steps.jsonl', row)
            brief = {'arm_id': row['arm_id'], 'source_step': row['step'], 'score': row['score'],
                     'model_calls': backend.calls, 'actions': row['actions'],
                     'outcomes': row['outcomes'], 'backend': backend.stats()}
            write_json(out / 'status.json', {'status': 'running', **brief})
            print(json.dumps({'event': 'step_completed', **brief}, ensure_ascii=False), flush=True)

        def on_arm(result):
            arm_id = result['summary']['arm_id']
            write_json(out / f'{arm_id}_histories_after.json', result['histories_after'])
            append_json(out / 'arms.jsonl', result['summary'])
            print(json.dumps({'event': 'arm_completed', **result['summary']}, ensure_ascii=False), flush=True)

        runs = run_cases(backend, prepared, on_step, on_arm)
        calls = [json.loads(line) for line in (out / 'inference.jsonl').read_text().splitlines()]
        if len(calls) != backend.calls or backend.calls != 30 * sum(len(r['steps']) for r in runs):
            raise RuntimeError('Completed steps and actual call count disagree')
        verify_continuation_prepared(out)
        result = {'arms': [r['summary'] for r in runs], 'model_calls': backend.calls,
                  'max_model_calls': MAX_CALLS, 'new_physical_steps': sum(len(r['steps']) for r in runs),
                  'reproduction': reproduction(runs, prepared, calls), 'symbolic_started': False}
        write_json(out / 'results.json', result)
        write_json(out / 'backend_stats.json', backend.stats())
        write_json(out / 'status.json', {'status': 'completed', 'completed': datetime.now().astimezone().isoformat(),
                                         'model_calls': backend.calls, 'arms': result['arms'], 'symbolic_started': False})
        print(json.dumps({'event': 'completed', 'model_calls': backend.calls, 'out': str(out)}, ensure_ascii=False), flush=True)
    except BaseException as error:
        write_json(out / 'status.json', {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                                         'backend': backend.stats() if backend is not None else None})
        raise
    finally:
        if backend is not None:
            backend.log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'execute'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == 'prepare':
        manifest = prepare(out)
        (ROOT / 'LATEST_CONTINUATION').write_text(str(out) + '\n')
        print(json.dumps({'out': str(out), 'max_model_calls': manifest['max_backend_calls']}, ensure_ascii=False))
    else:
        execute(out)


if __name__ == '__main__':
    main()
