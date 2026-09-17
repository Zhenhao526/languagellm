"""Run the redesigned pilot locally; all controls have independent histories."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time

from .agents import build_prompt, remember
from .backend import Backend
from .environment import AGENTS, RULES, World
from .protocol import run_communication, run_natural_communication

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent/'qwen_collect_pilot/models/Qwen3.5-9B-8bit'


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def append_json(path, data):
    with path.open('a') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def inference_seed(group, episode, step, window, agent, stage):
    # Same planned seed for paired conditions; each inference reseeds locally.
    key = f'{group}/{episode}/{step}/{window}/{agent}/{stage}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big') % (2**31-1)


def run_episode(backend, run_dir, *, condition, group, episode, seed, variant, histories, phase):
    world = World(seed, variant=variant)
    label_base = dict(phase=phase, condition=condition, group=group, episode=episode, seed=seed)
    all_messages, timeline = [], []
    started = time.perf_counter()
    natural = condition in ('natural', 'full_information')
    write_json(run_dir/f'world_{phase}_{condition}_{group}_{episode}.json', {'initial_state':world.state_dict(), 'witness':world.witness})
    write_json(run_dir/f'checkpoint_{phase}_{condition}_{group}_{episode}_before.json', histories)
    step = 0
    while not world.done:
        step += 1
        observations = {a:(world.full_information_observe(a) if condition=='full_information' else world.observe(a)) for a in AGENTS}
        menus = {a:world.action_menu(a) for a in AGENTS}
        state_before = world.state_dict()
        hist_before = deepcopy(histories)
        def decide_message(agent, window, remaining, visible):
            prompt = build_prompt(agent, rules=RULES, observation=observations[agent], history=histories[agent],
                transcript=visible, condition=condition, window=window, remaining=remaining)
            return backend.decide(prompt, mode='natural_message' if natural else 'message',
                limit=min(32, remaining) if remaining is not None else 32,
                temperature=0.7, seed=inference_seed(group,episode,step,window,agent,'message'),
                label={**label_base,'step':step,'window':window,'agent':agent})
        communication = run_natural_communication(decide_message, step=step) if natural else run_communication(condition, decide_message, step=step)
        actions, selections = {}, {}
        for agent in AGENTS:
            prompt = build_prompt(agent, rules=RULES, observation=observations[agent], history=histories[agent],
                transcript=communication['transcripts'][agent], condition=condition, menu=menus[agent])
            choice = int(backend.decide(prompt, mode='action', choices=[x['id'] for x in menus[agent]],
                seed=inference_seed(group,episode,step,0,agent,'action'),
                label={**label_base,'step':step,'window':None,'agent':agent}))
            selected = next(a for a in menus[agent] if a['id']==choice)
            actions[agent] = deepcopy(selected['action'])
            selections[agent] = deepcopy(selected)
        # All inputs were constructed before any actions were committed.
        feedback = world.step(actions)
        end = {'目标完成比例':world.score, '已用动作步':step} if world.done else None
        remember(histories, observations=observations, messages=communication['messages'], actions=actions, feedback=feedback, episode_end=end)
        all_messages.extend(communication['messages'])
        record = {**label_base, 'step':step, 'observations':observations, 'menus':menus,
            'messages':communication['messages'], 'channel_usage':communication.get('usage'),
            'selections':selections, 'actions':actions, 'feedback':feedback, 'score':world.score,
            'state_before':state_before, 'state_after':world.state_dict()}
        timeline.append(record)
        append_json(run_dir/'steps.jsonl', record)
        # Retain predecision inputs for frozen action-level interventions, never fed to training.
        if step in (1, 4, 8, 12):
            write_json(run_dir/f'probe_{phase}_{condition}_{group}_{episode}_{step}.json',
                {**record, 'private_histories_before_step':hist_before})
        progress = {**label_base, 'event':'step_completed', 'step':step, 'score':world.score,
            'elapsed_seconds':time.perf_counter()-started, 'backend':backend.stats()}
        print(json.dumps(progress, ensure_ascii=False), flush=True)
        write_json(run_dir/'status.json', {'status':'running', **progress})
    result = {**label_base, 'variant':variant, 'score':world.score, 'steps':step, 'success':world.score==1,
        'seconds':time.perf_counter()-started, 'messages':all_messages}
    append_json(run_dir/'episodes.jsonl', result)
    write_json(run_dir/f'checkpoint_{phase}_{condition}_{group}_{episode}_after.json', histories)
    write_json(run_dir/'backend_stats.json', backend.stats())
    print(json.dumps({'event':'episode_completed', **{k:v for k,v in result.items() if k!='messages'}},ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['calibration','pilot'], default='calibration')
    parser.add_argument('--conditions', default='natural')
    parser.add_argument('--groups', default='17')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--variants', help='Comma-separated task variants, fixed before the run; length must equal episodes')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    conditions = args.conditions.split(',')
    groups = list(map(int,args.groups.split(',')))
    if any(c not in ('immediate','delayed','natural','full_information') for c in conditions):
        parser.error('Unknown condition')
    run_dir = args.run_dir or ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = run_dir/'code_snapshot'
    snapshot.mkdir()
    for path in ROOT.glob('*.py'):
        shutil.copy2(path, snapshot/path.name)
    variants = list(map(int,args.variants.split(','))) if args.variants else list(range(args.episodes))
    if len(variants) != args.episodes:
        parser.error('--variants must have exactly --episodes entries')
    scenarios = [{'episode':ep+1,'seed':6101+variant*47,'variant':variant} for ep,variant in enumerate(variants)]
    write_json(run_dir/'manifest.json', dict(schema_version=1, phase=args.phase, conditions=conditions,
        groups=groups, episodes_per_group=args.episodes, scenarios=scenarios,
        model=str(args.model.resolve()), model_commit='16daa4818c54ce5f5436f929d52542eb65bbed9d',
        alphabet='@#%&*+=~', max_message_symbols=32, per_agent_step_symbols=64, windows=4,
        private_analysis_max_tokens=256, private_analysis_temperature=0, message_temperature=.7,
        action_temperature=0, natural_message_max_tokens=96, history='all own raw episodic observations/actions/feedback and delivered broadcasts; no analysis',
        controls='independent histories; natural language is not bandwidth matched',
        created=datetime.now().isoformat(), preregistered_order='group then episode then alternating condition order',
        scope='engineering and descriptive pilot; no statistical language-emergence claim'))
    backend = Backend(args.model, run_dir/'inference.jsonl')
    memories = {(c,g):{a:[] for a in AGENTS} for c in conditions for g in groups}
    try:
        for gi, group in enumerate(groups):
            for sc in scenarios:
                order = conditions if (gi+sc['episode'])%2 else list(reversed(conditions))
                for condition in order:
                    run_episode(backend, run_dir, condition=condition, group=group, phase=args.phase,
                        histories=memories[(condition,group)], **sc)
        write_json(run_dir/'status.json', {'status':'completed', 'backend':backend.stats()})
    except BaseException as error:
        write_json(run_dir/'status.json', {'status':'failed','error_type':type(error).__name__,'error':str(error),'backend':backend.stats()})
        raise
    finally:
        write_json(run_dir/'backend_stats.json', backend.stats())
        from .analyze import write_report
        write_report(run_dir)
        from .inspect_run import inspect
        inspect(run_dir)
    (ROOT/'LATEST_RUN').write_text(str(run_dir.resolve())+'\n')
    print(json.dumps({'event':'run_completed','run_dir':str(run_dir.resolve())}),flush=True)


if __name__ == '__main__':
    main()
