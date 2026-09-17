"""Run a frozen capability gate, then the authorized symbol study if passed."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from .environment import World

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verify_frozen():
    frozen = json.loads((ROOT/'core_source_frozen.json').read_text())
    for name, expected in frozen['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen source differs: {name}')
    actual = hashlib.sha256((ROOT/'experiment_locked.json').read_bytes()).hexdigest()
    if actual != frozen['experiment_sha256']:
        raise RuntimeError('Locked experiment configuration changed')


def check_capability(run_dir, condition, planned):
    """Check terminal delivery states and fresh-history provenance, not reports."""
    manifest = json.loads((run_dir/'manifest.json').read_text())
    status = json.loads((run_dir/'status.json').read_text())
    rows = [json.loads(x) for x in (run_dir/'episodes.jsonl').read_text().splitlines()]
    steps = [json.loads(x) for x in (run_dir/'steps.jsonl').read_text().splitlines()]
    expected = [(i+1, sc['seed'], sc['variant']) for i, sc in enumerate(planned)]
    expected_manifest = [{'episode':episode,'seed':seed,'variant':variant} for episode,seed,variant in expected]
    identities = [(r['episode'], r['seed'], r['variant']) for r in rows]
    if (status['status'] != 'completed' or manifest['history_mode'] != 'independent_episodes'
            or manifest.get('phase') != 'calibration' or manifest.get('conditions') != [condition]
            or manifest.get('groups') != [17] or manifest.get('scenarios') != expected_manifest
            or sorted(identities) != sorted(expected) or len(rows) != len(expected)
            or any(r['phase'] != 'calibration' or r['condition'] != condition or r['group'] != 17 for r in rows)):
        raise RuntimeError('Capability run does not match its completed locked design')
    cases = []
    for row in rows:
        episode = row['episode']
        history = json.loads((run_dir/f'checkpoint_calibration_{condition}_17_{episode}_before.json').read_text())
        if history != {a:[] for a in 'ABC'}:
            raise RuntimeError('Capability scenario did not start with fresh private histories')
        trajectory = [s for s in steps if s['condition'] == condition and s['episode'] == episode and s['group'] == 17]
        if (not trajectory or [s['step'] for s in trajectory] != list(range(1,row['steps']+1))
                or any(s.get('seed') != row['seed'] or s.get('phase') != 'calibration' for s in trajectory)):
            raise RuntimeError('Incomplete capability trajectory')
        replay = World(row['seed'], variant=row['variant'])
        for recorded in trajectory:
            if recorded.get('state_before') != replay.state_dict():
                raise RuntimeError('Capability state_before differs from deterministic replay')
            feedback = replay.step(recorded['actions'])
            recorded_score = recorded.get('score')
            if (recorded.get('state_after') != replay.state_dict() or recorded.get('feedback') != feedback
                    or not isinstance(recorded_score,(int,float)) or not math.isfinite(recorded_score)
                    or abs(recorded_score-replay.score) > 1e-12):
                raise RuntimeError('Capability transition differs from deterministic replay')
        final = trajectory[-1]['state_after']
        if (final.get('seed') != row['seed'] or final.get('variant') != row['variant']
                or final.get('t') != row['steps'] or final.get('max_steps') != 12
                or any(trajectory[-1]['feedback'].get(a,{}).get('done') is not True for a in 'ABC')):
            raise RuntimeError('Capability terminal state or feedback is inconsistent')
        goals = final['goals']
        if (not goals or any(type(g.get('quantity')) is not int or g['quantity'] <= 0
                or type(g.get('delivered')) is not int or not 0 <= g['delivered'] <= g['quantity'] for g in goals)):
            raise RuntimeError('Invalid capability goal counts')
        all_delivered = all(g['delivered'] >= g['quantity'] for g in goals)
        score = sum(g['delivered'] for g in goals)/sum(g['quantity'] for g in goals)
        if (not isinstance(row['score'],(int,float)) or not math.isfinite(row['score'])
                or abs(score-row['score']) > 1e-12 or row['success'] is not all_delivered
                or abs(score-trajectory[-1]['score']) > 1e-12):
            raise RuntimeError('Episode summary disagrees with actual terminal deliveries')
        cases.append({'episode':episode, 'seed':row['seed'], 'variant':row['variant'],
            'score':row['score'], 'steps':row['steps'], 'all_goals_delivered':all_delivered,
            'fresh_histories_verified':True, 'passed':all_delivered and row['steps'] <= 12})
    return {'condition':condition, 'cases':cases, 'passed':all(c['passed'] for c in cases)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-dir', type=Path)
    args = parser.parse_args()
    verify_frozen()
    config = json.loads((ROOT/'experiment_locked.json').read_text())
    batch = args.batch_dir or ROOT/'results'/datetime.now().strftime('%Y%m%d_%H%M%S')
    batch.mkdir(parents=True, exist_ok=False)
    for name in ('experiment_locked.json','core_source_frozen.json'):
        shutil.copy2(ROOT/name, batch/name)
    (ROOT/'LATEST_BATCH').write_text(str(batch.resolve())+'\n')
    directories = []
    gates = []

    def status(stage, **extra):
        value = {'updated':datetime.now().isoformat(), 'stage':stage, **extra}
        write_json(batch/'pipeline_status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def execute(stage, command):
        verify_frozen()
        status(stage, command=command)
        subprocess.run(command, cwd=WORK, check=True)

    def run_stage(condition, phase, scenarios, history_mode, directory):
        directories.append(directory)
        execute(condition if phase == 'calibration' else 'symbolic_pilot', [sys.executable,'-m','qwen_language_v3.run',
            '--phase',phase,'--conditions',condition,'--groups','17','--episodes',str(len(scenarios)),
            '--variants',','.join(str(s['variant']) for s in scenarios),
            '--seeds',','.join(str(s['seed']) for s in scenarios),
            '--history-mode',history_mode,'--run-dir',str(directory.resolve())])

    def finish(outcome):
        available = [p for p in directories if (p/'manifest.json').exists()]
        if available:
            execute('combined_report', [sys.executable,'-m','qwen_language_v3.report_suite',
                *(str(p.resolve()) for p in available),'--out',str((batch/'summary').resolve())])
            for directory in available:
                execute('replay_export', [sys.executable,'-m','qwen_language_v3.replay_snapshot',
                    str(directory.resolve()),'--out',str((batch/'replays'/directory.name/'replay.html').resolve())])
        status('completed', outcome=outcome, gates=gates,
            symbolic_started=any(p.name=='symbolic_immediate' for p in directories),
            report=str((batch/'summary/report.md').resolve()))

    try:
        for condition in ('full_information','natural'):
            directory = batch/f'capability_{condition}'
            run_stage(condition, 'calibration', config['capability_scenarios'], 'independent_episodes', directory)
            gate = check_capability(directory, condition, config['capability_scenarios'])
            gates.append(gate)
            write_json(batch/'capability_gates.json', gates)
            # Both cases in the current tier finish even if the first fails.
            if not gate['passed']:
                finish(f'{condition}_capability_not_passed')
                return
        pilot = batch/'symbolic_immediate'
        run_stage('immediate', 'pilot', config['symbolic_scenarios'], 'continuous', pilot)
        for selection in config['frozen_probes']:
            execute('frozen_probe', [sys.executable,'-m','qwen_language_v3.probe',str(pilot.resolve()),
                '--episode',str(selection['episode']),'--step',str(selection['step'])])
        finish('symbolic_study_completed')
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error), gates=gates)
        raise


if __name__ == '__main__':
    main()
