"""Finish the current, authorized pilot sequentially; never resume/overwrite a run."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
CONTROL = ROOT / 'results/calibration_complete_rules_20260914'
PILOT = ROOT / 'results/pilot_immediate_delayed_20260914'
REPORT = ROOT / 'results/final_summary_20260914'
STATUS = ROOT / 'results/pipeline_status_20260914.json'


def status(stage, **extra):
    data = {'updated': datetime.now().isoformat(), 'stage': stage, **extra}
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(STATUS)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def verify_core():
    frozen = json.loads((ROOT / 'core_source_frozen.json').read_text())
    for name, expected in frozen['source_sha256'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Frozen source changed: {name}')


def execute(stage, command):
    status(stage, command=command)
    subprocess.run(command, cwd=WORK, check=True)


def main():
    if PILOT.exists():
        raise FileExistsError(f'Pilot directory exists; refusing overwrite: {PILOT}')
    verify_core()
    status('waiting_for_current_control', control=str(CONTROL))
    while True:
        current = json.loads((CONTROL / 'status.json').read_text())
        if current['status'] == 'completed':
            # Let the original process finish its report and release model memory.
            processes = subprocess.check_output(['ps', '-axo', 'command='], text=True)
            if not any('python -m qwen_language_v2.run --phase calibration' in line
                       and 'calibration_complete_rules_20260914' in line
                       for line in processes.splitlines()):
                break
        elif current['status'] != 'running':
            raise RuntimeError(f'Control did not complete: {current["status"]}')
        time.sleep(5)
    verify_core()
    locked = json.loads((ROOT / 'pilot_budget_locked.json').read_text())
    execute('symbolic_pilot', [sys.executable, '-m', 'qwen_language_v2.run',
        '--phase', 'pilot', '--conditions', ','.join(locked['conditions']),
        '--groups', ','.join(map(str, locked['group_seeds'])),
        '--episodes', str(locked['episodes_per_group']),
        '--variants', ','.join(map(str, locked['scenario_variants'])),
        '--run-dir', str(PILOT)])
    verify_core()
    selection = locked['probe_selection']
    execute('frozen_probe', [sys.executable, '-m', 'qwen_language_v2.probe', str(PILOT),
        '--episode', str(selection['episode']), '--step', str(selection['physical_step'])])
    execute('combined_report', [sys.executable, '-m', 'qwen_language_v2.report_suite',
        str(ROOT / 'results/calibration_natural_20260914'),
        str(ROOT / 'results/calibration_clarified_20260914'), str(CONTROL), str(PILOT),
        '--out', str(REPORT)])
    status('completed', pilot=str(PILOT), report=str(REPORT / 'report.md'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc))
        raise
