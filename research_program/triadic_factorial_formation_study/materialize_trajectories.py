"""Write corrected trajectory JSON lines into the corrected run tree."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os

from .materialize_corrected import stream_runs, require, sha


def materialize(corrected_run):
    corrected_run = Path(corrected_run).resolve()
    execution = corrected_run / 'execution'
    rows = 0
    for row in stream_runs(execution / 'results.json'):
        directory = execution / f"seed_{int(row['seed'])}_{row['condition']}"
        require(directory.is_dir() and not directory.is_symlink(), f'Corrected run directory missing: {directory}')
        path = directory / 'trajectory.jsonl'
        require(path.exists() and (path.is_symlink() or path.is_file()), f'Corrected trajectory missing: {path}')
        if path.is_symlink():
            path.unlink()
        temporary = path.with_name(path.name + '.tmp')
        require(not temporary.exists(), f'Temporary trajectory already exists: {temporary}')
        with temporary.open('x', encoding='utf-8') as stream:
            for trajectory in row['trajectory']:
                stream.write(json.dumps(trajectory, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
                stream.write('\n')
        os.replace(temporary, path)
        rows += 1
    require(rows == 128, f'Expected 128 trajectories, got {rows}')
    return rows


def main(corrected_run, output='trajectory_materialization_receipt_001'):
    corrected_run = Path(corrected_run).resolve()
    receipt = corrected_run / output
    require(not receipt.exists(), 'Never overwrite trajectory receipt')
    rows = materialize(corrected_run)
    value = dict(status='passed', materialized_trajectories=rows,
                 corrected_results_sha256=sha(corrected_run / 'execution' / 'results.json'),
                 copied_npz=0, copied_checkpoints=0,
                 at=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corrected-run', required=True)
    parser.add_argument('--output', default='trajectory_materialization_receipt_001')
    args = parser.parse_args()
    print(json.dumps(main(args.corrected_run, args.output), ensure_ascii=False))
