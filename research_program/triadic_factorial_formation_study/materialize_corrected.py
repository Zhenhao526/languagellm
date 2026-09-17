"""Materialize corrected per-run JSON identities without copying large arrays."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stream_runs(path):
    """Yield objects in the top-level runs array without loading the 3.5 GB file."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8') as stream:
        marker = '"runs":'
        buffer = ''
        marker_index = -1
        while marker_index < 0:
            chunk = stream.read(1 << 20)
            require(chunk, 'Top-level runs array missing')
            buffer += chunk
            search = 0
            while True:
                candidate = buffer.find(marker, search)
                if candidate < 0:
                    break
                tail = buffer[candidate + len(marker):].lstrip()
                if tail.startswith('['):
                    marker_index = candidate
                    break
                search = candidate + len(marker)
        buffer = buffer[marker_index + len(marker):]
        while True:
            if not buffer:
                buffer = stream.read(1 << 20)
                require(buffer, 'Runs array truncated')
            buffer = buffer.lstrip()
            if not buffer:
                continue
            if buffer[0] == '[':
                buffer = buffer[1:]
                break
            raise ValueError('Runs array syntax')
        while True:
            while True:
                buffer = buffer.lstrip()
                if buffer:
                    break
                chunk = stream.read(1 << 20)
                require(chunk, 'Runs array truncated')
                buffer += chunk
            if buffer[0] == ']':
                return
            while True:
                try:
                    value, consumed = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    # A run object can span many chunks; retain the partial
                    # object and append until JSON becomes complete.
                    chunk = stream.read(1 << 20)
                    require(chunk, 'Run object truncated')
                    buffer += chunk
            yield value
            buffer = buffer[consumed:]
            while True:
                buffer = buffer.lstrip()
                if buffer:
                    break
                chunk = stream.read(1 << 20)
                require(chunk, 'Runs separator missing')
                buffer += chunk
            require(buffer[0] in ',]', 'Runs separator')
            if buffer[0] == ']':
                return
            buffer = buffer[1:]


def materialize(corrected_run, source_run):
    corrected_run = Path(corrected_run).resolve()
    source_run = Path(source_run).resolve()
    source_execution = source_run / 'execution'
    destination_execution = corrected_run / 'execution'
    total = destination_execution / 'results.json'
    require(total.exists(), 'Corrected total result missing')
    rows = 0
    for row in stream_runs(total):
        seed = int(row['seed'])
        condition = row['condition']
        source_dir = source_execution / f'seed_{seed}_{condition}'
        destination_dir = destination_execution / f'seed_{seed}_{condition}'
        require(source_dir.is_dir(), f'Source run missing: {source_dir}')
        if destination_dir.is_symlink():
            destination_dir.unlink()
        require(not destination_dir.exists(), f'Destination run already materialized: {destination_dir}')
        destination_dir.mkdir()
        for source_file in source_dir.iterdir():
            if source_file.name == 'result.json':
                continue
            os.symlink(source_file.resolve(), destination_dir / source_file.name)
        with (destination_dir / 'result.json').open('x', encoding='utf-8') as stream:
            json.dump(row, stream, indent=2, ensure_ascii=False, allow_nan=False)
        rows += 1
    require(rows == 128, f'Expected 128 corrected run rows, got {rows}')
    return rows


def main(corrected_run, source_run, output='materialize_receipt_001'):
    corrected_run = Path(corrected_run).resolve()
    receipt_path = corrected_run / output
    require(not receipt_path.exists(), 'Never overwrite materialization receipt')
    count = materialize(corrected_run, source_run)
    value = dict(status='passed', materialized_runs=count,
                 corrected_results_sha256=sha(corrected_run / 'execution' / 'results.json'),
                 source_execution=str(Path(source_run).resolve() / 'execution'),
                 npz_copies=0, checkpoint_copies=0, training_log_copies=0,
                 reused_files='Absolute symlinks to the immutable original execution tree.',
                 at=datetime.now(timezone.utc).isoformat())
    receipt_path.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corrected-run', required=True)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--output', default='materialize_receipt_001')
    args = parser.parse_args()
    print(json.dumps(main(args.corrected_run, args.source_run, args.output), ensure_ascii=False))
