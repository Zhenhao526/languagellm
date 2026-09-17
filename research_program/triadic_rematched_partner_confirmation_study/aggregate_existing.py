"""Recover a completed grid from per-run JSON files only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import design, metrics


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    execution = source / 'execution'; runs = []
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            path = execution / f'seed_{seed}_{condition}' / 'result.json'
            if not path.is_file():
                raise FileNotFoundError(path)
            runs.append(json.loads(path.read_text()))
    if len(runs) != 64:
        raise ValueError('Expected 64 completed runs')
    result = dict(status='completed_after_json_only_aggregation', source=str(source), runs=runs,
                  primary=metrics.summarize(runs), no_model_calls=True, no_optimizer_updates=True,
                  recovery_note='Read per-run result.json files after all training workers completed; no checkpoints or model outputs were read.')
    result_path = output / 'results.json'; result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_runs=len(runs), model_forwards=0, optimizer_updates=0,
                   output_results_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.source, args.output)
