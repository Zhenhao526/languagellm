"""JSON-only aggregation for the neutral-action pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import design, metrics


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    raw = json.loads((source / 'execution/results.json').read_text())
    expected = len(design.SEEDS) * len(design.CONDITIONS)
    if raw.get('status') != 'completed' or len(raw.get('runs', [])) != expected:
        raise ValueError('Incomplete pilot execution')
    result = dict(status='completed_json_only_aggregation', source=str(source), runs=raw['runs'], primary=metrics.summarize(raw['runs']),
                  no_model_calls=True, no_optimizer_updates=True, recovery_note='Read completed per-run JSON only; no forward pass or training update.')
    path = output / 'results.json'; path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_rows=expected, model_forwards=0, optimizer_updates=0,
                   output_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), no_model_calls=True)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True); args = parser.parse_args(); main(args.source, args.output)
