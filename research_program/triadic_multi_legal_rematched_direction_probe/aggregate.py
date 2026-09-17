"""JSON-only aggregation for the rematched directional probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import design, metrics


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    raw = json.loads((source / 'execution/results.json').read_text())
    if raw.get('status') != 'completed' or not raw.get('posthoc'):
        raise ValueError('Incomplete posthoc execution')
    rows = raw['rows']
    if len(rows) != len(design.SEEDS) * len(design.CONDITIONS):
        raise ValueError('Incomplete policy grid')
    result = dict(status='completed_json_only_posthoc_aggregation', source=str(source), rows=rows,
                  primary=metrics.summarize(rows), no_model_calls=False, no_optimizer_updates=True,
                  recovery_note='Read policy rows only; no optimizer updates or training calls.')
    result_path = output / 'results.json'; result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_rows=len(rows), model_forwards=raw.get('budget', {}).get('total_new_module_samples', 0),
                   optimizer_updates=0, output_results_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(), posthoc=True)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True); args = parser.parse_args(); main(args.source, args.output)
