"""JSON-only comparison of the FI and earlier PL alternative-partner runs.

The two information conditions were trained as separate populations.  This
script reports descriptive same-seed contrasts and does not label them as a
within-population causal effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import metrics


def require(ok, message):
    if not ok:
        raise ValueError(message)


KEYS = ('q_rate', 'conditional_q_rate', 'target_pair_legal_rate',
        'proposal_legal_rate', 'physical_execution_rate', 'engagement_rate',
        'third_agent_neutral_rate')


def load(path):
    data = json.loads(Path(path).read_text())
    require(data.get('status') == 'completed_after_json_only_aggregation',
            'Input is not a completed JSON-only aggregation: ' + str(path))
    require(len(data.get('runs', [])) == 64, 'Each information input must contain 64 runs')
    return data


def index(data):
    result = {(int(r['seed']), r['schedule'], bool(r['live'])): r for r in data['runs']}
    require(len(result) == 64, 'Run keys are not a complete 16×2×2 grid')
    return result


def endpoint(run, key):
    return float(run['final']['new_layouts'][key])


def trajectory(run, key):
    rows = sorted(run['trajectory'], key=lambda row: row['update'])
    return np.asarray([row['target_trajectory'][key] for row in rows], dtype=np.float64)


def contrast_stats(values):
    return metrics.stats(np.asarray(values, dtype=np.float64))


def main(fi_path, pl_path, output):
    output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    fi = load(fi_path); pl = load(pl_path); fi_by = index(fi); pl_by = index(pl)
    seeds = sorted({key[0] for key in fi_by})
    require(seeds == sorted({key[0] for key in pl_by}) and len(seeds) == 16,
            'FI and PL seed sets differ')

    cells = {}
    for schedule in ('static', 'rematched'):
        for live in (False, True):
            label = f'{schedule}_{"live" if live else "silent"}'
            fi_values = {key: [endpoint(fi_by[s, schedule, live], key) for s in seeds] for key in KEYS}
            pl_values = {key: [endpoint(pl_by[s, schedule, live], key) for s in seeds] for key in KEYS}
            cells[label] = dict(
                FI={key: dict(mean=float(np.mean(value)), statistics=contrast_stats(value)) for key, value in fi_values.items()},
                PL={key: dict(mean=float(np.mean(value)), statistics=contrast_stats(value)) for key, value in pl_values.items()},
                FI_minus_PL={key: dict(mean=float(np.mean(np.asarray(fi_values[key]) - np.asarray(pl_values[key]))),
                                       statistics=contrast_stats(np.asarray(fi_values[key]) - np.asarray(pl_values[key])))
                             for key in KEYS},
            )

    # Same-seed contrasts over training time, reported as descriptive AUCs.
    time_cells = {}
    for schedule in ('static', 'rematched'):
        for live in (False, True):
            label = f'{schedule}_{"live" if live else "silent"}'
            time_cells[label] = {}
            for key in KEYS:
                fi_auc = [metrics.centered_auc(trajectory(fi_by[s, schedule, live], key)) for s in seeds]
                pl_auc = [metrics.centered_auc(trajectory(pl_by[s, schedule, live], key)) for s in seeds]
                # AUC is centered within each population at update 0, so this
                # contrast describes change-from-initialization, not a causal
                # information effect.
                delta = np.asarray(fi_auc) - np.asarray(pl_auc)
                time_cells[label][key] = dict(FI_centered_AUC=contrast_stats(fi_auc),
                                               PL_centered_AUC=contrast_stats(pl_auc),
                                               FI_minus_PL_centered_AUC=contrast_stats(delta))

    channel_effects = {}
    for information, source in (('FI', fi_by), ('PL', pl_by)):
        channel_effects[information] = {}
        for schedule in ('static', 'rematched'):
            label = f'{schedule}_live_minus_silent'
            channel_effects[information][label] = {}
            for key in KEYS:
                values = [endpoint(source[s, schedule, True], key) - endpoint(source[s, schedule, False], key)
                          for s in seeds]
                channel_effects[information][label][key] = dict(mean=float(np.mean(values)), statistics=contrast_stats(values))

    result = dict(
        status='completed_json_only_information_comparison',
        FI_source=str(Path(fi_path).resolve()), PL_source=str(Path(pl_path).resolve()),
        FI_aggregation_sha256=hashlib.sha256(Path(fi_path).read_bytes()).hexdigest(),
        PL_aggregation_sha256=hashlib.sha256(Path(pl_path).read_bytes()).hexdigest(),
        seeds=seeds, cells=cells, centered_time_AUC=time_cells, channel_effects=channel_effects,
        interpretation_boundary='FI−PL contrasts pair separate trained populations and are descriptive; they are not a within-run causal estimate or evidence of language.',
        no_model_calls=True, no_optimizer_updates=True,
    )
    path = output / 'comparison.json'; path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_runs=128, output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   no_model_calls=True, no_optimizer_updates=True)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fi', required=True); parser.add_argument('--pl', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.fi, args.pl, args.output)
