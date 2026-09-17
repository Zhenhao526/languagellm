"""JSON-only posthoc diagnostics for conditional legal-plan coordination."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import metrics


def main(results_path, output):
    data = json.loads(Path(results_path).read_text()); runs = data['runs']; by = {(r['seed'], r['live']): r for r in runs}; rows = []
    for seed in sorted({r['seed'] for r in runs}):
        series = {}
        for live in (False, True):
            run = by[seed, live]; traj = sorted(run['trajectory'], key=lambda x: x['update'])
            q = np.asarray([x['target_trajectory']['q_rate'] for x in traj], dtype=np.float64)
            p = np.asarray([x['target_trajectory']['physical_execution_rate'] for x in traj], dtype=np.float64)
            series[live] = np.divide(q, p, out=np.zeros_like(q), where=p > 0)
        delta = series[True] - series[False]
        rows.append(dict(seed=seed, conditional_q_given_physical_AUC=metrics.centered_auc(delta), endpoint_interaction=float(delta[-1])))
    auc = np.asarray([r['conditional_q_given_physical_AUC'] for r in rows]); endpoint = np.asarray([r['endpoint_interaction'] for r in rows])
    def summary(values):
        mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd / math.sqrt(16); half = metrics.T15_975 * se
        return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15, ci95_lower=mean-half, ci95_upper=mean+half, positive_count=int((values > 0).sum()))
    result = dict(status='completed_json_only_posthoc_diagnostic', source=str(Path(results_path).resolve()), metric='Q conditional on any physical execution', auc=summary(auc), endpoint=summary(endpoint), by_seed=rows, no_model_calls=True, no_optimizer_updates=True, posthoc=True)
    output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False); path=output/'results.json'; path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n'); receipt=dict(status='passed',input_results_sha256=hashlib.sha256(Path(results_path).read_bytes()).hexdigest(),output_results_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),model_forwards=0,optimizer_updates=0,posthoc=True); (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(receipt,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--results',required=True); parser.add_argument('--output',required=True); args=parser.parse_args(); main(args.results,args.output)
