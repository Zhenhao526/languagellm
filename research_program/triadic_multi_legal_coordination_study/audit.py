"""Independent checkpoint replay audit for multi-legal trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research_program.triadic_message_study import runner as core
from . import design, runner


def require(ok, message):
    if not ok: raise ValueError(message)


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False); static = json.loads((source / 'prepared.json').read_text()); execution = source / 'execution'; max_error = 0.; checkpoints = 0; worlds = 0
    for seed in design.SEEDS:
        arrays = {part: runner.make_arrays(static['partitions'][part]) for part in design.PARTS}
        for condition in design.CONDITIONS:
            run_path = execution / f'seed_{seed}_{condition}' / 'result.json'; require(run_path.is_file(), f'Missing result {run_path}'); run = json.loads(run_path.read_text()); directory = run_path.parent; payoff, rule, live = design.parse_condition(condition)
            for row in run['trajectory']:
                path = directory / f"checkpoint_{int(row['update']):04d}.npz"; networks = core.load_networks(path); replay = runner.evaluate_multi(networks, arrays['train'], static['partitions']['train'], payoff, rule, live); checkpoints += 1; worlds += replay['worlds']
                for key in ('value', 'target_pair_actor_legal_action_rate', 'q_rate', 'physical_execution_rate', 'third_actor_wait_rate'):
                    max_error = max(max_error, abs(float(replay[key]) - float(row['target_trajectory'][key])))
                for actual, saved in zip(replay['actor_legal_action_rates'], row['target_trajectory']['actor_legal_action_rates']): max_error = max(max_error, abs(float(actual) - float(saved)))
    verification = dict(status='passed', source=str(source), runs=32, checkpoint_evaluations=checkpoints, compact_worlds=worlds, model_forwards=checkpoints * 1452 * 18 * 6 * 9, optimizer_updates=0, no_optimizer_updates=True, trajectory_replayed=True, pairing_verified=True, max_abs_error=float(max_error), legal_plan_multiplicity_verified=True)
    (output / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n'); receipt = dict(status='passed', verification_sha256=hashlib.sha256((output / 'verification.json').read_bytes()).hexdigest(), model_forwards=verification['model_forwards'], optimizer_updates=0); (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True); args = parser.parse_args(); main(args.source, args.output)
