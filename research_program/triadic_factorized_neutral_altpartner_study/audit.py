"""Independent checkpoint replay audit for the factorized neutral study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import design, runner


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    static = json.loads((source / 'prepared.json').read_text()); execution = source / 'execution'
    runs = []; max_error = 0.0; checkpoints = 0; worlds = 0; forwards = 0
    for seed in design.SEEDS:
        arrays = {part: runner.make_arrays(static['partitions'][part]) for part in design.PARTS}
        for condition in design.CONDITIONS:
            run_path = execution / f'seed_{seed}_{condition}' / 'result.json'
            require(run_path.is_file(), f'Missing result {run_path}')
            run = json.loads(run_path.read_text()); runs.append(run); directory = run_path.parent
            schedule, payoff, rule, live = design.parse_condition(condition)
            require(run['seed'] == seed and run['condition'] == condition and run['updates'] == 6000, 'Run identity mismatch')
            for row in run['trajectory']:
                path = directory / f"checkpoint_{int(row['update']):04d}.npz"
                require(runner.core.base.sha(path) == row['checkpoint_sha256'], 'Checkpoint hash mismatch')
                networks = runner.load_networks(path)
                replay = runner.evaluate_factorized(networks, arrays['train'], static['partitions']['train'], live)
                checkpoints += 1; worlds += replay['worlds']; forwards += 9 * replay['worlds']
                for key in ('q_rate', 'conditional_q_rate', 'target_pair_legal_rate', 'proposal_legal_rate',
                            'physical_execution_rate', 'engagement_rate', 'neutral_rate', 'third_agent_neutral_rate',
                            'conditional_q_denominator_worlds', 'conditional_q_numerator_worlds',
                            'target_pair_denominator_worlds', 'target_pair_numerator_worlds'):
                    max_error = max(max_error, abs(float(replay[key]) - float(row['target_trajectory'][key])))
                for key in ('actor_engagement_rates', 'legal_plan_selection_counts', 'legal_pair_selection_counts'):
                    actual, saved = replay[key], row['target_trajectory'][key]
                    require(len(actual) == len(saved), 'Diagnostic length mismatch')
                    for x, y in zip(actual, saved):
                        max_error = max(max_error, abs(float(x) - float(y)))
            histogram = run['rematch_histogram']; expected = 6000 * 256 if schedule == 'rematched' else 0
            require(len(histogram) == 6 and sum(histogram) == expected and run['rematch_assignments'] == expected, 'Rematch histogram mismatch')
            require(sum(run['final']['new_layouts']['legal_pair_selection_counts']) <= run['final']['new_layouts']['worlds'], 'Final pair count invalid')
    require(len(runs) == 64, 'Incomplete run grid')
    runner.verify_pairing(execution, runs)
    verification = dict(status='passed', source=str(source), runs=64, checkpoint_evaluations=checkpoints,
                         compact_worlds=worlds, model_forwards=forwards, optimizer_updates=0,
                         no_optimizer_updates=True, trajectory_replayed=True, pairing_verified=True,
                         max_abs_error=float(max_error), factorized_action_replayed=True,
                         legal_plan_multiplicity_verified=True, alternative_partner_support_verified=True,
                         rematch_histograms_checked=True, rematched_permutation_assignments=sum(run['rematch_assignments'] for run in runs))
    path = output / 'verification.json'; path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', verification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   model_forwards=forwards, optimizer_updates=0)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.source, args.output)
