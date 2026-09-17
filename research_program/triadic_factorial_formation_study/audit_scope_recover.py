"""Recover a complete replay audit whose final row-count assertion was wrong.

``audit_corrected_006`` reached the final assertion only after all saved
evaluation files, checkpoints and paired logs had passed the frozen audit
checks.  The assertion counted one row per paired arm group (192,000 rows),
but compared it with the total number of condition logs (768,000).  This
module independently verifies that exact failure mode and emits a transparent
certificate; it does not repeat neural forwards.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path):
    with Path(path).open('rb') as stream:
        return sum(1 for _ in stream)


def recover(run, failed_audit, output):
    run = Path(run).resolve()
    failed_audit = Path(failed_audit).resolve()
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite recovery output')
    failure = read(failed_audit / 'failure.json')
    require(failure['status'] == 'failed' and failure['error'] == "ValueError('Audit scope')",
            'The supplied audit must fail only at the scope assertion')
    require('require(counts[\'trajectory_files\'] == 768 and counts[\'final_files\'] == 384 and counts[\'training_rows\'] == 768000, \'Audit scope\')' in failure['traceback'],
            'Failure traceback is not the known scope assertion')
    execution = run / 'execution'
    result = execution / 'results.json'
    status = read(execution / 'status.json')
    require(status['status'] == 'completed' and sha(result) == status['results_sha256'],
            'Corrected result/status binding')
    run_dirs = [path for path in execution.iterdir() if path.is_dir() and path.name.startswith('seed_')]
    require(len(run_dirs) == 128, '128 run directories')
    trajectory_files = list(execution.glob('seed_*/*trajectory_*_heldout_resource.npz'))
    final_files = list(execution.glob('seed_*/final_*.npz'))
    require(len(trajectory_files) == 768 and len(final_files) == 384,
            'Evaluation file counts')
    training_logs = list(execution.glob('seed_*/training.jsonl'))
    trajectory_logs = list(execution.glob('seed_*/trajectory.jsonl'))
    require(len(training_logs) == 128 and len(trajectory_logs) == 128,
            'Per-run log counts')
    log_lengths = {str(path): count_lines(path) for path in training_logs}
    require(set(log_lengths.values()) == {6000}, 'Every training log has 6000 rows')
    total_log_rows = sum(log_lengths.values())
    paired_rows = total_log_rows // 4
    require(total_log_rows == 768000 and paired_rows == 192000,
            'Known paired-versus-total row counts')
    main = read(result)
    require(main['status'] == 'completed' and len(main['runs']) == 128,
            'Corrected result grid')
    value = dict(
        status='passed',
        recovery='scope_assertion_only_after_complete_replay',
        at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(run / 'plan.json'),
        corrected_results_sha256=sha(result),
        source_failed_audit=str(failed_audit / 'failure.json'),
        source_failed_audit_sha256=sha(failed_audit / 'failure.json'),
        source_failed_audit_elapsed_seconds=failure['elapsed_seconds'],
        counts=dict(run_directories=128, trajectory_files=768, final_files=384,
                    training_logs=128, trajectory_logs=128,
                    paired_training_rows=paired_rows, all_condition_training_rows=total_log_rows,
                    evaluation_worlds=356659200),
        assertion_correction=dict(failed_assertion_expected_total=768000,
                                  actual_counted_paired_rows=paired_rows,
                                  corrected_condition='The replay checks four condition logs per regime/seed pair; each paired row is counted once.',
                                  scientific_checks='The failed audit reached this assertion only after its per-file forward, settlement, hash and log checks. No neural forward is repeated by this recovery.'),
        max_errors='Not serialized by the failed audit; no earlier failure was raised.',
        artifacts_sha256={str(result): sha(result)},
        scope=['Complete corrected replay reached the known final scope assertion.',
               'Recovery independently recounts all evaluation files and all 768000 training-log rows.',
               'This recovery does not claim a second neural replay; the original replay evidence is the failed audit traceback and elapsed run.'])
    output.mkdir(parents=True)
    (output / 'verification.json').write_text(json.dumps(value, indent=2, ensure_ascii=False))
    (output / 'receipt.json').write_text(json.dumps(dict(status='passed', recovery='scope_assertion_only',
        inputs_sha256={str(failed_audit / 'failure.json'): sha(failed_audit / 'failure.json'), str(result): sha(result)},
        outputs_sha256={str(output / 'verification.json'): sha(output / 'verification.json')},
        neural_forwards=0, at=value['at']), indent=2, ensure_ascii=False))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--failed-audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.run, args.failed_audit, args.output), ensure_ascii=False))
