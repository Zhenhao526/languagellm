"""Explicit dtype correction for the frozen analysis's exact margin replay.

No frozen source is changed. Saved logits must round-trip exactly to float32.
Only the lookup verification subtraction is performed in its original dtype;
all original action, tie, measurement, SHA, grid and contrast checks still run.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import traceback
import numpy as np

STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
import analysis as original

BATCH = STUDY / 'results/gain_001'
PROBE = BATCH / 'probe'
RECOVERY = BATCH / 'analysis_recovery_001'


def main():
    require, sha, read = original.require, original.sha, original.read
    require(not RECOVERY.exists() and not (BATCH/'analysis.json').exists() and
            not (BATCH/'analysis.md').exists(), 'No overwrite or automatic retry')
    diagnosis = read(BATCH/'analysis_dtype_diagnosis.json')
    require(diagnosis['status'] == 'confirmed_margin_dtype_mismatch_only', 'Wait for independent dtype diagnosis')
    RECOVERY.mkdir()
    original.write_new(RECOVERY/'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        correction='Replay top-two subtraction in original float32 after exact logit round-trip check.',
        source_sha256=sha(Path(__file__)), original_analysis_sha256=sha(STUDY/'analysis.py'),
        original_failure_sha256=sha(BATCH/'analysis_failed_001.json'),
        independent_diagnosis_sha256=sha(BATCH/'analysis_dtype_diagnosis.json'),
        probe_results_sha256=sha(PROBE/'execution/results.json'),
        original_sources_changed=False, new_training=False, new_forward=False,
        changed_estimands_or_contrasts=False, check_relaxed=False))
    saved_lookup = original.measurement.lookup_from_logits
    checks = []

    def replay_float32(logits):
        values = np.asarray(logits, dtype=np.float64)
        replay = values.astype(np.float32)
        require(np.array_equal(values, replay.astype(np.float64)), 'JSON logits not exact float32 values')
        a, b = saved_lookup(values), saved_lookup(replay)
        require(a['actions'] == b['actions'] and a['unique_max_count'] == b['unique_max_count'],
                'Dtype correction must not change any physical action or maximum tie')
        delta = np.abs(np.asarray(a['top_two_margin'])-np.asarray(b['top_two_margin']))
        checks.append(dict(differing_margin_entries=int((delta != 0).sum()), max_margin_difference=float(delta.max())))
        return b

    original.measurement.lookup_from_logits = replay_float32
    try:
        result = original.analyze(BATCH, PROBE)
        require(len(checks) == 1024, 'Exactly all 1024 receiver lookup records must be replayed')
        correction = dict(recovery_plan_sha256=sha(RECOVERY/'plan.json'), records_checked=1024,
            changed_action_or_tie_entries=0, exact_float32_logit_roundtrip=True,
            differing_margin_entries=sum(c['differing_margin_entries'] for c in checks),
            max_margin_difference=max(c['max_margin_difference'] for c in checks),
            original_failure_retained=True, raw_records_changed=False, new_training_or_forward=False)
        result['implementation_correction'] = correction
        original.write_new(BATCH/'analysis.json', result)
        with (BATCH/'analysis.md').open('x') as stream:
            stream.write(original.markdown(result))
            stream.write('\n实现更正：原汇总将 JSON logits 读成 float64，导致与 float32 保存的前两名差值不能逐位相等。'
                         '此次在确认 logits 精确回转、动作及并列完全不变后，按原 float32 重算差值；所有原检查继续执行。'
                         '原失败、冻结源码和原探针均保留。见 analysis_recovery_001/plan.json。\n')
        original.write_new(RECOVERY/'result.json', dict(status='complete', **correction,
            analysis_sha256=sha(BATCH/'analysis.json'), completed_at=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(status='complete', output=str(BATCH/'analysis.json'), **correction)))
    except Exception as error:
        original.write_new(RECOVERY/'failure.json', dict(status='failed', error=repr(error),
            traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        original.measurement.lookup_from_logits = saved_lookup


if __name__ == '__main__':
    main()
