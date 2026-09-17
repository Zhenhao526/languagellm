"""Independent saved-JSON dtype diagnosis only. No model imports or forwards."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np

STUDY = Path(__file__).resolve().parent
BATCH = STUDY / 'results/gain_001'
PROBE = BATCH / 'probe'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())


def main():
    result_path = PROBE / 'execution/results.json'
    result = read(result_path)
    assert result['status'] == 'complete' and result['counts']['directions'] == 1024
    fields = ('actions', 'unique_max_count', 'top_two_margin')
    mismatch_records, mismatch_values = Counter(), Counter()
    rows, examples, seen = [], [], set()
    total_logits = total_actions = total_margins = 0
    maximum_error = 0.
    frozen = read(BATCH / 'manifest.json')['source_hashes']
    sources = {name: sha(STUDY / name) for name in ('analysis.py', 'probe.py')}
    for name, digest in sources.items(): assert frozen[str(STUDY / name)] == digest
    source_inputs = {str(path): sha(path) for path in
        (result_path, PROBE / 'plan.json', PROBE / 'freeze.json', BATCH / 'manifest.json')}
    assert read(PROBE / 'freeze.json')['plan_sha256'] == sha(PROBE / 'plan.json')
    assert result['plan_sha256'] == sha(PROBE / 'plan.json')
    for entry in result['records']:
        path = PROBE / 'execution' / entry['file']
        assert sha(path) == entry['sha256']
        record = read(path)
        key = tuple(record[k] for k in ('seed', 'condition', 'update', 'scout'))
        assert key == tuple(entry[k] for k in ('seed', 'condition', 'update', 'scout')) and key not in seen
        seen.add(key)
        receiver = record['receiver']
        assert set(receiver) == set(fields) | {'logits'}
        logits64 = np.asarray(receiver['logits'])
        assert logits64.dtype == np.float64 and logits64.shape == (49, 2, 6) and np.isfinite(logits64).all()
        logits32 = logits64.astype(np.float32)
        assert np.array_equal(logits64, logits32.astype(np.float64))
        derived = {}
        for label, logits in [('float64', logits64), ('float32', logits32)]:
            sorted_logits = np.sort(logits, axis=-1)
            derived[label] = dict(actions=logits.argmax(-1).tolist(),
                unique_max_count=(logits == logits.max(-1, keepdims=True)).sum(-1).tolist(),
                top_two_margin=(sorted_logits[..., -1]-sorted_logits[..., -2]).tolist())
        assert derived['float32'] == {key: receiver[key] for key in fields}
        assert derived['float64']['actions'] == receiver['actions']
        assert derived['float64']['unique_max_count'] == receiver['unique_max_count']
        assert np.all(np.asarray(receiver['unique_max_count']) == 1)
        differences = {}
        for field in fields:
            stored = np.asarray(receiver[field]); new = np.asarray(derived['float64'][field])
            mask = stored != new
            differences[field] = int(mask.sum())
            mismatch_values[field] += int(mask.sum())
            mismatch_records[field] += bool(mask.any())
            if field == 'top_two_margin':
                delta = np.abs(stored-new)
                maximum_error = max(maximum_error, float(delta.max()))
                if mask.any() and len(examples) < 5:
                    ix = tuple(np.argwhere(mask)[0])
                    code, goal = map(int, ix)
                    order = np.argsort(logits64[code, goal])
                    examples.append(dict(record=list(key), message=[code//7, code % 7], goal=goal,
                        largest=float(logits64[code, goal, order[-1]]),
                        second_largest=float(logits64[code, goal, order[-2]]),
                        stored_margin=float(stored[ix]), reconstructed_float64_margin=float(new[ix]),
                        reconstructed_float32_margin=float(derived['float32'][field][code][goal]),
                        absolute_difference=float(delta[ix])))
        rows.append(dict(seed=key[0], condition=key[1], update=key[2], scout=key[3],
            file=entry['file'], sha256=entry['sha256'], float64_mismatch_value_counts=differences,
            logits_exact_float32_roundtrip=True, all_float32_fields_exact=True,
            actions_and_unique_max_counts_exact_both_dtypes=True))
        total_logits += logits64.size; total_actions += 98; total_margins += 98
    seeds = (28101, 28102, 28103, 28104)
    conditions = tuple(f"{('split'+str(s)) if s else 'full'}_{kind}_gain{gain}"
                       for s, kind, gain in product((1, 2, 3, 0), ('additive', 'joint'), (1, 3)))
    assert seen == set(product(seeds, conditions, (0, 100, 300, 600, 1200, 1800, 2100, 2400), range(2)))
    assert len(rows) == 1024 and not ('torch' in sys.modules or 'camp' in sys.modules)
    for path, digest in source_inputs.items(): assert sha(Path(path)) == digest
    for name, digest in sources.items(): assert sha(STUDY / name) == digest
    output = dict(status='confirmed_margin_dtype_mismatch_only', created_at=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)), frozen_source_sha256=sources, inputs_sha256=source_inputs,
        completed_records=1024, exact_float32_roundtrip_logit_values=total_logits,
        actions_exact_both_dtypes=total_actions, unique_max_counts_exact_both_dtypes=total_actions,
        float32_margins_exact=total_margins, receiver_maximum_ties=0,
        float64_mismatch_record_counts={k: int(mismatch_records[k]) for k in fields},
        float64_mismatch_value_counts={k: int(mismatch_values[k]) for k in fields},
        max_absolute_float64_margin_difference=maximum_error, examples=examples, records=rows,
        cause='JSON list -> np.asarray defaults to float64. Saved top_two_margin subtracts float32 NumPy logits. Ordering is unchanged, but subtraction rounding differs.',
        permitted_recovery='For this consistency check only, restore JSON logits to float32 before the existing lookup function. Keep exact equality, original failure and frozen files. No tolerance relaxation or rerun.',
        scope='All 1024 saved receiver records and their input hashes; not a rerun of full J/U/N analysis or model outputs.',
        model_imports=0, weight_loads=0, neural_forward_calls=0, frozen_files_modified=0)
    json_path, md_path = BATCH / 'analysis_dtype_diagnosis.json', BATCH / 'analysis_dtype_diagnosis.md'
    assert not json_path.exists() and not md_path.exists()
    with json_path.open('x') as stream: json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ['# 原分析失败：接收 margin 的 dtype 核查', '',
        '全 1024 份接收记录已独立检查。差异只在 float64 重算的 `top_two_margin`；动作与唯一最大值计数没有差异。', '',
        f"- {total_logits:,} 个 JSON logits 均可精确转为 float32 再转回 float64，未丢失任何存储数值。",
        f"- {total_actions:,} 个动作及同数目的唯一最大值计数，在两种 dtype 下均与存档逐项相等；并列最大值为 0。",
        f"- 原 float32 减法精确重现全部 {total_margins:,} 个 margin。",
        f"- 默认 float64 减法在 {mismatch_records['top_two_margin']} 份记录、{mismatch_values['top_two_margin']:,} 个 margin 上产生差异，最大绝对差 {maximum_error:.17g}。", '',
        '原探针将 float32 logits 直接做减法并保存结果；JSON 不保存 dtype，原分析把列表交给 `np.asarray` 后变为 float64。浮点减法的舍入改变了部分 margin，精确字典比较因此失败。', '',
        '可以在显式恢复包装器中，仅为这一项一致性检查把 JSON logits 恢复为 float32，再执行原精确比较。无需放宽误差、重跑探针或修改当前测量定义。原失败与原分析文件必须保留，恢复输出另存。', '',
        '核查包含完整网格、1024 记录 SHA、原分析／探针冻结 SHA 和读入前后的来源 SHA。没有导入 Torch、读取权重、执行神经前向或改写冻结文件。此诊断不代替后续完整 J／U／N 分析。', '',
        f'诊断 JSON SHA256：`{sha(json_path)}`。逐记录差异和数值例子见同名 JSON。', '']
    with md_path.open('x') as stream: stream.write('\n'.join(lines))
    print(json.dumps({k: v for k, v in output.items() if k not in ('records', 'inputs_sha256', 'examples')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
