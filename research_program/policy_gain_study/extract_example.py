"""One predetermined descriptive case from saved probe records, no new inference.

Selection fixed after 8/64 runs finished, before this case's message probe: first seed, split1, scout0,
first held-out map, first fixed validation photo pair, all 4 cells/all 8 times.
Not selected for success, representativeness or a visually interesting string.
"""
from pathlib import Path
from itertools import product
import hashlib
import json

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / 'results/gain_001'
PROBE = BATCH / 'probe'
SEED, SCOUT, PHOTO_INDEX = 28101, 0, 0
CELLS = [('additive', 1), ('additive', 3), ('joint', 1), ('joint', 3)]
UPDATES = [0, 100, 300, 600, 1200, 1800, 2100, 2400]


def main():
    dest = BATCH / 'descriptive_example'
    assert not dest.exists()
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan = read(PROBE / 'plan.json')
    result = read(PROBE / 'execution/results.json')
    assert result['status'] == 'complete'
    run = next(r for r in plan['runs'] if r['seed'] == SEED and r['condition'] == 'split1_additive_gain1')
    map_id = run['heldout_map_ids'][0]
    target = plan['map_table'][map_id]
    rows, inputs = [], {}
    for kind, gain in CELLS:
        condition = f'split1_{kind}_gain{gain}'
        for update in UPDATES:
            entry = next(r for r in result['records'] if r['seed'] == SEED and r['condition'] == condition
                         and r['update'] == update and r['scout'] == SCOUT)
            path = PROBE / 'execution' / entry['file']
            assert sha(path) == entry['sha256']
            record = read(path)
            message = record['emitted'][map_id][PHOTO_INDEX]
            decoded = record['receiver']['actions'][message[0]*7+message[1]]
            available = any(actions == target for actions in record['receiver']['actions'])
            correct = [a == t for a, t in zip(decoded, target)]
            rows.append(dict(kind=kind, gain=gain, update=update, message=message,
                             decoded_food_water_sites=decoded, success_by_resource=correct,
                             both_correct=all(correct), any_complete_correct_code=available))
            inputs[str(path)] = sha(path)
    output = dict(selection=dict(seed=SEED, split=1, scout=SCOUT, collector=1-SCOUT,
        map_id=map_id, target_food_water_sites=target, photo_index=PHOTO_INDEX,
        photo_pair=plan['validation_photo_pairs'][PHOTO_INDEX], all_cells=CELLS, all_updates=UPDATES,
        rationale='Deterministic first-index display chosen at 8/64 runs complete, before this case message probe; no success selection. Not registered before training.'),
        rows=rows, inputs_sha256=inputs, source_sha256=sha(Path(__file__)),
        scope='One descriptive held-out map/photo case, not an independent test or a representative population estimate.')
    dest.mkdir()
    (dest/'example.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 一个预先指定情境的消息变化', '',
        f'种子 {SEED}、split1、发送主体0→接收主体1；第一张留出地图与第一组固定验证照片。',
        f'本图食物地点={target[0]}、水地点={target[1]}。下表地点和符号均为模型原始从0开始的整数编号。', '',
        '选择规则在8/64组完成时、该情境消息探针执行前写定；不是训练前登记的主要测量。保留全部四格和八个检查点，不按成功或故事性挑选。这个单例不代表四个种子的总体模式。', '',
        '| 收益 | 权重 | 步数 | 自然消息 | 解读为食物/水地点 | 两项均对 | 存在完整正确码 |',
        '|---|---:|---:|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['kind']} | {r['gain']} | {r['update']} | {r['message']} | "
                     f"{r['decoded_food_water_sites']} | {'是' if r['both_correct'] else '否'} | "
                     f"{'是' if r['any_complete_correct_code'] else '否'} |")
    lines += ['', '“存在完整正确码”只考察当前接收表；“两项均对”还要求发送者对这一照片实际发出了可用码。'
              '这段轨迹不能据以给单个符号命名为词，也未检验语法或自主组合。所有行由冻结探针JSON直接读取，没有新模型调用。', '']
    (dest/'example.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='complete', output=str(dest), rows=len(rows))))


if __name__ == '__main__':
    main()
