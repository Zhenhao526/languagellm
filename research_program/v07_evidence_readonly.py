"""Post-hoc v0.7 JSON integration; no model imports or full audit replay.

Run from the project root with Python 3.9+; optional --out creates a new JSON.
Existing v0.7 files are read only. Decoder coverage is a new post-hoc summary,
not a claim that the original v0.7 analysis preregistered this statistic.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'redesign_v0.7/results/binding_001'
SEEDS = (26101, 26102, 26103, 26104)
REPS = ('identity', 'permute', 'orthogonal')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def integrate():
    summary = read(BASE / 'analysis/summary.json')
    protocol = read(BASE / 'protocol_analysis.json')
    audit = read(BASE / 'audit_execution.json')
    controls = read(BASE / 'individual_controls/summary.json')
    for field, filename in (('protocol', 'protocol_analysis.json'),
                            ('execution_audit', 'audit_execution.json'),
                            ('controls', 'individual_controls/summary.json')):
        require(summary[field]['source_sha256'] == digest(BASE / filename), field + ' source link')
    require(audit['audit_script_sha256'] == digest(ROOT / 'redesign_v0.7/audit_execution.py'), 'audit source')
    require(audit['status'] == 'passed' and audit['failures'] == [] and
            audit['completed_runs'] == 52 and sum(audit['checks'].values()) == 874023, 'recorded prior audit')
    expected = {(s, f'split{k}_{r}') for s in SEEDS for k in (1, 2, 3) for r in REPS}
    expected |= {(s, f'full_{r}') for s in SEEDS for r in (*REPS, 'blocked')}
    require({(r['seed'], r['condition']) for r in protocol['runs']} == expected and
            len(protocol['runs']) == len(summary['runs']) == 52, 'complete social grid')
    evidence, per_seed, endpoint = [], defaultdict(list), defaultdict(list)
    directions = recodings = decoder_rows = natural_rows = 0
    for run in protocol['runs']:
        seed, condition = run['seed'], run['condition']
        folder = BASE / f's{seed}_{condition}'
        cfg, result = read(folder / 'config.json'), read(folder / 'result.json')
        require(cfg['seed'] == result['seed'] == seed and cfg['condition'] == result['condition'] == condition and
                cfg['plan'] == result['plan'] == run['plan'], 'run identity')
        require(cfg['updates'] == result['updates'] == 2400 and cfg['batch'] == result['batch'] == 512,
                'fixed social budget')
        require(len(run['directions']) == 2, 'two directions')
        rep, split = run['plan']['representation'], run['plan']['split']
        if split:
            for subset, key in (('seen', 'normal/train_maps'), ('unseen', 'normal/heldout_maps')):
                counts = result['scores']['normal']['map_groups'][subset]
                endpoint[(rep, seed, key)].append(counts['correct'] / counts['n'])
        for d in run['directions']:
            directions += 1
            ma, table = d['menu_audit'], d['receiver_decoder_table']
            require(ma['all_menu_permutations_physically_equivalent'] is True and
                    ma['enumerated_messages'] == 49 and ma['goals'] == 2 and ma['menus'] == 720,
                    'stored full-menu equivalence')
            require(len(table) == 49 and {tuple(t['message']) for t in table} == set(product(range(7), repeat=2)),
                    'complete code space')
            decoder = {}
            for t in table:
                decoder_rows += 1
                actions = t['actions_by_goal']
                require(len(actions) == 2 and all(0 <= a < 6 for a in actions), 'decoder action range')
                for goal in (0, 1):
                    require(t['menu_action_counts_by_goal'][goal] ==
                            [720 if a == actions[goal] else 0 for a in range(6)], 'menu-independent physical action')
                decoder[tuple(t['message'])] = tuple(actions)
            re = d['whole_message_recoding']
            require(re['replicates'] == len(re['records']) == 100, 'full reference budget')
            for r in re['records']:
                require(sorted(r['old_to_new_code']) == list(range(49)), 'recoding is a whole-code bijection')
                recodings += 1
            phase = d['phases']['validation']
            require(len(phase['photo_pairs']) == 16 and len(phase['codebook']) == 60, 'validation scope')
            by_map = defaultdict(list)
            for row in phase['codebook']:
                natural_rows += 1
                target = (row['food_location'], row['water_location'])
                require(sum(m['count'] for m in row['delivered_messages']) == 16, 'natural photo count')
                successes = sum(m['count'] for m in row['delivered_messages'] if decoder[tuple(m['message'])] == target)
                require(row['both']['numerator'] == successes and row['both']['denominator'] == 16,
                        'reconstructed natural same-message success')
                by_map[row['map_id']].append(row)
            if not split:
                continue
            U = N = no_code_failures = 0
            for map_id in run['heldout_map_ids']:
                rows = by_map[map_id]
                require(len(rows) == 2 and {r['sender_goal'] for r in rows} == {0, 1}, 'both sender goal axes')
                require(rows[0]['delivered_messages'] == rows[1]['delivered_messages'], 'sender goal hidden')
                target = (rows[0]['food_location'], rows[0]['water_location'])
                covered = target in decoder.values()
                count = sum(r['both']['numerator'] for r in rows)
                require(covered or count == 0, 'natural success belongs to receiver coverage')
                U += int(covered)
                N += count
                no_code_failures += 0 if covered else 32
            require(N == phase['cross_goal_groups']['unseen']['same_message_correct_for_both_goals']['numerator'],
                    'raw natural group total')
            item = dict(seed=seed, condition=condition, split=split, scout=d['scout'], collector=d['collector'],
                        U=U, U_denominator=6, N=N, N_denominator=192, no_code_failures=no_code_failures)
            evidence.append(item)
            per_seed[(rep, seed)].append(item)
    groups = {}
    for rep in REPS:
        rows = [r for r in evidence if r['condition'].endswith('_' + rep)]
        U, N, failures_without_code = (sum(r[k] for r in rows) for k in ('U', 'N', 'no_code_failures'))
        values = []
        for seed in SEEDS:
            sr = per_seed[(rep, seed)]
            require(len(sr) == 6, 'three splits, two directions per seed')
            values.append(dict(seed=seed, U=sum(r['U'] for r in sr) / 36,
                               N=sum(r['N'] for r in sr) / 1152))
        groups[rep] = dict(U=U, U_denominator=144, U_rate=U / 144, N=N, N_denominator=4608,
                           N_rate=N / 4608, no_code_failures=failures_without_code,
                           natural_failures=4608 - N,
                           no_code_share_of_natural_failures=failures_without_code / (4608 - N), per_seed=values)
        for key in ('normal/train_maps', 'normal/heldout_maps'):
            vals = [mean(endpoint[(rep, s, key)]) for s in SEEDS]
            require(all(abs(a-b) < 1e-12 for a,b in zip(vals, summary['groups'][rep]['metrics'][key]['values'])),
                    'raw endpoint counts reproduce seed means')
        require(abs(N / 4608 - summary['protocol']['groups'][rep]['metrics']['unseen_both']['mean']) < 1e-12,
                'natural protocol mean agrees with summary')
    require(directions == 104 and recodings == 10400 and decoder_rows == 5096 and natural_rows == 6240,
            'complete JSON record coverage')
    require(controls['complete'] and controls['passed'] and len(controls['results']) == 12, 'personal control scope')
    personal = []
    for r in controls['results']:
        require(r == read(BASE / 'individual_controls' / f"s{r['seed']}_{r['representation']}" / 'result.json'),
                'personal control source record')
        for who in (0,1):
            normal = r['scores']['normal']['per_agent'][who]
            erased = r['scores']['erase_memory']['per_agent'][who]
            require(normal['correct'] == normal['n'] == 9600 and erased['correct'] == 1600 and erased['n'] == 9600,
                    'personal exact counts')
            personal.append(dict(seed=r['seed'], representation=r['representation'], agent=who,
                                 normal_correct=9600, erased_correct=1600, n=9600))
    source_paths = [BASE / f for f in ('protocol_analysis.json', 'analysis/summary.json', 'audit_execution.json',
                                      'individual_controls/summary.json')]
    source_paths += [ROOT / 'redesign_v0.7' / f for f in ('README.md', '固定执行方案.md', 'audit_execution.py')]
    return dict(status='passed_readonly_json_integration', generated_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=digest(Path(__file__)), source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        independent_pair_seeds=list(SEEDS), social_runs=52, personal_control_runs=12,
        direction_tables_checked=directions, menu_independent_decoder_rows_checked=decoder_rows,
        whole_code_bijections_checked=recodings, natural_codebook_rows_recounted=natural_rows,
        heldout_decoder_coverage_posthoc=groups, heldout_direction_records=evidence,
        personal_counts_from_original_json=personal,
        prior_full_audit_record=dict(status=audit['status'], checks=sum(audit['checks'].values()),
                                    training_updates=audit['train_updates_verified'],
                                    social_trace_rows=audit['final_trace_rows_recounted']),
        scope=['Only JSON/Markdown/Python source files read; no .pt, .npz or model loading.',
               'Prior 874023-check audit is read and source-hash linked, not re-executed here.',
               'Coverage is a new post-hoc summary of saved greedy 49-code receiver functions; no retraining or new inference.',
               'Menu invariance checked from all saved 720-menu action counts, not by rerunning neural decisions.',
               'N retains duplicated hidden sender-goal axis: 4608 records but 2304 distinct photo/map/direction contexts per representation.',
               'The 52 conditions share four prepared pair seeds; personal controls and directions are not new social replicates.'],
        new_model_calls=0, new_training_updates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = integrate()
    if args.out:
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'heldout_decoder_coverage_posthoc')}, ensure_ascii=False, indent=2))
