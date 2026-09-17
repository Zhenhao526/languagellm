"""Describe retained contact failures by scene, directly from final traces."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    folder = parser.parse_args().directory.resolve()
    results = json.loads((folder / 'results.json').read_text())
    output = {'scope': 'Descriptive final full-task greedy evaluation; cases and edges are not independent populations.', 'runs': []}
    for run in results:
        entry = {'seed': run['seed'], 'condition': run['condition'], 'pairs': []}
        directory = folder / f"{run['condition']}_s{run['seed']}" / 'traces'
        for pair in run['evaluation']['pairs']:
            i, j = pair['agents']
            path = directory / f'pair_{i}{j}_full_normal_trace.jsonl'
            groups = {}
            contexts = {}
            total = 0
            successes = 0
            for line in path.read_text().splitlines():
                row = json.loads(line)
                local = [sum(k) for k in row['kinds']]
                mixed = sum(k == 1 for k in local)
                group = ('both_restricted', 'one_restricted', 'both_mixed')[mixed]
                context = '_'.join(str(k) for k in local)
                success = int(row['success'])
                assert success == int(row['selected_kinds'][0] != row['selected_kinds'][1])
                for bucket, key in ((groups, group), (contexts, context)):
                    record = bucket.setdefault(key, {'n': 0, 'successes': 0, 'failures': 0})
                    record['n'] += 1
                    record['successes'] += success
                    record['failures'] += 1 - success
                total += 1
                successes += success
            assert total == run['evaluation']['cases_per_pair_task_mode']
            assert successes / total == pair['tasks']['full']['normal']['mean_reward_per_step']
            for record in list(groups.values()) + list(contexts.values()):
                record['success_rate'] = record['successes'] / record['n']
            entry['pairs'].append({'agents': [i, j], 'group': pair['group'], 'n': total,
                                   'successes': successes, 'failures': total - successes,
                                   'by_scene_type': groups, 'by_local_kind_sum': contexts,
                                   'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        output['runs'].append(entry)
    (folder / 'contact_failure_summary.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'runs': len(output['runs']), 'normal_full_cases_scored': sum(p['n'] for r in output['runs'] for p in r['pairs'])}))


if __name__ == '__main__':
    main()
