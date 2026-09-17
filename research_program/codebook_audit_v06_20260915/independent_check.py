"""Recompute held-out counts directly from saved tables; no audit-module import."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'redesign_v0.6/results/recombination_001/protocol_analysis.json'
OUT = Path(__file__).resolve().parent


def main():
    source = json.loads(SOURCE.read_text())
    reported = json.loads((OUT / 'codebook_audit.json').read_text())
    groups = defaultdict(lambda: [0, 0, 0, 0, 0])
    per_seed = defaultdict(lambda: [0, 0, 0, 0, 0])
    for run in source['runs']:
        if not run['heldout_map_ids']:
            continue
        family = run['condition'].split('_', 1)[1]
        for direction in run['directions']:
            lookup = {tuple(r['message']): tuple(r['actions_by_goal'])
                      for r in direction['receiver_decoder_table']}
            assert len(lookup) == 49
            grid = direction['phases']['validation']['codebook']
            for map_id in run['heldout_map_ids']:
                paired = [r for r in grid if r['map_id'] == map_id]
                assert len(paired) == 2
                target = (paired[0]['food_location'], paired[0]['water_location'])
                available = any(action == target for action in lookup.values())
                successes, unsupported, not_selected = 0, 0, 0
                for record in paired:
                    assert target == (record['food_location'], record['water_location'])
                    for message in record['delivered_messages']:
                        count = message['count']
                        if lookup[tuple(message['message'])] == target:
                            successes += count
                        elif available:
                            not_selected += count
                        else:
                            unsupported += count
                assert successes + unsupported + not_selected == 32
                values = [successes, unsupported, not_selected, int(available), 1]
                for accumulator in (groups[family], per_seed[family, run['seed']]):
                    for i, value in enumerate(values):
                        accumulator[i] += value
    for row in reported['heldout_validation_by_family']:
        c = row['pooled_descriptive_counts']
        expect = [c['natural_success']['numerator'], c['receiver_unreachable_failure']['numerator'],
                  c['available_code_not_delivered_failure']['numerator'],
                  row['receiver_reachable_maps']['numerator'], row['receiver_reachable_maps']['denominator']]
        assert groups[row['family']] == expect
    for row in reported['heldout_validation_by_seed']:
        c = row['natural_decomposition']
        expect = [c['natural_success']['numerator'], c['receiver_unreachable_failure']['numerator'],
                  c['available_code_not_delivered_failure']['numerator'],
                  row['receiver_reachable_maps']['numerator'], row['receiver_reachable_maps']['denominator']]
        assert per_seed[row['family'], row['seed']] == expect
    for name, expected in reported['source_files_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    result = {'verified_at': datetime.now().astimezone().isoformat(), 'status': 'verified',
              'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'scope': 'Held-out validation codebook: direct raw-table recomputation of pooled and all16 seed-family counts.',
              'columns': ['natural_success', 'no_correct_code', 'correct_code_not_selected', 'reachable_maps', 'maps'],
              'by_family': dict(groups), 'seed_family_rows_checked': len(per_seed),
              'new_model_calls': 0,
              'limits': 'Uses the existing receiver tables; does not reload checkpoint weights or reconstruct photo identities.'}
    (OUT / 'independent_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
