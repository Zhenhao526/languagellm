"""Check preparation lineage without treating saved copies as new populations."""
from pathlib import Path
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(state):
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def main():
    historical = []
    for batch in ('pilot_001', 'partners_001'):
        for path in sorted((ROOT / 'results' / batch).glob('prepared_s*.pt')):
            for who, state in enumerate(torch.load(path, weights_only=True)):
                historical.append({'path': str(path), 'agent': who, 'sha256': fingerprint(state)})
    fresh = []
    for batch, count in (('formation_confirm_001', 10), ('mechanism_origins_001', 7)):
        paths = sorted((ROOT / 'results' / batch).glob('prepared_s*.pt'))
        assert len(paths) == count, f'{batch}: preparation set incomplete'
        for path in paths:
            seed = int(path.stem.split('_s')[-1])
            for who, state in enumerate(torch.load(path, weights_only=True)):
                value = fingerprint(state)
                fresh.append({'batch': batch, 'seed': seed, 'agent': who, 'path': str(path),
                              'sha256': value, 'historical_matches': [x for x in historical if x['sha256'] == value]})
    assert len({x['sha256'] for x in fresh}) == len(fresh), 'New cohorts share preparation tensors'
    collisions = [x for x in fresh if x['historical_matches']]
    assert {(x['batch'], x['seed'], x['agent']) for x in collisions} == {
        ('formation_confirm_001', 1202, 0), ('formation_confirm_001', 1202, 1)}
    report = {'status': 'passed_with_disclosed_historical_overlap', 'new_prepared_agents': len(fresh),
              'within_new_cohorts_all_preparations_unique': True,
              'historical_overlap_agents': len(collisions), 'overlaps': collisions,
              'meaning': 'Only non-social preparation repeats; no socially learned protocol inherited. Keep planned ten-seed analysis and disclose the supplementary nine-seed sensitivity.',
              'new_preparations': fresh}
    path = Path(__file__).with_name('初始化溯源核查.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'new_prepared_agents', 'historical_overlap_agents')}))


if __name__ == '__main__':
    main()
