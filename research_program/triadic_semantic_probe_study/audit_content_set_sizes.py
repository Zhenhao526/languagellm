"""Read only the frozen dataset; describe content action-set sizes, no policy I/O."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_features(mask):
    actions = [a for a in range(17) if int(mask) & (1 << a)]
    assert actions and 0 not in actions
    sites = {(a - 1) // 4 for a in actions}
    destinations = {((a - 1) % 4) // 2 for a in actions}
    partner_slots = {(a - 1) % 2 for a in actions}
    assert len(destinations) == len(partner_slots) == 1
    assert len(sites) == int(mask).bit_count() == len(actions)
    return len(actions)


def audit(dataset, out):
    dataset, out = Path(dataset).resolve(), Path(out).resolve()
    assert not out.exists(), 'Do not overwrite a previous audit'
    manifest = json.loads((dataset / 'manifest.json').read_text())
    inputs = {str(dataset / 'manifest.json'): sha(dataset / 'manifest.json')}
    for name in ('cases.json', 'train.npz', 'heldout_layouts.npz'):
        digest = sha(dataset / name)
        assert digest == manifest['outputs'][name]['sha256']
        inputs[str(dataset / name)] = digest
    cases = json.loads((dataset / 'cases.json').read_text())
    selected = [c for c in cases if c['classification'] == 'content']
    assert len(selected) == 120 and all(c['ecology'] == 'unique' for c in selected)
    canonical = {}
    for axis in range(3):
        rows = [c for c in selected if c['axis_index'] == axis]
        distribution = Counter(tuple(mask_features(m) for m in c['canonical_success_action_masks']) for c in rows)
        weighted = [sum(Fraction(c['content_within_axis_weight_fraction']) * mask_features(c['canonical_success_action_masks'][e]) for c in rows) for e in range(2)]
        assert sum(Fraction(c['content_within_axis_weight_fraction']) for c in rows) == 1
        canonical[str(axis)] = dict(axis=rows[0]['axis'], case_count=len(rows),
            pair_size_distribution={','.join(map(str, k)): v for k, v in sorted(distribution.items())},
            weighted_endpoint_means_fraction=list(map(str, weighted)))
    partitions = {}
    for part in ('train', 'heldout_layouts'):
        with np.load(dataset / (part + '.npz'), allow_pickle=False) as data:
            axes = {}
            for axis in range(3):
                ids = np.flatnonzero((data['classification'] == 1) & (data['axis'] == axis))
                distribution, weighted = Counter(), [Fraction(0), Fraction(0)]
                endpoint_distribution = [Counter(), Counter()]
                mass = Fraction(0)
                for row in ids:
                    case = cases[int(data['case_index'][row])]
                    assert case['classification'] == 'content' and case['axis_index'] == axis
                    sizes = tuple(mask_features(m) for m in data['success_action_masks'][row])
                    assert sizes == tuple(mask_features(m) for m in case['canonical_success_action_masks'])
                    weight = Fraction(1, int(data['active_weight_denominator'][row]))
                    assert float(weight) == float(data['content_within_axis_weight'][row])
                    mass += weight
                    distribution[sizes] += 1
                    for endpoint in range(2):
                        weighted[endpoint] += weight * sizes[endpoint]
                        endpoint_distribution[endpoint][sizes[endpoint]] += 1
                assert mass == 1
                axes[str(axis)] = dict(axis=canonical[str(axis)]['axis'], rows=len(ids),
                    pair_size_distribution={','.join(map(str, k)): v for k, v in sorted(distribution.items())},
                    endpoint_size_distributions=[dict(sorted(d.items())) for d in endpoint_distribution],
                    axis_weight_sum_fraction=str(mass),
                    weighted_endpoint_means_fraction=list(map(str, weighted)),
                    weighted_two_endpoint_mean_fraction=str(sum(weighted) / 2),
                    weighted_two_endpoint_mean=float(sum(weighted) / 2))
            partitions[part] = axes
    report = dict(status='passed_dataset_only_exhaustive_content_set_size_audit',
        source_sha256=inputs, audit_source_sha256=sha(__file__), canonical=canonical,
        partitions=partitions, legal_set='Projection of all team-full-success joint plans onto the listener action; not the whole 17-action menu.',
        action_decoding='For action a>0: site=(a-1)//4, destination=((a-1)%4)//2, partner slot=(a-1)%2.',
        every_content_endpoint_has_single_destination_and_partner=True,
        action_count_equals_distinct_valid_site_count=True,
        weights='Frozen per-axis weights; six ordered sender/listener strata equal, then cases/layouts/owners uniform; two endpoints averaged equally.',
        policy_results_read=False, policy_success_selection=False, neural_forward_calls=0,
        scope='Static support difference only; set cardinality alone neither identifies learning preference nor measures total task difficulty.')
    text = ('预定的 120 个内容案例中，种类轴 24 例及长短轴 24 例的两端满分合法听者行动集合均为 (1,1)，目的地轴 72 例均为 (2,2)，这一分布在训练布局 12,960 行和留出布局 4,320 行的全部内容展开中一致。\n\n'
            '按冻结的轴内权重，两端各自及两端平均的集合大小在种类、长短、目的地轴分别精确为 1、1、2；每端搭档与目的地均唯一，因此目的地轴的两个行动对应两个不同合法物资位置，并非两个目的地。\n\n'
            '三个轴的合法行动集合大小未配平，且集合大小不能穷尽任务难度，因此不能仅以目的地轴约 13% 与其他轴约 2% 的适切率差直接推断学习偏好；本补记只读取冻结数据集，未读取或筛选模型成功、未做模型前向。\n')
    out.mkdir(parents=True)
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / '三轴合法行动集合补记.md').write_text(text)
    print(json.dumps(dict(status=report['status'], canonical=canonical, output=str(out)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    audit(args.dataset, args.out)
