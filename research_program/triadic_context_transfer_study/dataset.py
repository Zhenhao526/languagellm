"""Static content-case expansion and all-other-layout donor indices.

Only frozen environment/dataset definitions are read; no policy records or weights.
All truth, indices, eligibility, and labels in this module are researcher-side.
"""
from collections import Counter
from hashlib import sha256
from itertools import permutations
from pathlib import Path
import argparse
import json
import numpy as np

from research_program.triadic_action_dependency_study import dataset as source_dataset
from research_program.triadic_action_dependency_study import environment as env

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'research_program/triadic_action_dependency_study/results/context_001'
PARTITIONS = source_dataset.PARTITIONS
AXES = source_dataset.AXES
RANK_PREFIX = 'triadic_context_transfer_layout_rank_v1'
SCHEMA = 'triadic_context_transfer_dataset_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def compact(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


def layout_mapping(layouts):
    """[L-1,L]: each nonzero cyclic rank shift is a fixed-point-free bijection."""
    layouts = [list(map(int, x)) for x in layouts]
    require(len(layouts) > 1 and len({tuple(x) for x in layouts}) == len(layouts), 'Need distinct layouts')
    require(all(sorted(x) == list(range(4)) for x in layouts), 'Invalid layout permutation')
    ranking = sorted(range(len(layouts)), key=lambda i: (
        sha256((RANK_PREFIX + '|' + compact(layouts[i])).encode()).hexdigest(), layouts[i]))
    mappings = np.empty((len(layouts) - 1, len(layouts)), dtype=np.int16)
    for shift in range(1, len(layouts)):
        for rank, index in enumerate(ranking):
            mappings[shift - 1, index] = ranking[(rank + shift) % len(layouts)]
    return mappings, ranking


def remap_actions(canonical_actions, layout_positions):
    """Broadcast canonical [...,3] actions to actual material sites.

    Canonical site's integer is material ID; partner/destination stay unchanged.
    """
    actions = np.asarray(canonical_actions)
    positions = np.asarray(layout_positions)
    active = actions != 0
    materials = np.maximum(actions - 1, 0) // 4
    site = np.take_along_axis(positions, materials, axis=-1)
    return np.where(active, 1 + site * 4 + (actions - 1) % 4, 0).astype(np.int8)


def make_spec(part_spec):
    """Pure construction: {'metadata': JSON-ready, 'arrays': NumPy arrays}.

    N = content case × recipient layout × owner, without direction.
    Endpoint order 0/1 is the unchanged original case order. K = L-1.
    """
    all_rows = source_dataset.content_pairs(part_spec)['rows']
    selected = [(i, row) for i, row in enumerate(all_rows) if row['classification'] == 'content']
    require(bool(selected), 'No content cases')
    rows = [row for _, row in selected]
    layouts = np.asarray(part_spec['layouts'], dtype=np.int16)
    owners = np.asarray(part_spec['private_sites'], dtype=np.int16)
    require(sorted(map(tuple, owners.tolist())) == sorted(permutations((1, 2, 3))), 'Expected all six owner permutations')
    mappings, ranking = layout_mapping(layouts)
    c, l, o = len(rows), len(layouts), len(owners)
    b, n, k = l * o, c * l * o, l - 1
    case_index = np.repeat(np.arange(c, dtype=np.int32), b)
    layout_index = np.tile(np.repeat(np.arange(l, dtype=np.int16), o), c)
    owner_index = np.tile(np.arange(o, dtype=np.int8), c * l)
    need_ids = np.asarray([r['endpoint_need_indices'] for r in rows], dtype=np.int32)[case_index]
    endpoint_indices = (need_ids * l + layout_index[:, None]) * o + owner_index[:, None]
    canonical = np.asarray([r['correct_actions'] for r in rows], dtype=np.int8)[case_index]
    positions = np.argsort(layouts, axis=1).astype(np.int8)
    correct = remap_actions(canonical, np.broadcast_to(positions[layout_index, None, :], (n, 2, 4)))
    target_materials_by_case = np.asarray([[env.full_success_plans(need)[0][2]
        for need in row['needs']] for row in rows], dtype=np.int8)
    target_materials = target_materials_by_case[case_index]
    recipient_target_sites = np.take_along_axis(positions[layout_index], target_materials, axis=1)
    donor_layouts = mappings[:, layout_index]
    donor_endpoint_indices = (need_ids[None, :, :] * l + donor_layouts[:, :, None]) * o + owner_index[None, :, None]
    donor_positions = positions[donor_layouts]
    donor_correct = remap_actions(np.broadcast_to(canonical[None], (k, n, 2, 3)),
        np.broadcast_to(donor_positions[:, :, None, :], (k, n, 2, 4)))
    donor_target_sites = np.take_along_axis(donor_positions,
        np.broadcast_to(target_materials[None], (k, n, 2)), axis=-1)
    changed = donor_target_sites != recipient_target_sites[None]
    eligible = changed.all(axis=2)
    eligible_count = eligible.sum(axis=0).astype(np.int16)
    require(np.all(eligible_count > 0), 'Some case-recipient background lacks a both-target-site-changing donor')
    axis = np.asarray([r['axis_index'] for r in rows], dtype=np.int8)[case_index]
    sender = np.asarray([r['sender'] for r in rows], dtype=np.int8)[case_index]
    listener = np.asarray([r['listener'] for r in rows], dtype=np.int8)[case_index]
    denominator = np.asarray([r['within_axis_case_weight_denominator'] for r in rows], dtype=np.int32)[case_index]
    base_weight = 1.0 / (denominator.astype(np.float64) * b)
    for a in range(3):
        require(abs(base_weight[axis == a].sum() - 1) <= 1e-12, 'Missing axis/S-L strata or invalid weight')
    own_site = owners[owner_index, listener]
    view_equal = ((layouts[donor_layouts, 0] == layouts[layout_index, 0][None]) &
        (layouts[donor_layouts, own_site[None]] == layouts[layout_index, own_site][None]))
    arrays = dict(case_index=case_index,
        source_case_index=np.asarray([i for i, _ in selected], dtype=np.int32)[case_index],
        axis=axis, sender=sender, listener=listener,
        endpoint_indices=endpoint_indices.astype(np.int32), correct_actions=correct,
        target_materials=target_materials,
        recipient_layout_index=layout_index, owner_index=owner_index,
        donor_layout_index=mappings, donor_endpoint_indices=donor_endpoint_indices.astype(np.int32),
        donor_correct_actions=donor_correct, target_site_changed=changed,
        eligible=eligible, eligible_donor_count=eligible_count,
        base_within_axis_weight=base_weight, view_equal_LL=view_equal)
    per_axis = {}
    for a, name in enumerate(AXES):
        mask = axis == a
        per_axis[name] = dict(case_count=sum(r['axis_index'] == a for r in rows),
            recipient_rows=int(mask.sum()), eligible_min=int(eligible_count[mask].min()),
            eligible_max=int(eligible_count[mask].max()),
            eligible_count_histogram={str(v): int(count) for v, count in zip(*np.unique(eligible_count[mask], return_counts=True))},
            all_other_rows=int(k * mask.sum()), eligible_rows=int(eligible[:, mask].sum()),
            within_axis_base_weight_sum=float(base_weight[mask].sum()),
            all_other_both_targets_change_weight=float(np.sum(base_weight[mask] * eligible[:, mask].mean(axis=0))),
            all_other_listener_LL_view_equal_weight=float(np.sum(base_weight[mask] * view_equal[:, mask].mean(axis=0))))
    metadata = dict(schema=SCHEMA, partition=part_spec['partition'], part_spec=part_spec,
        content_cases=rows, source_case_indices=[i for i, _ in selected],
        row_order='content case in original order, recipient layout, owner; no direction dimension in N',
        endpoint_order='Original case endpoint 0/1; same donor uses b, opposite donor uses 1-b.',
        donor_order='K axis is nonzero SHA-rank cyclic shift 1..L-1; donor_layout_index is [K,L].',
        rank_prefix=RANK_PREFIX, rank_indices=ranking,
        rank_hash_serialization='prefix + | + compact JSON actual layout; SHA256 then lex layout',
        n_rows=n, n_cases=c, n_layouts=l, n_owners=o, n_donors=k, by_axis=per_axis,
        weighting=dict(base='Within each axis: six ordered sender/listener pairs equal, cases within each stratum equal, recipient layouts and owners equal.',
            primary_per_direction='base_within_axis_weight * eligible / eligible_donor_count / 2',
            all_other_per_direction='base_within_axis_weight / n_donors / 2',
            macro='Average the three axis results; divide by 3 only here.',
            normalization_tolerance=1e-12),
        eligibility='Both endpoint target materials change site; all original cases and recipient backgrounds retained; no learned-output filter.',
        correctness_scope='For endpoint b, its donor true correct action differs from its recipient true correct action. Thus copying the corresponding opposite-need donor correct action cannot reach the recipient counterfactual target. A same-need donor correct action may hit that target when the two materials exchange sites; actual erroneous donor actions can also coincide with it. Evaluate actual same/opposite donor-action sets separately, never use them as dataset filtering.',
        view_equal_LL_scope='Only the designated listener material observation is compared; own need and owner are already unchanged. It does not assert equality of messages or action input.',
        axes=list(AXES), neural_forward_calls=0, training_calls=0, policy_outputs_read=0,
        array_schema={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()})
    return dict(metadata=metadata, arrays=arrays)


def load_source(source=SOURCE):
    source = Path(source).resolve()
    paths = [source / name for name in ('freeze.json', 'plan.json', 'prepared.json')]
    freeze, plan, prepared = [json.loads(p.read_text()) for p in paths]
    require(sha(paths[1]) == freeze['plan_sha256'], 'Frozen source plan mismatch')
    require(sha(paths[2]) == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Frozen source prepared mismatch')
    sources = {str(p): sha(p) for p in paths}
    for module in (source_dataset, env):
        path = Path(module.__file__).resolve()
        require(sha(path) == plan['sources'][str(path.relative_to(ROOT))], 'Frozen source module mismatch')
        sources[str(path)] = sha(path)
    return prepared, sources


def prepare(out, source=SOURCE):
    out = Path(out).resolve()
    require(not out.exists(), 'Refuse to overwrite any prior preparation')
    prepared, sources = load_source(source)
    sources[str(Path(__file__).resolve())] = sha(__file__)
    out.mkdir(parents=True)
    records = {}
    try:
        for part in PARTITIONS:
            result = make_spec(prepared['partitions'][part])
            metadata_path, array_path = out / (part + '.json'), out / (part + '.npz')
            metadata_path.write_text(json.dumps(result['metadata'], ensure_ascii=False, indent=2) + '\n')
            np.savez_compressed(array_path, **result['arrays'])
            records[part] = dict(metadata_path=str(metadata_path), metadata_sha256=sha(metadata_path),
                array_path=str(array_path), array_sha256=sha(array_path),
                n_rows=result['metadata']['n_rows'], n_donors=result['metadata']['n_donors'],
                by_axis=result['metadata']['by_axis'])
        for path, digest in sources.items():
            require(sha(path) == digest, 'Source changed during preparation')
        outputs = {}
        for record in records.values():
            for kind in ('metadata', 'array'):
                path = Path(record[kind + '_path'])
                outputs[path.name] = dict(path=str(path), sha256=record[kind + '_sha256'], bytes=path.stat().st_size)
        manifest = dict(status='prepared_static_only', schema=SCHEMA, source_sha256=sources,
            partitions=records, outputs=outputs, all_content_cases_and_backgrounds_retained=True,
            neural_forward_calls=0, training_calls=0, policy_outputs_read=0)
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(status=manifest['status'], partitions={p: r['n_rows'] for p, r in records.items()})))
        return manifest
    except BaseException as exc:
        (out / 'failure.json').write_text(json.dumps(dict(status='failed', error=repr(exc),
            source_sha256=sources, completed_partitions=records), ensure_ascii=False, indent=2) + '\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare',))
    parser.add_argument('--out', required=True)
    parser.add_argument('--source', default=str(SOURCE))
    arguments = parser.parse_args()
    prepare(arguments.out, arguments.source)
