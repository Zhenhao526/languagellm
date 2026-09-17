"""Static discovery/validation indices for single-position message probes.

No policy message, action, model parameter, or experimental outcome is read.
prepare is an explicit command and never runs on import or in make_prepared.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import argparse
import json
import numpy as np
from research_program.triadic_action_dependency_study import dataset as task
from research_program.triadic_action_dependency_study import environment as env

ROOT = Path(__file__).resolve().parents[2]
CONTEXT = ROOT / 'research_program/triadic_action_dependency_study/results/context_001'
TRANSFER_DATA = ROOT / 'research_program/triadic_context_transfer_study/dataset_001'
TRANSFER_MANIFEST_SHA = '71d1b249f2fba78c070c88efc45aec190134596ca53b40989ff742a55f5bf6d3'
CASE_SALT = 'triadic_position_reuse_case_v1'
BACKGROUND_SALT = 'triadic_position_reuse_background_v1'
DONOR_SALT = 'triadic_position_reuse_donor_v1'
AXES = ('kind', 'length', 'destination')
DISCOVERY_PARTITION = 'train'
VALIDATION_PARTITION = 'new_needs_and_layouts'
SCHEMA = 'triadic_position_reuse_dataset_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rank(salt, payload):
    text = salt + '|' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    return sha256(text.encode()).hexdigest()


def identity(case):
    """Semantic identity, independent of source row numbering or policy outputs."""
    return [case['axis_index'], case['sender'], case['listener'], case['needs'][0], case['needs'][1]]


def load_sources(context=CONTEXT, transfer_data=TRANSFER_DATA):
    context, transfer_data = Path(context).resolve(), Path(transfer_data).resolve()
    mp = transfer_data / 'manifest.json'
    require(sha(mp) == TRANSFER_MANIFEST_SHA, 'Unexpected frozen transfer dataset manifest')
    manifest = read(mp)
    paths = {str(mp): sha(mp)}
    for filename, digest in manifest['source_sha256'].items():
        require(sha(filename) == digest, 'Transfer source changed: ' + filename)
        paths[filename] = digest
    for name, record in manifest['outputs'].items():
        path = transfer_data / name
        require(sha(path) == record['sha256'], 'Transfer static artifact changed: ' + name)
        paths[str(path)] = record['sha256']
    freeze, plan, prepared = [read(context / n) for n in ('freeze.json', 'plan.json', 'prepared.json')]
    require(sha(context / 'plan.json') == freeze['plan_sha256'], 'Context plan changed')
    require(sha(context / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Context prepared changed')
    for module in (task, env):
        path = Path(module.__file__).resolve()
        require(sha(path) == plan['sources'][str(path.relative_to(ROOT))], 'Original task source changed')
    parts = {}
    for part in (DISCOVERY_PARTITION, VALIDATION_PARTITION):
        meta_path, arrays_path = transfer_data / (part + '.json'), transfer_data / (part + '.npz')
        meta = read(meta_path)
        require(meta['part_spec'] == prepared['partitions'][part], 'Transfer/context static spec mismatch')
        with np.load(arrays_path, allow_pickle=False) as handle:
            arrays = {name: handle[name] for name in handle.files}
        parts[part] = dict(metadata=meta, arrays=arrays)
    return dict(context_prepared=prepared, parts=parts, source_sha256=paths)


def extract_rows(source, selected):
    a = source['arrays']
    selected = np.asarray(selected, dtype=np.int64)
    require(selected.ndim == 1 and np.all((selected >= 0) & (selected < len(a['case_index']))), 'Invalid source row')
    fields = ('case_index', 'axis', 'sender', 'listener', 'endpoint_indices', 'correct_actions',
              'target_materials', 'recipient_layout_index', 'owner_index')
    arrays = {key: np.asarray(a[key][selected]).copy() for key in fields}
    arrays['source_spec_row'] = selected.astype(np.int32)
    arrays['source_all_case_index'] = a['source_case_index'][selected].copy()
    actions = arrays['correct_actions']
    listeners = arrays['listener']
    listener_actions = actions[np.arange(len(selected))[:, None], np.arange(2)[None, :], listeners[:, None]]
    require(np.all(listener_actions > 0), 'Content must have active listeners in both endpoints')
    arrays['listener_correct_actions'] = listener_actions
    arrays['target_sites'] = ((listener_actions - 1) // 4).astype(np.int8)
    arrays['target_destinations'] = (((listener_actions - 1) % 4) // 2).astype(np.int8)
    peers = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int8)
    partners = peers[listeners[:, None], (listener_actions - 1) % 2]
    attributes = np.stack((arrays['target_materials'] // 2, arrays['target_materials'] % 2,
                           arrays['target_destinations'], partners), axis=-1).astype(np.int8)
    arrays['truth_attributes'] = attributes
    arrays['truth_field_change_mask'] = attributes[:, 0] != attributes[:, 1]
    return arrays


def field_change_counts(arrays):
    mask = arrays['truth_field_change_mask']
    expected = np.eye(4, dtype=bool)[arrays['axis']]
    rows = {}
    for axis, name in enumerate(AXES):
        take = arrays['axis'] == axis
        patterns = Counter(''.join(str(int(v)) for v in row) for row in mask[take])
        rows[name] = dict(row_count=int(take.sum()), mask_counts=dict(sorted(patterns.items())),
            only_expected_axis_changes=int(np.sum(np.all(mask[take] == expected[take], axis=1))))
    return dict(field_order=['material_kind', 'material_length', 'destination', 'partner'],
        by_axis=rows, other_change_pattern_rows=int(np.sum(np.any(mask != expected, axis=1))),
        wait_rows=int(np.sum(np.any(arrays['listener_correct_actions'] == 0, axis=1))),
        changes_used_for_case_filtering=False,
        interpretation='Actual material attributes and decoded destination/partner, not raw site IDs. All selected rows retained regardless of mask; unchanged truth fields may define a later conjunction.')


def discovery_spec(source):
    a, meta = source['arrays'], source['metadata']
    arrays = extract_rows(source, np.arange(len(a['case_index'])))
    arrays['base_within_axis_weight'] = a['base_within_axis_weight'].copy()
    require(len(arrays['case_index']) == 209952, 'Unexpected full training discovery support')
    for axis in range(3):
        weights = arrays['base_within_axis_weight']
        require(abs(weights[arrays['axis'] == axis].sum() - 1) <= 1e-12, 'Discovery axis not balanced')
        for sender in range(3):
            take = (arrays['axis'] == axis) & (arrays['sender'] == sender)
            require(abs(weights[take].sum() * 3 - 1) <= 1e-12, 'Discovery sender-conditional weights invalid')
    return dict(arrays=arrays, metadata=dict(partition=DISCOVERY_PARTITION,
        n_rows=len(arrays['case_index']), n_directional_rows=2 * len(arrays['case_index']),
        content_cases=meta['content_cases'], source_spec='Frozen transfer train.npz row order, every row retained.',
        truth_field_change_counts=field_change_counts(arrays),
        weighting='Each axis has total base weight1. For policy/sender/axis response, select fixed sender then multiply base by3; two listeners equal, their cases uniform, every108 layout-owner backgrounds uniform. Compute the three response rates separately before target minus other-two mean.',
        source_part_spec=meta['part_spec'],
        policy_messages_read=False, policy_actions_read=False, neural_forward_calls=0))


def validation_spec(source):
    a, meta = source['arrays'], source['metadata']
    cases, spec = meta['content_cases'], meta['part_spec']
    layouts, owners = spec['layouts'], spec['private_sites']
    nl, no = len(layouts), len(owners)
    strata = {(axis, sender, listener): [] for axis in range(3)
              for sender in range(3) for listener in range(3) if sender != listener}
    for ci, case in enumerate(cases):
        strata[case['axis_index'], case['sender'], case['listener']].append(ci)
    source_rows, selected_indices, shifts = [], [], []
    case_records, background_records = [], []
    for stratum in sorted(strata):
        candidates = sorted(strata[stratum], key=lambda ci: (rank(CASE_SALT, identity(cases[ci])), identity(cases[ci])))
        require(len(candidates) >= 4, 'A validation axis/S-L stratum has fewer than four cases')
        for case_rank, ci in enumerate(candidates[:4]):
            case = cases[ci]; ident = identity(case); selected_case_index = len(case_records)
            backgrounds = sorted(((li, oi) for li in range(nl) for oi in range(no)),
                key=lambda x: (rank(BACKGROUND_SALT, [ident, layouts[x[0]], owners[x[1]]]), layouts[x[0]], owners[x[1]]))
            chosen = []
            for li, oi in backgrounds:
                if li not in {x[0] for x in chosen}:
                    chosen.append((li, oi))
                if len(chosen) == 2:
                    break
            require(len(chosen) == 2, 'Two distinct recipient layouts unavailable')
            case_records.append(dict(selected_case_index=selected_case_index, source_case_index=ci,
                case_rank_within_stratum=case_rank, case_sha256=rank(CASE_SALT, ident),
                identity=ident, case=case))
            for background_rank, (li, oi) in enumerate(chosen):
                source_row = (ci * nl + li) * no + oi
                require(int(a['case_index'][source_row]) == ci and int(a['recipient_layout_index'][source_row]) == li
                        and int(a['owner_index'][source_row]) == oi, 'Unexpected frozen source row order')
                candidates = [k for k in range(nl - 1) if bool(a['eligible'][k, source_row])]
                require(bool(candidates), 'Selected background lacks eligible donor')
                def donor_key(k):
                    dl = int(a['donor_layout_index'][k, li])
                    return rank(DONOR_SALT, [ident, layouts[li], owners[oi], layouts[dl]]), layouts[dl]
                shift = min(candidates, key=donor_key)
                dl = int(a['donor_layout_index'][shift, li])
                source_rows.append(source_row); selected_indices.append(selected_case_index); shifts.append(shift)
                background_records.append(dict(row_index=len(source_rows)-1, selected_case_index=selected_case_index,
                    source_spec_row=source_row, recipient_layout_index=li, owner_index=oi,
                    background_rank_within_case=background_rank,
                    background_sha256=rank(BACKGROUND_SALT, [ident, layouts[li], owners[oi]]),
                    donor_layout_index=dl, donor_shift_index=shift, eligible_donor_count=len(candidates),
                    donor_sha256=donor_key(shift)[0]))
    rows = np.asarray(source_rows, dtype=np.int64); shift = np.asarray(shifts, dtype=np.int16)
    arrays = extract_rows(source, rows)
    n = len(rows)
    require(n == 144 and len(case_records) == 72, 'Expected72 cases at two backgrounds each')
    arrays.update(selected_case_index=np.asarray(selected_indices, dtype=np.int16),
        donor_shift_index=shift,
        donor_endpoint_indices=a['donor_endpoint_indices'][shift, rows].copy(),
        donor_correct_actions=a['donor_correct_actions'][shift, rows].copy(),
        donor_layout_index=a['donor_layout_index'][shift, arrays['recipient_layout_index']].copy(),
        eligible_donor_count=a['eligible_donor_count'][rows].copy(),
        view_equal_LL=a['view_equal_LL'][shift, rows].copy(),
        base_within_axis_weight=np.full(n, 1 / 48, dtype=np.float64))
    require(np.all(a['eligible'][shift, rows]), 'Chosen donor fails the two-material site-change condition')
    donor_listener_actions = arrays['donor_correct_actions'][np.arange(n)[:, None], np.arange(2)[None], arrays['listener'][:, None]]
    arrays['donor_listener_correct_actions'] = donor_listener_actions
    arrays['donor_target_sites'] = ((donor_listener_actions - 1) // 4).astype(np.int8)
    require(np.all(arrays['donor_target_sites'] != arrays['target_sites']), 'Target material site did not change')
    for axis in range(3):
        require(sum(arrays['axis'] == axis) == 48, 'Validation axis count mismatch')
        for sender in range(3):
            for listener in range(3):
                if sender != listener:
                    require(np.sum((arrays['axis'] == axis) & (arrays['sender'] == sender) & (arrays['listener'] == listener)) == 8, 'Validation S-L count mismatch')
    return dict(arrays=arrays, metadata=dict(partition=VALIDATION_PARTITION,
        n_rows=n, n_directional_rows=2 * n, n_cases=len(case_records),
        truth_field_change_counts=field_change_counts(arrays),
        selected_cases=case_records, selected_backgrounds=background_records,
        case_salt=CASE_SALT, background_salt=BACKGROUND_SALT, donor_salt=DONOR_SALT,
        hash_serialization='salt + | + compact JSON payload, UTF-8, no whitespace; SHA256 then semantic payload lex tie-break.',
        case_hash_payload='[axis_index,sender,listener,needs_endpoint0,needs_endpoint1]',
        background_hash_payload='[case_identity,actual_recipient_layout,actual_owner_assignment]',
        donor_hash_payload='[case_identity,actual_recipient_layout,actual_owner_assignment,actual_donor_layout]',
        row_order='axis, sender, listener, selected case hash rank, selected background hash rank; no direction axis in N.',
        weighting='Within each axis144/3=48 base rows equal at1/48. Each direction has1/96. Three-axis macro divides by3. For discovered-axis q all evaluated-axis r rows and denominators are reused, never selected by output.',
        background_rule='Two distinct recipient LAYOUTS per case; owner may coincide. Donor remains in same held layout partition, owner unchanged; select one from preexisting eligible donors.',
        source_part_spec=spec, unobserved_combination_policy='No natural-message occurrence or success criterion enters eligibility.',
        inference_budget_scope='This static dataset fixes288 directions per policy only. Position/arm enumeration and aliases are controlled separately by the frozen runner; no inference executed here.'))


def make_prepared(bundle):
    discovery = discovery_spec(bundle['parts'][DISCOVERY_PARTITION])
    validation = validation_spec(bundle['parts'][VALIDATION_PARTITION])
    dspec = discovery['metadata']['source_part_spec']; vspec = validation['metadata']['source_part_spec']
    dn, vn = set(map(tuple, dspec['needs'])), set(map(tuple, vspec['needs']))
    dl, vl = set(map(tuple, dspec['layouts'])), set(map(tuple, vspec['layouts']))
    require(not dn & vn and not dl & vl, 'Discovery/validation source need or layout overlap')
    selected_needs = {tuple(n) for row in validation['metadata']['selected_cases'] for n in row['case']['needs']}
    require(selected_needs <= vn and not selected_needs & dn, 'Selected validation needs leak into discovery')
    orbit_by_need = {tuple(n): tuple(orbit['canonical']) for orbit in bundle['context_prepared']['need_orbits'] for n in orbit['members']}
    d_orbits, v_orbits = {orbit_by_need[n] for n in dn}, {orbit_by_need[n] for n in selected_needs}
    require(not d_orbits & v_orbits, 'Demand relation orbits overlap')
    proof = dict(discovery_source_need_count=len(dn), validation_source_need_count=len(vn),
        selected_validation_unique_need_triples=len(selected_needs),
        discovery_orbit_count=len(d_orbits), selected_validation_orbit_count=len(v_orbits),
        discovery_validation_need_intersection=[], discovery_validation_orbit_intersection=[],
        discovery_validation_layout_intersection=[], discovery_layout_count=len(dl),
        validation_source_layout_count=len(vl),
        selected_recipient_layout_counts=dict(sorted(Counter(map(int, validation['arrays']['recipient_layout_index'])).items())),
        selected_donor_layout_counts=dict(sorted(Counter(map(int, validation['arrays']['donor_layout_index'])).items())),
        selected_owner_counts=dict(sorted(Counter(map(int, validation['arrays']['owner_index'])).items())),
        individual_needs_shared=True,
        boundary='Joint three-agent need-relation orbits and layouts are disjoint; individual24 needs/material values are not. Policies and aggregate prior scores have already been developed, so this is isolation from position discovery, not a new independent policy cohort.',
        policy_messages_read=False, policy_actions_read=False, neural_forward_calls=0, training_calls=0)
    for part in (discovery, validation):
        part['metadata']['array_schema'] = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in part['arrays'].items()}
    return dict(schema=SCHEMA, discovery=discovery, validation=validation, separation_proof=proof,
                source_sha256=dict(bundle['source_sha256']))


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), 'Refuse to overwrite any previous preparation')
    prepared = make_prepared(load_sources())
    sources = dict(prepared['source_sha256']); sources[str(Path(__file__).resolve())] = sha(__file__)
    out.mkdir(parents=True)
    outputs = {}
    try:
        for label in ('discovery', 'validation'):
            data = prepared[label]
            mp, ap = out / (label + '.json'), out / (label + '.npz')
            mp.write_text(json.dumps(data['metadata'], ensure_ascii=False, indent=2) + '\n')
            np.savez_compressed(ap, **data['arrays'])
            for path in (mp, ap):
                outputs[path.name] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        proof_path = out / 'separation_proof.json'
        proof_path.write_text(json.dumps(prepared['separation_proof'], ensure_ascii=False, indent=2) + '\n')
        outputs[proof_path.name] = dict(path=str(proof_path), sha256=sha(proof_path), bytes=proof_path.stat().st_size)
        for path, digest in sources.items():
            require(sha(path) == digest, 'Source changed during static preparation')
        manifest = dict(status='prepared_static_only', schema=SCHEMA, source_sha256=sources,
            outputs=outputs, discovery_rows=prepared['discovery']['metadata']['n_rows'],
            validation_rows=144, validation_directional_rows=288,
            neural_forward_calls=0, training_calls=0, policy_messages_read=False, policy_actions_read=False)
        (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(status=manifest['status'], discovery_rows=manifest['discovery_rows'], validation_rows=144)))
        return manifest
    except BaseException as error:
        (out / 'failure.json').write_text(json.dumps(dict(status='failed', error=repr(error), source_sha256=sources, outputs=outputs), ensure_ascii=False, indent=2) + '\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare',))
    parser.add_argument('--out', required=True)
    arguments = parser.parse_args()
    prepare(arguments.out)
