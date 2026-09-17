"""Static checkpoint inventory and index pools; no parameters or policies loaded.

Only the two previously frozen static NPZ files are read as arrays. Checkpoint
NPZ files are byte-hashed, never opened with np.load. No forward or training.
"""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from research_program.triadic_action_dependency_study import dataset as task
from research_program.triadic_action_dependency_study import environment as env

ROOT = Path(__file__).resolve().parents[2]
POSITION = ROOT / 'research_program/triadic_position_reuse_study/results/position_001'
CONTEXT = ROOT / 'research_program/triadic_action_dependency_study/results/context_001'
POSITION_PLAN_SHA = '87bf3b412c10fc7a27539b90542e2ec67165e1b2ad4b19476748f2cac2a81eaa'
CONTEXT_AUDIT_SHA = '7c0eab86edff602ec6def4e54e92cacf3403cb495e39b195594b0185caeb250a'
UPDATES = (0, 100, 500, 1500, 3000, 6000)
SEEDS = (51101, 51102, 51103, 51104)
CONDITIONS = ('PL_silent', 'PL_live', 'LL_silent', 'LL_live')
HELDOUT = 'new_needs_and_layouts'
SCHEMA = 'triadic_formation_trajectory_static_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_arrays(path):
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def validation_pool(validation, part_spec):
    """Ascending original full-domain IDs; repeated row references stay intact."""
    recipient = np.asarray(validation['endpoint_indices'])
    donor = np.asarray(validation['donor_endpoint_indices'])
    require(recipient.ndim == 2 and recipient.shape[1] == 2 and donor.shape == recipient.shape,
            'Recipient/donor endpoints must have the same N x 2 shape')
    require(recipient.dtype.kind in 'iu' and donor.dtype.kind in 'iu', 'Endpoint indices must be integers')
    require(np.all((recipient >= 0) & (recipient < part_spec['world_count']))
            and np.all((donor >= 0) & (donor < part_spec['world_count'])), 'Endpoint index outside original partition')
    indices = np.unique(np.concatenate((recipient.ravel(), donor.ravel())))
    recipient_rows, donor_rows = np.searchsorted(indices, recipient), np.searchsorted(indices, donor)
    require(np.array_equal(indices[recipient_rows], recipient) and np.array_equal(indices[donor_rows], donor),
            'Pool does not exactly reconstruct original endpoint references')
    states = task.pack_states(part_spec, indices)
    require(len(np.unique(states, axis=0)) == len(indices), 'Original index/state mapping is not one-to-one')
    arrays = dict(validation_pool_endpoint_indices=indices.astype(np.int32),
        validation_pool_states=states,
        recipient_pool_rows=recipient_rows.astype(np.int32), donor_pool_rows=donor_rows.astype(np.int32))
    metadata = dict(partition=HELDOUT, original_world_count=part_spec['world_count'],
        undirected_rows=len(recipient), directional_rows=recipient.size,
        recipient_unique_worlds=len(np.unique(recipient)), donor_unique_worlds=len(np.unique(donor)),
        recipient_donor_intersection_worlds=len(np.intersect1d(recipient, donor)),
        unique_natural_worlds=len(indices), total_endpoint_references=recipient.size + donor.size,
        pool_order='Ascending full double-held endpoint index; original row and direction order preserved by maps.',
        filtering='Only exact identical world IDs deduplicated; no truth, action, message, or outcome filtering.')
    return dict(arrays=arrays, metadata=metadata)


def computation_budget(train_worlds, validation_worlds):
    """Natural-message accounting only, not a runner execution authorization."""
    full_cells = len(SEEDS) * len(CONDITIONS) * len(UPDATES)
    new_cells_if_final_reused = len(SEEDS) * len(CONDITIONS) * (len(UPDATES) - 1)
    def counts(cells):
        return dict(policy_checkpoint_cells=cells,
            training_message_world_samples=cells * train_worlds,
            training_message_module_samples=cells * train_worlds * 6,
            validation_natural_world_samples=cells * validation_worlds,
            validation_natural_module_samples=cells * validation_worlds * 9)
    return dict(full_six_checkpoint_matrix=counts(full_cells),
        first_five_if_6000_saved_outputs_reused=counts(new_cells_if_final_reused),
        training_modules_per_world=6, validation_modules_per_world=9,
        description='Training domain: three sender1 + three sender2 only; no action heads. Validation pool: these six plus three action heads. Silent still generates own messages. 6000 reuse is an option for the runner, not performed by static preparation.',
        excludes='Position/whole-packet interventions, sham checks, independent audit replay and any extra action evaluation are not included.',
        actual_forward_worlds=0, actual_module_samples=0, actual_training_updates=0)


def load_sources(position=POSITION, context=CONTEXT):
    position, context = Path(position).resolve(), Path(context).resolve()
    sources = {}
    def check(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, 'Changed source: ' + str(path))
        sources[str(path)] = digest
        return read(path)

    freeze = check(position / 'freeze.json')
    require(freeze['plan_sha256'] == POSITION_PLAN_SHA, 'Unexpected position-study freeze')
    old_plan = check(position / 'plan.json', POSITION_PLAN_SHA)
    position_results_path = position / 'execution/results.json'
    # Hash the earlier completed result for the runner's 6000 reference lookup;
    # do not deserialize its outcomes or selection statistics here.
    sources[str(position_results_path)] = sha(position_results_path)
    old_frozen_path = position / 'frozen_spec.json'
    frozen = check(old_frozen_path, old_plan['prepared_files_sha256'][str(old_frozen_path)])
    require(frozen['discovery_started'] is False, 'Position design was not bound before discovery')
    for module in (task, env):
        path = Path(module.__file__).resolve()
        sources[str(path)] = sha(path)
        require(sources[str(path)] == old_plan['sources_sha256'][str(path)], 'Changed original task interface')
    manifest_path = position / 'dataset/manifest.json'
    manifest = check(manifest_path, old_plan['prepared_files_sha256'][str(manifest_path)])
    static_paths, static = {}, {}
    for name in ('discovery.json', 'discovery.npz', 'validation.json', 'validation.npz', 'separation_proof.json'):
        path = position / 'dataset' / name
        digest = sha(path)
        require(digest == manifest['outputs'][name]['sha256'] == old_plan['prepared_files_sha256'][str(path)],
                'Changed prior static dataset artifact: ' + name)
        sources[str(path)] = digest; static_paths[name] = str(path)
    for name in ('discovery', 'validation'):
        static[name] = dict(metadata=read(static_paths[name + '.json']), arrays=load_arrays(static_paths[name + '.npz']))

    cf = check(context / 'freeze.json', manifest['source_sha256'][str(context / 'freeze.json')])
    cp = check(context / 'plan.json', cf['plan_sha256'])
    prepared = check(context / 'prepared.json', cf['prepared_sha256'])
    require(cf['prepared_sha256'] == cp['prepared_sha256'], 'Context prepared identity')
    for key, part in (('discovery', 'train'), ('validation', HELDOUT)):
        require(static[key]['metadata']['source_part_spec'] == prepared['partitions'][part], 'Static/context partition mismatch')
    audit_path = context / 'audit_execution_001/verification.json'
    audit = check(audit_path, CONTEXT_AUDIT_SHA)
    require(audit['status'] == 'passed' and audit['plan_sha256'] == cf['plan_sha256'], 'Original execution audit did not pass')
    original_results_path = context / 'execution/results.json'
    original_results = check(original_results_path, audit['artifacts_sha256'][str(original_results_path)])
    require(original_results['status'] == 'completed' and original_results['completed_run_count'] == 24,
            'Original context matrix incomplete')
    require(original_results['plan_sha256'] == cf['plan_sha256'], 'Original results plan mismatch')
    cohorts = {(p['seed'], p['condition']): p for p in old_plan['policies']}
    require(set(cohorts) == {(s, c) for s in SEEDS for c in CONDITIONS}, 'Changed position policy cohort')
    records = {(r['seed'], r['condition']): r for r in original_results['runs']}
    checkpoints = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            directory = context / 'execution' / f'seed_{seed}_{condition}'
            run_path = directory / 'result.json'
            run = check(run_path, audit['artifacts_sha256'][str(run_path)])
            require(run == records[seed, condition] and run['updates'] == 6000, 'Original run/main metadata mismatch')
            require([r['update'] for r in run['monitor']] == list(UPDATES), 'Missing, duplicate, or reordered checkpoint metadata')
            old_policy = cohorts[seed, condition]
            for monitor in run['monitor']:
                update = monitor['update']; path = directory / f'checkpoint_{update:04d}.npz'
                digest = sha(path)  # Byte identity only: never parameter deserialization.
                require(digest == monitor['checkpoint_sha256'] == audit['artifacts_sha256'][str(path)],
                        'Checkpoint bytes disagree with original run and audit: ' + str(path))
                sources[str(path)] = digest
                record = dict(seed=seed, condition=condition, update=update, path=str(path),
                    sha256=digest, bytes=path.stat().st_size, source_run_record=str(run_path),
                    source_main_result=str(original_results_path), source_audit=str(audit_path),
                    array_parameters_loaded=False, neural_forward_calls=0)
                if update == 6000:
                    require(digest == run['final_checkpoint_sha256'] == old_plan['inputs_sha256'][str(path)]
                            and str(path) == old_policy['checkpoint'], 'Final checkpoint does not match position-study source')
                    record['saved_natural_6000'] = {part: dict(path=old_policy['endpoints'][part],
                        sha256=old_plan['inputs_sha256'][old_policy['endpoints'][part]],
                        current_bytes_read=False, boundary='Identity recorded in prior freeze; this static inventory does not read or verify endpoint output bytes.')
                        for part in ('train', HELDOUT)}
                    record.update(prior_policy_tag=f'seed_{seed}_{condition}',
                        source_position_result=str(position_results_path),
                        source_position_result_sha256=sources[str(position_results_path)],
                        source_position_policy_directory=str(position / 'execution' / f'seed_{seed}_{condition}'),
                        source_position_selection=old_policy['selection'],
                        source_position_selection_sha256=old_plan['prepared_files_sha256'][old_policy['selection']])
                checkpoints.append(record)
    return dict(discovery=static['discovery'], validation=static['validation'],
        separation_proof=read(static_paths['separation_proof.json']), static_paths=static_paths,
        context_prepared=prepared, checkpoints=checkpoints, source_sha256=sources)


def make_prepared(bundle):
    discovery, validation = bundle['discovery'], bundle['validation']
    train_spec = bundle['context_prepared']['partitions']['train']
    pool = validation_pool(validation['arrays'], bundle['context_prepared']['partitions'][HELDOUT])
    pool['arrays']['train_world_indices'] = np.arange(train_spec['world_count'], dtype=np.int32)
    ds_indices = discovery['arrays']['endpoint_indices']
    require(ds_indices.shape == (209952, 2) and train_spec['world_count'] == 419904,
            'Changed complete training discovery support')
    require(np.all((ds_indices >= 0) & (ds_indices < train_spec['world_count'])), 'Discovery reference outside training support')
    require(pool['metadata']['unique_natural_worlds'] == 554 and pool['metadata']['undirected_rows'] == 144,
            'Changed fixed validation world pool')
    pool['metadata'].update(schema=SCHEMA, training_partition='train', training_world_count=train_spec['world_count'],
        training_spec=train_spec, validation_spec=bundle['context_prepared']['partitions'][HELDOUT],
        discovery_rows=209952, discovery_endpoint_references=419904,
        discovery_unique_worlds=len(np.unique(ds_indices)),
        training_domain='All 419904 original training worlds in original full-domain index order, including worlds not referenced by discovery pairs.',
        budget=computation_budget(train_spec['world_count'], len(pool['arrays']['validation_pool_endpoint_indices'])),
        selected_natural_strategy='first_five_if_6000_saved_outputs_reused',
        no_new_truth_generation=True, no_sample_reselection=True)
    return dict(schema=SCHEMA, discovery=discovery, validation=validation, pool=pool,
        checkpoints=bundle['checkpoints'], separation_proof=bundle['separation_proof'],
        source_sha256=dict(bundle['source_sha256']), static_paths=dict(bundle['static_paths']))


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), 'Refuse overwrite of static preparation')
    prepared = make_prepared(load_sources())
    out.mkdir(parents=True)
    for name, source in prepared['static_paths'].items():
        shutil.copyfile(source, out / name)  # Existing truth and row order byte-for-byte.
    def write(name, value):
        with (out / name).open('x', encoding='utf8') as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
    np.savez_compressed(out / 'pool.npz', **prepared['pool']['arrays'])
    write('pool.json', prepared['pool']['metadata'])
    write('checkpoints.json', prepared['checkpoints'])
    outputs = {p.name: dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)
               for p in sorted(out.iterdir()) if p.is_file()}
    for path, digest in prepared['source_sha256'].items():
        require(sha(path) == digest, 'Input changed during static preparation: ' + path)
    write('manifest.json', dict(schema=SCHEMA, status='prepared_static_only',
        at=datetime.now(timezone.utc).isoformat(), source_sha256=prepared['source_sha256'],
        dataset_source_sha256=sha(__file__), outputs=outputs,
        checkpoint_count=len(prepared['checkpoints']), policy_count=16, updates=list(UPDATES),
        validation_unique_natural_worlds=554, training_world_count=419904,
        neural_forward_calls=0, training_updates=0, checkpoint_parameters_loaded=False,
        policy_message_action_arrays_loaded=False, validation_rows_reselected=False))
    return read(out / 'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare'])
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = prepare(args.out)
    print(json.dumps({k: result[k] for k in ('status', 'checkpoint_count', 'validation_unique_natural_worlds')}, ensure_ascii=False))
