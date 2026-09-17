"""Independent reconstruction of mechanism exposure, data, freezes and RNGs."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PAIRS = tuple(itertools.combinations(range(4), 2))
ORIGINAL = ((0, 1), (2, 3))
FLAGS = {'both_learn': {'project': False, 'sender': True, 'actor': True, 'value': True},
         'sender_only': {'project': False, 'sender': True, 'actor': False, 'value': True},
         'listener_only': {'project': False, 'sender': False, 'actor': True, 'value': True},
         'behavior_frozen': {'project': False, 'sender': False, 'actor': False, 'value': True},
         'all_plastic': dict.fromkeys(('project', 'sender', 'actor', 'value'), True)}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def manifests(folder):
    for filename, expected in read(folder / 'source_hashes.json').items():
        assert sha(folder / 'source' / filename) == expected
    for filename, expected in read(folder / 'batch_artifact_hashes.json').items():
        assert sha(folder / filename) == expected
    assert read(folder / 'source_validation.json')['passed']


def compare_states(left, right):
    assert len(left) == len(right) == 4
    assert all(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a) for a, b in zip(left, right))


def reconstruct(seed, rows, pools, total, mechanism=True):
    """Independent source scene enumeration; no training sampler imports."""
    full = np.asarray([s for s in itertools.product((0, 1), repeat=4) if len(set(s)) == 2], dtype=np.int64).reshape(14, 2, 2)
    same = full[..., 0] == full[..., 1]
    course = full[same.sum(1) == 1]
    rng_seed = seed + (2000 if mechanism else 0)
    world = np.random.default_rng(rng_seed * 10000 + 101)
    layout = np.random.default_rng(rng_seed * 10000 + 302)
    matching_rng = np.random.default_rng(rng_seed * 10000 + 301)
    rotation = np.concatenate([matching_rng.permutation(3) for _ in range(total // 3)])
    matchings = np.asarray([[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]])
    edges = np.zeros((4, 4), dtype=np.int64)
    history = {0: edges.copy()}
    assert len(rows) == total
    for offset, row in enumerate(rows):
        update = offset + 1
        assert row['update'] == update
        scenes = full if mechanism or update > 600 else course
        kinds = scenes[world.integers(len(scenes), size=2048)].reshape(1024, 2, 2, 2)
        ids = np.empty(kinds.shape, dtype=np.int64)
        for q in (0, 1):
            mask = kinds == q
            ids[mask] = world.choice(pools[q], int(mask.sum()))
        bits = layout.integers(2, size=3)
        index = int(rotation[offset]) if mechanism else 0
        assignment = matchings[index].copy()
        if bits[0]:
            assignment = assignment[::-1].copy()
        for slot in (0, 1):
            if bits[slot + 1]:
                assignment[slot] = assignment[slot, ::-1]
        assert row['world_slots_sha256'] == digest(kinds, ids)
        assert row['matching_index'] == index and row['slot_assignment'] == assignment.tolist()
        assert row['layout_bits'] == bits.tolist()
        assert row['task'] == ('full' if mechanism or update > 600 else 'curriculum')
        assert row['stage'] == ('mechanism_pure_reward' if mechanism else
                                'course' if update <= 600 else 'full_exploration' if update <= 900 else 'pure_reward')
        assert row['aux_weight'] == 0
        assert row['action_entropy_weight'] == (0 if mechanism or update > 900 else .05)
        if mechanism:
            assert row['lifetime_update'] == update + 1200
        assert [a['agent'] for a in row['agents']] == [0, 1, 2, 3]
        for slot, pair in enumerate(assignment):
            outcome = row['pair_outcomes'][slot]
            assert outcome['agents'] == pair.tolist() and outcome['slot'] == slot
            assert outcome['success'] == outcome['successes'] / 1024
            edges[pair[0], pair[1]] += 1
            edges[pair[1], pair[0]] += 1
            for local, who in enumerate(pair):
                entry = row['agents'][who]
                partner = int(pair[1 - local])
                assert entry['partner'] == partner
                assert entry['received_sha256'] == row['agents'][partner]['sent_sha256']
                assert entry['sampled_success'] == outcome['success']
                assert entry['private_input_sha256'] == digest(kinds[:, slot, local], ids[:, slot, local])
                if mechanism:
                    assert entry['mixed_resource_diagnostics']['n'] == int((kinds[:, slot, local].sum(-1) == 1).sum())
        history[update] = edges.copy()
    policies = []
    for who in range(4):
        rng = np.random.default_rng(rng_seed * 10000 + 102 + who)
        rng.bit_generator.advance(2 * 1024 * total)
        policies.append(rng.bit_generator.state)
    return {'edges': edges, 'history': history, 'world': world.bit_generator.state,
            'layout': layout.bit_generator.state, 'policies': policies, 'rotation': rotation}


def audit(folder):
    manifests(folder)
    config = read(folder / 'config.json')
    assert config['confirmatory_seeds'] == [404, 505, 606, 707, 808, 909, 1001]
    assert config['exploratory_seeds'] == [101, 202, 303]
    assert config['additional_updates'] == 600 and config['batch_size_per_agent'] == 1024
    assert config['training_seed_offset'] == 2000
    assert config['trainability_by_condition'] == FLAGS
    assert read(folder / 'completed.json')['runs'] == 47
    entries = read(folder / 'data_manifest.json')['images']
    pools = {q: np.asarray([i for i, e in enumerate(entries) if e['split'] == 'train' and e['category'] == category])
             for q, category in enumerate(('food', 'water'))}
    aggregated = {(r['seed'], r['condition']): r for r in read(folder / 'results.json')}
    records, origin_records = [], []
    old_contact = folder.parent / 'contact_001'
    old_config = read(old_contact / 'config.json')
    for key in ('additional_updates', 'batch_size_per_agent', 'horizon', 'learning_rate', 'action_entropy_coefficient',
                'signalling_coefficient', 'training_seed_offset', 'checkpoint_evaluation_seed_offset',
                'evaluation_seed_offset', 'checkpoint_evaluation_n', 'evaluation_n', 'intervention_n', 'alignment_n',
                'protocol_probe_n', 'protocol_support_n', 'protocol_seed_offset', 'fresh_adam_all_conditions'):
        assert old_config[key] == config[key]
    for name, expected in read(old_contact / 'source_hashes.json').items():
        assert sha(old_contact / 'source' / name) == expected == sha(folder / 'source' / name)
    references = read(folder / 'all_plastic_references.json')
    assert len(references) == 10
    reused = [r for r in references if r['reused']]
    assert [r['seed'] for r in reused] == config['exploratory_seeds']
    for reference in reused:
        path = Path(reference['path'])
        assert path == old_contact / f"fixed_to_rotating_s{reference['seed']}"
        assert sha(path / 'result.json') == reference['result_sha256']
        assert sha(path / 'checkpoint_0600.pt') == reference['checkpoint_sha256']
        assert sha(path / 'initial.pt') == sha(folder / f"both_learn_s{reference['seed']}/initial.pt")
    origins = folder.parent / config['origins_batch']
    manifests(origins)
    preparations = read(origins / 'preparations.json')
    assert [r['seed'] for r in preparations] == config['confirmatory_seeds']
    assert len(preparations) == 7 and all(r['passed'] and not r['reused_agents'] for r in preparations)
    for entry in preparations:
        assert entry['source_seeds'] == [entry['seed'], entry['seed'] + 1000]
        assert len(entry['agents']) == 4 and all(r['heldout_need_sensitive_choice'] >= .8 for r in entry['agents'])
        assert all(r['sender_head_unchanged_during_preparation'] for r in entry['agents'])
        assert sha(origins / f"prepared_s{entry['seed']}.pt") == entry['prepared_sha256']
    for seed in config['confirmatory_seeds']:
        path = origins / f'fixed_s{seed}'
        rows = [json.loads(line) for line in (path / 'training_metrics.jsonl').read_text().splitlines()]
        checked = reconstruct(seed, rows, pools, 1200, mechanism=False)
        result = read(path / 'result.json')
        assert result['edge_training_updates'] == checked['edges'].tolist()
        assert result['updates'] == 1200
        compare_states(torch.load(origins / f'prepared_s{seed}.pt', weights_only=True), torch.load(path / 'checkpoint_0000.pt', weights_only=True))
        for row in read(path / 'learning_curve.json'):
            assert row['edge_training_updates'] == checked['history'][row['update']].tolist()
        origin_records.append({'seed': seed, 'updates': 1200, 'all_inputs_and_exposure_reconstructed': True,
                               'individual_preparation_gate_passed': True, 'no_reused_agents': True})
    for seed in config['seeds']:
        conditions = config['conditions'] + (['all_plastic'] if seed in config['confirmatory_seeds'] else [])
        first_rows = None
        frozen_sender_messages = None
        for condition in conditions:
            path = folder / f'{condition}_s{seed}'
            rows = [json.loads(line) for line in (path / 'training_metrics.jsonl').read_text().splitlines()]
            # Reconstruct one identical external stream per population and verify
            # all other conditions byte-for-byte in every external input field.
            if first_rows is None:
                checked = reconstruct(seed, rows, pools, 600)
                first_rows = rows
            else:
                assert len(rows) == len(first_rows) == 600
                for row, base in zip(rows, first_rows):
                    for key in ('update', 'lifetime_update', 'task', 'stage', 'aux_weight', 'action_entropy_weight',
                                'matching_index', 'layout_bits', 'slot_assignment', 'world_slots_sha256'):
                        assert row[key] == base[key]
                    for a, b in zip(row['agents'], base['agents']):
                        assert a['private_input_sha256'] == b['private_input_sha256']
                        assert a['partner'] == b['partner'] and a['agent'] == b['agent']
                        assert a['received_sha256'] == row['agents'][a['partner']]['sent_sha256']
                        assert a['mixed_resource_diagnostics']['n'] == b['mixed_resource_diagnostics']['n']
            for row in rows:
                for agent in row['agents']:
                    assert all(agent['gradient_norms'][name] == 0 for name, active in FLAGS[condition].items() if not active)
                    partner = agent['partner']
                    assert agent['sampled_success'] == row['agents'][partner]['sampled_success']
                assert row['sampled_training_success'] == np.mean([p['success'] for p in row['pair_outcomes']])
            if condition in ('listener_only', 'behavior_frozen'):
                messages = [[a['sent_sha256'] for a in row['agents']] for row in rows]
                if frozen_sender_messages is None:
                    frozen_sender_messages = messages
                else:
                    assert messages == frozen_sender_messages
            result, metadata = read(path / 'result.json'), read(path / 'origin.json')
            assert result == aggregated[seed, condition]
            assert metadata['trainability'] == result['trainability'] == FLAGS[condition]
            assert metadata['initial_optimizer_state_entries'] == [0] * 4 and metadata['fresh_adam']
            original_path = Path(metadata['origin_path'])
            assert sha(path / 'initial.pt') == sha(original_path) == metadata['origin_sha256'] == metadata['initial_sha256']
            original = torch.load(original_path, weights_only=True)
            curve = read(path / 'learning_curve.json')
            assert [r['update'] for r in curve] == config['checkpoints']
            assert curve[0]['evaluation'] == read(original_path.parent / 'learning_curve.json')[-1]['evaluation']
            old_edges = np.asarray(read(original_path.parent / 'result.json')['edge_training_updates'])
            for entry in curve:
                update = entry['update']
                states = torch.load(path / f'checkpoint_{update:04d}.pt', weights_only=True)
                assert entry['edge_training_updates'] == checked['history'][update].tolist()
                assert entry['cumulative_edge_training_updates'] == (old_edges + checked['history'][update]).tolist()
                for before, after in zip(original, states):
                    assert all(torch.isfinite(p).all() for p in after.values())
                    for name, value in before.items():
                        if not FLAGS[condition][name.split('.')[0]]:
                            assert torch.equal(value, after[name]), f'Frozen tensor changed: {path.name} {update} {name}'
                if condition == 'behavior_frozen':
                    assert entry['evaluation'] == curve[0]['evaluation']
            state = torch.load(path / 'final_training_state.pt', weights_only=True)
            final = torch.load(path / 'checkpoint_0600.pt', weights_only=True)
            compare_states(state['agents'], final)
            assert state['condition'] == condition and state['trainability'] == FLAGS[condition]
            assert state['world_rng'] == checked['world'] and state['layout_rng'] == checked['layout']
            assert state['policy_rngs'] == checked['policies']
            np.testing.assert_array_equal(state['matching_schedule'], checked['rotation'])
            assert state['edge_training_updates'].tolist() == checked['edges'].tolist()
            assert result['individual_training_choices_per_agent'] == 614400 and result['joint_dyad_training_steps'] == 1228800
            for agent, optimizer in zip(final, state['optimizers']):
                group = optimizer['param_groups'][0]
                assert group['lr'] == .0003 and len(group['params']) == 14
                for identifier, (name, value) in zip(group['params'], agent.items()):
                    active = FLAGS[condition][name.split('.')[0]]
                    assert (identifier in optimizer['state']) == active
                    if active:
                        assert int(optimizer['state'][identifier]['step']) == 600
                        assert optimizer['state'][identifier]['exp_avg'].shape == value.shape
                        assert all(torch.isfinite(optimizer['state'][identifier][moment]).all()
                                   for moment in ('exp_avg', 'exp_avg_sq'))
            for filename, expected in read(path / 'artifact_hashes.json').items():
                assert sha(path / filename) == expected
            records.append({'seed': seed, 'condition': condition, 'updates': 600, 'frozen_parameters_exact_at_all_checkpoints': True,
                            'inactive_parameters_have_no_optimizer_state_or_gradient': True, 'same_private_inputs_and_all_rngs_verified': True,
                            'behavior_frozen_checkpoint_scores_exactly_constant': condition == 'behavior_frozen'})
            print(json.dumps({'execution_audited': path.name}), flush=True)
    output = {'status': 'passed', 'origins': origin_records, 'runs': records, 'new_training_runs': len(records) + len(origin_records),
              'reused_all_plastic_references': len(reused), 'reused_reference_sources_and_runtime_design_match': True,
              'mechanism_training_updates': len(records) * 600, 'origin_training_updates': len(origin_records) * 1200,
              'scope': 'Independent scene/photo/layout reconstruction, routing, budgets, freeze tensors, optimizer and RNG audits; communication replay is recorded separately.'}
    (folder / '机制训练独立核查.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=ROOT / 'results/mechanism_001')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.directory)
    print(json.dumps({'status': result['status'], 'new_training_runs': result['new_training_runs']}))
