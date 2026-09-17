"""Fixed-budget readaptation of saved partner conventions; no new auxiliaries."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import shutil
import time

import numpy as np
import torch

from run_pilot import ImageBank, write_json
from run_curriculum import private_public
from run_partners import (load_population, matching_schedule, assign_slots,
                          rollout, optimize, digest)
from resource_env import sample_scenes
from partner_eval import checkpoint_population, evaluate_population
from contact_eval import protocol_reference, protocol_drift, ORIGINAL_PAIRS, ALL_PAIRS

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'results/partners_001'
CONFIG = {
    'seeds': [101, 202, 303],
    'conditions': ['fixed_to_fixed', 'fixed_to_rotating', 'rotating_to_rotating'],
    'source_batch': 'partners_001', 'origin_update': 1200, 'additional_updates': 600,
    'batch_size_per_agent': 1024, 'population_size': 4, 'horizon': 16,
    'learning_rate': .0003, 'action_entropy_coefficient': 0., 'signalling_coefficient': 0.,
    'gamma': 0., 'capacity': 1, 'initial_inventory': [0, 0], 'task': 'full',
    'checkpoints': [0, 1, 5, 10, 20, 50, 100, 200, 400, 600],
    'checkpoint_evaluation_n': 512, 'evaluation_n': 4096,
    'intervention_n': 2048, 'alignment_n': 2048,
    'protocol_probe_n': 1024, 'protocol_support_n': 2048,
    'protocol_seed_offset': 900000, 'training_seed_offset': 2000,
    'checkpoint_evaluation_seed_offset': 700000, 'evaluation_seed_offset': 800000,
    'fresh_adam_all_conditions': True,
    'optimizer_limitation': 'Origin saved parameters only. All conditions restart Adam, not bitwise uninterrupted training.',
    'policy_sampling': 'Sample learned sender and action probabilities; no added entropy reward.',
    'partner_identity_input': False, 'signal_vocabulary_size': 5,
    'architecture': 'unchanged ResourceAgent, four independent 83527-parameter interfaces',
    'backbone': 'frozen official DINOv2 ViT-L/14, cached original-photo features',
    'training_device': 'cpu', 'torch': str(torch.__version__), 'python': platform.python_version(),
    'fixed_budget_no_success_based_stopping': True,
    'primary_comparison': 'fixed_to_rotating versus fixed_to_fixed; same fixed origin',
    'reference_condition': 'rotating_to_rotating has a different origin and prior partner exposure',
    'primary_metric': 'full-task normal success and normal-minus-shuffle across original cross four pairs',
    'replicate_unit': 'population seed; pairs, agents, directions and cases are not independent replicates',
    'engineering_edge_criteria': {'full_success': .85, 'shuffle_gap': .1, 'direction_shuffle_gap': .1},
    'source_data_paired': 'Same world/photo slots and layouts; changing assignment can change per-agent observations.',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def origin_condition(condition):
    if condition not in CONFIG['conditions']:
        raise ValueError('unknown contact condition')
    return 'rotating' if condition == 'rotating_to_rotating' else 'fixed'


def contact_schedule(seed, condition, config=CONFIG):
    origin_condition(condition)
    rotation = matching_schedule(seed + config['training_seed_offset'], config['additional_updates'])
    return np.zeros_like(rotation) if condition == 'fixed_to_fixed' else rotation


def initial_edges(condition, config=CONFIG):
    edges = np.zeros((4, 4), dtype=np.int64)
    pairs = ALL_PAIRS if origin_condition(condition) == 'rotating' else ORIGINAL_PAIRS
    count = config['origin_update'] // 3 if origin_condition(condition) == 'rotating' else config['origin_update']
    for i, j in pairs:
        edges[i, j] = edges[j, i] = count
    return edges


def seen_pairs(edges):
    """Pairs actually encountered, rather than all allowed future partners."""
    return tuple((i, j) for i, j in ALL_PAIRS if edges[i, j] > 0)


@torch.no_grad()
def mixed_diagnostics(agents, result, kinds, assignment):
    """Measure uncertainty only when the agent can choose different resources."""
    records = [None] * 4
    for pair_slot, pair in enumerate(assignment):
        for agent_slot, who in enumerate(pair):
            mask = torch.from_numpy(kinds[:, pair_slot, agent_slot].sum(-1) == 1)
            count = int(mask.sum())
            probability = agents[who].act(*result['representations'][who], result['received'][who]).softmax(-1)
            records[who] = {
                'n': count,
                'action_entropy': float(result['act_outputs'][who][2][mask].mean()) if count else None,
                'maximum_action_probability': float(probability[mask].max(-1).values.mean()) if count else None,
                'sender_entropy': float(result['send_outputs'][who][2][mask].mean()) if count else None,
            }
    return records


def validate_source(source=SOURCE):
    """Fail before training if the unchanged implementation/data no longer match."""
    expected = json.loads((source / 'source_hashes.json').read_text())
    actual = {name: sha(ROOT / name) for name in expected}
    if actual != expected:
        raise RuntimeError('Origin implementation hash mismatch')
    data = {}
    for name in ('manifest.json', 'features.npz', 'encoder_report.json'):
        data[name] = sha(ROOT / 'data' / name)
        if data[name] != sha(source / ('data_' + name)):
            raise RuntimeError('Origin data mismatch: ' + name)
    if json.loads((source / 'completed.json').read_text())['runs'] != 6:
        raise RuntimeError('Origin batch incomplete')
    return {'passed': True, 'source_hashes': actual, 'data_hashes': data}


def save_resume_state(path, agents, optimizers, completed, world_rng, layout_rng, policy_rngs, rotation, edges):
    # All elements support weights_only=True; NumPy states are primitive PCG64 dictionaries.
    torch.save({'agents': [a.state_dict() for a in agents],
                'optimizers': [o.state_dict() for o in optimizers],
                'additional_updates_completed': completed,
                'world_rng': world_rng.bit_generator.state,
                'layout_rng': layout_rng.bit_generator.state,
                'policy_rngs': [r.bit_generator.state for r in policy_rngs],
                'torch_rng_state': torch.get_rng_state(),
                'matching_schedule': torch.from_numpy(rotation.copy()),
                'edge_training_updates': torch.from_numpy(edges.copy())}, path)


def train(seed, condition, bank, out, config=CONFIG, source=SOURCE):
    origin = source / f'{origin_condition(condition)}_s{seed}'
    prepared = origin / f"checkpoint_{config['origin_update']:04d}.pt"
    origin_result = json.loads((origin / 'result.json').read_text())
    previous_eval = json.loads((origin / 'learning_curve.json').read_text())[-1]['evaluation']
    out.mkdir(exist_ok=False)
    shutil.copyfile(prepared, out / 'initial.pt')
    shutil.copyfile(origin / 'result.json', out / 'origin_result.json')
    agents = load_population(prepared)
    optimizers = [torch.optim.Adam(a.parameters(), lr=config['learning_rate']) for a in agents]
    if any(o.state for o in optimizers):
        raise RuntimeError('Expected fresh Adam states')
    total, n = config['additional_updates'], config['batch_size_per_agent']
    rng_seed = seed + config['training_seed_offset']
    rotation = contact_schedule(seed, condition, config)
    layout_rng = np.random.default_rng(rng_seed * 10000 + 302)
    world_rng = np.random.default_rng(rng_seed * 10000 + 101)
    policy_rngs = [np.random.default_rng(rng_seed * 10000 + 102 + i) for i in range(4)]
    public, _ = private_public(n, config['horizon'])
    edges = np.zeros((4, 4), dtype=np.int64)
    old_edges = initial_edges(condition, config)
    if old_edges.tolist() != origin_result['edge_training_updates']:
        raise RuntimeError('Origin exposure mismatch')
    old_pairs = ALL_PAIRS if origin_condition(condition) == 'rotating' else ORIGINAL_PAIRS
    reference = protocol_reference(agents, bank, seed + config['protocol_seed_offset'],
                                   n=config['protocol_probe_n'], trained_pairs=old_pairs,
                                   support_n=config['protocol_support_n'], horizon=config['horizon'])
    torch.save(reference, out / 'protocol_reference.pt')
    origin_metadata = {'seed': seed, 'condition': condition, 'origin_path': str(prepared),
                       'origin_sha256': sha(prepared), 'initial_sha256': sha(out / 'initial.pt'),
                       'origin_result_sha256': sha(origin / 'result.json'),
                       'origin_edge_training_updates': old_edges.tolist(),
                       'fresh_adam': True, 'initial_optimizer_state_entries': [len(o.state) for o in optimizers],
                       'protocol_reference_sha256': sha(out / 'protocol_reference.pt'),
                       'protocol_probe_sha256': reference['probe']['sha256'],
                       'protocol_support_probe_sha256': reference['support_probe']['sha256'],
                       'training_seed': rng_seed}
    write_json(out / 'origin.json', origin_metadata)
    curve = []
    start = time.monotonic()
    with (out / 'training_metrics.jsonl').open('x') as handle:
        for completed in range(total + 1):
            if completed in config['checkpoints']:
                if completed == 0:
                    shutil.copyfile(prepared, out / 'checkpoint_0000.pt')
                else:
                    torch.save([a.state_dict() for a in agents], out / f'checkpoint_{completed:04d}.pt')
                evaluation = checkpoint_population(agents, bank, seed + config['checkpoint_evaluation_seed_offset'],
                                                     n=config['checkpoint_evaluation_n'], horizon=config['horizon'])
                if completed == 0:
                    passed = evaluation == previous_eval
                    write_json(out / 'zero_update_validation.json', {
                        'passed': passed, 'comparison': 'Exact complete same-case checkpoint evaluation against origin update 1200',
                        'origin_learning_curve_sha256': sha(origin / 'learning_curve.json')})
                    if not passed:
                        raise RuntimeError('Zero-update evaluation differs from source endpoint')
                drift = protocol_drift(agents, reference, current_trained_pairs=seen_pairs(old_edges + edges))
                record = {'update': completed, 'lifetime_update': config['origin_update'] + completed,
                          'stage': 'contact_pure_reward', 'task': 'full', 'aux_weight': 0., 'action_entropy_weight': 0.,
                          'edge_training_updates': edges.tolist(),
                          'cumulative_edge_training_updates': (old_edges + edges).tolist(),
                          'evaluation': evaluation, 'protocol_drift': drift}
                curve.append(record)
                write_json(out / 'learning_curve.json', curve)
                group = evaluation['aggregates']
                print(json.dumps({'seed': seed, 'condition': condition, 'update': completed,
                                  **{g: group[g]['tasks']['full']['normal']['mean_reward_per_step'] for g in ('all', 'original', 'cross')}}), flush=True)
            if completed == total:
                break
            update = completed + 1
            matching_index = int(rotation[completed])
            layout_bits = layout_rng.integers(2, size=3)
            assignment = assign_slots(matching_index, layout_bits)
            kinds = sample_scenes(world_rng, 2 * n).reshape(n, 2, 2, 2)
            features, image_ids = bank.sample(kinds, 'train', world_rng)
            result = rollout(agents, features, kinds, assignment, public, policy_rngs)
            diagnostics = mixed_diagnostics(agents, result, kinds, assignment)
            agent_metrics = optimize(agents, optimizers, result, entropy_weight=0.)
            for pair_slot, pair in enumerate(assignment):
                edges[pair[0], pair[1]] += 1
                edges[pair[1], pair[0]] += 1
                for agent_slot, who in enumerate(pair):
                    agent_metrics[who]['private_input_sha256'] = digest(kinds[:, pair_slot, agent_slot], image_ids[:, pair_slot, agent_slot])
                    agent_metrics[who]['mixed_resource_diagnostics'] = diagnostics[who]
            metrics = {'update': update, 'lifetime_update': config['origin_update'] + update,
                       'stage': 'contact_pure_reward', 'task': 'full', 'aux_weight': 0., 'action_entropy_weight': 0.,
                       'matching_index': matching_index, 'layout_bits': layout_bits.tolist(),
                       'slot_assignment': assignment.tolist(), 'world_slots_sha256': digest(kinds, image_ids),
                       'sampled_training_success': float(np.mean([p['success'] for p in result['pair_outcomes']])),
                       'pair_outcomes': result['pair_outcomes'], 'agents': agent_metrics}
            handle.write(json.dumps(metrics, allow_nan=False) + '\n')
            if update % 100 == 0:
                handle.flush()
    save_resume_state(out / 'final_training_state.pt', agents, optimizers, total,
                      world_rng, layout_rng, policy_rngs, rotation, edges)
    final = {'seed': seed, 'condition': condition, 'updates': total,
             'lifetime_updates': config['origin_update'] + total,
             'individual_training_choices_per_agent': total * n, 'joint_dyad_training_steps': total * n * 2,
             'edge_training_updates': edges.tolist(), 'edge_joint_cases': (edges * n).tolist(),
             'cumulative_edge_training_updates': (edges + old_edges).tolist(),
             'origin': origin_metadata, 'final_aux_weight': 0., 'final_action_entropy_weight': 0.,
             'trainable_parameters_per_agent': [sum(p.numel() for p in a.parameters()) for a in agents],
             'protocol_drift': curve[-1]['protocol_drift'],
             'evaluation': evaluate_population(agents, bank, seed + config['evaluation_seed_offset'], n=config['evaluation_n'],
                                               trace_dir=out / 'traces', intervention_n=config['intervention_n'],
                                               alignment_n=config['alignment_n'], horizon=config['horizon'])}
    final['seconds'] = time.monotonic() - start
    write_json(out / 'result.json', final)
    write_json(out / 'artifact_hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='contact_001')
    args = parser.parse_args()
    torch.set_num_threads(4)
    validation = validate_source()
    out = ROOT / 'results' / args.name
    out.mkdir(exist_ok=False)
    write_json(out / 'source_validation.json', validation)
    config = copy.deepcopy(CONFIG)
    write_json(out / 'config.json', config)
    source = out / 'source'
    source.mkdir()
    for name in ('run_contact.py', 'contact_eval.py', 'run_partners.py', 'partner_eval.py',
                 'run_curriculum.py', 'curriculum_eval.py', 'run_pilot.py', 'agents.py', 'resource_env.py'):
        shutil.copyfile(ROOT / name, source / name)
    write_json(out / 'source_hashes.json', {p.name: sha(p) for p in source.iterdir()})
    shutil.copyfile(ROOT / '已有约定接触实验_执行方案.md', out / 'execution_plan.md')
    for name in ('manifest.json', 'features.npz', 'encoder_report.json'):
        shutil.copyfile(ROOT / 'data' / name, out / ('data_' + name))
    bank = ImageBank()
    np.savez_compressed(out / 'feature_scaling.npz', center=bank.center, scale=bank.scale)
    scaling = np.load(SOURCE / 'feature_scaling.npz')
    if not np.array_equal(bank.center, scaling['center']) or not np.array_equal(bank.scale, scaling['scale']):
        raise RuntimeError('Feature scaling differs from origin')
    results = []
    start = time.monotonic()
    for seed in config['seeds']:
        for condition in config['conditions']:
            results.append(train(seed, condition, bank, out / f'{condition}_s{seed}', config))
            write_json(out / 'results.json', results)
    write_json(out / 'completed.json', {'status': 'completed', 'runs': len(results), 'seconds': time.monotonic() - start})
    write_json(out / 'batch_artifact_hashes.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})


if __name__ == '__main__':
    main()
