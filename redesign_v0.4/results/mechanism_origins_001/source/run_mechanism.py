"""Predeclared sender/listener plasticity factorial; old implementations unchanged."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import shutil
import time

import numpy as np
import torch

import run_contact as contact
import run_partners as partners
from run_pilot import ImageBank, individual_practice, make_agents, write_json
from contact_eval import protocol_reference, protocol_drift, ORIGINAL_PAIRS
from mechanism_bounds import frozen_function_bounds


ROOT = Path(__file__).resolve().parent
EXPLORATORY = [101, 202, 303]
CONFIRMATORY = [404, 505, 606, 707, 808, 909, 1001]
FACTORIAL = ['both_learn', 'sender_only', 'listener_only', 'behavior_frozen']
FLAGS = {
    'both_learn': {'project': False, 'sender': True, 'actor': True, 'value': True},
    'sender_only': {'project': False, 'sender': True, 'actor': False, 'value': True},
    'listener_only': {'project': False, 'sender': False, 'actor': True, 'value': True},
    'behavior_frozen': {'project': False, 'sender': False, 'actor': False, 'value': True},
    'all_plastic': {'project': True, 'sender': True, 'actor': True, 'value': True},
}
CONFIG = {**copy.deepcopy(contact.CONFIG),
          'seeds': EXPLORATORY + CONFIRMATORY, 'exploratory_seeds': EXPLORATORY,
          'confirmatory_seeds': CONFIRMATORY, 'conditions': FACTORIAL,
          'reference_condition': 'all_plastic', 'trainability_by_condition': FLAGS,
          'source_batch': 'partners_001 and mechanism_origins_001, fixed origins only',
          'primary_comparison': 'Fixed-project 2x2 sender and conditional action-function plasticity; seven predeclared confirmatory populations reported separately.',
          'reference_comparison': 'all_plastic versus both_learn changes permission to update the shared visual projection.',
          'source_data_paired': 'All mechanism conditions share exact partner schedule, world/photo slots, layout, and private inputs.',
          'scope': 'Checkpoint contact mechanisms, not sufficient non-language abilities or the origin of human language.',
          'reuse_original_all_plastic': 'contact_001 fixed_to_rotating seeds 101,202,303; independently verified same-stream implementation.',
          'fresh_adam_all_conditions': True}
SOURCE_FILES = ('run_mechanism.py', 'mechanism_bounds.py', 'run_contact.py', 'contact_eval.py',
                'run_partners.py', 'partner_eval.py', 'run_curriculum.py', 'curriculum_eval.py',
                'run_pilot.py', 'agents.py', 'resource_env.py')


def cohort(seed):
    if seed in EXPLORATORY:
        return 'exploratory'
    if seed in CONFIRMATORY:
        return 'confirmatory'
    raise ValueError('Undeclared population seed')


def configure(agents, condition):
    if condition not in FLAGS:
        raise ValueError('Unknown plasticity condition')
    for agent in agents:
        for name, enabled in FLAGS[condition].items():
            getattr(agent, name).requires_grad_(enabled)
    partners.assert_independent(agents)
    return [torch.optim.Adam(a.parameters(), lr=CONFIG['learning_rate']) for a in agents]


def module_digests(agents):
    result = []
    for agent in agents:
        modules = {}
        for name in FLAGS['all_plastic']:
            h = hashlib.sha256()
            for key, value in getattr(agent, name).state_dict().items():
                h.update(key.encode())
                h.update(value.detach().cpu().numpy().tobytes())
            modules[name] = h.hexdigest()
        result.append(modules)
    return result


def assert_frozen(agents, condition, initial, drift=None):
    current = module_digests(agents)
    for who, agent in enumerate(agents):
        for name, enabled in FLAGS[condition].items():
            assert all(p.requires_grad == enabled for p in getattr(agent, name).parameters())
            if not enabled:
                assert current[who][name] == initial[who][name], f'Frozen {name} changed'
                assert all(p.grad is None for p in getattr(agent, name).parameters())
    if drift is not None and not FLAGS[condition]['project']:
        if not FLAGS[condition]['sender']:
            assert all(s['overall']['mean_probability_total_variation'] == 0 for s in drift['senders'])
        if not FLAGS[condition]['actor']:
            assert all(symbol['mean_action_probability_total_variation'] == 0
                       for receiver in drift['listeners'] for context in receiver['by_own_context'].values()
                       for symbol in context['all_five_symbols'])
    return {'passed': True, 'module_sha256': current,
            'frozen_modules': [name for name, enabled in FLAGS[condition].items() if not enabled]}


def save_state(path, agents, optimizers, condition, completed, world, layout, policies, rotation, edges):
    contact.save_resume_state(path, agents, optimizers, completed, world, layout, policies, rotation, edges)
    saved = torch.load(path, weights_only=True)
    saved['condition'] = condition
    saved['trainability'] = FLAGS[condition]
    saved['optimizer_parameter_names'] = [[name for name, _ in agent.named_parameters()] for agent in agents]
    torch.save(saved, path)


def restore_state(path):
    from agents import ResourceAgent
    saved = torch.load(path, map_location='cpu', weights_only=True)
    agents = [ResourceAgent() for _ in range(4)]
    for agent, state in zip(agents, saved['agents']):
        agent.load_state_dict(state, strict=True)
    assert saved['trainability'] == FLAGS[saved['condition']]
    optimizers = configure(agents, saved['condition'])
    for optimizer, state in zip(optimizers, saved['optimizers']):
        optimizer.load_state_dict(state)
    def rng(state):
        generator = np.random.default_rng()
        generator.bit_generator.state = copy.deepcopy(state)
        return generator
    return agents, optimizers, rng(saved['world_rng']), rng(saved['layout_rng']), [rng(s) for s in saved['policy_rngs']], saved


def snapshot(out, config):
    out.mkdir(exist_ok=False)
    write_json(out / 'config.json', config)
    (out / 'source').mkdir()
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / name, out / 'source' / name)
    write_json(out / 'source_hashes.json', {name: contact.sha(out / 'source' / name) for name in SOURCE_FILES})
    for name in ('manifest.json', 'features.npz', 'encoder_report.json'):
        shutil.copyfile(ROOT / 'data' / name, out / ('data_' + name))
    plan = ROOT / '完整研究_执行方案.md'
    if not plan.exists():
        raise RuntimeError('The complete research execution plan must exist before a formal run')
    shutil.copyfile(plan, out / 'execution_plan.md')
    bank = ImageBank()
    np.savez_compressed(out / 'feature_scaling.npz', center=bank.center, scale=bank.scale)
    write_json(out / 'source_validation.json', contact.validate_source())
    return bank


def prepare_new_population(seed, bank, out, config):
    agents, practices = [], []
    for source_seed in (seed, seed + config['preparation_seed_offset']):
        pair = make_agents(source_seed)
        before = [copy.deepcopy(a.sender.state_dict()) for a in pair]
        report = individual_practice(pair, bank, source_seed, updates=config['preparation_updates'],
                                     batch=config['preparation_batch_size'])
        for i, entry in enumerate(report):
            entry['agent'] = len(agents) + i
            entry['initialization_and_practice_seed'] = source_seed
            entry['sender_head_unchanged_during_preparation'] = all(
                torch.equal(value, before[i][key]) for key, value in pair[i].sender.state_dict().items())
            assert entry['sender_head_unchanged_during_preparation']
        agents.extend(pair)
        practices.extend(report)
    partners.assert_independent(agents)
    path = out / f'prepared_s{seed}.pt'
    torch.save([a.state_dict() for a in agents], path)
    result = {'seed': seed, 'cohort': cohort(seed), 'agents': practices, 'reused_agents': [],
              'source_seeds': [seed, seed + config['preparation_seed_offset']],
              'prepared_sha256': contact.sha(path), 'gate': config['preparation_accuracy_gate'],
              'passed': all(r['heldout_need_sensitive_choice'] >= config['preparation_accuracy_gate'] for r in practices),
              'note': 'Only sender head is untrained during individual practice; its shared input projection can change.'}
    write_json(out / f'preparation_s{seed}.json', result)
    return path, result


def create_origins(name='mechanism_origins_001'):
    config = copy.deepcopy(partners.CONFIG)
    config.update(seeds=CONFIRMATORY, conditions=['fixed'],
                  preparation='All four agents newly initialized; no seed selection or retries.',
                  source_course_batch='Original 600 course + 300 full exploration + 300 full pure-reward schedule.')
    out = ROOT / 'results' / name
    bank = snapshot(out, config)
    start, paths, reports = time.monotonic(), {}, []
    # Prepare all declared populations before starting any social training.
    for seed in config['seeds']:
        path, report = prepare_new_population(seed, bank, out, config)
        paths[seed] = path
        reports.append(report)
        write_json(out / 'preparations.json', reports)
    if not all(r['passed'] for r in reports):
        write_json(out / 'blocked.json', {'reason': 'At least one predeclared preparation failed; no social training started.',
                                         'failed_seeds': [r['seed'] for r in reports if not r['passed']]})
        raise RuntimeError('Preparation gate failed; all declared outcomes retained')
    results = []
    for seed in config['seeds']:
        result = partners.train(seed, 'fixed', bank, paths[seed], out / f'fixed_s{seed}', config)
        result['cohort'] = cohort(seed)
        write_json(out / f'fixed_s{seed}' / 'result.json', result)
        results.append(result)
        write_json(out / 'results.json', results)
    write_json(out / 'completed.json', {'status': 'completed', 'runs': len(results), 'seconds': time.monotonic() - start})
    write_json(out / 'batch_artifact_hashes.json', {p.name: contact.sha(p) for p in out.iterdir() if p.is_file()})


def origin_folder(seed, origins_name='mechanism_origins_001'):
    batch = 'partners_001' if seed in EXPLORATORY else origins_name
    cohort(seed)
    return ROOT / 'results' / batch / f'fixed_s{seed}'


def train(seed, condition, bank, out, config=CONFIG, origin=None):
    origin = Path(origin) if origin is not None else origin_folder(seed)
    prepared = origin / 'checkpoint_1200.pt'
    original = json.loads((origin / 'result.json').read_text())
    previous_eval = json.loads((origin / 'learning_curve.json').read_text())[-1]['evaluation']
    out.mkdir(exist_ok=False)
    shutil.copyfile(prepared, out / 'initial.pt')
    shutil.copyfile(origin / 'result.json', out / 'origin_result.json')
    agents = partners.load_population(prepared)
    optimizers = configure(agents, condition)
    assert all(not optimizer.state for optimizer in optimizers)
    initial = module_digests(agents)
    total, n = config['additional_updates'], config['batch_size_per_agent']
    rng_seed = seed + config['training_seed_offset']
    rotation = partners.matching_schedule(rng_seed, total)
    layout = np.random.default_rng(rng_seed * 10000 + 302)
    world = np.random.default_rng(rng_seed * 10000 + 101)
    policies = [np.random.default_rng(rng_seed * 10000 + 102 + i) for i in range(4)]
    public, _ = contact.private_public(n, config['horizon'])
    edges = np.zeros((4, 4), dtype=np.int64)
    old_edges = np.asarray(original['edge_training_updates'], dtype=np.int64)
    assert np.array_equal(old_edges, contact.initial_edges('fixed_to_fixed', config))
    reference = protocol_reference(agents, bank, seed + config['protocol_seed_offset'], n=config['protocol_probe_n'],
                                   trained_pairs=ORIGINAL_PAIRS, support_n=config['protocol_support_n'], horizon=config['horizon'])
    torch.save(reference, out / 'protocol_reference.pt')
    metadata = {'seed': seed, 'cohort': cohort(seed), 'condition': condition, 'origin_path': str(prepared),
                'origin_sha256': contact.sha(prepared), 'initial_sha256': contact.sha(out / 'initial.pt'),
                'origin_result_sha256': contact.sha(origin / 'result.json'),
                'origin_edge_training_updates': old_edges.tolist(), 'fresh_adam': True,
                'initial_optimizer_state_entries': [len(o.state) for o in optimizers],
                'protocol_reference_sha256': contact.sha(out / 'protocol_reference.pt'),
                'protocol_probe_sha256': reference['probe']['sha256'],
                'protocol_support_probe_sha256': reference['support_probe']['sha256'],
                'training_seed': rng_seed, 'trainability': FLAGS[condition], 'initial_module_sha256': initial}
    write_json(out / 'origin.json', metadata)
    curve, start = [], time.monotonic()
    with (out / 'training_metrics.jsonl').open('x') as handle:
        for completed in range(total + 1):
            if completed in config['checkpoints']:
                if completed == 0:
                    shutil.copyfile(prepared, out / 'checkpoint_0000.pt')
                else:
                    torch.save([a.state_dict() for a in agents], out / f'checkpoint_{completed:04d}.pt')
                evaluation = contact.checkpoint_population(agents, bank, seed + config['checkpoint_evaluation_seed_offset'],
                                                           n=config['checkpoint_evaluation_n'], horizon=config['horizon'])
                if completed == 0:
                    passed = evaluation == previous_eval
                    write_json(out / 'zero_update_validation.json', {'passed': passed,
                        'origin_learning_curve_sha256': contact.sha(origin / 'learning_curve.json')})
                    assert passed, 'Zero-update evaluation differs from origin'
                drift = protocol_drift(agents, reference, current_trained_pairs=contact.seen_pairs(old_edges + edges))
                frozen = assert_frozen(agents, condition, initial, drift)
                row = {'update': completed, 'lifetime_update': 1200 + completed, 'stage': 'mechanism_pure_reward',
                       'task': 'full', 'aux_weight': 0., 'action_entropy_weight': 0.,
                       'edge_training_updates': edges.tolist(), 'cumulative_edge_training_updates': (old_edges + edges).tolist(),
                       'evaluation': evaluation, 'protocol_drift': drift, 'freeze_verification': frozen}
                curve.append(row)
                write_json(out / 'learning_curve.json', curve)
                print(json.dumps({'seed': seed, 'condition': condition, 'update': completed,
                                  **{g: evaluation['aggregates'][g]['tasks']['full']['normal']['mean_reward_per_step']
                                     for g in ('all', 'original', 'cross')}}), flush=True)
            if completed == total:
                break
            update, matching = completed + 1, int(rotation[completed])
            bits = layout.integers(2, size=3)
            assignment = partners.assign_slots(matching, bits)
            kinds = contact.sample_scenes(world, 2 * n).reshape(n, 2, 2, 2)
            features, ids = bank.sample(kinds, 'train', world)
            result = partners.rollout(agents, features, kinds, assignment, public, policies)
            diagnostics = contact.mixed_diagnostics(agents, result, kinds, assignment)
            metrics = partners.optimize(agents, optimizers, result, entropy_weight=0.)
            for slot, pair in enumerate(assignment):
                edges[pair[0], pair[1]] += 1
                edges[pair[1], pair[0]] += 1
                for local, who in enumerate(pair):
                    metrics[who]['private_input_sha256'] = partners.digest(kinds[:, slot, local], ids[:, slot, local])
                    metrics[who]['mixed_resource_diagnostics'] = diagnostics[who]
                    assert all(metrics[who]['gradient_norms'][name] == 0 for name, active in FLAGS[condition].items() if not active)
            row = {'update': update, 'lifetime_update': 1200 + update, 'stage': 'mechanism_pure_reward', 'task': 'full',
                   'aux_weight': 0., 'action_entropy_weight': 0., 'matching_index': matching, 'layout_bits': bits.tolist(),
                   'slot_assignment': assignment.tolist(), 'world_slots_sha256': partners.digest(kinds, ids),
                   'sampled_training_success': float(np.mean([p['success'] for p in result['pair_outcomes']])),
                   'pair_outcomes': result['pair_outcomes'], 'agents': metrics}
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            if update % 100 == 0:
                handle.flush()
    save_state(out / 'final_training_state.pt', agents, optimizers, condition, total, world, layout, policies, rotation, edges)
    final = {'seed': seed, 'cohort': cohort(seed), 'condition': condition, 'updates': total, 'lifetime_updates': 1200 + total,
             'individual_training_choices_per_agent': total * n, 'joint_dyad_training_steps': total * n * 2,
             'edge_training_updates': edges.tolist(), 'edge_joint_cases': (edges * n).tolist(),
             'cumulative_edge_training_updates': (old_edges + edges).tolist(), 'origin': metadata,
             'final_aux_weight': 0., 'final_action_entropy_weight': 0., 'trainability': FLAGS[condition],
             'trainable_parameters_per_agent': [sum(p.numel() for p in a.parameters() if p.requires_grad) for a in agents],
             'policy_trainable_parameters_per_agent': [sum(p.numel() for name in ('project', 'sender', 'actor')
                                                          for p in getattr(a, name).parameters() if p.requires_grad) for a in agents],
             'protocol_drift': curve[-1]['protocol_drift'], 'freeze_verification': assert_frozen(agents, condition, initial),
             'evaluation': contact.evaluate_population(agents, bank, seed + config['evaluation_seed_offset'], n=config['evaluation_n'],
                    trace_dir=out / 'traces', intervention_n=config['intervention_n'], alignment_n=config['alignment_n'], horizon=config['horizon'])}
    final['seconds'] = time.monotonic() - start
    write_json(out / 'result.json', final)
    write_json(out / 'artifact_hashes.json', {p.name: contact.sha(p) for p in out.iterdir() if p.is_file()})
    return final


def run_batch(name='mechanism_001', origins_name='mechanism_origins_001'):
    out = ROOT / 'results' / name
    config = copy.deepcopy(CONFIG)
    config['origins_batch'] = origins_name
    bank = snapshot(out, config)
    results, refs, start = [], [], time.monotonic()
    for seed in config['seeds']:
        origin = origin_folder(seed, origins_name)
        agents = partners.load_population(origin / 'checkpoint_1200.pt')
        bound = frozen_function_bounds(agents, bank, horizon=config['horizon'])
        bound.update(seed=seed, cohort=cohort(seed), origin_sha256=contact.sha(origin / 'checkpoint_1200.pt'))
        write_json(out / f'frozen_function_bounds_s{seed}.json', bound)
        for condition in FACTORIAL + ([] if seed in EXPLORATORY else ['all_plastic']):
            results.append(train(seed, condition, bank, out / f'{condition}_s{seed}', config, origin))
            write_json(out / 'results.json', results)
        if seed in EXPLORATORY:
            folder = ROOT / 'results/contact_001' / f'fixed_to_rotating_s{seed}'
            ref = json.loads((folder / 'result.json').read_text())
            assert ref['origin']['origin_sha256'] == contact.sha(origin / 'checkpoint_1200.pt')
            refs.append({'seed': seed, 'cohort': cohort(seed), 'condition': 'all_plastic', 'reused': True,
                         'path': str(folder), 'result_sha256': contact.sha(folder / 'result.json'),
                         'checkpoint_sha256': contact.sha(folder / 'checkpoint_0600.pt'),
                         'verification': 'Same origin, budget, optimizer reset, source, and RNG streams; small-run exact-replay test saved before formal execution.'})
        else:
            refs.append({'seed': seed, 'cohort': cohort(seed), 'condition': 'all_plastic', 'reused': False,
                         'path': str(out / f'all_plastic_s{seed}')})
        write_json(out / 'all_plastic_references.json', refs)
    write_json(out / 'completed.json', {'status': 'completed', 'runs': len(results), 'reused_references': 3,
                                      'populations': len(config['seeds']), 'seconds': time.monotonic() - start})
    write_json(out / 'batch_artifact_hashes.json', {p.name: contact.sha(p) for p in out.iterdir() if p.is_file()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['origins', 'mechanism'])
    parser.add_argument('--name')
    parser.add_argument('--origins-name', default='mechanism_origins_001')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == 'origins':
        create_origins(args.name or 'mechanism_origins_001')
    else:
        run_batch(args.name or 'mechanism_001', args.origins_name)


if __name__ == '__main__':
    main()
