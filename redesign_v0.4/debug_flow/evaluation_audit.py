"""Read-only pilot evaluator audit. Oracle results are engineering controls only."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from run_pilot import ImageBank, evaluate, make_agents, public_tensor
from resource_env import FEASIBLE_SCENES, one_step_coordination_bounds


class ImageLookupOracle:
    """Hand-written protocol, with exact image-feature -> metadata lookup.

    Grants perfect category recognition; never evidence of learned language.
    Only the two local features enter observe; partner state arrives by message.
    """
    def __init__(self, bank, identity):
        self.bank = bank
        self.identity = identity
        self.labels = torch.tensor([0 if e['category'] == 'food' else 1 for e in bank.entries])
        self.received_log = []

    def observe(self, own_features, public):
        shape = own_features.shape[:2]
        flat = own_features.reshape(-1, own_features.shape[-1])
        distances = torch.cdist(flat, self.bank.features)
        indices = distances.argmin(-1)
        # Inputs here were copied unchanged from the bank, so exact matching
        # verifies the image route without fitting a category classifier.
        assert torch.equal(flat, self.bank.features[indices])
        kinds = self.labels[indices].reshape(shape)
        return kinds, kinds.sum(1)

    def send(self, local):
        # 1=FF, 2=mixed, 3=WW. 0 denotes absent information.
        logits = torch.full((len(local), 5), -30.)
        logits[torch.arange(len(local)), local + 1] = 30.
        return logits

    def act(self, options, local, received):
        self.received_log.append(received.tolist())
        # Fixed identity breaks ties if both have both resources.
        preferred = torch.full((len(local),), self.identity, dtype=torch.int64)
        preferred[received == 1] = 1  # Partner has only food.
        preferred[received == 3] = 0  # Partner has only water.
        choice = (options == preferred[:, None]).to(torch.int64).argmax(-1)
        logits = torch.full((len(local), 2), -30.)
        logits[torch.arange(len(local)), choice] = 30.
        return logits


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_agents(path):
    agents = make_agents(1)
    states = torch.load(path, map_location='cpu', weights_only=True)
    for agent, state in zip(agents, states):
        agent.load_state_dict(state, strict=True)
        agent.eval()
    return agents, states


def metrics(result):
    return {k: result[k] for k in ('mean_reward_per_step', 'balanced_gathering', 'shortage_per_episode', 'greedy')}


def main():
    torch.set_num_threads(4)
    bank = ImageBank()
    pilot = ROOT / 'results/pilot_001'
    out = {'scope': 'Engineering process audit only; no social training or model adaptation.',
           'source_hash_matches': {}, 'exact_bounds': one_step_coordination_bounds()}
    archived_hashes = json.loads((pilot / 'source_hashes.json').read_text())
    for name in ('run_pilot.py', 'resource_env.py', 'agents.py'):
        out['source_hash_matches'][name] = digest(ROOT / name) == archived_hashes[name]
    out['feature_archive_equal'] = bool(np.array_equal(
        np.load(ROOT / 'data/features.npz')['features'],
        np.load(pilot / 'data_features.npz')['features']))
    oracle_results = {}
    for condition, mode in [('communicate', 'normal'), ('communicate', 'shuffle'), ('silent', 'normal')]:
        agents = [ImageLookupOracle(bank, i) for i in range(2)]
        result, trace = evaluate(agents, bank, condition, 812345, n=512, horizon=16, mode=mode, trace=True)
        route_correct = all(
            np.array_equal(np.asarray(agents[i].received_log[step]), np.asarray(row['delivered'])[:, 1-i])
            for i in range(2) for step, row in enumerate(trace))
        honest_send = all(np.array_equal(np.asarray(row['sent']), np.asarray(row['kinds']).sum(2) + 1) for row in trace)
        oracle_results[f'{condition}_{mode}'] = {**metrics(result), 'partner_delivery_verified': route_correct,
                                               'local_only_protocol_send_verified': honest_send}
        assert route_correct and honest_send
    assert oracle_results['communicate_normal']['balanced_gathering'] == 1.
    assert oracle_results['communicate_shuffle']['balanced_gathering'] < .95
    assert oracle_results['silent_normal']['balanced_gathering'] < .8
    out['handwritten_image_lookup_protocol'] = oracle_results
    out['handwritten_image_lookup_protocol_caveat'] = 'Category labels and message meanings explicitly supplied to an engineering oracle; these are not trained model results.'
    # Exhaust every feasible scene rather than relying solely on a random batch.
    exact_features, _ = bank.sample(FEASIBLE_SCENES, 'test', np.random.default_rng(929))
    exact_agents = [ImageLookupOracle(bank, i) for i in range(2)]
    exact_reps = [exact_agents[i].observe(exact_features[:, i], public_tensor(np.zeros((14, 2), dtype=np.int64), 1, 1)) for i in range(2)]
    exact_sent = [exact_agents[i].send(exact_reps[i][1]).argmax(-1) for i in range(2)]
    exact_actions = np.column_stack([exact_agents[i].act(*exact_reps[i], exact_sent[1-i]).argmax(-1).numpy() for i in range(2)])
    exact_selected = np.take_along_axis(FEASIBLE_SCENES, exact_actions[..., None], axis=-1)[..., 0]
    out['exhaustive_oracle'] = {'scenes': 14, 'balanced_scenes': int((exact_selected[:, 0] != exact_selected[:, 1]).sum())}
    out['checkpoint_audit'] = []
    out['recomputed_learning_curves'] = []
    out['independent_saved_trace_audit'] = []
    final_by_seed = {}
    trace_by_seed = {}
    for seed in (101, 202, 303):
        final_by_seed[seed] = {}
        trace_by_seed[seed] = {}
        for condition in ('communicate', 'silent'):
            path = pilot / f'{condition}_s{seed}'
            archived_curve = {r['update']: r for r in json.loads((path / 'learning_curve.json').read_text())}
            initial_agents, initial_states = load_agents(path / 'initial.pt')
            final_agents, final_states = load_agents(path / 'checkpoint_0600.pt')
            final_by_seed[seed][condition] = final_states
            diffs = []
            for index in range(2):
                diffs.append({'agent': index,
                              'changed_tensors': sum(not torch.equal(initial_states[index][k], final_states[index][k]) for k in initial_states[index]),
                              'total_tensors': len(initial_states[index]),
                              'parameter_l2_change': float(torch.cat([(final_states[index][k] - initial_states[index][k]).flatten() for k in initial_states[index]]).norm())})
            record = {'seed': seed, 'condition': condition, 'parameter_changes': diffs,
                      'separate_agent_storage': all(p0.data_ptr() != p1.data_ptr() for p0, p1 in zip(final_agents[0].parameters(), final_agents[1].parameters())),
                      'checkpoint_sha256': digest(path / 'checkpoint_0600.pt')}
            archived_final = json.loads((path / 'result.json').read_text())
            # Independently score original traces using only selected resource
            # categories, without calling transition or evaluate aggregators.
            raw_rows = [json.loads(line) for line in (path / 'heldout_trace.jsonl').read_text().splitlines()]
            all_kinds = np.asarray([row['kinds'] for row in raw_rows])
            all_actions = np.asarray([row['actions'] for row in raw_rows])
            selected = np.take_along_axis(all_kinds, all_actions[..., None], axis=-1)[..., 0]
            balanced = selected[..., 0] != selected[..., 1]
            recorded_reward = np.asarray([row['reward'] for row in raw_rows])
            assert np.array_equal(balanced.astype(float), recorded_reward)
            assert float(balanced.mean()) == archived_final['normal']['balanced_gathering']
            trace_by_seed[seed][condition] = (all_kinds, selected)
            # Opposite fixed resource preferences explain the observed data;
            # identity orientation can differ across training seeds.
            preference_matches = []
            for preferred_a in (0, 1):
                target = np.stack([
                    np.where((all_kinds[..., i, :] == pref).any(-1), pref, 1 - pref)
                    for i, pref in enumerate((preferred_a, 1 - preferred_a))
                ], axis=-1)
                preference_matches.append(float((selected == target).mean()))
            out['independent_saved_trace_audit'].append({
                'seed': seed, 'condition': condition, 'steps': int(balanced.size),
                'selected_resources_recomputed_success': float(balanced.mean()),
                'every_reward_equals_direct_balanced_indicator': True,
                'fixed_opposite_preference_best_match': max(preference_matches),
                'agent_0_preferred_resource': int(np.argmax(preference_matches)),
            })
            for mode, greedy, archive_key in [('normal', True, 'normal'), ('shuffle', True, 'shuffle'), ('normal', False, 'stochastic')]:
                result, _ = evaluate(final_agents, bank, condition, seed + 800000, n=512, horizon=16, mode=mode, greedy=greedy)
                record[f'{archive_key}_recomputed'] = metrics(result)
                record[f'{archive_key}_matches_saved'] = result['balanced_gathering'] == archived_final[archive_key]['balanced_gathering']
                assert record[f'{archive_key}_matches_saved']
            out['checkpoint_audit'].append(record)
            curve = []
            for update in range(0, 601, 100):
                agents, _ = load_agents(path / f'checkpoint_{update:04d}.pt')
                greedy_result, _ = evaluate(agents, bank, condition, seed + 700000, n=128, horizon=16, greedy=True)
                stochastic_result, _ = evaluate(agents, bank, condition, seed + 700000, n=128, horizon=16, greedy=False)
                row = {'update': update, 'greedy_success': greedy_result['balanced_gathering'],
                       'stochastic_success': stochastic_result['balanced_gathering'],
                       'greedy_matches_saved': greedy_result['balanced_gathering'] == archived_curve[update]['normal']['balanced_gathering']}
                assert row['greedy_matches_saved']
                curve.append(row)
            out['recomputed_learning_curves'].append({'seed': seed, 'condition': condition, 'curve': curve})
    out['paired_condition_checkpoint_independence'] = []
    for seed, conditions in final_by_seed.items():
        out['paired_condition_checkpoint_independence'].append({'seed': seed,
            'same_evaluation_scenes': bool(np.array_equal(trace_by_seed[seed]['communicate'][0], trace_by_seed[seed]['silent'][0])),
            'same_selected_resources': bool(np.array_equal(trace_by_seed[seed]['communicate'][1], trace_by_seed[seed]['silent'][1])),
            'agents': [
            {'agent': i, 'differing_tensors': sum(not torch.equal(conditions['communicate'][i][k], conditions['silent'][i][k]) for k in conditions['communicate'][i]),
             'parameter_l2_difference': float(torch.cat([(conditions['communicate'][i][k] - conditions['silent'][i][k]).flatten() for k in conditions['communicate'][i]]).norm())}
            for i in range(2)]})
    (Path(__file__).parent / 'evaluation_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'oracle': oracle_results, 'all_archived_results_reproduced': True,
                      'output': str(Path(__file__).parent / 'evaluation_audit.json')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
