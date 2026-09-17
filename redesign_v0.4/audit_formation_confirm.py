"""Read-only independent RNG, inference and scoring audit of formation runs.

Does not call run_formation_confirm.train or curriculum_eval.evaluate. Reuses the
previous independent auditor's saved-data reader and float32 categorical sampler,
and the archived network class solely to execute frozen saved parameters.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from audit_partner_communication import (SavedBank, FULL_SCENES, COURSE_SCENES,
                                         replay_draw, audit_intervention)

ROOT = Path(__file__).resolve().parent
MODES = ('normal', 'shuffle', 'blank', 'stochastic', 'stochastic_blank')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(actual, expected, context):
    assert np.allclose(actual, expected, rtol=0, atol=1e-7), context


def state_equal(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key]), key


@torch.no_grad()
def replay(agents, case, seed):
    features, public = case['features'], case['public']
    reps = [a.observe(features[:, i], public) for i, a in enumerate(agents)]
    logits = [a.send(reps[i][1]) for i, a in enumerate(agents)]
    results = {}
    for mode in MODES:
        greedy = mode not in ('stochastic', 'stochastic_blank')
        rngs = [np.random.default_rng(seed + 11001 + i) for i in range(2)]
        sent = np.column_stack([replay_draw(logits[i], rngs[i], greedy).numpy() for i in range(2)])
        delivered = sent.copy()
        if mode in ('blank', 'stochastic_blank'):
            delivered[:] = 0
        elif mode == 'shuffle':
            rng = np.random.default_rng(seed + 22001)
            for time in np.unique(case['remaining']):
                indices = np.flatnonzero(case['remaining'] == time)
                for sender in range(2):
                    delivered[indices, sender] = sent[rng.permutation(indices), sender]
        acts = np.column_stack([replay_draw(agents[i].act(*reps[i], torch.from_numpy(delivered[:, 1-i])),
                                           rngs[i], greedy).numpy() for i in range(2)])
        selected = np.take_along_axis(case['kinds'], acts[..., None], axis=-1)[..., 0]
        success = selected[:, 0] != selected[:, 1]
        results[mode] = {'sent': sent, 'delivered': delivered, 'actions': acts,
                         'selected_kinds': selected, 'success': success}
    return results


def audit_stats(saved, actual, case):
    assert saved['external_cases_sha256'] == case['hash']
    same(saved['mean_reward_per_step'], actual['success'].mean(), 'aggregate success')
    same(saved['balanced_gathering'], actual['success'].mean(), 'balanced score')
    same(saved['shortage_per_step'], 1-actual['success'].mean(), 'shortage')
    same(saved['overflow_per_step'], 1-actual['success'].mean(), 'overflow')
    same_resource = case['kinds'][..., 0] == case['kinds'][..., 1]
    for d in saved['by_direction']:
        sender = d['restricted_sender']
        mask = same_resource[:, sender] & ~same_resource[:, 1-sender]
        assert int(mask.sum()) == d['n']
        same(d['success'], actual['success'][mask].mean(), 'direction score')
        for r in d['by_restricted_resource']:
            sub = mask & (case['kinds'][:, sender, 0] == r['resource'])
            assert int(sub.sum()) == r['n']
            same(r['success'], actual['success'][sub].mean(), 'direction/resource score')
    table = np.zeros((2, 3, 5), np.int64)
    for i in range(2):
        np.add.at(table[i], (case['kinds'][:, i].sum(1), actual['sent'][:, i]), 1)
    np.testing.assert_array_equal(table, saved['symbols_by_local_resource_set'])


def audit_native(record, condition):
    for task in ('curriculum', 'full'):
        native = record['native_task'][task]
        gkey = 'blank' if condition == 'course_blocked' else 'normal'
        skey = 'stochastic_blank' if condition == 'course_blocked' else 'stochastic'
        assert native['greedy_diagnostic_key'] == gkey and native['stochastic_diagnostic_key'] == skey
        for key, mode in (('greedy_success', gkey), ('stochastic_success', skey)):
            expected = 1. if condition == 'course_substitutable' else record[task][mode]['mean_reward_per_step']
            same(native[key], expected, f'native {condition}/{task}/{key}')


def reconstruct_training(folder, config, bank):
    output = []
    pools = {k: np.flatnonzero((bank.splits == 'train') & (bank.labels == k)) for k in (0, 1)}
    n = config['batch_size']
    zero_hash = hashlib.sha256(np.zeros(n, np.int64).tobytes()).hexdigest()
    for seed in config['seeds']:
        rows = {c: [json.loads(line) for line in (folder / f'{c}_s{seed}/training_metrics.jsonl').read_text().splitlines()]
                for c in config['conditions']}
        for condition, metrics in rows.items():
            assert [r['update'] for r in metrics] == list(range(1, config['updates']+1))
            world = np.random.default_rng(seed*10000+101)
            policies = [np.random.default_rng(seed*10000+102+i) for i in range(2)]
            for update, row in enumerate(metrics, 1):
                course = condition != 'direct_communication' and update <= config['course_updates']
                scenes = COURSE_SCENES if course else FULL_SCENES
                kinds = scenes[world.integers(len(scenes), size=n)].copy()
                ids = np.empty(kinds.shape, np.int64)
                for k in (0, 1):
                    mask = kinds == k
                    ids[mask] = world.choice(pools[k], int(mask.sum()))
                expected = hashlib.sha256(kinds.tobytes()+ids.tobytes()).hexdigest()
                assert row['world_input_sha256'] == expected
                assert row['world_rng_after'] == world.bit_generator.state
                assert row['task'] == ('curriculum' if course else 'full')
                assert row['action_entropy_weight'] == (.05 if update <= 900 else 0)
                assert row['aux_weight'] == 0
                if condition == 'course_substitutable':
                    assert row['native_training_success'] == 1
                for i in range(2):
                    policies[i].random((n, 1))
                    policies[i].random((n, 1))
                    a = row['agents'][i]
                    assert a['policy_rng_after'] == policies[i].bit_generator.state
                    if condition == 'course_blocked':
                        assert a['received_sha256'] == zero_hash and a['received_nonzero'] == 0
                    else:
                        assert a['received_sha256'] == row['agents'][1-i]['sent_sha256']
            final_state = torch.load(folder / f'{condition}_s{seed}/training_state_final.pt', weights_only=True)
            assert final_state['completed_updates'] == config['updates']
            assert final_state['world_rng'] == world.bit_generator.state
            assert final_state['policy_rng'] == [p.bit_generator.state for p in policies]
            for opt in final_state['optimizers']:
                assert all(int(value['step']) == config['updates'] for value in opt['state'].values())
        course = rows['course_communication']
        for c in ('course_blocked', 'course_substitutable'):
            assert [r['world_input_sha256'] for r in course] == [r['world_input_sha256'] for r in rows[c]]
        assert [r['world_input_sha256'] for r in course[600:]] == [r['world_input_sha256'] for r in rows['direct_communication'][600:]]
        output.append({'seed': seed, 'updates_per_condition': config['updates'],
                       'prepared_weight_and_rng_pairing': True, 'blocked_received_zero_every_update': True,
                       'substitutable_native_reward_one_every_update': True,
                       'course_controls_worlds_identical': True, 'direct_worlds_identical_after_600': True})
    return output


@torch.no_grad()
def audit(folder):
    config = read(folder/'config.json')
    completed = read(folder/'completed.json')
    assert completed['runs'] == len(config['seeds'])*len(config['conditions']) and completed['status'] == 'completed'
    for name, expected in read(folder/'source_hashes.json').items():
        assert sha(folder/'source'/name) == expected
    for name in ('manifest.json', 'features.npz', 'encoder_report.json'):
        assert sha(folder/('data_'+name)) == sha(ROOT/'data'/name)
    bank = SavedBank(folder)
    overlap_source = ROOT / 'results/partners_001/prepared_s202.pt'
    overlap = torch.load(folder/'prepared_s1202.pt', weights_only=True)
    previous = torch.load(overlap_source, weights_only=True)
    state_equal(overlap, previous[2:])
    module_spec = importlib.util.spec_from_file_location('formation_archived_agents', folder/'source/agents.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    training = reconstruct_training(folder, config, bank)
    outputs, case_count, checkpoint_case_count = [], 0, 0
    preparation = read(folder/'preparation.json')
    for seed in config['seeds']:
        prepared = torch.load(folder/f'prepared_s{seed}.pt', weights_only=True)
        assert all(p['heldout_need_sensitive_choice'] >= config['practice_min_accuracy'] for p in preparation[str(seed)]['practice'])
        assert preparation[str(seed)]['sender_head_unchanged']
        for condition in config['conditions']:
            run = folder/f'{condition}_s{seed}'
            final, curve = read(run/'result.json'), read(run/'learning_curve.json')
            state_equal(prepared, torch.load(run/'initial.pt', weights_only=True))
            assert [r['update'] for r in curve] == config['checkpoints']
            agents = [module.ResourceAgent(), module.ResourceAgent()]
            for checkpoint in curve:
                states = torch.load(run/f"checkpoint_{checkpoint['update']:04d}.pt", weights_only=True)
                for a, state in zip(agents, states):
                    a.load_state_dict(state, strict=True)
                for task in ('curriculum', 'full'):
                    case = bank.cases(seed+700000, config['checkpoint_evaluation_n'], task, config['horizon'])
                    actual = replay(agents, case, seed+700000)
                    for mode in MODES:
                        audit_stats(checkpoint[task][mode], actual[mode], case)
                        checkpoint_case_count += config['checkpoint_evaluation_n']
                audit_native(checkpoint, condition)
            state_equal([a.state_dict() for a in agents], torch.load(run/'training_state_final.pt', weights_only=True)['agents'])
            for task in ('curriculum', 'full'):
                case = bank.cases(seed+800000, config['evaluation_n'], task, config['horizon'])
                actual = replay(agents, case, seed+800000)
                for mode in MODES:
                    audit_stats(final[task][mode], actual[mode], case)
                    trace_name = f'final_{task}_{mode}_trace.jsonl'
                    assert sha(run/trace_name) == read(run/'trace_hashes.json')[trace_name]
                    rows = [json.loads(line) for line in (run/trace_name).read_text().splitlines()]
                    assert len(rows) == config['evaluation_n']
                    for i, row in enumerate(rows):
                        assert row['case'] == i
                        assert row['remaining'] == int(case['remaining'][i])
                        for key in ('kinds', 'image_ids', 'inventory'):
                            assert row[key] == case[key][i].tolist()
                        for key in ('sent', 'delivered', 'actions', 'selected_kinds'):
                            assert row[key] == actual[mode][key][i].tolist()
                        assert row['success'] == bool(actual[mode]['success'][i])
                        assert row['reward'] == float(actual[mode]['success'][i])
                        assert row['native_reward'] == (1. if condition == 'course_substitutable' else row['reward'])
                        assert row['native_channel'] == (condition != 'course_blocked' or mode in ('blank', 'stochastic_blank'))
                    case_count += len(rows)
            intervention = copy.deepcopy(final['intervention'])
            for d in intervention['directions']:
                d['population_sender'] = d['sender']
                d['population_receiver'] = d['receiver']
            audit_intervention(agents, (0, 1), bank, seed+900000, intervention, config['intervention_n'], config['horizon'])
            audit_native(final, condition)
            outputs.append({'seed': seed, 'condition': condition, 'checkpoints_replayed': len(curve),
                            'final_traces_replayed': 10, 'both_direction_all_five_symbol_interventions_replayed': True})
            print(json.dumps(outputs[-1]), flush=True)
    report = {'status': 'passed', 'runs': outputs, 'training': training,
              'final_cases_replayed': case_count, 'checkpoint_cases_replayed': checkpoint_case_count,
              'checkpoint_count': len(outputs)*len(config['checkpoints']),
              'source_and_data_hashes_verified': True,
              'preparation_overlap_verified': {'seed': 1202, 'previous_agents': [2, 3],
                                                'source': str(overlap_source), 'source_sha256': sha(overlap_source),
                                                'scope': 'non-social preparation only; no prior social checkpoint loaded'},
              'method': 'archived neural policy + separately reconstructed cases, float32 sampling, delivery and resource scoring; no training or evaluation runner called'}
    audit_source = folder / 'audit_source'
    audit_source.mkdir(exist_ok=True)
    for name in ('audit_formation_confirm.py', 'audit_partner_communication.py'):
        shutil.copyfile(ROOT / name, audit_source / name)
    report['audit_source_sha256'] = {p.name: sha(p) for p in audit_source.iterdir() if p.suffix == '.py'}
    (folder/'formation_confirm_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    text = (f'# 形成确认批次只读核查\n\n{len(outputs)}组全部通过。'
            f'重新构造{len(outputs)*config["updates"]:,}条训练更新的世界、照片和随机流；'
            f'独立重放{case_count:,}个终点案例和{checkpoint_case_count:,}个检查点评估案例，'
            f'共{len(outputs)*len(config["checkpoints"])}份检查点。\n\n'
            '恒0条件从第1步始终收到0，原任务按概率采样评估也保持0；可替代条件原生回报每步为1。'
            '同一种子四条件准备权重相同，课程条件世界批次相同，直接完整条件第601步起的批次相同。'
            '所有五符号双方向干预、原生指标和公共资源平衡诊断均已核查。\n\n'
            '另逐tensor确认1202准备权重与旧partners_001种子202的C/D相同；未加载旧社会学习终点。'
            '本轮10群体内部仍分别训练；历史准备起点并非全部首次使用，不能与旧批次作为完全独立来源合并。\n\n'
            '核查使用保存的网络类推理，环境案例、采样和评分来自独立实现；未调用正式训练器或正式评估器。'
            '这是实现与记录核查，不增加独立群体数或图像样本量。\n')
    (folder/'形成确认_只读核查.md').write_text(text)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='?', type=Path, default=ROOT/'results/formation_confirm_001')
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = audit(args.directory)
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs','training')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
