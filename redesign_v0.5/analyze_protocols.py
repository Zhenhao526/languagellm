"""Read-only Stage A protocol probes with calibration/validation separation.

Does not train or modify checkpoints. Fixed recipient contexts retain the same
goal, empty inventory/history and action-menu permutation during token swaps.
Run after completed runs exist; partial results are explicitly marked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import re

import numpy as np
import torch

from camp import MAPS, HOLDOUT, TRAIN_MAPS, ImageBank, remake_agents, projected_banks, scene_visual
from run_stages import A


ROOT = Path(__file__).resolve().parent
MENUS = np.asarray(list(permutations(range(4))), dtype=np.int64)
ASSIGNMENTS = ((0, 1), (1, 0))  # token slot for food location, water location
RECODING_CONDITIONS = ('known_sequence', 'hidden_sequence', 'hidden_sequence_holdout')
EXTENDED_RECODING_CONDITIONS = RECODING_CONDITIONS + ('hidden_blocked', 'hidden_sequence_direct', 'hidden_sequence_mixed')
RECODING_REPLICATES = 100
EVALUATION_AMENDMENT = dict(
    decision_time_utc='2026-09-15T07:59:59Z',
    timing='Added during Stage A training, after the first seed completed, before the formal protocol analysis and before any recoding-reference results were computed',
    reason='A nonzero local token-swap score can arise by chance in a finite holistic codebook; compare against bijective whole-message recodings preserving natural behavior',
    changes_training=False,
    scope=list(RECODING_CONDITIONS), replicates_per_direction=RECODING_REPLICATES,
    inference='A conditional recoding reference, not new training seeds, a population confidence interval, or proof of syntax',
)
EXPLORATORY_EXTENSION = dict(
    decision_time_utc='2026-09-15T08:06:14Z',
    timing='Added after observing all raw protocol scores and the initial three-condition recoding results',
    reason='Compare course, direct and mixed training against their own task-behavior-preserving recoding references rather than comparing raw fragment scores alone',
    scope=list(EXTENDED_RECODING_CONDITIONS),
    newly_added_conditions=['hidden_blocked', 'hidden_sequence_direct', 'hidden_sequence_mixed'],
    replicates_per_direction=RECODING_REPLICATES, changes_training=False,
    status='Post-result exploratory extension; not part of the original pre-analysis scope',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_ids(messages, vocab):
    values = np.zeros(messages.shape[:-1], dtype=np.int64)
    for slot in range(messages.shape[-1]):
        values = values * vocab + messages[..., slot]
    return values


def proportion(values, eligible=None):
    values = np.asarray(values, dtype=bool)
    if eligible is not None:
        values = values[np.asarray(eligible, dtype=bool)]
    n = int(values.size)
    correct = int(values.sum())
    return dict(numerator=correct, denominator=n, rate=correct / n if n else None)


def pooled(rows, field):
    correct = sum(row[field]['numerator'] for row in rows)
    n = sum(row[field]['denominator'] for row in rows)
    return dict(numerator=correct, denominator=n, rate=correct / n if n else None)


@torch.no_grad()
def receiver_table(agent):
    """Enumerate every message, both goals, and all 24 private menus."""
    messages = np.asarray(list(product(range(agent.vocab), repeat=agent.length)), dtype=np.int64)
    indices = np.asarray(list(product(range(len(messages)), range(2), range(24))), dtype=np.int64)
    m, g, order = indices.T
    logits, _ = agent.receive(
        torch.from_numpy(messages[m]), torch.from_numpy(np.eye(2, dtype=np.float32)[g]),
        torch.zeros(len(indices), 2), torch.zeros(len(indices), 14), torch.from_numpy(MENUS[order]))
    choices = logits.argmax(-1).numpy()
    physical = MENUS[order, choices].reshape(len(messages), 2, 24)
    table = []
    for code, message in enumerate(messages):
        table.append(dict(message=message.tolist(), physical_actions_by_goal_and_menu=physical[code].tolist(),
                          menu_sensitive_goal_count=sum(len(set(row.tolist())) > 1 for row in physical[code])))
    return physical, table


@torch.no_grad()
def natural_messages(agent, projected, photo_pairs, plan):
    """Same photo pair is used in original/donor maps: only location changes."""
    indices = np.asarray(list(product(range(12), range(len(photo_pairs)), range(2))), dtype=np.int64)
    maps, photos, goals = indices.T
    positions, ids = MAPS[maps], photo_pairs[photos]
    h = agent.observe(scene_visual(positions, ids, projected))
    goal = torch.from_numpy(np.eye(2, dtype=np.float32)[goals])
    if not plan['known']:
        goal = torch.zeros_like(goal)
    emitted, _, _, _ = agent.send(h, goal, torch.zeros(len(indices), 2), np.random.default_rng(810071), True)
    emitted = emitted.numpy().reshape(12, len(photo_pairs), 2, agent.length)
    delivered = np.zeros_like(emitted) if plan.get('blocked') else emitted.copy()
    return emitted, delivered


def cross_goal_report(messages, decoder, vocab):
    codes = code_ids(messages, vocab)
    actions = decoder[codes]  # map, photos, sender goal, recipient goal, menu
    wanted = MAPS[:, None, None, :, None]
    correctness = actions == wanted
    native = np.stack([correctness[:, :, goal, goal] for goal in range(2)], axis=2)
    switched = np.stack([correctness[:, :, goal, 1 - goal] for goal in range(2)], axis=2)
    both = correctness.all(axis=3)
    return dict(
        native_goal=proportion(native), goal_switched_with_message_fixed=proportion(switched),
        same_message_correct_for_both_goals=proportion(both),
        correct_after_switch_when_native_correct=proportion(switched, native),
        message_unchanged_across_sender_goal=proportion((messages[:, :, 0] == messages[:, :, 1]).all(-1)),
        per_sender_goal=[dict(sender_goal=goal, native=proportion(native[:, :, goal]),
                              switched=proportion(switched[:, :, goal]), both=proportion(both[:, :, goal]))
                         for goal in range(2)],
    )


def codebook(emitted, delivered, decoder, vocab):
    rows = []
    for map_id, world in enumerate(MAPS):
        for goal in range(2):
            emitted_counts = Counter(map(tuple, emitted[map_id, :, goal].tolist()))
            delivered_counts = Counter(map(tuple, delivered[map_id, :, goal].tolist()))
            codes = code_ids(delivered[map_id, :, goal], vocab)
            correct = decoder[codes] == world[None, :, None]
            rows.append(dict(
                map_id=map_id, food_location=int(world[0]), water_location=int(world[1]), sender_goal=goal,
                emitted_messages=[dict(message=list(message), count=count) for message, count in sorted(emitted_counts.items())],
                delivered_messages=[dict(message=list(message), count=count) for message, count in sorted(delivered_counts.items())],
                native=proportion(correct[:, goal]), switched=proportion(correct[:, 1 - goal]),
                both=proportion(correct.all(1)),
            ))
    return rows


def paired_maps(kind, allowed_maps):
    return [(first, second) for first in allowed_maps for second in allowed_maps
            if MAPS[first, kind] != MAPS[second, kind]
            and MAPS[first, 1 - kind] == MAPS[second, 1 - kind]]


def fragment_report(messages, decoder, vocab, assignment, allowed_maps, calibration_support):
    reports = []
    nmenus = decoder.shape[-1]
    for kind, slot in enumerate(assignment):
        pairs = paired_maps(kind, allowed_maps)
        grid = np.asarray(list(product(range(len(pairs)), range(messages.shape[1]), range(2))), dtype=np.int64)
        pair_id, photo_id, sender_goal = grid.T
        original_maps = np.asarray([pairs[i][0] for i in pair_id])
        donor_maps = np.asarray([pairs[i][1] for i in pair_id])
        original = messages[original_maps, photo_id, sender_goal]
        donor = messages[donor_maps, photo_id, sender_goal]
        hybrid = original.copy()
        hybrid[:, slot] = donor[:, slot]
        original_actions = decoder[code_ids(original, vocab)]
        donor_actions = decoder[code_ids(donor, vocab)]
        hybrid_codes = code_ids(hybrid, vocab)
        hybrid_actions = decoder[hybrid_codes]
        original_correct = original_actions == MAPS[original_maps, :, None]
        donor_correct = donor_actions == MAPS[donor_maps, :, None]
        original_both = original_correct.all(1)
        donor_both = donor_correct.all(1)
        natural_both = original_both & donor_both
        row = np.arange(len(grid))
        native_both = original_correct[row, sender_goal] & donor_correct[row, sender_goal]
        changed_resource_correct = hybrid_actions[:, kind] == MAPS[donor_maps, kind, None]
        changed_resource_moved = changed_resource_correct & original_correct[:, kind]
        other_resource_correct = hybrid_actions[:, 1 - kind] == MAPS[original_maps, 1 - kind, None]
        other_action_preserved = other_resource_correct & original_correct[:, 1 - kind]
        strict = changed_resource_moved & other_action_preserved
        hybrid_both_correct = (hybrid_actions == MAPS[donor_maps, :, None]).all(1)
        support = np.isin(hybrid_codes, list(calibration_support))[:, None].repeat(nmenus, 1)
        changed_token = (original[:, slot] != donor[:, slot])[:, None].repeat(nmenus, 1)
        reports.append(dict(
            resource=kind, token_slot=slot, directed_map_pairs=len(pairs), photo_pairs=messages.shape[1],
            paired_contexts_without_menu=len(grid), menus_per_context=nmenus,
            strict_transfer_all=proportion(strict),
            hybrid_correct_for_both_goals_all=proportion(hybrid_both_correct),
            changed_resource_points_to_donor=proportion(changed_resource_correct),
            unchanged_resource_stays_correct=proportion(other_action_preserved),
            natural_both_maps_both_goals_correct=proportion(natural_both),
            strict_transfer_when_both_maps_both_goals_correct=proportion(strict, natural_both),
            strict_transfer_when_both_native_goals_correct=proportion(strict, native_both),
            natural_both_maps_native_goal_correct=proportion(native_both),
            token_actually_changed=proportion(changed_token),
            hybrid_in_greedy_calibration_support=proportion(support),
            strict_transfer_in_calibration_support=proportion(strict, support),
            strict_transfer_outside_calibration_support=proportion(strict, ~support),
        ))
    return dict(assignment_food_water_slots=list(assignment), resources=reports,
                strict_transfer_all=pooled(reports, 'strict_transfer_all'),
                strict_transfer_when_both_maps_both_goals_correct=pooled(reports, 'strict_transfer_when_both_maps_both_goals_correct'),
                hybrid_correct_for_both_goals_all=pooled(reports, 'hybrid_correct_for_both_goals_all'))


def choose_assignment(messages, decoder, vocab, allowed_maps, support):
    candidates = [fragment_report(messages, decoder, vocab, assignment, allowed_maps, support)
                  for assignment in ASSIGNMENTS]
    # Tie is resolved a priori by the first assignment, never by validation.
    index = max(range(2), key=lambda i: candidates[i]['strict_transfer_all']['rate'])
    return ASSIGNMENTS[index], candidates


def heldout_stitch_report(messages, decoder, vocab, assignment, support):
    """All seen-map donor combinations, not hand-picked successful examples."""
    rows = []
    nmenus = decoder.shape[-1]
    for target in HOLDOUT:
        food_sources = [i for i in TRAIN_MAPS if MAPS[i, 0] == MAPS[target, 0]]
        water_sources = [i for i in TRAIN_MAPS if MAPS[i, 1] == MAPS[target, 1]]
        grid = np.asarray(list(product(food_sources, water_sources, range(messages.shape[1]), range(2))), dtype=np.int64)
        food_map, water_map, photo, sender_goal = grid.T
        food_message = messages[food_map, photo, sender_goal]
        water_message = messages[water_map, photo, sender_goal]
        hybrid = np.empty_like(food_message)
        hybrid[:, assignment[0]] = food_message[:, assignment[0]]
        hybrid[:, assignment[1]] = water_message[:, assignment[1]]
        hybrid_codes = code_ids(hybrid, vocab)
        actions = decoder[hybrid_codes]
        food_natural = decoder[code_ids(food_message, vocab)] == MAPS[food_map, :, None]
        water_natural = decoder[code_ids(water_message, vocab)] == MAPS[water_map, :, None]
        source_both = food_natural.all(1) & water_natural.all(1)
        sources_component_correct = food_natural[:, 0] & water_natural[:, 1]
        correct = actions == MAPS[target][None, :, None]
        natural_target = decoder[code_ids(messages[target, photo, sender_goal], vocab)] == MAPS[target][None, :, None]
        in_support = np.isin(hybrid_codes, list(support))[:, None].repeat(nmenus, 1)
        rows.append(dict(
            map_id=int(target), food_location=int(MAPS[target, 0]), water_location=int(MAPS[target, 1]),
            food_source_maps=list(map(int, food_sources)), water_source_maps=list(map(int, water_sources)),
            contexts_without_menu=len(grid), menus_per_context=nmenus,
            both_goals_correct_all=proportion(correct.all(1)),
            per_goal_correct_all=proportion(correct),
            both_source_messages_both_goals_correct=proportion(source_both),
            both_goals_correct_when_sources_both_goals_correct=proportion(correct.all(1), source_both),
            both_goals_correct_when_source_components_correct=proportion(correct.all(1), sources_component_correct),
            natural_target_message_both_goals_correct=proportion(natural_target.all(1)),
            hybrid_in_greedy_calibration_support=proportion(in_support),
        ))
    return dict(assignment_food_water_slots=list(assignment), target_maps=rows,
                both_goals_correct_all=pooled(rows, 'both_goals_correct_all'),
                per_goal_correct_all=pooled(rows, 'per_goal_correct_all'),
                both_goals_correct_when_sources_both_goals_correct=pooled(rows, 'both_goals_correct_when_sources_both_goals_correct'))


def recoding_summary(observed, values):
    if observed is None:
        assert all(value is None for value in values)
        return dict(observed=None, reason='zero eligible denominator in observed and every recoding')
    array = np.asarray(values, dtype=np.float64)
    assert np.isfinite(array).all()
    return dict(observed=float(observed), reference_mean=float(array.mean()),
                reference_min=float(array.min()), reference_max=float(array.max()),
                reference_quantiles={str(q): float(np.quantile(array, q)) for q in (.025, .25, .5, .75, .975)},
                observed_minus_reference_mean=float(observed - array.mean()),
                fraction_reference_strictly_below_observed=float((array < observed).mean()),
                fraction_reference_at_or_below_observed=float((array <= observed).mean()),
                reference_at_or_above_observed_count=int((array >= observed).sum()),
                reference_count=len(values))


def whole_message_recoding_reference(messages, decoder, assignment, selection_maps, *,
                                     seed, replicates=RECODING_REPLICATES, stitch=False):
    """Change code coordinates while preserving every natural message action.

    permutation[old_code] = new_code. Receiver_new[new_code] = Receiver_old[old_code].
    The resulting virtual protocols preserve the whole-message behavior exactly;
    they do not correspond to retraining the neural receiver on renamed tokens.
    """
    assert decoder.shape[0] == 25 and messages['calibration'].shape[-1] == 2
    permutation_invariant = bool(np.all(decoder == decoder[:, :, :1]))
    # Only collapse menu copies after checking all 25 codes and both goals.
    scoring_decoder = decoder[:, :, :1] if permutation_invariant else decoder
    codes = {phase: code_ids(value, 5) for phase, value in messages.items()}
    support = set(codes['calibration'][selection_maps].flatten().tolist())
    observed = fragment_report(messages['validation'], scoring_decoder, 5, assignment, np.arange(12), support)
    observed_stitch = heldout_stitch_report(messages['validation'], scoring_decoder, 5, assignment, support) if stitch else None
    rng = np.random.default_rng(seed)
    records = []
    for repeat in range(replicates):
        permutation = rng.permutation(25)
        inverse = np.argsort(permutation)
        new_full_decoder = decoder[inverse]
        assert np.array_equal(new_full_decoder[permutation], decoder)
        new_decoder = new_full_decoder[:, :, :1] if permutation_invariant else new_full_decoder
        renamed = {}
        for phase, original_codes in codes.items():
            renamed_codes = permutation[original_codes]
            renamed[phase] = np.stack((renamed_codes // 5, renamed_codes % 5), -1)
            assert np.array_equal(new_full_decoder[renamed_codes], decoder[original_codes])
            original_frequency = np.bincount(original_codes.flatten(), minlength=25)
            renamed_frequency = np.bincount(renamed_codes.flatten(), minlength=25)
            assert np.array_equal(renamed_frequency[permutation], original_frequency)
        new_support = set(permutation[list(support)].tolist())
        new_assignment, calibration = choose_assignment(renamed['calibration'], new_decoder, 5,
                                                        selection_maps, new_support)
        validation = fragment_report(renamed['validation'], new_decoder, 5, new_assignment,
                                     np.arange(12), new_support)
        assert validation['strict_transfer_all']['denominator'] == observed['strict_transfer_all']['denominator']
        assert validation['strict_transfer_when_both_maps_both_goals_correct']['denominator'] == observed['strict_transfer_when_both_maps_both_goals_correct']['denominator']
        row = dict(replicate=repeat, old_to_new_code=permutation.tolist(),
                   calibration_candidate_strict_rates=[r['strict_transfer_all']['rate'] for r in calibration],
                   selected_assignment_food_water_slots=list(new_assignment),
                   validation_strict_transfer_all=validation['strict_transfer_all'],
                   validation_strict_transfer_when_both_maps_both_goals_correct=validation['strict_transfer_when_both_maps_both_goals_correct'])
        if stitch:
            stitched = heldout_stitch_report(renamed['validation'], new_decoder, 5, new_assignment, new_support)
            row['validation_stitch_both_goals_correct_all'] = stitched['both_goals_correct_all']
        records.append(row)
    result = dict(
        rng_seed=int(seed), replicates=replicates, full_message_capacity=25,
        menu_invariance_checked_for_all_codes_and_goals=permutation_invariant,
        menus_used_per_counterfactual=scoring_decoder.shape[-1],
        original_menus_enumerated=decoder.shape[-1],
        preservation_checks='All 25 code/goal/menu action functions and all natural calibration/validation actions and code frequencies match under bijective relabeling',
        calibration_selection_repeated_for_every_recoding=True,
        observed_calibration_assignment_food_water_slots=list(assignment),
        selection_map_ids=list(map(int, selection_maps)),
        strict_transfer_all=recoding_summary(observed['strict_transfer_all']['rate'],
                                            [row['validation_strict_transfer_all']['rate'] for row in records]),
        strict_transfer_when_both_maps_both_goals_correct=recoding_summary(
            observed['strict_transfer_when_both_maps_both_goals_correct']['rate'],
            [row['validation_strict_transfer_when_both_maps_both_goals_correct']['rate'] for row in records]),
        records=records,
    )
    if stitch:
        result['heldout_stitch_both_goals_correct_all'] = recoding_summary(
            observed_stitch['both_goals_correct_all']['rate'],
            [row['validation_stitch_both_goals_correct_all']['rate'] for row in records])
    return result


@torch.no_grad()
def analyze_run(path, batch, bank, splits, recoding_conditions=RECODING_CONDITIONS):
    config = json.loads((path / 'config.json').read_text())
    if config['source_hashes']['camp.py'] != sha(ROOT / 'camp.py'):
        raise RuntimeError(f'Camp implementation changed since training: {path}')
    seed, plan = config['seed'], config['plan']
    prepared_path, final_path = batch / f'prepared_{seed}.pt', path / 'final.pt'
    prepared = torch.load(prepared_path, weights_only=True)
    agents = remake_agents(seed, prepared, plan['vocab'], plan['length'])
    states = torch.load(final_path, weights_only=True)
    for agent, state in zip(agents, states):
        agent.load_state_dict(state)
        agent.eval()
    banks = projected_banks(agents, bank)
    directions = []
    for scout in range(2):
        collector = 1 - scout
        decoder, table = receiver_table(agents[collector])
        messages = {}
        phase_results = {}
        for phase, pairs in splits.items():
            emitted, delivered = natural_messages(agents[scout], banks[scout], pairs, plan)
            messages[phase] = delivered
            phase_results[phase] = dict(photo_pairs=pairs.tolist(),
                cross_goal=cross_goal_report(delivered, decoder, plan['vocab']),
                codebook=codebook(emitted, delivered, decoder, plan['vocab']))
        direction = dict(scout=scout, collector=collector, receiver_decoder_table=table, phases=phase_results)
        if plan['length'] == 2 and plan['vocab'] == 5:
            selection_maps = TRAIN_MAPS if plan.get('holdout') else np.arange(12)
            support = set(code_ids(messages['calibration'][selection_maps], plan['vocab']).flatten().tolist())
            assignment, candidates = choose_assignment(messages['calibration'], decoder, plan['vocab'], selection_maps, support)
            direction['fragments'] = dict(
                selection_unit='one assignment per sender/receiver direction, independently of the other direction',
                selection_map_ids=selection_maps.tolist(), selection_metric='unfiltered strict transfer rate',
                calibration_candidates=candidates, selected_assignment_food_water_slots=list(assignment),
                validation=fragment_report(messages['validation'], decoder, plan['vocab'], assignment, np.arange(12), support),
                greedy_calibration_message_support=sorted(support),
            )
            if plan.get('holdout'):
                direction['heldout_stitch_validation'] = heldout_stitch_report(
                    messages['validation'], decoder, plan['vocab'], assignment, support)
            if config['condition'] in recoding_conditions:
                recoding_seed = 810000000 + seed * 100 + 10 * EXTENDED_RECODING_CONDITIONS.index(config['condition']) + scout
                direction['whole_message_recoding'] = whole_message_recoding_reference(
                    messages, decoder, assignment, selection_maps, seed=recoding_seed,
                    stitch=bool(plan.get('holdout')))
                assert direction['whole_message_recoding']['strict_transfer_all']['observed'] == direction['fragments']['validation']['strict_transfer_all']['rate']
        directions.append(direction)
    return dict(seed=seed, condition=config['condition'], plan=plan,
                final_checkpoint=str(final_path), final_sha256=sha(final_path),
                prepared_checkpoint=str(prepared_path), prepared_sha256=sha(prepared_path),
                directions=directions)


def run_synthetic_checks():
    """Oracle and null protocols test scoring without any learned checkpoint."""
    decoder = np.empty((25, 2, 24), np.int64)
    for first, second in product(range(5), repeat=2):
        decoder[5 * first + second, 0] = min(first, 3)
        decoder[5 * first + second, 1] = min(second, 3)
    messages = np.tile(MAPS[:, None, None, :], (1, 16, 2, 1))
    support = set(code_ids(messages, 5).flatten().tolist())
    cross = cross_goal_report(messages, decoder, 5)
    assert cross['same_message_correct_for_both_goals']['rate'] == 1
    assignment, candidates = choose_assignment(messages, decoder, 5, np.arange(12), support)
    assert assignment == (0, 1)
    assert candidates[0]['strict_transfer_all']['rate'] == 1
    assert candidates[1]['strict_transfer_all']['rate'] == 0
    assert candidates[0]['strict_transfer_all']['denominator'] == 36864
    stitched = heldout_stitch_report(messages, decoder, 5, assignment, support)
    assert stitched['both_goals_correct_all']['rate'] == 1
    assert stitched['both_goals_correct_all']['denominator'] == 12288
    constant = np.zeros_like(messages)
    null = fragment_report(constant, decoder, 5, (0, 1), np.arange(12), {0})
    assert null['strict_transfer_all']['rate'] == 0
    assert null['strict_transfer_when_both_maps_both_goals_correct']['denominator'] == 0
    assert null['strict_transfer_when_both_maps_both_goals_correct']['rate'] is None
    instruction_messages = messages.copy()
    instruction_decoder = decoder.copy()
    for location in range(4):
        instruction_decoder[5 * location + 4, :, :] = location
    for goal in range(2):
        instruction_messages[:, :, goal, 0] = MAPS[:, goal, None]
        instruction_messages[:, :, goal, 1] = 4
    instruction = cross_goal_report(instruction_messages, instruction_decoder, 5)
    assert instruction['native_goal']['rate'] == 1
    assert instruction['goal_switched_with_message_fixed']['rate'] == 0
    recoding = whole_message_recoding_reference(
        dict(calibration=messages, validation=messages), decoder, assignment,
        np.arange(12), seed=810009971, replicates=10, stitch=True)
    assert recoding['strict_transfer_all']['observed'] == 1
    assert recoding['strict_transfer_all']['reference_mean'] < 1
    assert recoding['heldout_stitch_both_goals_correct_all']['observed'] == 1
    assert recoding['menus_used_per_counterfactual'] == 1
    assert all(sorted(row['old_to_new_code']) == list(range(25)) for row in recoding['records'])
    null_recoding = whole_message_recoding_reference(
        dict(calibration=constant, validation=constant), decoder, (0, 1),
        np.arange(12), seed=810009972, replicates=2)
    assert null_recoding['strict_transfer_all']['reference_max'] == 0
    assert null_recoding['strict_transfer_when_both_maps_both_goals_correct']['observed'] is None
    return dict(status='passed', cases=['compositional oracle', 'wrong token assignment',
                'heldout stitching oracle', 'constant-message null with zero eligible denominator',
                'goal-specific instruction with perfect native but zero switched accuracy',
                'bijective recodings preserve every natural behavior and frequency',
                'menu collapse only after full equivalence verification',
                'recoded zero-denominator subset remains undefined'])


def percentage(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def markdown(result):
    lines = ['# A阶段协议与成分干预分析', '',
             f"完成运行数：{len(result['runs'])}；尚未完成：{len(result['missing_runs'])}。状态：{result['status']}。", '',
             '本分析只读取检查点，不更新模型。旧test每类前4张用于选择符号位置分配，后4张用于验证；每组枚举16个照片对、12张地图、2种发送时目标及2个信息方向，并枚举24种接收者行动菜单。它们仍是既有测试照片，不是新采集的外部确认集。', '',
             '目标切换保持消息不变，只改变接收者需求。片段干预在同一照片对下配对仅一个资源位置不同的地图，只替换一个符号；库存、历史和菜单保持不变。严格成功要求该资源的动作从旧正确位置转向新位置，同时另一资源的正确动作保持不变。', '',
             '符号位置分配按每个通信方向分别选定，只用校准数据中未筛选的严格成功率；平分时固定选“第一符号对应食物位置”。组合留出条件的分配选择只使用8张训练地图。验证集不重新选择分配。', '',
             '| 种子 | 条件 | 方向 | 原目标正确 | 固定消息换目标 | 同消息两个目标都正确 | 单符号严格迁移（全部） | 单符号严格迁移（两图两目标原本均正确） |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for run in result['runs']:
        for direction in run['directions']:
            cross = direction['phases']['validation']['cross_goal']
            fragment = direction.get('fragments', {}).get('validation')
            if fragment:
                overall = fragment['strict_transfer_all']
                selected = fragment['strict_transfer_when_both_maps_both_goals_correct']
                ftext = f"{percentage(overall['rate'])} ({overall['numerator']}/{overall['denominator']})"
                stext = f"{percentage(selected['rate'])} ({selected['numerator']}/{selected['denominator']})"
            else:
                ftext = stext = '不适用'
            lines.append(f"| {run['seed']} | {run['condition']} | {direction['scout']}→{direction['collector']} | "
                         f"{percentage(cross['native_goal']['rate'])} | {percentage(cross['goal_switched_with_message_fixed']['rate'])} | "
                         f"{percentage(cross['same_message_correct_for_both_goals']['rate'])} | {ftext} | {stext} |")
    lines += ['', '表中原目标正确率来自确定的平衡枚举，不必等于原始随机终点评估。每个分母是干预案例数，不是独立训练重复数；统计重复单位仍是种子。零分母保留为空，不改写为0%。', '',
              '完整JSON保留校准和验证词典、每个消息在两个目标下的动作、两种候选分配、各资源迁移、未经筛选和不同自然消息正确性子集的分母，以及干预消息是否在有限校准最大概率消息集合中出现。该集合不是模型概率支持集，不可称为从未产生过的消息。', '',
              '只有保留上下文的成分干预与跨目标表现共同支持时，才讨论有限任务中的成分复用。成功不等于语法；失败可能涉及原协议本来不完整、整体编码或干预组合缺少训练经验。']
    recodings = [(run, direction) for run in result['runs'] for direction in run['directions']
                 if 'whole_message_recoding' in direction]
    if recodings:
        scope_text = '全部六类双符号条件' if result.get('exploratory_extension') else '主已知目标、隐藏目标和组合留出序列条件'
        lines += ['', '## 保持任务表现的完整消息重编码基准', '',
                  f'对{scope_text}，每方向固定100次25码随机双射。发送方整个消息按双射改名，接收映射作逆变换，逐例核查原生与跨目标动作、完整消息频率及25码容量保持不变。每次重新使用相同校准规则选择符号位置，再在验证图片上固定评分。', '',
                  '这是一组虚拟重编码协议，未训练新模型。它破坏原来的符号片段组织，同时保留完整消息能完成的事情。只有确认全部25码、两个目标在24种菜单下动作相同后，才将重复菜单压缩为一个以减少计算；24种菜单不构成独立证据。', '',
                  '| 种子 | 条件 | 方向 | 观察到的严格迁移 | 重编码均值 | 重编码全范围 | 重编码2.5%–97.5%分位 | 观察值减均值 | 参考值≥观察值次数 |',
                  '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for run, direction in recodings:
            reference = direction['whole_message_recoding']['strict_transfer_all']
            q = reference['reference_quantiles']
            lines.append(f"| {run['seed']} | {run['condition']} | {direction['scout']}→{direction['collector']} | "
                         f"{percentage(reference['observed'])} | {percentage(reference['reference_mean'])} | "
                         f"{percentage(reference['reference_min'])}–{percentage(reference['reference_max'])} | "
                         f"{percentage(q['0.025'])}–{percentage(q['0.975'])} | "
                         f"{100 * reference['observed_minus_reference_mean']:.2f}个百分点 | "
                         f"{reference['reference_at_or_above_observed_count']}/{reference['reference_count']} |")
        lines += ['', '上述范围是给定协议的重编码分布，不是训练种子的置信区间，也不是论文级显著性结论。100次重编码分辨率有限；约25%的局部替换成功不能脱离此基准直接解释为组合结构。JSON同时保留全部置换、校准选择和验证分数。', '',
                  '评价补充时间：2026-09-15 15:59:59（北京时间）。当时A阶段仍在训练，首种子已完成；本项在正式协议分析和任何重编码基准结果计算之前固定。原因是有限整体码表可能偶然支持局部交换；没有修改训练、预算或种子，不将这项运行中补充称为最初方案已注册的指标。']
        if result.get('exploratory_extension'):
            lines += ['', '## 看到结果后的探索性扩展', '',
                      '2026-09-15 16:06:14（北京时间），看到全部原始片段分数及初版三条件重编码结果之后，将同一基准扩展到恒通道、直接训练和混合呈现。原始结果文件保留；本扩展不属于最初评价范围，不能描述为运行前已计划覆盖全部条件。', '',
                      '| 条件 | 三种子平均严格迁移 | 各自重编码均值 | 观察值减自身基准 |',
                      '| --- | ---: | ---: | ---: |']
            for condition in EXTENDED_RECODING_CONDITIONS:
                rows = [d['whole_message_recoding']['strict_transfer_all'] for run, d in recodings if run['condition'] == condition]
                observed = np.mean([row['observed'] for row in rows])
                baseline = np.mean([row['reference_mean'] for row in rows])
                lines.append(f'| {condition} | {percentage(observed)} | {percentage(baseline)} | {100 * (observed - baseline):.2f}个百分点 |')
            lines += ['', '每种子两个通信方向等权，三种子等权。观察值减基准仍会受可正确处理的情境数量影响，不能单凭差值将训练顺序解释为抽象组合能力的因果变化；全部案例与自然消息正确子集均保留。', '',
                      '| 种子 | 课程：观察值减基准 | 直接：观察值减基准 | 混合：观察值减基准 |',
                      '| --- | ---: | ---: | ---: |']
            for seed in result['expected_seeds']:
                values = []
                for condition in ('hidden_sequence', 'hidden_sequence_direct', 'hidden_sequence_mixed'):
                    rows = [d['whole_message_recoding']['strict_transfer_all'] for run, d in recodings
                            if run['seed'] == seed and run['condition'] == condition]
                    values.append(np.mean([row['observed_minus_reference_mean'] for row in rows]))
                lines.append(f"| {seed} | " + ' | '.join(f'{100 * value:.2f}个百分点' for value in values) + ' |')
    holdouts = [(run, direction) for run in result['runs'] for direction in run['directions']
                if 'heldout_stitch_validation' in direction]
    if holdouts:
        lines += ['', '## 训练地图成分拼接到留出地图', '',
                  '只对实际留出了四张地图的条件报告。两个供体都来自8张训练地图，穷举全部合资格供体组合；分配仍由训练地图的校准照片决定。', '',
                  '| 种子 | 方向 | 拼接后两个目标都正确（全部） | 两供体两目标原本均正确子集 |',
                  '| --- | --- | ---: | ---: |']
        for run, direction in holdouts:
            row = direction['heldout_stitch_validation']
            a, b = row['both_goals_correct_all'], row['both_goals_correct_when_sources_both_goals_correct']
            lines.append(f"| {run['seed']} | {direction['scout']}→{direction['collector']} | {percentage(a['rate'])} ({a['numerator']}/{a['denominator']}) | {percentage(b['rate'])} ({b['numerator']}/{b['denominator']}) |")
    if result['missing_runs']:
        lines += ['', '未完成的运行：' + '、'.join(result['missing_runs']) + '。本表不能作为全部条件的最终比较。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path, nargs='?', default=ROOT / 'results/progression_001')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--extended-null', action='store_true', help='Post-result exploratory reference for all six two-token conditions; writes separate files')
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks = run_synthetic_checks()
    if args.self_test:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return
    batch = args.batch.resolve()
    prefix = 'protocol_analysis_extended_null' if args.extended_null else 'protocol_analysis'
    json_path, md_path = batch / f'{prefix}.json', batch / f'{prefix}.md'
    if json_path.exists() or md_path.exists():
        raise SystemExit(f'Refusing to overwrite existing analysis: {prefix}')
    original_path = batch / 'protocol_analysis.json'
    original = json.loads(original_path.read_text()) if args.extended_null else None
    original_hash = sha(original_path) if args.extended_null else None
    expected_seeds = {int(re.fullmatch(r'prepared_(\d+)\.pt', p.name).group(1)) for p in batch.glob('prepared_*.pt')}
    for invocation_path in batch.glob('invocation_*.json'):
        invocation = json.loads(invocation_path.read_text())
        if 'A' in invocation['stages']:
            expected_seeds.update(invocation['seeds'])
    seeds = sorted(expected_seeds)
    paths, missing = [], []
    for seed in seeds:
        for condition in A:
            path = batch / 'A' / f's{seed}_{condition}'
            if all((path / name).exists() for name in ('config.json', 'result.json', 'final.pt')):
                paths.append(path)
            else:
                missing.append(path.name)
    if not paths:
        raise SystemExit('No completed Stage A runs: no analysis result written.')
    bank = ImageBank()
    for kind in range(2):
        assert len(bank.pools['test', kind]) == 8, 'Expected existing 8 held-out photos per resource'
    splits = {phase: np.asarray(list(product(bank.pools['test', 0][subset], bank.pools['test', 1][subset])), dtype=np.int64)
              for phase, subset in [('calibration', slice(0, 4)), ('validation', slice(4, 8))]}
    assert not (set(splits['calibration'].flatten()) & set(splits['validation'].flatten()))
    result = dict(status='partial' if missing else 'complete', batch=str(batch),
                  expected_seeds=seeds,
                  evaluation_amendment=EVALUATION_AMENDMENT,
                  analysis_sha256=sha(__file__), camp_sha256=sha(ROOT / 'camp.py'),
                  feature_sha256=sha(ROOT.parent / 'redesign_v0.4/data/features.npz'),
                  synthetic_checks=checks, missing_runs=missing,
                  protocol='greedy sender and receiver; exact 24-menu average; fixed recipient context',
                  repeated_unit='trained seed; cases and menus are not independent training repeats',
                  photo_splits={phase: pairs.tolist() for phase, pairs in splits.items()}, runs=[])
    if args.extended_null:
        result['exploratory_extension'] = EXPLORATORY_EXTENSION
        result['original_analysis'] = dict(path=str(original_path), sha256=original_hash)
    for path in paths:
        result['runs'].append(analyze_run(path, batch, bank, splits,
            EXTENDED_RECODING_CONDITIONS if args.extended_null else RECODING_CONDITIONS))
        print(f"Analyzed {path.name}", flush=True)
    if args.extended_null:
        original_runs = {(row['seed'], row['condition']): row for row in original['runs']}
        compared_directions = 0
        for row in result['runs']:
            before = original_runs[row['seed'], row['condition']]
            for previous, current in zip(before['directions'], row['directions']):
                for key, value in previous.items():
                    assert current[key] == value, f'Original result changed: {row["condition"]}/{key}'
                compared_directions += 1
        assert sha(original_path) == original_hash
        result['original_values_reproduced_exactly'] = dict(directions=compared_directions,
            original_recoding_directions=sum('whole_message_recoding' in d for r in original['runs'] for d in r['directions']),
            original_file_unchanged=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    md_path.write_text(markdown(result), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], analyzed=len(paths), missing=len(missing),
                         output=str(json_path)), ensure_ascii=False))


if __name__ == '__main__':
    main()
