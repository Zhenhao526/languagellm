"""Read-only v0.7 visual-interface protocol probes, fixed before this batch's training.

Does not train or modify checkpoints. Fixed recipient contexts retain the same
goal, empty inventory/history and action-menu permutation during token swaps.
Run after completed runs exist; partial results are explicitly marked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
from itertools import permutations, product
import json
from pathlib import Path
import re

import numpy as np
import torch

from camp import SITES, HISTORY, MAPS, split_maps, ImageBank, remake_agents, projected_banks, scene_visual
from run_experiment import CONDITIONS, SEEDS


ROOT = Path(__file__).resolve().parent
MENUS = np.asarray(list(permutations(range(SITES))), dtype=np.int64)
ASSIGNMENTS = ((0, 1), (1, 0))  # token slot for food location, water location
VOCAB = 7
CAPACITY = VOCAB ** 2
RECODING_REPLICATES = 100
ANALYSIS_PLAN = dict(
    timing='Fixed before v0.7 formal training; identical protocol tests adapted from v0.6',
    scope='All 52 planned runs and 104 directions; each receives 100 whole-code recoding references',
    replicates_per_direction=RECODING_REPLICATES, changes_training=False,
    selection='Per-direction assignment selected on training maps and calibration photos only',
    primary_units='Four training seeds; splits and directions are within-seed repeated measures',
    reference_interpretation='Conditional coordinate-recoding distribution, not a confidence interval over independently trained agents',
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
    """Enumerate all 720 menus in chunks before any menu collapse."""
    messages = np.asarray(list(product(range(agent.vocab), repeat=agent.length)), dtype=np.int64)
    indices = np.asarray(list(product(range(len(messages)), range(2), range(len(MENUS)))), dtype=np.int64)
    physical = np.empty(len(indices), dtype=np.int64)
    for start in range(0, len(indices), 4096):
        end = min(start + 4096, len(indices))
        m, g, order = indices[start:end].T
        logits, _ = agent.receive(
            torch.from_numpy(messages[m]), torch.from_numpy(np.eye(2, dtype=np.float32)[g]),
            torch.zeros(end - start, 2), torch.zeros(end - start, HISTORY), torch.from_numpy(MENUS[order]))
        physical[start:end] = MENUS[order, logits.argmax(-1).numpy()]
    physical = physical.reshape(len(messages), 2, len(MENUS))
    invariant = bool(np.all(physical == physical[:, :, :1]))
    audit = dict(enumerated_messages=len(messages), goals=2, menus=len(MENUS),
                 cases=len(indices), all_menu_permutations_physically_equivalent=invariant,
                 physical_action_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
                 menus_used_after_check=1 if invariant else len(MENUS))
    table = []
    for code, message in enumerate(messages):
        row = dict(message=message.tolist(),
                   actions_by_goal=physical[code, :, 0].tolist(),
                   menu_action_counts_by_goal=[np.bincount(actions, minlength=SITES).tolist() for actions in physical[code]])
        if not invariant:
            row['physical_actions_by_goal_and_menu'] = physical[code].tolist()
        table.append(row)
    return physical[:, :, :1] if invariant else physical, table, audit


@torch.no_grad()
def natural_messages(agent, projected, photo_pairs, plan):
    """Same photo pair is used in original/donor maps: only location changes."""
    indices = np.asarray(list(product(range(len(MAPS)), range(len(photo_pairs)), range(2))), dtype=np.int64)
    maps, photos, goals = indices.T
    positions, ids = MAPS[maps], photo_pairs[photos]
    h = agent.observe(scene_visual(positions, ids, projected))
    goal = torch.from_numpy(np.eye(2, dtype=np.float32)[goals])
    if not plan['known']:
        goal = torch.zeros_like(goal)
    emitted, _, _, _ = agent.send(h, goal, torch.zeros(len(indices), 2), np.random.default_rng(810071), True)
    emitted = emitted.numpy().reshape(len(MAPS), len(photo_pairs), 2, agent.length)
    delivered = np.zeros_like(emitted) if plan.get('blocked') else emitted.copy()
    return emitted, delivered


def cross_goal_report(messages, decoder, vocab, map_ids=None):
    map_ids = np.arange(len(MAPS)) if map_ids is None else np.asarray(map_ids)
    if len(map_ids) == 0:
        return None
    messages = messages[map_ids]
    codes = code_ids(messages, vocab)
    actions = decoder[codes]  # map, photos, sender goal, recipient goal, menu
    wanted = MAPS[map_ids, None, None, :, None]
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


@lru_cache(None)
def fragment_indices(kind, allowed_maps, nphotos):
    pairs = paired_maps(kind, allowed_maps)
    grid = np.asarray(list(product(range(len(pairs)), range(nphotos), range(2))), dtype=np.int64)
    assert len(grid), 'No one-factor map pairs in requested support'
    pair_id, photo_id, sender_goal = grid.T
    original_maps = np.asarray([pairs[i][0] for i in pair_id])
    donor_maps = np.asarray([pairs[i][1] for i in pair_id])
    return len(pairs), original_maps, donor_maps, photo_id, sender_goal


def fragment_report(messages, decoder, vocab, assignment, allowed_maps, calibration_support):
    reports = []
    nmenus = decoder.shape[-1]
    for kind, slot in enumerate(assignment):
        npairs, original_maps, donor_maps, photo_id, sender_goal = fragment_indices(
            kind, tuple(map(int, allowed_maps)), messages.shape[1])
        ncontexts = len(photo_id)
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
        row = np.arange(ncontexts)
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
            resource=kind, token_slot=slot, directed_map_pairs=npairs, photo_pairs=messages.shape[1],
            paired_contexts_without_menu=ncontexts, menus_per_context=nmenus,
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


@lru_cache(None)
def stitch_indices(target, train_maps, nphotos):
    food_sources = [i for i in train_maps if MAPS[i, 0] == MAPS[target, 0]]
    water_sources = [i for i in train_maps if MAPS[i, 1] == MAPS[target, 1]]
    grid = np.asarray(list(product(food_sources, water_sources, range(nphotos), range(2))), dtype=np.int64)
    return food_sources, water_sources, grid


def heldout_stitch_report(messages, decoder, vocab, assignment, support, train_maps, heldout_maps):
    """All seen-map donor combinations, not hand-picked successful examples."""
    rows = []
    nmenus = decoder.shape[-1]
    assert len(heldout_maps) > 0
    for target in heldout_maps:
        food_sources, water_sources, grid = stitch_indices(
            int(target), tuple(map(int, train_maps)), messages.shape[1])
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
                                     seed, replicates=RECODING_REPLICATES, heldout_maps=()):
    """Change code coordinates while preserving every natural message action.

    permutation[old_code] = new_code. Receiver_new[new_code] = Receiver_old[old_code].
    The resulting virtual protocols preserve the whole-message behavior exactly;
    they do not correspond to retraining the neural receiver on renamed tokens.
    """
    assert decoder.shape[0] == CAPACITY and messages['calibration'].shape[-1] == 2
    permutation_invariant = bool(np.all(decoder == decoder[:, :, :1]))
    # receiver_table already enumerated every menu before any earlier collapse.
    scoring_decoder = decoder[:, :, :1] if permutation_invariant else decoder
    codes = {phase: code_ids(value, VOCAB) for phase, value in messages.items()}
    support = set(codes['calibration'][selection_maps].flatten().tolist())
    observed = fragment_report(messages['validation'], scoring_decoder, VOCAB, assignment, np.arange(len(MAPS)), support)
    stitch = len(heldout_maps) > 0
    observed_stitch = heldout_stitch_report(messages['validation'], scoring_decoder, VOCAB, assignment, support,
                                           selection_maps, heldout_maps) if stitch else None
    rng = np.random.default_rng(seed)
    records = []
    for repeat in range(replicates):
        permutation = rng.permutation(CAPACITY)
        inverse = np.argsort(permutation)
        new_full_decoder = decoder[inverse]
        assert np.array_equal(new_full_decoder[permutation], decoder)
        new_decoder = new_full_decoder[:, :, :1] if permutation_invariant else new_full_decoder
        renamed = {}
        for phase, original_codes in codes.items():
            renamed_codes = permutation[original_codes]
            renamed[phase] = np.stack((renamed_codes // VOCAB, renamed_codes % VOCAB), -1)
            assert np.array_equal(new_full_decoder[renamed_codes], decoder[original_codes])
            original_frequency = np.bincount(original_codes.flatten(), minlength=CAPACITY)
            renamed_frequency = np.bincount(renamed_codes.flatten(), minlength=CAPACITY)
            assert np.array_equal(renamed_frequency[permutation], original_frequency)
        new_support = set(permutation[list(support)].tolist())
        new_assignment, calibration = choose_assignment(renamed['calibration'], new_decoder, VOCAB,
                                                        selection_maps, new_support)
        validation = fragment_report(renamed['validation'], new_decoder, VOCAB, new_assignment,
                                     np.arange(len(MAPS)), new_support)
        assert validation['strict_transfer_all']['denominator'] == observed['strict_transfer_all']['denominator']
        assert validation['strict_transfer_when_both_maps_both_goals_correct']['denominator'] == observed['strict_transfer_when_both_maps_both_goals_correct']['denominator']
        row = dict(replicate=repeat, old_to_new_code=permutation.tolist(),
                   calibration_candidate_strict_rates=[r['strict_transfer_all']['rate'] for r in calibration],
                   selected_assignment_food_water_slots=list(new_assignment),
                   validation_strict_transfer_all=validation['strict_transfer_all'],
                   validation_strict_transfer_when_both_maps_both_goals_correct=validation['strict_transfer_when_both_maps_both_goals_correct'])
        if stitch:
            stitched = heldout_stitch_report(renamed['validation'], new_decoder, VOCAB, new_assignment, new_support,
                                             selection_maps, heldout_maps)
            row['validation_stitch_both_goals_correct_all'] = stitched['both_goals_correct_all']
        records.append(row)
    result = dict(
        rng_seed=int(seed), replicates=replicates, full_message_capacity=CAPACITY,
        menu_invariance_checked_for_all_codes_and_goals=permutation_invariant,
        menus_used_per_counterfactual=scoring_decoder.shape[-1],
        original_menus_enumerated=len(MENUS),
        preservation_checks='All 49 code/goal action functions and all natural calibration/validation actions and code frequencies match under bijective relabeling; see direction-level full-menu audit',
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
def analyze_run(path, batch, bank, splits):
    config = json.loads((path / 'config.json').read_text())
    assert config['source_hashes']['camp.py'] == sha(ROOT / 'camp.py'), f'Changed camp implementation: {path}'
    assert config['map_table'] == MAPS.tolist()
    seed, plan = config['seed'], config['plan']
    assert plan == CONDITIONS[config['condition']], f'Plan differs from declared condition: {path}'
    train_maps, heldout_maps = split_maps(plan['split'])
    assert config['train_map_ids'] == train_maps.tolist()
    assert config['heldout_map_ids'] == heldout_maps.tolist()
    prepared_path, final_path = batch / f'prepared_{seed}.pt', path / 'final.pt'
    assert config['prepared_source']['sha256'] == sha(prepared_path)
    prepared = torch.load(prepared_path, weights_only=True)
    agents = remake_agents(seed, prepared, plan['vocab'], plan['length'], representation=plan['representation'])
    for agent, state in zip(agents, torch.load(final_path, weights_only=True)):
        expected_transform = agent.input_transform.clone()
        agent.load_state_dict(state)
        assert torch.equal(agent.input_transform, expected_transform), f'Changed private input transform: {path}'
        agent.eval()
    banks = projected_banks(agents, bank)
    directions = []
    for scout in range(2):
        collector = 1 - scout
        decoder, table, menu_audit = receiver_table(agents[collector])
        messages, phase_results = {}, {}
        for phase, pairs in splits.items():
            emitted, delivered = natural_messages(agents[scout], banks[scout], pairs, plan)
            messages[phase] = delivered
            phase_results[phase] = dict(photo_pairs=pairs.tolist(),
                cross_goal=cross_goal_report(delivered, decoder, plan['vocab']),
                cross_goal_groups={name: cross_goal_report(delivered, decoder, plan['vocab'], ids)
                                   for name, ids in [('seen', train_maps), ('unseen', heldout_maps)]},
                codebook=codebook(emitted, delivered, decoder, plan['vocab']))
        direction = dict(scout=scout, collector=collector, receiver_decoder_table=table,
                         menu_audit=menu_audit, phases=phase_results)
        if plan['length'] == 2:
            assert plan['vocab'] == VOCAB
            support = set(code_ids(messages['calibration'][train_maps], VOCAB).flatten().tolist())
            assignment, candidates = choose_assignment(messages['calibration'], decoder, VOCAB, train_maps, support)
            direction['fragments'] = dict(
                selection_unit='one assignment per communication direction',
                selection_map_ids=train_maps.tolist(), selection_metric='unfiltered strict transfer rate',
                calibration_candidates=candidates, selected_assignment_food_water_slots=list(assignment),
                validation=fragment_report(messages['validation'], decoder, VOCAB, assignment, np.arange(len(MAPS)), support),
                validation_seen_map_pairs=fragment_report(messages['validation'], decoder, VOCAB, assignment, train_maps, support),
                greedy_calibration_message_support=sorted(support))
            if len(heldout_maps):
                direction['heldout_stitch_validation'] = heldout_stitch_report(
                    messages['validation'], decoder, VOCAB, assignment, support, train_maps, heldout_maps)
            recoding_seed = 910000000 + seed * 100 + 10 * list(CONDITIONS).index(config['condition']) + scout
            direction['whole_message_recoding'] = whole_message_recoding_reference(
                messages, decoder, assignment, train_maps, seed=recoding_seed, heldout_maps=heldout_maps)
            assert direction['whole_message_recoding']['strict_transfer_all']['observed'] == direction['fragments']['validation']['strict_transfer_all']['rate']
        directions.append(direction)
    return dict(seed=seed, condition=config['condition'], plan=plan,
                train_map_ids=train_maps.tolist(), heldout_map_ids=heldout_maps.tolist(),
                final_checkpoint=str(final_path), final_sha256=sha(final_path),
                prepared_checkpoint=str(prepared_path), prepared_sha256=sha(prepared_path), directions=directions)


def run_synthetic_checks():
    decoder = np.empty((CAPACITY, 2, 1), np.int64)
    for first, second in product(range(VOCAB), repeat=2):
        decoder[VOCAB * first + second, 0] = min(first, SITES - 1)
        decoder[VOCAB * first + second, 1] = min(second, SITES - 1)
    messages = np.tile(MAPS[:, None, None, :], (1, 16, 2, 1))
    support = set(code_ids(messages, VOCAB).flatten().tolist())
    cross = cross_goal_report(messages, decoder, VOCAB)
    assert cross['same_message_correct_for_both_goals']['rate'] == 1
    assert cross['native_goal']['denominator'] == 30 * 16 * 2
    for split in (1, 2, 3):
        train, heldout = split_maps(split)
        assignment, candidates = choose_assignment(messages, decoder, VOCAB, train, support)
        assert assignment == (0, 1)
        assert candidates[0]['strict_transfer_all']['rate'] == 1
        assert candidates[1]['strict_transfer_all']['rate'] == 0
        assert candidates[0]['strict_transfer_all']['denominator'] == 4608
        full = fragment_report(messages, decoder, VOCAB, assignment, np.arange(30), support)
        assert full['strict_transfer_all']['denominator'] == 7680
        stitched = heldout_stitch_report(messages, decoder, VOCAB, assignment, support, train, heldout)
        assert stitched['both_goals_correct_all']['rate'] == 1
        assert stitched['both_goals_correct_all']['denominator'] == 3072
        assert cross_goal_report(messages, decoder, VOCAB, heldout)['native_goal']['denominator'] == 192
    constant = np.zeros_like(messages)
    null = fragment_report(constant, decoder, VOCAB, (0, 1), np.arange(30), {0})
    assert null['strict_transfer_all']['rate'] == 0
    assert null['strict_transfer_when_both_maps_both_goals_correct']['rate'] is None
    train, heldout = split_maps(1)
    recoding = whole_message_recoding_reference(dict(calibration=messages, validation=messages),
        decoder, (0, 1), train, seed=910009971, replicates=4, heldout_maps=heldout)
    assert recoding['strict_transfer_all']['observed'] == 1
    assert recoding['strict_transfer_all']['reference_mean'] < 1
    assert recoding['heldout_stitch_both_goals_correct_all']['observed'] == 1
    assert all(sorted(row['old_to_new_code']) == list(range(CAPACITY)) for row in recoding['records'])
    # Receiver enumeration checks the actual six-site API with random test weights.
    from camp import CampAgent
    from torch import nn
    torch.manual_seed(910009973)
    agent = CampAgent(nn.Sequential(nn.Linear(1024, 64), nn.Tanh()))
    actual_decoder, _, audit = receiver_table(agent)
    assert audit['cases'] == 49 * 2 * 720
    assert audit['all_menu_permutations_physically_equivalent']
    assert actual_decoder.shape == (49, 2, 1)
    return dict(status='passed', cases=['all three matching splits', 'training-only token assignment',
        'oracle full and heldout counts', 'zero-denominator null', '49-code bijection invariance',
        'all 720 menus checked through actual receiver before collapse'])


def percentage(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def count_text(value):
    return '不适用' if value is None else f"{percentage(value['rate'])} ({value['numerator']}/{value['denominator']})"


def markdown(result):
    lines = ['# v0.7 视觉接口、组合留出与完整码重编码分析', '',
        f"完成运行：{len(result['runs'])}；计划中尚未纳入：{len(result['missing_runs'])}。状态：{result['status']}。", '',
        '本次52个运行均使用双7符号，结构与49码重编码评价在v0.7训练前固定。旧test每类前4张为分析校准、后4张为验证，每组16个照片对；这些旧照片已经用于既往探索，不是新图像外部确认。', '',
        '符号位置只在每个划分的24张训练地图校准；split0使用全部30图。验证在两种目标和两种通信方向进行。每个49码、两个目标均完整检查720种菜单的物理动作等变性，完全一致后才折叠重复菜单。病例数不是独立训练样本量。', '',
        '本批发送者均不知道目标，因此均匀切换目标后的平均正确率与原目标平均值相等并不单独证明复用。主要结合“同一消息对两目标同时正确”、单片段的严格动作改变、训练地图供体拼接和完整码重编码基准判断。', '',
        '| 种子 | 条件 | 方向 | 训练图自然正确 | 留出图自然正确 | 留出图同消息双目标正确 | 严格片段（全30图） | 训练供体拼接到留出图双目标正确 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for run in result['runs']:
        for direction in run['directions']:
            groups = direction['phases']['validation']['cross_goal_groups']
            unseen = groups['unseen']
            fragment = direction.get('fragments', {}).get('validation', {}).get('strict_transfer_all')
            stitch = direction.get('heldout_stitch_validation', {}).get('both_goals_correct_all')
            lines.append(f"| {run['seed']} | {run['condition']} | {direction['scout']}→{direction['collector']} | "
                f"{percentage(groups['seen']['native_goal']['rate'])} | "
                f"{percentage(unseen['native_goal']['rate']) if unseen else '不适用'} | "
                f"{percentage(unseen['same_message_correct_for_both_goals']['rate']) if unseen else '不适用'} | "
                f"{count_text(fragment)} | {count_text(stitch)} |")
    lines += ['', 'identity、permute和orthogonal使用相同双7符号信道；只在共享非线性编码前改变固定视觉槽变换。Q在该非线性之前可逆，不能因此断言整个学得接口信息无损。每种子三个全地图接口和一个full_blocked均只计为一个split0运行，不复制到三个留出划分。', '',
        '## 保持原生行为的完整消息重编码', '',
        '每个两符号运行的每个方向100次49码双射，同步使用接收逆映射，保持原生/跨目标动作与完整消息频率。每次重新在原训练地图校准，固定后评分验证照片；这是假想重编码协议，不是重训网络。', '',
        '| 种子 | 条件 | 方向 | 严格迁移观察值 | 重编码均值 | 2.5%–97.5%分位 | 观察值减均值 | 拼接观察值 | 拼接重编码均值 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for run in result['runs']:
        for direction in run['directions']:
            if 'whole_message_recoding' not in direction:
                continue
            ref = direction['whole_message_recoding']
            strict = ref['strict_transfer_all']; q = strict['reference_quantiles']
            stitch = ref.get('heldout_stitch_both_goals_correct_all')
            lines.append(f"| {run['seed']} | {run['condition']} | {direction['scout']}→{direction['collector']} | "
                f"{percentage(strict['observed'])} | {percentage(strict['reference_mean'])} | "
                f"{percentage(q['0.025'])}–{percentage(q['0.975'])} | {100*strict['observed_minus_reference_mean']:.2f}个百分点 | "
                f"{percentage(stitch['observed']) if stitch else '不适用'} | {percentage(stitch['reference_mean']) if stitch else '不适用'} |")
    lines += ['', '全30图片段配对可能包含留出地图自然消息，因此它本身不是只用训练供体的新组合测试；独立的拼接指标仅使用训练地图供体。JSON分别保留全部配对、训练图内配对、自然消息正确子集及零分母。', '',
        '参考分位是给定协议的重编码分布，不是四个训练种子的置信区间。多个划分、两个方向与720种菜单都不能当作独立种子。跨划分汇总应先在每个种子内等权平均，再比较四个种子；单个成功协议只作为有限环境的例子，不命名为语法。']
    if result['missing_runs']:
        lines += ['', '尚未纳入：' + '、'.join(result['missing_runs']) + '。']
    return '\n'.join(lines) + '\n'


def cached_analysis(path, batch, bank, splits, fingerprint):
    cache = batch / 'protocol_cache'
    cache.mkdir(exist_ok=True)
    target = cache / f'{path.name}.json'
    identity = dict(fingerprint, config=sha(path / 'config.json'), final=sha(path / 'final.pt'),
                    prepared=sha(batch / f"prepared_{json.loads((path / 'config.json').read_text())['seed']}.pt"))
    if target.exists():
        previous = json.loads(target.read_text())
        if previous['fingerprint'] == identity:
            return previous['run'], True
    run = analyze_run(path, batch, bank, splits)
    target.write_text(json.dumps(dict(fingerprint=identity, run=run), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Preserve the exact analysis implementation used for these results.
    snapshot = cache / f"analysis_{fingerprint['analysis']}.py"
    if not snapshot.exists():
        snapshot.write_text(Path(__file__).read_text(), encoding='utf-8')
    return run, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path, nargs='?', default=ROOT / 'results/binding_001')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--validate-run', help='Analyze/cache one completed flat run, reporting interface checks only')
    parser.add_argument('--full-only', action='store_true', help='Incremental split0 subset; final analysis reuses these run caches')
    args = parser.parse_args()
    torch.set_num_threads(1)
    checks = run_synthetic_checks()
    if args.self_test:
        print(json.dumps(checks, ensure_ascii=False, indent=2)); return
    batch = args.batch.resolve()
    expected = set()
    for invocation_path in batch.glob('invocation_*.json'):
        invocation = json.loads(invocation_path.read_text())
        expected.update(product(invocation['seeds'], invocation['conditions']))
    if not expected:
        expected = set(product(SEEDS, CONDITIONS))
    expected_order = sorted(expected, key=lambda pair: (pair[0], list(CONDITIONS).index(pair[1])))
    selected, missing = [], []
    for seed, condition in expected_order:
        path = batch / f's{seed}_{condition}'
        eligible = not args.full_only or int(CONDITIONS[condition]['split']) == 0
        complete = all((path / name).exists() for name in ('config.json', 'result.json', 'final.pt'))
        if eligible and complete:
            selected.append(path)
        else:
            missing.append(path.name)
    if args.validate_run:
        selected = [batch / args.validate_run]
        assert all((selected[0] / name).exists() for name in ('config.json', 'result.json', 'final.pt'))
    if not selected:
        raise SystemExit('No completed requested runs; no analysis result written')
    bank = ImageBank()
    assert all(len(bank.pools['test', kind]) == 8 for kind in range(2))
    splits = {phase: np.asarray(list(product(bank.pools['test', 0][subset], bank.pools['test', 1][subset])), dtype=np.int64)
              for phase, subset in [('calibration', slice(0, 4)), ('validation', slice(4, 8))]}
    assert not (set(splits['calibration'].flatten()) & set(splits['validation'].flatten()))
    fingerprint = dict(analysis=sha(__file__), camp=sha(ROOT/'camp.py'),
        features=sha(ROOT.parent/'redesign_v0.4/data/features.npz'),
        photo_manifest=sha(ROOT.parent/'redesign_v0.4/data/manifest.json'))
    result = dict(status='partial' if missing else 'complete', batch=str(batch),
        expected_seeds=sorted({seed for seed, condition in expected}), expected_run_count=len(expected),
        analysis_plan=ANALYSIS_PLAN, fingerprints=fingerprint, synthetic_checks=checks,
        photo_splits={phase: pairs.tolist() for phase, pairs in splits.items()},
        statistical_unit='training seed; split0 controls appear once per seed', missing_runs=missing, runs=[])
    for path in selected:
        run, reused = cached_analysis(path, batch, bank, splits, fingerprint)
        result['runs'].append(run)
        if args.validate_run:
            print(json.dumps(dict(status='interface_validation_passed', run=path.name,
                directions=len(run['directions']), menu_checks=[d['menu_audit'] for d in run['directions']],
                calibration_maps=len(run['train_map_ids']), heldout_maps=len(run['heldout_map_ids']),
                recoding_replicates=[d.get('whole_message_recoding', {}).get('replicates') for d in run['directions']],
                cache_reused=reused), ensure_ascii=False, indent=2))
            return
        print(('Reused ' if reused else 'Analyzed ') + path.name, flush=True)
    prefix = 'protocol_analysis_full_only' if args.full_only else 'protocol_analysis'
    (batch / f'{prefix}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (batch / f'{prefix}.md').write_text(markdown(result), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], runs=len(result['runs']), missing=len(missing),
                         output=str(batch / f'{prefix}.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
