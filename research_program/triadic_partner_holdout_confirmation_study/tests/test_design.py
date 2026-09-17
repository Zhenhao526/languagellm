from research_program.triadic_partner_holdout_confirmation_study import design


def test_pair_rotation_and_counts():
    prepared = design.make_prepared()
    assert prepared['support_count'] == 5376
    assert prepared['train_need_count'] == 3584
    assert prepared['heldout_need_count'] == 1792
    assert sorted(set(prepared['heldout_pair_by_seed'].values())) == [0, 1, 2]
    assert len(prepared['runs']) == 128


def test_seed_partition_shapes():
    for seed in design.SEEDS:
        parts = design.build_seed_partitions(seed)
        assert parts['train']['world_count'] == 3584 * 18 * 6
        assert parts['heldout_pair']['world_count'] == 1792 * 18 * 6
        assert parts['heldout_pair_new_layout']['world_count'] == 1792 * 6 * 6
        assert parts['seen_control_new_layout']['world_count'] == 3584 * 6 * 6


def test_conditions_are_paired():
    assert len(design.CONDITIONS) == 8
    assert set(design.parse_condition(c)[0] for c in design.CONDITIONS) == {'partial', 'all_or_nothing'}
