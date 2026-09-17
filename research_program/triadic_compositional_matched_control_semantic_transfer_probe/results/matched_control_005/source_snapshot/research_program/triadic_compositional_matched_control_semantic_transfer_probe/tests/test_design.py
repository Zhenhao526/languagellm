from research_program.triadic_compositional_matched_control_semantic_transfer_probe import design


def test_condition_grid_and_sources():
    assert len(design.CONDITIONS) == 8
    assert len(design.make_prepared()["sources"]) == 64


def test_matched_source_families_share_support_metadata():
    static = design.make_prepared()
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            for live in design.LIVES:
                seen = static["sources"][f"{seed}:seen_joint_only:{design.condition('seen_joint_only', schedule, live)}"]
                full = static["sources"][f"{seed}:all_joint:{design.condition('all_joint', schedule, live)}"]
                assert seen["seed"] == full["seed"] == seed
                assert seen["schedule"] == full["schedule"] == schedule
                assert seen["live"] == full["live"] == live
