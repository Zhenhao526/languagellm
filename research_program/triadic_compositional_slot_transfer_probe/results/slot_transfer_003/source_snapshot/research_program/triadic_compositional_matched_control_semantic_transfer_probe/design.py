"""Frozen matched-control sources and heldout semantic-transfer cases."""
from __future__ import annotations

import json
from pathlib import Path

from research_program.triadic_compositional_holdout_semantic_transfer_probe import design as case_design

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(66701, 66709))
ARMS = ("seen_joint_only", "all_joint")
SCHEDULES = ("static", "rematched")
LIVES = (True, False)
CONDITIONS = tuple(
    f"{arm}_{schedule}_PL_{'live' if live else 'silent'}"
    for arm in ARMS for schedule in SCHEDULES for live in LIVES
)
CHUNK_SIZE = 2048
SHAM_ROWS_PER_SENDER = 128
SEEN_ROOT = ROOT / "triadic_new_receiver_compositional_holdout_study/results/holdout_001"
ALL_ROOT = ROOT / "triadic_protocol_chain_study/results/chain_001"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def condition(arm, schedule, live):
    require(arm in ARMS and schedule in SCHEDULES, "Unknown source condition")
    return f"{arm}_{schedule}_PL_{'live' if live else 'silent'}"


def source_path(seed, arm, schedule, live):
    if arm == "seen_joint_only":
        name = f"seed_{seed}_{schedule}_new_receiver_compositional_seen_joint_only_PL_{'live' if live else 'silent'}"
        return SEEN_ROOT / "execution" / name
    name = f"seed_{seed}_generation2_replace_A_{schedule}_PL_{'live' if live else 'silent'}"
    return ALL_ROOT / "execution" / name


def source_result(seed, arm, schedule, live):
    path = source_path(seed, arm, schedule, live) / "result.json"
    require(path.is_file(), f"Missing source result: {path}")
    return json.loads(path.read_text(encoding="utf8")), path


def make_prepared():
    base = case_design.make_prepared()
    sources = {}
    for seed in SEEDS:
        for arm in ARMS:
            for schedule in SCHEDULES:
                for live in LIVES:
                    result, result_path = source_result(seed, arm, schedule, live)
                    checkpoint = result_path.parent / "checkpoint_6000.npz"
                    require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
                    require(result["final_checkpoint_sha256"], "Missing checkpoint digest")
                    sources[f"{seed}:{arm}:{condition(arm, schedule, live)}"] = dict(
                        seed=seed, arm=arm, schedule=schedule, live=bool(live),
                        condition=condition(arm, schedule, live),
                        checkpoint=str(checkpoint.resolve()), result=str(result_path.resolve()),
                        checkpoint_sha256=result["final_checkpoint_sha256"],
                    )
    return dict(
        schema="triadic_compositional_matched_control_semantic_transfer_prepared_v1",
        seeds=list(SEEDS), arms=list(ARMS), schedules=list(SCHEDULES), lives=["live", "silent"],
        conditions=list(CONDITIONS), heldout_spec=base["heldout_spec"], cases=base["cases"],
        sources=sources, case_source="triadic_compositional_holdout_semantic_transfer_probe",
        scientific_question="Does the heldout aligned-minus-placebo transfer deficit survive a matched full-combination receiver control?",
        primary="all-joint A minus seen-joint-only A aligned-minus-placebo plan transfer",
        matching="same parent B/C endpoint, same fresh A initialization, same world/message/rematch streams; only A training support differs",
        no_training_updates=True,
        claim_boundary="This is a behavioral matched-control diagnostic, not lexical, compositional-syntax or language-origin evidence.",
    )


if __name__ == "__main__":
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
