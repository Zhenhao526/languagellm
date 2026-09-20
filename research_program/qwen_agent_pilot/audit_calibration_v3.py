"""Replay and validate a completed v3 Qwen channel-calibration matrix."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .calibration import (
    CALIBRATION_VERSION,
    CODEBOOK,
    CONDITIONS,
    DEFAULT_SEEDS,
    _condition_order,
    _summarize,
)
from .environment import PILOT_EPISODES, make_episode, score_episode
from .pilot import ALPHABET, MAX_MESSAGE_LENGTH, MIN_MESSAGE_LENGTH, valid_message


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _same(label: str, actual: object, expected: object) -> None:
    if isinstance(actual, bool) or isinstance(expected, bool):
        _require(actual is expected, f"{label}: {actual!r} != {expected!r}")
    elif isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        _require(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12),
                 f"{label}: {actual!r} != {expected!r}")
    elif isinstance(actual, dict) and isinstance(expected, dict):
        _require(actual.keys() == expected.keys(), f"{label}: dictionary keys differ")
        for key in expected:
            _same(f"{label}.{key}", actual[key], expected[key])
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        _require(len(actual) == len(expected), f"{label}: list lengths differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _same(f"{label}[{index}]", left, right)
    else:
        _require(actual == expected, f"{label}: {actual!r} != {expected!r}")


def audit_matrix(matrix: dict) -> dict:
    _require(matrix.get("calibration_version") == CALIBRATION_VERSION,
             "unexpected calibration version")
    seeds = matrix.get("seeds")
    _require(seeds == list(DEFAULT_SEEDS), f"unexpected paired seeds: {seeds!r}")
    _require(matrix.get("conditions") == list(CONDITIONS), "condition list differs from frozen protocol")
    _require(matrix.get("model_revision") == "16daa4818c54ce5f5436f929d52542eb65bbed9d",
             "model revision differs from the pinned revision")
    _require(matrix.get("temperature") == 0.35 and matrix.get("max_tokens") == 120,
             "generation settings differ from the frozen protocol")
    _require(matrix.get("message_alphabet") == "@#%&+=?~", "message alphabet differs")
    _require(matrix.get("message_length") == [MIN_MESSAGE_LENGTH, MAX_MESSAGE_LENGTH],
             "message length differs")
    _require(matrix.get("known_codebook") == {str(key): value for key, value in CODEBOOK.items()},
             "fixed codebook differs")
    _require(matrix.get("paired_decoding_seeds") is True, "paired decoding seeds were not recorded")
    _require(matrix.get("raw_model_completions_retained") is False,
             "raw model completions should not be retained")
    _require(matrix.get("chain_of_thought_retained") is False,
             "hidden reasoning should not be retained")
    expected_order = {
        str(seed): _condition_order(index) for index, seed in enumerate(seeds)
    }
    _same("condition_order_by_seed", matrix.get("condition_order_by_seed"), expected_order)

    runs = matrix.get("runs", [])
    expected_pairs = {(seed, condition) for seed in seeds for condition in CONDITIONS}
    observed_pairs = [(run.get("seed"), run.get("condition")) for run in runs]
    _require(len(runs) == len(expected_pairs), f"expected 12 runs; found {len(runs)}")
    _require(len(set(observed_pairs)) == len(observed_pairs), "duplicate seed-condition run")
    _require(set(observed_pairs) == expected_pairs, "seed-condition matrix is incomplete")
    sorted_pairs = [(seed, condition) for seed in seeds for condition in CONDITIONS]
    _require(observed_pairs == sorted_pairs, "archived run order is not canonical")

    records_by_run: dict[tuple[int, str], list[dict]] = {}
    totals = defaultdict(lambda: {
        "episodes": 0,
        "joint_successes": 0,
        "designated_item_correct": 0,
        "designated_destination_correct": 0,
        "designated_helper_both_correct": 0,
        "unassigned_waited": 0,
        "valid_messages": 0,
        "message_count": 0,
        "codebook_exact_encodings": 0,
        "helper_actions": 0,
        "helper_action_schemas_valid": 0,
        "owner_json_valid": 0,
        "owner_calls": 0,
        "helper_json_valid": 0,
    })
    replayed_episodes = 0
    summary_recomputations = 0

    for run in runs:
        seed, condition = run["seed"], run["condition"]
        key = (seed, condition)
        records = run.get("records", [])
        _require(run.get("episodes") == PILOT_EPISODES,
                 f"{key}: expected {PILOT_EPISODES} episodes")
        _require(run.get("model_calls") == 3 * PILOT_EPISODES,
                 f"{key}: unexpected model call count")
        _require(len(records) == PILOT_EPISODES, f"{key}: record count differs")
        _require([record.get("episode") for record in records] == list(range(PILOT_EPISODES)),
                 f"{key}: episode indices are incomplete or duplicated")
        records_by_run[key] = records

        for index, record in enumerate(records):
            episode = make_episode(index, seed)
            schedule_fields = {
                "episode": episode["episode"],
                "block": episode["block"],
                "meaning_id": episode["meaning_id"],
                "owner": episode["owner"],
                "goal": episode["goal"],
                "public_board": episode["scene"],
                "target_item_id": episode["target_item_id"],
                "destinations": episode["destinations"],
            }
            for field, expected in schedule_fields.items():
                _same(f"{key} episode {index} {field}", record.get(field), expected)

            message = record.get("message")
            if condition in {"blank", "oracle_decoded"}:
                _require(message == "", f"{key} episode {index}: closed channel carried content")
                _require(record.get("message_valid") is None,
                         f"{key} episode {index}: closed-channel message was scored as a symbol")
                _require(record.get("encoder_correct") is None,
                         f"{key} episode {index}: closed-channel encoder was scored")
            else:
                is_valid = valid_message(message)
                _require(record.get("message_valid") is is_valid,
                         f"{key} episode {index}: recorded symbol validity differs")
                _require(all(char in ALPHABET for char in message),
                         f"{key} episode {index}: out-of-alphabet content crossed the channel")
                if condition == "known_codebook":
                    encoder_correct = message == CODEBOOK[episode["meaning_id"]]
                    _require(record.get("encoder_correct") is encoder_correct,
                             f"{key} episode {index}: codebook encoder score differs")
                else:
                    _require(record.get("encoder_correct") is None,
                             f"{key} episode {index}: free symbols cannot use codebook score")

            _require(isinstance(record.get("owner_json_valid"), bool),
                     f"{key} episode {index}: owner JSON validity is not boolean")
            helpers = set(episode["helpers"])
            actions = record.get("actions", {})
            helper_results = record.get("helper_results", {})
            _require(set(actions) == helpers and set(helper_results) == helpers,
                     f"{key} episode {index}: helper identities differ from schedule")
            for helper in helpers:
                result = helper_results[helper]
                _require(isinstance(result.get("json_valid"), bool)
                         and isinstance(result.get("action_valid"), bool),
                         f"{key} episode {index} helper {helper}: missing schema flags")
                action = actions[helper]
                if action is not None:
                    _require(isinstance(action, dict) and set(action) == {"item_id", "destination"},
                             f"{key} episode {index} helper {helper}: malformed canonical action")
                    _require(action["item_id"] in {row["item_id"] for row in episode["scene"]},
                             f"{key} episode {index} helper {helper}: unknown item id")
                    _require(isinstance(action["destination"], str),
                             f"{key} episode {index} helper {helper}: destination is not a string")

            expected_outcome = score_episode(episode, actions)
            _same(f"{key} episode {index} outcome", record.get("outcome"), expected_outcome)
            replayed_episodes += 1
            total = totals[condition]
            total["episodes"] += 1
            total["joint_successes"] += int(expected_outcome["success"])
            total["designated_item_correct"] += int(expected_outcome["designated_item_correct"])
            total["designated_destination_correct"] += int(expected_outcome["designated_destination_correct"])
            total["designated_helper_both_correct"] += int(expected_outcome["designated_helper_correct"])
            total["unassigned_waited"] += int(expected_outcome["unassigned_helper_waited"])
            total["owner_json_valid"] += int(record["owner_json_valid"])
            total["owner_calls"] += 1
            if condition in {"known_codebook", "free_symbols"}:
                total["message_count"] += 1
                total["valid_messages"] += int(record["message_valid"] is True)
                total["codebook_exact_encodings"] += int(record["encoder_correct"] is True)
            for helper_result in helper_results.values():
                total["helper_actions"] += 1
                total["helper_action_schemas_valid"] += int(helper_result["action_valid"])
                total["helper_json_valid"] += int(helper_result["json_valid"])

        recomputed = _summarize(
            condition,
            seed,
            records,
            run["wall_time_seconds"],
            run["prompt_tokens_reported_by_server"],
            run["completion_tokens_reported_by_server"],
        )
        _same(f"{key} summary", {k: v for k, v in run.items() if k != "records"},
              {k: v for k, v in recomputed.items() if k != "records"})
        summary_recomputations += 1

    paired_schedule_comparisons = 0
    for seed in seeds:
        reference = records_by_run[(seed, CONDITIONS[0])]
        for condition in CONDITIONS[1:]:
            compared = records_by_run[(seed, condition)]
            for ref_record, record in zip(reference, compared):
                for field in ("episode", "block", "meaning_id", "owner", "goal",
                              "public_board", "target_item_id", "destinations"):
                    _same(f"paired schedule seed {seed} condition {condition} {field}",
                          record[field], ref_record[field])
                paired_schedule_comparisons += 1

    expected_contrasts = [
        {
            "seed": seed,
            "success_rates": {
                condition: next(run for run in runs
                               if run["seed"] == seed and run["condition"] == condition)["team_success_rate"]
                for condition in CONDITIONS
            },
            "free_minus_blank_success_rate": (
                next(run for run in runs if run["seed"] == seed and run["condition"] == "free_symbols")["team_success_rate"]
                - next(run for run in runs if run["seed"] == seed and run["condition"] == "blank")["team_success_rate"]
            ),
            "free_minus_codebook_success_rate": (
                next(run for run in runs if run["seed"] == seed and run["condition"] == "free_symbols")["team_success_rate"]
                - next(run for run in runs if run["seed"] == seed and run["condition"] == "known_codebook")["team_success_rate"]
            ),
            "free_minus_oracle_success_rate": (
                next(run for run in runs if run["seed"] == seed and run["condition"] == "free_symbols")["team_success_rate"]
                - next(run for run in runs if run["seed"] == seed and run["condition"] == "oracle_decoded")["team_success_rate"]
            ),
        }
        for seed in seeds
    ]
    _same("paired_seed_contrasts", matrix.get("paired_seed_contrasts"), expected_contrasts)

    return {
        "status": "passed",
        "calibration_version": CALIBRATION_VERSION,
        "protocol_source_commit": matrix.get("protocol_source_commit"),
        "model_revision": matrix["model_revision"],
        "seeds": seeds,
        "conditions": list(CONDITIONS),
        "runs_checked": len(runs),
        "episodes_replayed": replayed_episodes,
        "paired_schedule_comparisons": paired_schedule_comparisons,
        "run_summaries_recomputed": summary_recomputations,
        "raw_model_completions_retained": matrix["raw_model_completions_retained"],
        "chain_of_thought_retained": matrix["chain_of_thought_retained"],
        "condition_totals": dict(totals),
        "free_symbol_seed_metrics": [
            {
                "seed": run["seed"],
                "successes": round(run["team_success_rate"] * run["episodes"]),
                "valid_messages": round(run["valid_message_rate_nonblank_channels"] * run["episodes"]),
                "unique_messages": run["unique_messages"],
                "sender_meaning_exact_repeat_rate": run["sender_meaning_exact_repeat_rate"],
                "cross_sender_agreement_final_block": run["cross_sender_agreement_final_block"],
                "message_owner_mutual_information_bits": run["message_owner_mutual_information_bits"],
                "message_meaning_mutual_information_bits": run["message_meaning_mutual_information_bits"],
            }
            for run in runs if run["condition"] == "free_symbols"
        ],
        "known_codebook_seed_metrics": [
            {
                "seed": run["seed"],
                "successes": round(run["team_success_rate"] * run["episodes"]),
                "exact_encodings": round(run["known_codebook_encoder_accuracy"] * run["episodes"]),
            }
            for run in runs if run["condition"] == "known_codebook"
        ],
        "oracle_seed_metrics": [
            {
                "seed": run["seed"],
                "successes": round(run["team_success_rate"] * run["episodes"]),
            }
            for run in runs if run["condition"] == "oracle_decoded"
        ],
        "paired_seed_contrasts": expected_contrasts,
        "screening_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results_path = Path(args.results)
    matrix = json.loads(results_path.read_text())
    report = audit_matrix(matrix)
    report["results_path"] = str(results_path)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "runs_checked": report["runs_checked"],
        "episodes_replayed": report["episodes_replayed"],
        "paired_schedule_comparisons": report["paired_schedule_comparisons"],
        "out": args.out,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
