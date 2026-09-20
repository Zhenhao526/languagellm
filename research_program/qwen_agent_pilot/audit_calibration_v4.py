"""Replay and integrity audit for the v4 Qwen calibration records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .calibration_v4 import (
    CALIBRATION_VERSION, CODEBOOK, CONDITIONS, MAX_MESSAGE_LENGTH,
    MIN_MESSAGE_LENGTH, MODEL_REVISION, summarize,
)
from .factorized_environment import (
    AGENTS, ATTRIBUTES, DESTINATIONS, OBJECTS, TOTAL_EPISODES,
    make_episode, score_episode,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _eq(left, right, path: str, errors: list[str]) -> None:
    if left != right:
        errors.append(f"{path}: stored value does not match deterministic replay")


def _audit_run(condition: str, seed: int, episodes: list[dict], summary: dict,
               errors: list[str]) -> dict:
    if condition not in CONDITIONS:
        errors.append(f"unknown condition {condition}")
    if len(episodes) != TOTAL_EPISODES:
        errors.append(f"{condition}/{seed}: expected {TOTAL_EPISODES} episodes, got {len(episodes)}")

    paired_rows = []
    for index, row in enumerate(episodes):
        expected = make_episode(index, seed)
        for key in ("episode", "block", "owner", "meaning_id", "payload_id", "helpers",
                    "goal", "scene", "target_item_id", "destinations"):
            stored_key = "public_board" if key == "scene" else key
            expected_value = list(expected[key]) if key in {"helpers", "destinations"} else expected[key]
            _eq(row.get(stored_key), expected_value,
                f"{condition}/{seed}/episode {index}/{stored_key}", errors)
        paired_rows.append((row.get("episode"), row.get("block"), row.get("owner"),
                            row.get("meaning_id"), row.get("public_board")))

        actions = row.get("actions", {})
        _eq(set(actions), set(expected["helpers"]),
            f"{condition}/{seed}/episode {index}/action helper ids", errors)
        replayed = score_episode(expected, actions)
        _eq(row.get("outcome"), replayed, f"{condition}/{seed}/episode {index}/outcome", errors)

        message = row.get("message")
        if condition == "oracle_decoded":
            if message != "" or row.get("message_valid") is not True:
                errors.append(f"{condition}/{seed}/episode {index}: oracle must have an empty channel")
        else:
            valid = (
                isinstance(message, str)
                and MIN_MESSAGE_LENGTH <= len(message) <= MAX_MESSAGE_LENGTH
                and all(char in "@#%&+=?~" for char in message)
            )
            if row.get("message_valid") is not valid:
                errors.append(f"{condition}/{seed}/episode {index}: message validity flag mismatch")
            if not valid and message != "":
                errors.append(f"{condition}/{seed}/episode {index}: invalid signal was not suppressed")
        if condition == "known_codebook":
            _eq(row.get("codebook_correct"), message == CODEBOOK[expected["meaning_id"]],
                f"{condition}/{seed}/episode {index}/codebook accuracy", errors)

        outputs = row.get("helper_outputs", {})
        _eq(set(outputs), set(expected["helpers"]),
            f"{condition}/{seed}/episode {index}/helper outputs", errors)
        for agent, output in outputs.items():
            guess = output.get("interpretation", {})
            target = {
                "object": expected["goal"]["object"],
                "attribute": expected["goal"]["attribute"],
                "responsible_helper": expected["goal"]["partner"],
                "destination": expected["goal"]["destination"],
            }
            for field, value in guess.items():
                domain = {
                    "object": OBJECTS, "attribute": ATTRIBUTES,
                    "responsible_helper": AGENTS, "destination": DESTINATIONS,
                }.get(field)
                if domain is None or (value is not None and value not in domain):
                    errors.append(f"{condition}/{seed}/episode {index}/{agent}: invalid interpretation field")
            feedback = row.get("feedback", {}).get(agent)
            if not isinstance(feedback, dict):
                errors.append(f"{condition}/{seed}/episode {index}/{agent}: missing environment feedback")
            elif condition.endswith("component"):
                public = feedback.get("public_component_feedback", {})
                if "should_act" in json.dumps(public):
                    errors.append(f"{condition}/{seed}/episode {index}: component feedback leaks direct role label")
                _eq(public, {
                    helper: {
                        "responsibility_action_correct": detail["responsibility_correct"],
                        "item_matches": detail["item_correct"],
                        "destination_matches": detail["destination_correct"],
                    } for helper, detail in replayed["helper_feedback"].items()
                }, f"{condition}/{seed}/episode {index}/component feedback", errors)
            elif "public_component_feedback" in feedback:
                errors.append(f"{condition}/{seed}/episode {index}: scalar feedback contains component scores")

    expected_summary = summarize(condition, seed, episodes,
                                 summary.get("wall_time_seconds", 0.0),
                                 summary.get("prompt_tokens_reported_by_server", 0),
                                 summary.get("completion_tokens_reported_by_server", 0))
    checked_summary_keys = (
        "episodes", "model_calls", "joint_successes", "designated_item_correct",
        "designated_destination_correct", "unassigned_waited", "owner_json_valid",
        "owner_message_valid", "helper_json_valid", "helper_interpretation_valid",
        "helper_action_valid", "exact_codebook_encodings", "helper_field_accuracy",
        "exact_helper_interpretations", "helper_interpretations", "distinct_nonempty_messages",
        "message_entropy_bits", "message_meaning_mutual_information_bits",
        "message_payload_mutual_information_bits", "message_partner_mutual_information_bits",
        "message_requester_mutual_information_bits", "within_owner_repeat_exact",
        "within_owner_repeat_groups", "final_block_shared_form_groups",
        "final_block_semantic_groups", "by_block",
    )
    for key in checked_summary_keys:
        if summary.get(key) != expected_summary.get(key):
            # Float arithmetic is deterministic here, so exact equality is expected.
            errors.append(f"{condition}/{seed}/summary/{key}: summary does not replay")
    return {"condition": condition, "seed": seed, "episodes_checked": len(episodes),
            "paired_rows": paired_rows}


def audit(results_path: Path, audit_path: Path) -> dict:
    source = json.loads(results_path.read_text())
    errors: list[str] = []
    if source.get("calibration_version") != CALIBRATION_VERSION:
        errors.append("calibration version does not match v4")
    if source.get("model_revision") != MODEL_REVISION:
        errors.append("model revision does not match pinned v4 revision")

    if "runs" in source:
        runs = source["runs"]
    else:
        runs = [{"summary": source.get("summary"), "episodes": source.get("episodes", [])}]
    checked = []
    by_seed: dict[int, dict[str, list[tuple]]] = {}
    for run in runs:
        summary = run.get("summary") or {}
        condition, seed = summary.get("condition"), summary.get("seed")
        result = _audit_run(condition, seed, run.get("episodes", []), summary, errors)
        checked.append({key: value for key, value in result.items() if key != "paired_rows"})
        by_seed.setdefault(seed, {})[condition] = result["paired_rows"]

    for seed, condition_rows in by_seed.items():
        if len(condition_rows) < 2:
            continue
        baseline_condition = next(iter(condition_rows))
        baseline = condition_rows[baseline_condition]
        for condition, rows in condition_rows.items():
            if rows != baseline:
                errors.append(f"seed {seed}: episode schedule differs between {baseline_condition} and {condition}")

    status = "passed" if not errors else "failed"
    audit_record = {
        "status": status,
        "calibration_version": CALIBRATION_VERSION,
        "results_file": str(results_path),
        "results_sha256": _sha256(results_path),
        "runs_checked": len(checked),
        "episodes_replayed": sum(run["episodes_checked"] for run in checked),
        "run_checks": checked,
        "paired_seed_count": len(by_seed),
        "raw_model_completions_retained": False,
        "chain_of_thought_retained": False,
        "errors": errors,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_record, ensure_ascii=False, indent=2) + "\n")
    return audit_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = audit(args.results, args.out)
    print(json.dumps({key: record[key] for key in
                      ("status", "runs_checked", "episodes_replayed", "errors")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
