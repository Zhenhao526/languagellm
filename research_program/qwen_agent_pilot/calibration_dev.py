"""Run the v2 known-codebook development gate, excluded from paired analyses."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .calibration import CALIBRATION_VERSION, _git_commit, run_condition
from .pilot import MODEL_ID

DEVELOPMENT_SEED = 20260928
THRESHOLDS = {
    "known_codebook_encoder_accuracy": 35 / 36,
    "designated_helper_both_correct_rate": 29 / 36,
    "unassigned_wait_rate": 29 / 36,
    "team_success_rate": 27 / 36,
}


def evaluate_gate(run: dict) -> dict:
    checks = {
        metric: {
            "observed": run[metric],
            "threshold": threshold,
            "passed": run[metric] >= threshold,
        }
        for metric, threshold in THRESHOLDS.items()
    }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = run_condition(args.base_url, args.model, args.seed, "known_codebook",
                        args.temperature, args.max_tokens, args.timeout)
    payload = {
        "calibration_version": CALIBRATION_VERSION,
        "role": "development_only_not_in_paired_matrix",
        "protocol_source_commit": _git_commit(),
        "model": args.model,
        "model_revision": "16daa4818c54ce5f5436f929d52542eb65bbed9d",
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "condition": "known_codebook",
        "gate_thresholds": THRESHOLDS,
        "gate": evaluate_gate(run),
        "raw_model_completions_retained": False,
        "chain_of_thought_retained": False,
        "run": run,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)
    print(json.dumps({"status": "gate_passed" if payload["gate"]["passed"] else "gate_failed",
                      "checks": payload["gate"]["checks"], "out": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
