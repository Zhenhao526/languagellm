"""Create a compact, auditable summary for the FI single-slot probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FAMILIES = {
    "slot_payload_zero": [f"slot_payload_zero_{i}" for i in range(4)],
    "slot_symbol_shift": [f"slot_symbol_shift_{i}" for i in range(4)],
    "slot_symbol_xor4": [f"slot_symbol_xor4_{i}" for i in range(4)],
}
METRICS = ("q_rate", "conditional_q_rate", "physical_execution_rate", "proposal_legal_rate", "target_pair_legal_rate")


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(run, audit, out):
    run = Path(run).resolve()
    audit = Path(audit).resolve()
    out = Path(out).resolve()
    require(not out.exists(), "Never overwrite summary output")
    execution = json.loads((run / "execution/results.json").read_text())
    audit_v = json.loads((audit / "verification.json").read_text())
    require(execution.get("status") == "completed_json_only_single_slot_probe", "Unexpected run status")
    require(audit_v.get("status") == "passed" and float(audit_v.get("max_abs_error", 1.0)) == 0.0, "Audit must pass exactly")
    require(execution.get("policy_blocks") == 64 and execution.get("intervention_rows") == 2304, "Unexpected grid")
    require(audit_v.get("rows_replayed") == 2304 and audit_v.get("evaluations") == 1152, "Audit grid mismatch")
    summaries = execution["summaries"]
    rows = {}
    for schedule in ("static", "rematched"):
        for channel in ("live", "own"):
            for family, transforms in FAMILIES.items():
                family_key = f"{schedule}_{channel}_{family}"
                require(family_key in summaries, f"Missing {family_key}")
                rows[family_key] = {
                    "schedule": schedule,
                    "trained_channel": channel,
                    "family": family,
                    "transforms": transforms,
                    "metrics": {metric: summaries[family_key][metric] for metric in METRICS},
                }
                for transform in transforms:
                    key = f"{schedule}_{channel}_{transform}"
                    require(key in summaries, f"Missing {key}")
                    rows[key] = {
                        "schedule": schedule,
                        "trained_channel": channel,
                        "family": family,
                        "transform": transform,
                        "metrics": {metric: summaries[key][metric] for metric in METRICS},
                    }
    summary = {
        "schema": "triadic_fi_single_slot_summary_v1",
        "run": str(run),
        "run_results_sha256": sha(run / "execution/results.json"),
        "audit": str(audit),
        "audit_verification_sha256": sha(audit / "verification.json"),
        "policy_blocks": execution["policy_blocks"],
        "intervention_rows": execution["intervention_rows"],
        "live_intervention_rows": execution["live_intervention_rows"],
        "alias_rows": execution["alias_rows"],
        "endpoint_worlds": execution["endpoint_worlds"],
        "worlds": int(execution["intervention_rows"] * execution["endpoint_worlds"]),
        "model_forward_samples": execution["model_forward_samples"],
        "optimizer_updates": execution["optimizer_updates"],
        "families": FAMILIES,
        "metrics": list(METRICS),
        "effect_definition": "natural minus intervened, in proportion units; positive means the selected-slot intervention lowers the metric",
        "independent_seed_unit": "per-seed mean over the three selected senders; t intervals across 16 paired initializations",
        "interpretation": "Post-hoc local sensitivity of a fixed FI task policy. It is not evidence for lexical meaning, compositional grammar or language origin.",
        "rows": rows,
    }
    out.mkdir(parents=True)
    path = out / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    receipt = {
        "status": "passed",
        "output_sha256": sha(path),
        "run_results_sha256": summary["run_results_sha256"],
        "audit_verification_sha256": summary["audit_verification_sha256"],
        "policy_blocks": summary["policy_blocks"],
        "intervention_rows": summary["intervention_rows"],
        "optimizer_updates": summary["optimizer_updates"],
    }
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.run, args.audit, args.out)
