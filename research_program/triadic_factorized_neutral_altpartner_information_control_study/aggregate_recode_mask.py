"""Aggregate the audited FI recode/mask endpoint rows without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research_program.triadic_factorized_neutral_altpartner_information_control_study import design, metrics


TRANSFORMS = tuple([f"perm_{i:02d}" for i in range(6)] + ["position_rotation", "position_reverse", "cross_payload_zero", "cross_visibility_zero"])
FAMILIES = {"symbol_ensemble": [f"perm_{i:02d}" for i in range(6)], "position": ["position_rotation", "position_reverse"], "field": ["cross_payload_zero", "cross_visibility_zero"]}
PRIMARY = ("q_rate", "conditional_q_rate", "target_pair_legal_rate", "proposal_legal_rate", "physical_execution_rate", "engagement_rate", "third_agent_neutral_rate")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    x = np.asarray(values, dtype=np.float64); require(x.shape == (16,), "Expected sixteen seed values")
    return metrics.stats(x)


def aggregate_values(policies, schedule, channel, transform, key, senders=None):
    selected = [p for p in policies if p["schedule"] == schedule and p["trained_channel"] == channel]
    selected = {int(p["seed"]): p for p in selected}; require(len(selected) == 16, "Sixteen policy blocks")
    out = []
    for seed in design.SEEDS:
        rows = [r for r in selected[seed]["rows"] if r["transform"] == transform and (senders is None or int(r["sender"]) in senders)]
        require(len(rows) == (3 if senders is None else len(senders)), "Sender rows")
        out.append(float(np.mean([r["natural_minus_intervened"].get(key, 0.0) for r in rows])))
    return out


def natural_intervened_values(policies, schedule, channel, transform, key):
    selected = {int(p["seed"]): p for p in policies if p["schedule"] == schedule and p["trained_channel"] == channel}; require(len(selected) == 16, "Sixteen policies")
    natural, intervened = [], []
    for seed in design.SEEDS:
        rows = [r for r in selected[seed]["rows"] if r["transform"] == transform]
        natural.append(float(np.mean([r["natural"][key] for r in rows])))
        intervened.append(float(np.mean([r["intervened"].get(key, r["natural"][key]) for r in rows])))
    return natural, intervened


def main(run, audit, output):
    run = Path(run).resolve(); audit = Path(audit).resolve(); output = Path(output).resolve(); require(not output.exists(), "Never overwrite aggregate output"); output.mkdir(parents=True)
    result = json.loads((run / "execution/results.json").read_text()); verification = json.loads((audit / "verification.json").read_text())
    require(result["status"] == "completed_json_only_recode_mask_probe" and verification["status"] == "passed", "Completed audited probe required")
    policies = result["policies"]; require(len(policies) == 64, "Complete policy grid")
    summaries = {}
    for schedule in design.SCHEDULES:
        for channel in ("live", "own"):
            for transform in TRANSFORMS:
                label = f"{schedule}_{channel}_{transform}"; block = {"schedule": schedule, "trained_channel": channel, "transform": transform, "by_seed": {}}
                for key in PRIMARY:
                    values = aggregate_values(policies, schedule, channel, transform, key)
                    natural, intervened = natural_intervened_values(policies, schedule, channel, transform, key)
                    block["by_seed"][key] = values; block[key] = {"effect_natural_minus_intervened": stats(values), "natural": stats(natural), "intervened": stats(intervened)}
                summaries[label] = block
            for family, transforms in FAMILIES.items():
                label = f"{schedule}_{channel}_{family}"; block = {"schedule": schedule, "trained_channel": channel, "family": family, "transforms": transforms, "by_seed": {}}
                for key in PRIMARY:
                    values = []; natural_values = []; intervened_values = []
                    for seed in design.SEEDS:
                        rows = [r for p in policies if int(p["seed"]) == seed and p["schedule"] == schedule and p["trained_channel"] == channel for r in p["rows"] if r["transform"] in transforms]
                        require(len(rows) == 3 * len(transforms), "Family rows")
                        values.append(float(np.mean([r["natural_minus_intervened"].get(key, 0.0) for r in rows])))
                        natural_values.append(float(np.mean([r["natural"][key] for r in rows])))
                        intervened_values.append(float(np.mean([r["intervened"].get(key, r["natural"][key]) for r in rows])))
                    block["by_seed"][key] = values; block[key] = {"effect_natural_minus_intervened": stats(values), "natural": stats(natural_values), "intervened": stats(intervened_values)}
                summaries[label] = block
    sender = {}
    for schedule in design.SCHEDULES:
        for channel in ("live", "own"):
            for family, transforms in FAMILIES.items():
                sender[f"{schedule}_{channel}_{family}"] = {}
                for actor in range(3):
                    block = {}
                    for key in PRIMARY:
                        values = []
                        for seed in design.SEEDS:
                            policy = [p for p in policies if int(p["seed"]) == seed and p["schedule"] == schedule and p["trained_channel"] == channel][0]
                            rows = [r for r in policy["rows"] if int(r["sender"]) == actor and r["transform"] in transforms]; values.append(float(np.mean([r["natural_minus_intervened"].get(key, 0.0) for r in rows])))
                        block[key] = stats(values)
                    sender[f"{schedule}_{channel}_{family}"][str(actor)] = block
    summary = dict(schema="triadic_fi_recode_mask_summary_v1", run=str(run), run_results_sha256=sha(run / "execution/results.json"), audit=str(audit), audit_verification_sha256=sha(audit / "verification.json"), transforms=list(TRANSFORMS), families=FAMILIES, policy_blocks=64, intervention_rows=1920, independent_seed_unit="per-seed mean over the three selected senders; t intervals across 16 paired initializations", summaries=summaries, sender_effects=sender, interpretation="Positive effect means the natural route has a higher endpoint rate than the transformed route. This quantifies route/code sensitivity of a fixed task policy; it is not evidence for a lexicon or language origin.")
    path = output / "summary.json"; path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    receipt = dict(status="passed", output_sha256=sha(path), model_forward_samples=0, optimizer_updates=0, source_results_sha256=summary["run_results_sha256"], audit_verification_sha256=summary["audit_verification_sha256"])
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"); print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True); parser.add_argument("--audit", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); main(args.run, args.audit, args.output)
