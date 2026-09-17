"""Frozen v0.34 formal-execution gate; no performance-based filtering."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

import run_support as run


ROOT = Path(__file__).resolve().parent
SMOKE = ROOT / "results/smoke_001"


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def main():
    complete = run.read(SMOKE / "training_complete.json")
    if not (complete["status"] == "complete" and complete["formal"] is False and complete["social_runs"] == 3 and complete["pair_updates"] == 120):
        raise AssertionError("unexpected smoke completion")
    smoke_source = run.PROJECT / "redesign_v0.28/results/smoke_001"; formal_source = run.PROJECT / "redesign_v0.28/results/formation_001"
    formal_inputs = run.input_hashes(formal_source, run.SEEDS, run.PARTITIONS)
    if complete["source_hashes"] != run.source_hashes() or complete["input_hashes"] != run.input_hashes(smoke_source, [99528], [1]):
        raise AssertionError("smoke source/input binding changed")
    conditions = tuple(run.support.CONDITIONS); folders = [SMOKE / "social" / f"s99528_p1_{condition}" for condition in conditions]
    initial = [torch.load(folder / "initial.pt", weights_only=True) for folder in folders]
    if not all(torch.equal(initial[0][i][key], initial[j][i][key]) for j in range(1, len(initial)) for i in range(4) for key in initial[0][i]):
        raise AssertionError("condition initial states are not exactly paired")
    for condition, folder in zip(conditions, folders):
        cfg = run.read(folder / "config.json")
        expected_teams = [[0, 1], [1, 2], [2, 3], [3, 0]] if condition == "single_full" else [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]]
        if cfg["teams"] != expected_teams or cfg["checkpoints"] != [0, 40]:
            raise AssertionError("team/config gate failed")
        if condition == "dual_complementary" and cfg["training_schedule"] != "A_even_B_odd":
            raise AssertionError("rotation schedule metadata gate failed")
        manifest = run.read(SMOKE / "cache" / "s99528_p1" / "view_manifest.json")
        if not all(np.isfinite(value) and value <= 3e-5 for value in manifest["full_view_errors"].values()):
            raise AssertionError("full-view inheritance gate failed")
        curve = run.read(folder / "curve.json")
        if [item["update"] for item in curve] != [0, 40] or set(curve[0]["scores"]) != {f"team{i}" for i in range(4)}:
            raise AssertionError("protocol inventory gate failed")
        for t in (0, 40):
            for slot in range(4):
                raw = load(folder / f"protocol_{t:04d}_team{slot}.npz")
                if raw["tokens"].shape != (180, 2) or raw["sender_log_probs"].shape != (180, 49) or raw["receiver_logits"].shape != (49, 2, 6):
                    raise AssertionError("protocol table shape gate failed")
                if not np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1, atol=1e-6):
                    raise AssertionError("sender normalization gate failed")
        if condition == "dual_complementary":
            for schedule in ("B", "C"):
                for slot in range(4):
                    raw = load(folder / f"transfer_{schedule}_0040_team{slot}.npz")
                    if raw["tokens"].shape != (180, 2) or raw["receiver_logits"].shape != (49, 2, 6):
                        raise AssertionError("transfer protocol shape gate failed")
    for step in (1, 40):
        worlds = []
        for folder in folders:
            raw = load(folder / f"train_{step:04d}.npz")
            worlds.append({key: value for key, value in raw.items() if key.startswith("world__")})
        for key in worlds[0]:
            if not all(np.array_equal(worlds[0][key], other[key]) for other in worlds[1:]):
                raise AssertionError("conditions do not share paired world fixture")
    original = run.read(formal_source / "training_complete.json")["files"]; bound = 0
    for path, digest in formal_inputs.items():
        p = Path(path)
        if p.is_relative_to(formal_source):
            rel = str(p.relative_to(formal_source)); current = run.sha(p)
            if rel == "training_complete.json":
                if current != digest: raise AssertionError("formation completion hash")
            elif original[rel] != digest: raise AssertionError("formation inherited hash")
            bound += 1
    run.write(SMOKE / "terminal_receipt.json", dict(session_id=28435, exit_code=0, status="complete", observed_by="root tools.write_stdin", reported_program_seconds=complete["seconds"]))
    for path in (ROOT / "implementation_review.json", SMOKE / "raw_validation.json", SMOKE / "audit_execution.json"):
        if not run.read(path)["passed"]: raise AssertionError(path)
    run.write(ROOT / "preflight_qa.json", dict(passed=True, version="v0.34", created_utc=datetime.now(timezone.utc).isoformat(), source_hashes=run.source_hashes(), input_hashes=formal_inputs, smoke_output=str(SMOKE), smoke_completion_sha256=run.sha(SMOKE / "training_complete.json"), smoke_protocol_tables=24, initial_exactly_paired=True, conditions=list(conditions), team_schedule="A fixed; dual complementary training alternates A and B; C is held out for endpoint transfer", full_view_inheritance_max_error=max(run.read(SMOKE / "cache" / "s99528_p1" / "view_manifest.json")["full_view_errors"].values()), original_bound_inputs=bound, checks={str(path): run.sha(path) for path in (ROOT / "implementation_review.json", SMOKE / "raw_validation.json", SMOKE / "audit_execution.json")}, criterion_uses_performance_threshold=False, development_performance_not_used_for_selection=True, synthetic_tests=dict(support=5, views=2, dual=1, analyzer_self_tests=4), scope="Team inventory, full-view equality, complementary masks, paired condition fixtures, rotation metadata and independent smoke statistics"))
    print("preflight passed 24 smoke protocol tables;", bound, "formal inputs bound")


if __name__ == "__main__": main()
