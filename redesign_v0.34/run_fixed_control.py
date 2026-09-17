"""Run a same-namespace fixed-A complementary control.

This wrapper reuses the hash-bound v0.34 production runner in memory, limits
the formal matrix to dual_complementary, and overrides only the partner
schedule.  The wrapper receipt and rewritten config hashes make the control
condition explicit without changing the already sealed rotation batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import run_support as run


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    out = args.out.resolve()
    original_conditions = run.support.CONDITIONS
    original_slots = run.support.team_slots

    def fixed_slots(condition, step=None, schedule=None):
        if condition == "dual_complementary":
            return original_slots(condition, schedule="A")
        return original_slots(condition, step=step, schedule=schedule)

    run.support.CONDITIONS = ("dual_complementary",)
    run.support.team_slots = fixed_slots
    try:
        run.main()
    finally:
        run.support.CONDITIONS = original_conditions
        run.support.team_slots = original_slots

    # Make the control schedule explicit in each config and refresh the
    # generated file inventory.  No model or input file is changed here.
    for config_path in sorted((out / "social").glob("s*_p*_dual_complementary/config.json")):
        config = run.read(config_path); config["training_schedule"] = "A_only_fixed_control"; config["wrapper"] = "run_fixed_control.py"
        run.write(config_path, config)
    wrapper = {"status": "complete", "control": "fixed_A", "schedule": "A_every_update", "source": str(Path(__file__).resolve()), "source_sha256": sha(Path(__file__)), "production_runner": str((run.ROOT / "run_support.py").resolve()), "production_runner_sha256": sha(run.ROOT / "run_support.py"), "out": str(out)}
    run.write(out / "wrapper_receipt.json", wrapper)
    complete = run.read(out / "training_complete.json")
    complete["control"] = "fixed_A"
    complete["files"] = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "training_complete.json"}
    run.write(out / "training_complete.json", complete)
    print(json.dumps({"status": "complete", "control": "fixed_A", "social_runs": complete["social_runs"], "source_sha256": wrapper["source_sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
