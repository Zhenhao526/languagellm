"""Seal the formal v0.47 reset-granularity batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


VERSION = "v0.47-reset-granularity"
RESET_MODES = ("sender_only", "receiver_only", "both")


def read(path: Path): return json.loads(path.read_text())
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def assert_hash(path: Path, digest: str):
    if not path.is_file(): raise AssertionError(f"missing input: {path}")
    actual = sha(path)
    if actual != digest: raise AssertionError(f"hash mismatch: {path}: {actual} != {digest}")


def links(report: Path):
    for match in re.finditer(r"\]\((<?)([^)>]+)>?\)", report.read_text()):
        path = Path(match.group(2))
        if path.is_absolute(): yield path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); root = args.out.resolve(); out = root / "results" / "reset_granularity_001"; manifest = out / "completion_manifest.json"
    if manifest.exists(): raise FileExistsError(manifest)
    invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); analysis = read(out / "reset_granularity_analysis.json"); stats = read(out / "reset_granularity_statistics.json"); audit = read(out / "reset_granularity_audit.json"); visual = read(out / "visual_qa.json"); review = read(root / "结果审查.json"); report = out / "通信模块重置粒度与陌生主体恢复研究报告.md"
    required = (invocation.get("formal") is True, invocation.get("version") == VERSION, invocation.get("adaptation_schedule") == "random_ABC", tuple(invocation.get("reset_modes", ())) == RESET_MODES, invocation.get("updates") == 300, complete.get("status") == "complete", complete.get("formal") is True, complete.get("runs") == 1296, complete.get("updates_per_run") == 300, complete.get("trace_files_expected") == 10368, complete.get("protocol_files_expected") == 46656, analysis.get("status") == "complete", analysis.get("formal") is True, analysis.get("runs") == 1296, len(analysis.get("rows", [])) == 279936, len(analysis.get("sequence_rows", [])) == 3888, analysis.get("maximum_metric_absolute_difference") == 0.0, stats.get("status") == "complete", stats.get("formal") is True, stats.get("runs") == 1296, stats.get("bootstrap", {}).get("repetitions") == 20000, audit.get("status") == "complete", audit.get("formal") is True, audit.get("runs") == 1296, audit.get("coverage", {}).get("traces") == 10368, audit.get("coverage", {}).get("protocol_files") == 46656, audit.get("maximum_replay_absolute_difference") <= 1e-5, audit.get("production_modules_imported") is False, audit.get("model_calls") == 0, visual.get("status") == "passed_visual_qa", review.get("passed") is True, review.get("version") == VERSION, report.is_file())
    if not all(required): raise AssertionError("formal v0.47 analysis/audit/statistics/review required")
    for kind in ("source_hashes", "input_hashes"):
        for path, digest in invocation[kind].items(): assert_hash(Path(path), digest)
    for relative, digest in complete.get("files", {}).items(): assert_hash(out / relative, digest)
    report_links = list(links(report)); missing = [str(path) for path in report_links if not path.exists()]
    if missing: raise AssertionError(f"missing report links: {missing}")
    files = set()
    for path in root.rglob("*"):
        if not path.is_file() or ".venv" in path.parts or "__pycache__" in path.parts or path.name == "completion_manifest.json" or "failure" in path.name: continue
        if "results" in path.parts and any("smoke" in part or "probe" in part for part in path.parts): continue
        files.add(path.resolve())
    files.update(Path(path).resolve() for path in invocation["source_hashes"]); files.update(Path(path).resolve() for path in invocation["input_hashes"]); files.update(path.resolve() for path in report_links)
    artifacts = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(files) if path.is_file()]
    record = {"status": "complete_development_batch", "version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "study": "communication-side reset granularity during newcomer recovery inside frozen v0.43 resident cultures", "overall_research_goal_complete": False, "formal_output": str(out), "terminal": None, "counts": {"chains": complete["runs"], "visual_groups": 72, "resident_cultures": len(invocation["cultures"]), "reset_modes": len(invocation["reset_modes"]), "updates_per_chain": complete["updates_per_run"], "checkpoints": len(complete["checkpoints"]), "trace_files": complete["trace_files_expected"], "protocol_files": complete["protocol_files_expected"], "analysis_rows": len(analysis["rows"]), "sequence_rows": len(analysis["sequence_rows"])}, "quality": {"analysis_checks": analysis["checks"], "analysis_scalar_comparisons": analysis["scalar_comparisons"], "analysis_maximum_metric_absolute_difference": analysis["maximum_metric_absolute_difference"], "audit_checks": audit["checks"], "audit_scalar_comparisons": audit["scalar_comparisons"], "audit_maximum_replay_absolute_difference": audit["maximum_replay_absolute_difference"], "audit_production_modules_imported": audit["production_modules_imported"], "statistics_bootstrap_repetitions": stats["bootstrap"]["repetitions"], "visual_qa": visual["status"], "scientific_review": review["status"]}, "report": str(report), "review": str(root / "结果审查.json"), "artifacts": artifacts, "limitations": ["The newcomer inherits the resident visual frontend and all unreset communication modules.", "Residents are frozen during adaptation; no negotiation or repair is tested.", "Finite two-token, six-site protocol without delayed inventory or generational transmission.", "Bootstrap intervals are descriptive and do not replace a preregistered hierarchical model."], "next_step": "Add 600-update checkpoints and bounded resident adaptation, then introduce delayed resource consequences and generational transmission.", "excluded": ["results/*smoke*", "results/*probe*", "__pycache__", ".venv", "failure records", "this completion manifest"], "link_check": {"report_links": len(report_links), "missing": missing}}
    manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n"); print(json.dumps({"status": record["status"], "version": record["version"], "artifacts": len(artifacts), "training_files_rehashed": len(complete.get("files", {})), "manifest_sha256": sha(manifest)}, ensure_ascii=False))


if __name__ == "__main__": main()
