"""Bind current root annotations and local packet; perform no semantic relabeling."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib
import html
import json
import re

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "visual_confirmation_v2_water"
ROOT_ANNOTATOR = HERE.parents[1] / "redesign_v0.18" / "review_water_metadata.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def dump(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    checks = []

    def check(name, ok):
        checks.append({"name": name, "passed": bool(ok)})
        assert ok, name

    frozen = read(HERE / "input_freeze.json")
    current = {str(p.relative_to(SOURCE)): {"sha256": sha(p), "bytes": p.stat().st_size}
               for p in sorted(SOURCE.rglob("*")) if p.is_file()}
    check("all_77_source_files_unchanged", current == frozen["source_files"] and len(current) == 77)
    for name, expected in read(HERE / "packet_receipt.json")["artifact_hashes"].items():
        check("packet_receipt:" + name, sha(HERE / name) == expected)
    root_path = HERE / "root_independent_review.json"
    root_sha_before = sha(root_path)
    root = read(root_path)
    packet = read(HERE / "review_input.json")
    check("root_and_packet_both_79", len(root["rows"]) == len(packet["rows"]) == 79)
    check("root_annotator_code_sha", sha(ROOT_ANNOTATOR) == root["annotation_source_sha256"])
    check("root_labels_match_declared_counts", dict(Counter(x["water_text_evidence"] for x in root["rows"])) == root["counts"] == {"E": 28, "U": 43, "X": 8})
    for name, expected in root["inputs_sha256"].items():
        check("root_input:" + name, sha(SOURCE / name) == expected)
    for x, y in zip(root["rows"], packet["rows"]):
        r = y["original_manifest_record"]
        prefix = f"root_row_{x['order']:02d}:"
        check(prefix + "id_order", x["order"] == y["ordinal"] and x["id"] == y["id"])
        root_plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", r["description_html"]))).strip()
        collector_plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", "", r["description_html"]))).strip()
        check(prefix + "source_texts", x["title"] == y["title"] and x["description"] == root_plain and y["description"] == collector_plain)
        check(prefix + "identity_and_credit", x["artist_html"] == r["artist_html"] and x["credit_html"] == r["credit_html"] and x["author_keys"] == r["author_keys"])
        check(prefix + "original_sources", x["source_page"] == r["source_page"] and x["original_source_keys"] == r["original_source_keys"] and x["cluster_id"] == r["cluster_id"])
        check(prefix + "license_not_reassigned", x["license_as_recorded"] == r["license"])
        check(prefix + "no_pixel_or_model_verdict", x["pixel_review"] == "not_accessed" and x["model_use"] == "none" and x["author_identity"] == "not_independently_verified")
    for note in ["collector_note_revision.json", "collector_note_revision_002.json"]:
        r = read(HERE / note)
        check(note + ":original_preserved", sha(HERE / r["original_preserved"]) == r["original_sha256"])
    clarification = read(HERE / "annotation_clarification.json")
    check("root_clarification_keeps_labels", clarification["changed_rows"] == [61, 79] and clarification["no_new_inputs"])
    history_hashes = {sha(p) for p in (HERE / "annotation_history").rglob("*") if p.is_file()}
    check("original_root_json_and_code_preserved", {clarification["old_json_sha256"], clarification["old_code_sha256"]} <= history_hashes)
    for path in [HERE / "packet.md", HERE / "README.md", HERE / "元数据复核报告.md"]:
        targets = re.findall(r"\]\(<?(/[^>\n]+?)>?\)", path.read_text())
        for target in targets:
            check(f"local_link:{path.name}:{target}", Path(target).exists())
    check("root_file_unchanged_by_collector_qa", sha(root_path) == root_sha_before)
    dump("final_review_qa.json", {
        "status": "passed", "checks": len(checks), "failures": 0,
        "scope": "Current root annotations/source-field join, recorded hashes, revision preservation and links; no image acceptance",
        "root_annotation_sha256": root_sha_before, "collector_packet_source_qa_checks": 1517,
        "all_79_rationales_read_against_source_by_collector": True,
        "reading_note": "Collector checked translations/bounds; this is not a second independent review. Root clarified rows 61 and 79 with history preserved.",
        "new_network_requests": 0, "pixel_reads": 0, "model_calls": 0, "new_usable_images": 0,
        "source_files_unchanged": 77, "items": checks,
    })
    artifacts = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob("*"))
                 if p.is_file() and p.name != "completion_receipt.json" and "__pycache__" not in p.parts}
    dump("completion_receipt.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "metadata_review_package_complete",
        "artifact_hashes": artifacts, "external_root_annotation_source": {"path": str(ROOT_ANNOTATOR), "sha256": sha(ROOT_ANNOTATOR)},
        "original_representatives": 79, "original_v2_source_files_unchanged": 77,
        "new_usable_images": 0, "pixel_stage_started": False,
    })
    print(json.dumps({"status": "passed", "checks": len(checks), "artifacts": len(artifacts),
                      "root_annotation_sha256": root_sha_before,
                      "completion_receipt_sha256": sha(HERE / "completion_receipt.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
