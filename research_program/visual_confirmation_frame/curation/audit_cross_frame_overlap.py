"""Offline exact identity overlap; neither allocates nor changes either frame."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
from collections import Counter

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "own_candidates": ROOT / "research_program/visual_confirmation_frame/curation/ranking_001/candidate_order.json",
        "other_frame": ROOT / "paper_program/visual_confirmation_v1/sampling_frame.json",
        "other_manifest": ROOT / "paper_program/visual_confirmation_v1/manifest.json",
        "other_status": ROOT / "paper_program/visual_confirmation_v1/status.json",
    }
    raw = {k: p.read_bytes() for k, p in sources.items()}
    data = {k: json.loads(v) for k, v in raw.items()}
    assert data["other_status"]["status"] == "metadata_collection_complete"
    own = [r for rows in data["own_candidates"].values() for r in rows]
    other = data["other_manifest"]["images"]
    assert len(own) == 232 and len(other) == 800
    assert len({r["pageid"] for r in own}) == len(own)
    other_ids = Counter(r["pageid"] for r in other)
    rows = []
    for a in own:
        for b in other:
            same_page = a["pageid"] == b["pageid"]
            same_sha = bool(a["original_file_sha1"]) and a["original_file_sha1"] == b["original_sha1"]
            if same_page or same_sha:
                rows.append({"own_pageid": a["pageid"], "other_pageid": b["pageid"],
                             "own_review_id": a["review_id"], "own_label": a["label"],
                             "own_original_role": a["role"], "other_stratum": b["stratum"],
                             "other_provisional_assignment": b.get("provisional_assignment"),
                             "other_exclusion_reasons": b.get("exclusion_reasons", []),
                             "same_pageid": same_page, "same_original_sha1": same_sha})
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Exact page identity or original byte SHA1 only; not an author/series/perceptual duplicate audit.",
        "sources": {k: {"path": str(sources[k]), "sha256": hashlib.sha256(v).hexdigest()}
                    for k, v in raw.items()},
        "own_candidates": len(own), "other_metadata_candidates": len(other),
        "other_unique_pageids": len(other_ids),
        "other_repeated_pageid_rows": {str(k): n for k, n in other_ids.items() if n > 1},
        "overlap_pair_count": len(rows),
        "own_overlapping_candidates": len({r["own_pageid"] for r in rows}),
        "by_own_label": dict(Counter(r["own_label"] for r in rows)),
        "by_other_provisional_assignment": dict(Counter(str(r["other_provisional_assignment"]) for r in rows)),
        "same_page_different_sha_pairs": sum(r["same_pageid"] and not r["same_original_sha1"] for r in rows),
        "different_page_same_sha_pairs": sum(not r["same_pageid"] and r["same_original_sha1"] for r in rows),
        "overlaps": rows,
        "actions": "Read-only snapshots; no image requests, views, feature extraction, reallocation or exclusion changes.",
    }
    # Reject a torn snapshot of the other task's mutable metadata outputs.
    for k, p in sources.items():
        assert p.read_bytes() == raw[k], f"Source changed during audit: {k}"
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "overlap.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    n = result["own_overlapping_candidates"]
    assigned = result["by_other_provisional_assignment"]
    report = f"""# 两套候选框的精确身份重合

本次仅离线比较本项目固定232个提名，与另一任务已完成元数据的800个候选；不读取像素，不改变任何候选、排序或用途分配。

另一清单有800条类别行、{len(other_ids)}个不同pageid；重复pageid为 `{result['other_repeated_pageid_rows']}`。初次脚本把800行误当作800个不同文件，唯一性断言中止、未产生输出；本次明确保留类别行并分别计数。不能把同一文件在两个类别中的记录当作两份独立材料。

共有 **{n}个**本项目候选与另一框重合（pageid相同或原文件SHA1相同）。按本项目类别为 `{result['by_own_label']}`；另一任务的临时用途为 `{assigned}`。相同pageid但SHA1不同为{result['same_page_different_sha_pairs']}对，不同pageid但SHA1相同为{result['different_page_same_sha_pairs']}对。逐项身份及四个输入的快照哈希保存在 `overlap.json`。

这只确定候选身份重合，尚未证明发生模型信息泄露：两边的元数据候选和临时名额都不等于最终图片集。不过，两套框不能因为目录不同就当作独立来源。以后如共享训练、调试或验证材料，应先核对实际图片与用途，再决定哪些可作确认；本审计不替另一任务重新分配材料。未重合者也未获独立性证明，作者、拍摄系列、近重复和共同源图仍可能重合。

这是对另一任务当时状态的只读快照，后续文件变化需另记，不能把本文件当作对该任务后续行为的控制或保证。
"""
    (args.out / "重合核对.md").write_text(report)
    print(json.dumps({k: v for k, v in result.items() if k not in ("sources", "overlaps")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
