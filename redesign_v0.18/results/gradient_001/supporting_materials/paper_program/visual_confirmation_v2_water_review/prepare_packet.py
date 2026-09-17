"""Read-only metadata join. Writes only this new review directory; no network/pixels/models."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin
import hashlib
import html
import json
import re

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "visual_confirmation_v2_water"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    # Same plain-text transformation as the frozen collector.
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", "", str(value)))).strip()


def links(value):
    return [urljoin("https://commons.wikimedia.org", html.unescape(x))
            for x in re.findall(r'''href=["']([^"']+)''', str(value))]


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def inventory():
    return {str(p.relative_to(SOURCE)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(SOURCE.rglob("*")) if p.is_file()}


def md(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def local_link(label, path):
    return f"[{label}](<{path.resolve()}>)"


def main():
    initial_inventory = inventory()
    freeze_path = HERE / "input_freeze.json"
    if freeze_path.exists():
        frozen = read(freeze_path)
        assert frozen["source_files"] == initial_inventory, "Source changed since review freeze"
    else:
        write("input_freeze.json", {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "All existing v2 metadata files only; no source mutation or external access",
            "source_directory": str(SOURCE), "source_file_count": len(initial_inventory),
            "source_files": initial_inventory,
            "initial_builder_sha256": sha(Path(__file__)),
            "collector_notes_sha256": sha(HERE / "collector_notes.json"),
            "new_network_requests": 0, "pixel_reads": 0, "model_calls": 0,
        })
    checks = []

    def check(name, condition):
        checks.append({"check": name, "passed": bool(condition)})
        assert condition, name

    manifest = read(SOURCE / "manifest.json")
    all_groups = read(SOURCE / "author_groups.json")
    summary = read(SOURCE / "feasibility_summary.json")
    config = read(SOURCE / "config.json")
    rows = {x["id"]: x for x in manifest["images"]}
    groups = [g for g in all_groups["groups"] if g["eligible_representative"]]
    ids = [g["eligible_representative"] for g in groups]
    notes = {x[0]: x[1:] for x in read(HERE / "collector_notes.json")["notes"]}
    check("exactly_79_unique_representatives", len(ids) == len(set(ids)) == 79)
    check("order_equals_original_summary", ids == summary["provisional_cluster_representative_ids"])
    check("original_group_count_unchanged", all_groups["provisionally_eligible_clusters"] == 79)
    check("manual_notes_exactly_ordinal_1_to_79", set(notes) == set(range(1, 80)))
    for path, expected in read(SOURCE / "completion_receipt.json")["artifact_hashes"].items():
        check("original_completion_hash:" + path, sha(SOURCE / path) == expected)
    raw_paths = sorted({rows[i]["metadata_response"] for i in ids})
    raw_refs = {p: f"R{i:02d}" for i, p in enumerate(raw_paths, 1)}
    raw = {p: read(SOURCE / p) for p in raw_paths}
    output_rows = []
    auto_keys = ["metadata_provisionally_eligible", "author_metadata_parseable",
                 "license_metadata_supported", "scene_text_or_category_supported",
                 "scene_phrase_matches", "scene_category_sources", "scene_negative_terms",
                 "blocked_identity_or_source_keys", "exclusion_reasons"]
    packet = ["# 原水图元数据 79 个代表项：逐条复核材料", "",
              "本材料覆盖原 author_groups.json 的全部 79 个 eligible_representative，沿用原簇顺序，没有重选或补项。原自动计数保持 79；这些条目尚未成为可用图像。", "",
              "整理者是原元数据采集者；下列风险是采集者初评，不能称独立验收。根代理另行逐条判断。全程只读取冻结元数据：没有联网、下载、读像素或模型调用。咖啡/餐食旁明确有水时只标场景风险，不按标题自动排除。", "",
              "作者状态均未完成独立身份核实，parseable=true 仅复述原解析结果。许可仅为原页面元数据声明，不代表许可链已经验收。原拒绝词检查只作用于标题加描述的固定英文正则，未覆盖多语义、所有分类或图像内容。", "",
              "每条描述最多 400 字符，截断处明确标记；全文、完整字段、所有分类、原自动判断和原始来源 SHA256 均保存在 " + local_link("review_input.json", HERE / "review_input.json") + "。原始响应在文末逐项列出完整哈希；每条用 R 编号和 pageid 定位。", "",
              "输入冻结：" + local_link("input_freeze.json", freeze_path) + f"（原目录 {len(initial_inventory)} 个文件）。", ""]
    for ordinal, group in enumerate(groups, 1):
        r = rows[group["eligible_representative"]]
        prefix = f"row_{ordinal:02d}:"
        check(prefix + "representative_is_member", r["id"] in group["members"])
        check(prefix + "cluster_matches", r["cluster_id"] == group["cluster_id"])
        check(prefix + "original_eligible_not_blocked", r["metadata_provisionally_eligible"] and not group["blocked_by_old_or_reserved"])
        check(prefix + "raw_sha256", sha(SOURCE / r["metadata_response"]) == r["metadata_response_sha256"])
        pages = [p for p in raw[r["metadata_response"]]["query"]["pages"] if p["pageid"] == r["pageid"]]
        check(prefix + "one_raw_page", len(pages) == 1)
        page = pages[0]
        info = page["imageinfo"][0]
        check(prefix + "raw_title_and_revision", page["title"] == r["title"] and page.get("revisions", []) == r["page_revision"])
        check(prefix + "raw_all_extmetadata", info.get("extmetadata", {}) == r["extmetadata"])
        for source_key, result_key in [("url", "original_url"), ("sha1", "original_sha1"),
                                       ("descriptionurl", "source_page"), ("timestamp", "original_version_timestamp"),
                                       ("mime", "mime"), ("width", "width"), ("height", "height")]:
            check(prefix + source_key, info.get(source_key) == r[result_key])
        for source_key, result_key in [("Artist", "artist_html"), ("Credit", "credit_html"), ("ImageDescription", "description_html")]:
            check(prefix + source_key, info["extmetadata"].get(source_key, {}).get("value", "") == r[result_key])
        combined = clean(r["title"] + " " + r["description_html"])
        phrase_check = [m.group() for m in re.finditer(config["scene_phrase_pattern"], combined, re.I)]
        negative_check = sorted({m.group().casefold() for m in re.finditer(config["negative_pattern"], combined, re.I)})
        check(prefix + "fixed_english_phrase_recomputed", phrase_check == r["scene_phrase_matches"])
        check(prefix + "fixed_english_negative_recomputed", negative_check == r["scene_negative_terms"])
        cats = clean(r["extmetadata"].get("Categories", {}).get("value", "")).split("|")
        cats = [x for x in cats if x]
        water_axis, context_axis, risk = notes[ordinal]
        identity_note = "原 Artist/作者键可解析，未独立确认摄影者、原作身份或跨账号同一人；上传者字段不作为作者结论。"
        if ordinal == 34:
            identity_note = "Artist 是机构 Flickr 账号 SuperJet International；Credit 指向 Flickr 作品 5859312743。可追溯账号/作品声明，不等于已确认个人摄影者；上传机器人不是作者。"
        elif ordinal == 44:
            identity_note = "Artist 是机构 Flickr 账号 FinnishGovernment；描述署名 © Lauri Heikkinen，分类另有 Photographs by Lauri Heikkinen。原机构账号簇与摄影者身份不同层级，尚未完成个人作者簇归并/独立核实；© 不自动否定 CC BY 2.0。"
        elif ordinal == 74:
            identity_note = "Artist 原文只说原上传者 Pixel23，Credit 只说从 en.wikipedia 转入；并未明确摄影者或作品作者。原 parseable=true 和 commons-user:pixel23 是解析器输出，不足以确认作者身份；同时该链接实际指向 en.wikipedia，不能按键名前缀视作已核实 Commons 作者。"
        description = clean(r["description_html"])
        credit = clean(r["credit_html"])
        item = {
            "ordinal": ordinal, "id": r["id"], "pageid": r["pageid"], "title": r["title"],
            "cluster": group, "description": description, "description_display_limit": 400,
            "description_truncated_in_packet": len(description) > 400,
            "description_metadata_source": r["extmetadata"].get("ImageDescription", {}).get("source"),
            "artist": {"text": r["artist_text"], "html": r["artist_html"], "links": links(r["artist_html"]),
                       "original_author_keys": r["author_keys"], "original_parseable": r["author_metadata_parseable"],
                       "original_upload_user_not_author_inference": r["uploader_not_inferred_as_author"],
                       "identity_review": "not_independently_verified", "collector_note": identity_note},
            "credit": {"text": credit, "html": r["credit_html"], "links": links(r["credit_html"]),
                       "original_source_keys": r["original_source_keys"]},
            "license": {"name": r["license"], "url": r["license_url"], "attribution_required": r["attribution_required"],
                        "usage_terms": r["extmetadata"].get("UsageTerms", {}).get("value"),
                        "copyrighted": r["extmetadata"].get("Copyrighted", {}).get("value"),
                        "restrictions": r["extmetadata"].get("Restrictions", {}).get("value"),
                        "attribution": r["extmetadata"].get("Attribution", {}).get("value"),
                        "review_status": "stored_metadata_claim_only_not_independently_verified"},
            "categories": cats, "all_extmetadata": r["extmetadata"],
            "provenance": {k: r[k] for k in ["source_page", "page_revision", "original_url", "original_sha1",
                                            "original_version_timestamp", "mime", "width", "height", "frame_sources",
                                            "metadata_response", "metadata_response_sha256", "metadata_request_url"]},
            "raw_reference": raw_refs[r["metadata_response"]],
            "original_automatic": {k: r[k] for k in auto_keys},
            "collector_preliminary_risk": {
                "reviewer_role": "original_collector_not_independent", "water_evidence_axis": water_axis,
                "context_axis": context_axis, "note": risk,
                "evidence": [{"field": "title", "quote": r["title"]},
                             {"field": "extmetadata.ImageDescription.value", "quote": description or "[field absent/empty]"},
                             {"field": "extmetadata.Artist.value", "quote": r["artist_text"]},
                             {"field": "extmetadata.Credit.value", "quote": credit},
                             {"field": "extmetadata.Categories.value", "quote": " | ".join(cats)}],
                "english_phrase_absent": not bool(phrase_check),
                "automatic_support_only_nonbase_categories": not phrase_check and bool(r["scene_category_sources"]) and "Category:Glasses of water" not in r["scene_category_sources"],
                "visual_status": "not_viewed", "usable_image_verdict": None,
            },
            "independent_review_is_separate_file": "root_independent_review.json",
            "original_manifest_record": r,
        }
        output_rows.append(item)
        desc_short = description[:400] + ("…〔已截断，全文见 JSON〕" if len(description) > 400 else "")
        packet.extend([
            f"## {ordinal:02d} · {r['id']} · {md(r['title'])}", "",
            "描述（原 ImageDescription）：“" + md(desc_short or "〔缺失〕") + "”",
            "作者/来源：Artist=“" + md(r["artist_text"]) + "”；Credit=“" + md(credit or "〔缺失〕") + "”。" + identity_note,
            "许可声明：" + md(r["license"]) + "；署名要求=" + md(r["attribution_required"]) + "；" + f"[许可 URL]({r['license_url']})" + "。来源为已存 " + f"[Commons 页面]({r['source_page']})" + "；本次未访问。",
            "全部分类（原 Categories）：" + md(" | ".join(cats) or "〔空〕") + "。",
            "原自动：eligible=true；parseable=" + str(r["author_metadata_parseable"]).lower() + "；短语=" + md(json.dumps(phrase_check, ensure_ascii=False)) + "；入框分类=" + md(" | ".join(r["scene_category_sources"]) or "〔无〕") + "；拒绝词=" + md(json.dumps(negative_check, ensure_ascii=False)) + "；排除理由=" + md(json.dumps(r["exclusion_reasons"], ensure_ascii=False)) + "。",
            "采集者初评（非独立判定）：" + risk + " 证据即上列原描述/标题，作者风险另据 Artist、Credit 及分类；未看像素。",
            "原响应：" + local_link(raw_refs[r["metadata_response"]], SOURCE / r["metadata_response"]) + f"；pageid={r['pageid']}；SHA256 `{r['metadata_response_sha256']}`。", "",
        ])
    packet.extend(["# 原响应索引", "", "各条目的 pageid 指向相应 JSON query.pages；以下路径只链接既有本地 API 响应，不是图片。", ""])
    for path, ref in raw_refs.items():
        packet.append(f"- {ref}：" + local_link(path, SOURCE / path) + f"；SHA256 `{sha(SOURCE / path)}`。")
    output = {
        "status": "complete_collector_packet_only", "original_representatives": 79,
        "collector_is_not_independent_reviewer": True, "selection_performed": False,
        "original_automatic_count_unchanged": True,
        "input_freeze_sha256": sha(freeze_path), "source_directory": str(SOURCE),
        "source_files_frozen": len(initial_inventory), "metadata_raw_response_count": len(raw_refs),
        "fixed_regex": {k: config[k] for k in ["scene_phrase_pattern", "negative_pattern"]},
        "network_requests": 0, "pixel_reads": 0, "image_downloads": 0, "model_calls": 0,
        "new_usable_images_declared": 0, "all_pixel_and_source_acceptance_pending": True,
        "rows": output_rows,
    }
    write("review_input.json", output)
    (HERE / "packet.md").write_text("\n".join(packet) + "\n")
    check("packet_exact_79_numbered_entries", len(re.findall(r"^## \d{2} ·", (HERE / "packet.md").read_text(), re.M)) == 79)
    check("source_all_files_unchanged", inventory() == initial_inventory)
    check("original_root_annotation_not_read_by_builder", True)
    write("packet_qa.json", {
        "status": "passed", "scope": "join, raw-field, unchanged-source and packet integrity only; not semantic/image acceptance",
        "checks": len(checks), "failures": 0, "source_file_count": len(initial_inventory),
        "original_representative_count": 79, "raw_metadata_responses": len(raw_refs),
        "new_usable_images": 0, "network_requests": 0, "pixel_reads": 0, "model_calls": 0,
        "items": checks,
    })
    write("packet_receipt.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_hashes": {n: sha(HERE / n) for n in ["input_freeze.json", "collector_notes.json", "prepare_packet.py", "review_input.json", "packet.md", "packet_qa.json"]},
        "source_inventory_still_matches": True, "independent_root_annotation_not_modified": True,
        "no_pixel_stage_authorized": True,
    })
    print(json.dumps({"status": "passed", "checks": len(checks), "representatives": 79,
                      "source_files_unchanged": len(initial_inventory), "packet": str(HERE / "packet.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
