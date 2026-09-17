#!/usr/bin/env python3
"""Pure metadata nomination; never downloads, decodes, or classifies images.

Every hash tuple is compact UTF-8 JSON with ensure_ascii=False. Components are
computed on the entire source frame before any eligibility filter. No output
is a declaration that a photograph, source identity, or license was accepted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import unicodedata
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
FRAME = HERE.parent
SALT = "visual-curation-20260915-v1"
SOURCE_RECORD_SHA256 = "05b100ca038cb199c974ade0457203547c2767d3be3f31ec65b94241dd7aef5c"
CATEGORIES = {"Apples": "apple", "Bananas": "banana", "Oranges": "orange",
              "Glasses of water": "water"}
FOOD = ("apple", "banana", "orange")
LABELS = (*FOOD, "water")
MIMES = {"image/jpeg", "image/png", "image/tiff", "image/webp"}
OLD_FIELDS = ("old_title_exact_match", "old_title_normalized_match", "old_sha1_match",
              "old_author_key_match", "old_source_key_match")
MAX_FILE_BYTES = 16 * 1024**2
MAX_TOTAL_BYTES = 2 * 1024**3
TARGETS = {"apple": 28, "banana": 28, "orange": 28, "water": 84}
CAPS = {"apple": 42, "banana": 42, "orange": 42, "water": 126}
MARKER_FIELDS = ("Categories", "ObjectName", "ImageDescription")
# Closed pre-pixel list: do not add markers in response to candidate counts.
MARKERS = (
    ("ai_generated", r"\bAI[\s-]*generated\b"),
    ("artificial_intelligence_generated", r"\bartificial\s+intelligence[\s-]+generated\b"),
    ("midjourney", r"\bMidjourney\b"),
    ("stable_diffusion", r"\bStable\s+Diffusion\b"),
    ("dall_e", r"\bDALL[\s-]*E\b"),
    ("computer_generated", r"\bcomputer[\s-]*generated\b"),
    ("3d_render", r"\b3D[\s-]+render(?:ed|ing|s)?\b"),
    ("digital_illustration", r"\bdigital\s+illustrations?\b"),
    ("drawings_of", r"\bdrawings?\s+of\b"),
    ("paintings_of", r"\bpaintings?\s+of\b"),
)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def rank(kind, value):
    return digest([SALT, kind, value])


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


class Plain(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, text):
        self.parts.append(text)


class SourceLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self.links.extend(value for key, value in attrs if key.lower() == "href" and value)


def raw_commons_sources(row):
    result = defaultdict(list)
    for field in ("Credit", "Source"):
        parser = SourceLinks()
        parser.feed(str(row.get("extmetadata", {}).get(field, {}).get("value", "")))
        for href in parser.links:
            parsed = urlsplit(href if not href.startswith("//") else "https:" + href)
            path = unquote(parsed.path)
            if parsed.hostname == "commons.wikimedia.org" and path.startswith("/wiki/File:"):
                title = path[len("/wiki/"):].replace("_", " ")
                result["source_commons_file:" + normalized_title(title)].append(
                    {"field": field, "href": href, "raw_title": title})
    return result


def marker_hits(row):
    hits = []
    for field in MARKER_FIELDS:
        value = row.get("extmetadata", {}).get(field, {}).get("value", "")
        parser = Plain()
        parser.feed(str(value))
        text = " ".join(" ".join(parser.parts).split())
        for marker, pattern in MARKERS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                hits.append({"field": field, "marker": marker,
                             "match": match.group(),
                             "context": text[max(0, match.start()-60):match.end()+60]})
    return hits


def source_keys(row):
    keys = list(row["author_keys"]) + list(row["source_keys"])
    if row.get("original_file_sha1"):
        keys.append("sha1:" + row["original_file_sha1"])
    return sorted(set(keys))


def normalized_title(value):
    return unicodedata.normalize("NFKC", value).replace("_", " ").strip().casefold()


def global_components(rows, old_normalized_titles=()):
    """Return deterministic components on ALL rows, including rejected bridges."""
    byid = {r["pageid"]: r for r in rows}
    if len(byid) != len(rows):
        raise ValueError("duplicate pageid")
    keys_to_pages = defaultdict(list)
    titles_to_pages = defaultdict(set)
    for row in rows:
        for key in source_keys(row):
            keys_to_pages[key].append(row["pageid"])
        for title in (row["title"], row.get("current_title") or row["title"]):
            titles_to_pages[normalized_title(title)].add(row["pageid"])
    old_titles = set(old_normalized_titles)
    explicit_neighbors = defaultdict(set)
    explicit_links = defaultdict(list)
    for row in rows:
        raw_sources = raw_commons_sources(row)
        for key in row["source_keys"]:
            if not key.startswith("source_commons_file:"):
                continue
            title = key[len("source_commons_file:"):]
            targets = sorted(titles_to_pages.get(title, ()))
            raw_refs = raw_sources.get(key, [])
            matched_titles = [{"pageid": pid, "title": byid[pid]["title"],
                               "current_title": byid[pid].get("current_title")}
                              for pid in targets]
            exact = len(targets) == 1 and any(
                ref["raw_title"] in (byid[targets[0]]["title"].replace("_", " "),
                                      (byid[targets[0]].get("current_title") or "").replace("_", " "))
                for ref in raw_refs)
            link = {"from_pageid": row["pageid"], "source_key": key, "target_pageids": targets,
                    "raw_references": raw_refs, "matched_titles": matched_titles,
                    "edge_certainty": "exact_title" if exact else "normalized_uncertain" if len(targets) == 1 else "not_linked",
                    "points_to_old_normalized_title": title in old_titles,
                    "status": "unresolved_outside_frame" if not targets else
                              "unique_target" if len(targets) == 1 else "ambiguous_normalized_title"}
            explicit_links[row["pageid"]].append(link)
            if len(targets) == 1:
                explicit_neighbors[row["pageid"]].add(targets[0])
                explicit_neighbors[targets[0]].add(row["pageid"])
    remaining = set(byid)
    groups = []
    while remaining:
        stack = [min(remaining)]
        members, keys = set(), set()
        while stack:
            pageid = stack.pop()
            if pageid in members:
                continue
            members.add(pageid)
            remaining.discard(pageid)
            stack.extend(explicit_neighbors[pageid] - members)
            for key in source_keys(byid[pageid]):
                if key not in keys:
                    keys.add(key)
                    stack.extend(keys_to_pages[key])
        member_ids = sorted(members)
        component_id = digest(member_ids)
        old = [{"pageid": pid, "fields": [f for f in OLD_FIELDS if byid[pid].get(f)]}
               for pid in member_ids if any(byid[pid].get(f) for f in OLD_FIELDS)]
        links = [link for pid in member_ids for link in explicit_links[pid]]
        old += [{"pageid": link["from_pageid"], "fields": ["explicit_source_points_to_old_normalized_title"],
                 "source_key": link["source_key"]}
                for link in links if link["points_to_old_normalized_title"]]
        groups.append({"component_id": component_id, "component_rank": rank("component", component_id),
                       "pageids": member_ids, "keys": sorted(keys),
                       "explicit_commons_source_links": links,
                       "ambiguous_source_title": any(x["status"] == "ambiguous_normalized_title" for x in links),
                       "old_contaminated": bool(old), "old_evidence": old})
    return sorted(groups, key=lambda g: (g["component_rank"], g["component_id"]))


def metadata_decision(row, component):
    labels = sorted({CATEGORIES[c] for c in row["categories"]})
    reasons = []
    if row["status"] != "resolved":
        reasons.append("metadata_unresolved")
    if component["old_contaminated"]:
        reasons.append("global_component_old_material_or_author_or_source")
    if component["ambiguous_source_title"]:
        reasons.append("global_component_ambiguous_commons_source_target")
    if len(labels) != 1:
        reasons.append("category_overlap")
    if row["mime"] not in MIMES:
        reasons.append("mime_not_static_raster_candidate")
    if min(row.get("width", 0) or 0, row.get("height", 0) or 0) < 512:
        reasons.append("short_side_below_512")
    size = row.get("size")
    if type(size) is not int or not (0 < size <= MAX_FILE_BYTES):
        reasons.append("file_size_missing_or_outside_1_to_16MiB")
    if row.get("license_pattern_supported") is not True:
        reasons.append("license_pattern_not_supported")
    if not row["author_keys"]:
        reasons.append("author_keys_missing")
    hits = marker_hits(row)
    if hits:
        reasons.append("explicit_generated_or_nonphoto_marker")
    return {"pageid": row["pageid"], "title": row["title"], "categories": row["categories"],
            "labels_from_metadata": labels, "component_id": component["component_id"],
            "metadata_eligible": not reasons, "metadata_exclusions": reasons,
            "marker_hits": hits, "pool": None, "pool_exclusions": [],
            "nomination": None, "later_checks_status": "not_started_no_pixels"}


def build_candidates(rows, old_normalized_titles=()):
    groups = global_components(rows, old_normalized_titles)
    byid = {r["pageid"]: r for r in rows}
    group_by_page = {pid: g for g in groups for pid in g["pageids"]}
    inventory = {pid: metadata_decision(row, group_by_page[pid]) for pid, row in byid.items()}
    candidates_by_group = {}
    water_reserved = set()
    for group in groups:
        options = defaultdict(list)
        for pid in group["pageids"]:
            inv = inventory[pid]
            if inv["metadata_eligible"]:
                options[inv["labels_from_metadata"][0]].append(pid)
        if options.get("water"):
            water_reserved.add(group["component_id"])
        candidates_by_group[group["component_id"]] = {
            label: sorted(pids, key=lambda pid: (rank("file", pid), pid))
            for label, pids in options.items()}
    for pid, inv in inventory.items():
        if inv["metadata_eligible"]:
            label = inv["labels_from_metadata"][0]
            if label != "water" and inv["component_id"] in water_reserved:
                inv["pool_exclusions"].append("component_reserved_for_water")
            else:
                inv["pool"] = label

    orders = {label: [] for label in LABELS}
    used = set()

    def nominate(group, label):
        cid = group["component_id"]
        pid = candidates_by_group[cid][label][0]
        row = byid[pid]
        index = len(orders[label]) + 1
        review_id = "V_" + rank("review_id", pid)[:20]
        nomination = {
            "pageid": pid, "component_id": cid, "nomination_hash": group["component_rank"],
            "file_rank": rank("file", pid), "label": label, "fixed_order": index,
            "role": "primary" if index <= TARGETS[label] else "reserve",
            "review_id": review_id, "preview_file": f"previews/{review_id}.png",
            "title": row["title"], "original_url": row["original_url"],
            "description_url": row["description_url"], "original_file_sha1": row["original_file_sha1"],
            "mime": row["mime"], "width": row["width"], "height": row["height"], "size": row["size"],
            "license_short_name": row["license_short_name"], "license_url": row["license_url"],
            "author_keys": row["author_keys"], "source_keys": row["source_keys"]}
        orders[label].append(nomination)
        inventory[pid]["nomination"] = {k: nomination[k] for k in ("label", "fixed_order", "role", "review_id")}
        used.add(cid)

    for group in groups:
        if group["component_id"] in water_reserved and len(orders["water"]) < CAPS["water"]:
            nominate(group, "water")
    for _ in range(max(CAPS[label] for label in FOOD)):
        for label in FOOD:
            if len(orders[label]) >= CAPS[label]:
                continue
            for group in groups:
                cid = group["component_id"]
                if cid not in water_reserved and cid not in used and candidates_by_group[cid].get(label):
                    nominate(group, label)
                    break

    for pid, inv in inventory.items():
        if inv["pool"] is not None and inv["nomination"] is None:
            cid = inv["component_id"]
            inv["pool_exclusions"].append("component_already_nominated" if cid in used
                                          else "outside_fixed_nomination_cap")
    nominated = [n for label in LABELS for n in orders[label]]
    assert len(nominated) == len({n["component_id"] for n in nominated})
    assert len(nominated) == len({n["pageid"] for n in nominated})
    assert len(nominated) == len({n["review_id"] for n in nominated})
    assert len(nominated) <= 252
    reviewer = [{"review_id": n["review_id"], "preview_file": n["preview_file"]}
                for n in sorted(nominated, key=lambda n: (rank("presentation", n["review_id"]), n["review_id"]))]
    counts = {}
    for label in LABELS:
        eligible = [r for r in inventory.values() if r["metadata_eligible"] and r["labels_from_metadata"] == [label]]
        pool = [r for r in inventory.values() if r["pool"] == label]
        counts[label] = {"target_accepted": TARGETS[label], "nomination_cap": CAPS[label],
                         "metadata_eligible_files": len(eligible),
                         "metadata_eligible_global_components": len({r["component_id"] for r in eligible}),
                         "pool_files_after_water_priority": len(pool),
                         "pool_global_components_after_water_priority": len({r["component_id"] for r in pool}),
                         "nominated": len(orders[label]),
                         "primary": min(TARGETS[label], len(orders[label])),
                         "reserve": max(0, len(orders[label]) - TARGETS[label]),
                         "minimum_shortfall_before_visual_review": max(0, TARGETS[label] - len(orders[label]))}
    total_bytes = sum(n["size"] for n in nominated)
    summary = {"metadata_records": len(rows), "global_components_before_filtering": len(groups),
               "global_components_old_contaminated": sum(g["old_contaminated"] for g in groups),
               "explicit_commons_source_link_status_counts": dict(Counter(
                   link["status"] for g in groups for link in g["explicit_commons_source_links"])),
               "explicit_commons_source_edge_certainty_counts": dict(Counter(
                   link["edge_certainty"] for g in groups for link in g["explicit_commons_source_links"])),
               "water_reserved_global_components": len(water_reserved),
               "exclusion_counts_nonexclusive": dict(Counter(reason for r in inventory.values()
                                                              for reason in r["metadata_exclusions"])),
               "labels": counts, "nominated_files": len(nominated),
               "declared_original_bytes": total_bytes, "max_original_bytes": MAX_TOTAL_BYTES,
               "max_original_requests": 252, "declared_bytes_within_budget": total_bytes <= MAX_TOTAL_BYTES,
               "minimum_metadata_targets_available": all(len(orders[x]) >= TARGETS[x] for x in LABELS),
               "pixel_requests": 0, "pixels_viewed": 0, "visual_or_license_acceptance_complete": False,
               "material_sets_assigned": False}
    return {"summary": summary, "inventory": [inventory[pid] for pid in sorted(inventory)],
            "components": groups, "orders": orders, "reviewer_manifest": reviewer}


def prepare(out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"refuse to overwrite {out}")
    source = FRAME / "details_002/execution/records.json"
    if file_sha(source) != SOURCE_RECORD_SHA256:
        raise ValueError("source metadata differs from independently audited frame")
    rows = json.loads(source.read_text())
    if len(rows) != 1876 or any(r["status"] != "resolved" for r in rows):
        raise ValueError("expected complete 1876-file metadata frame")
    sources = [Path(__file__).resolve(), HERE / "plan.md", HERE / "test_prepare_candidates.py", source,
               FRAME / "details_002/execution/feasibility.json", FRAME / "details_002/execution/result.json",
               FRAME / "details_002/plan.json", FRAME / "details_002/独立核验.json",
               FRAME / "details_002/audit_metadata.py", FRAME / "details_001/old_index_evidence.json"]
    out.mkdir(parents=True)
    (out / "source_snapshot").mkdir()
    source_records = []
    for index, path in enumerate(sources):
        target = out / "source_snapshot" / f"{index:02d}_{path.name}"
        shutil.copyfile(path, target)
        source_records.append({"path": str(path), "sha256": file_sha(path),
                               "snapshot": str(target.relative_to(out)), "snapshot_sha256": file_sha(target)})
    freeze = {"created_at": datetime.now(timezone.utc).isoformat(),
              "status": "frozen_before_metadata_ranking", "source_files": source_records,
              "salt": SALT, "hash_encoding": "compact ensure_ascii=False UTF-8 JSON",
              "component_id_encoding": "sorted integer pageid array",
              "rank_encoding": "[salt,kind,value]", "markers": list(MARKERS),
              "metadata_only": True, "pixel_fetch_authorized_by_this_script": False}
    write_json(out / "freeze.json", freeze)
    try:
        old_index = json.loads((FRAME / "details_001/old_index_evidence.json").read_text())
        built = build_candidates(rows, old_index["normalized_titles"])
        write_json(out / "candidate_inventory.json", built["inventory"])
        write_json(out / "global_components.json", built["components"])
        write_json(out / "candidate_order.json", built["orders"])
        write_json(out / "reviewer_manifest.json", built["reviewer_manifest"])
        with (out / "review_template.jsonl").open("w", encoding="utf-8") as stream:
            for anonymous in built["reviewer_manifest"]:
                template = {**anonymous, "reviewer": None, "reviewed_at": None,
                            "visual_class": None, "visual_class_allowed": [*LABELS, "mixed", "uncertain"],
                            "real_photo": None, "main_resource_visible": None,
                            "target_occupancy_roughly_at_least_15_percent": None,
                            "severe_blur_exposure_or_occlusion": None, "mixed_target_resources": None,
                            "readable_label_text_or_prominent_logo": None,
                            "water_container_and_surface_visible_if_water": None,
                            "decision": None, "decision_allowed": ["pass", "reject", "uncertain"],
                            "reasons": [], "notes": None}
                stream.write(json.dumps(template, ensure_ascii=False) + "\n")
        outputs = {p.name: file_sha(p) for p in out.iterdir() if p.is_file()}
        result = {"status": "metadata_ranking_complete_pending_root_review",
                  "completed_at": datetime.now(timezone.utc).isoformat(),
                  "summary": built["summary"], "output_sha256": outputs,
                  "interpretation": "metadata components are not verified photographers; no pixels, downloads, models, acceptance, or material allocation occurred"}
        write_json(out / "result.json", result)
        return result
    except BaseException as exc:
        write_json(out / "failure.json", {"failed_at": datetime.now(timezone.utc).isoformat(),
                                          "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "ranking_001")
    args = parser.parse_args()
    result = prepare(args.out)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
