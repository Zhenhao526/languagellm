"""Synthetic metadata tests only: no requests, pixel reading, or model import."""
import copy
import unittest

import prepare_candidates as p


def row(pid, label="apple", author=None, sources=(), **extra):
    category = {v: k for k, v in p.CATEGORIES.items()}[label]
    result = {"pageid": pid, "title": f"File:Example {pid}.jpg", "categories": [category],
              "status": "resolved", "current_title": f"File:Example {pid}.jpg",
              "original_file_sha1": f"{pid:040x}", "author_keys": [author or f"artist_text:person {pid}"],
              "source_keys": list(sources), "mime": "image/jpeg", "width": 512, "height": 1024,
              "size": 1000, "license_pattern_supported": True, "extmetadata": {},
              "original_url": f"https://example.invalid/{pid}.jpg",
              "description_url": f"https://example.invalid/File:Example_{pid}.jpg",
              "license_short_name": "CC0", "license_url": "https://example.invalid/cc0"}
    result.update({key: False for key in p.OLD_FIELDS})
    result.update(extra)
    return result


class CandidateTests(unittest.TestCase):
    def test_old_bridge_retained_before_filtering(self):
        rows = [row(1, author="author:a"), row(2, author="author:a", sources=["source:x"],
                mime="image/svg+xml", old_sha1_match=True), row(3, sources=["source:x"])]
        built = p.build_candidates(rows)
        self.assertEqual(built["summary"]["global_components_before_filtering"], 1)
        self.assertTrue(all("global_component_old_material_or_author_or_source" in r["metadata_exclusions"]
                            for r in built["inventory"]))

    def test_explicit_parent_edge_and_old_title(self):
        parent = row(1)
        child = row(2, sources=["source_commons_file:file:example 1.jpg"])
        child["extmetadata"] = {"Credit": {"value": '<a href="https://commons.wikimedia.org/wiki/File:Example_1.jpg">source</a>'}}
        built = p.build_candidates([parent, child], ["file:example 1.jpg"])
        self.assertEqual(len(built["components"]), 1)
        link = built["components"][0]["explicit_commons_source_links"][0]
        self.assertEqual(link["edge_certainty"], "exact_title")
        self.assertTrue(built["components"][0]["old_contaminated"])

    def test_normalized_parent_uncertainty_preserved(self):
        parent = row(1, title="File:BANANAS.jpg", current_title="File:BANANAS.jpg")
        child = row(2, sources=["source_commons_file:file:bananas.jpg"],
                    extmetadata={"Credit": {"value": '<a href="https://commons.wikimedia.org/wiki/File:Bananas.JPG">source</a>'}})
        built = p.build_candidates([parent, child])
        self.assertEqual(len(built["components"]), 1)
        self.assertEqual(built["components"][0]["explicit_commons_source_links"][0]["edge_certainty"], "normalized_uncertain")

    def test_ambiguous_parent_not_guessed(self):
        a = row(1, title="File:X.jpg", current_title="File:X.jpg")
        b = row(2, title="File:x.jpg", current_title="File:x.jpg")
        c = row(3, sources=["source_commons_file:file:x.jpg"])
        built = p.build_candidates([a, b, c])
        self.assertEqual(len(built["components"]), 3)
        third = next(r for r in built["inventory"] if r["pageid"] == 3)
        self.assertIn("global_component_ambiguous_commons_source_target", third["metadata_exclusions"])

    def test_water_priority_and_no_substitute_from_component(self):
        rows = [row(1, author="author:shared"), row(2, "water", author="author:shared"),
                row(3, "water", author="author:shared")]
        built = p.build_candidates(rows)
        self.assertEqual(len(built["orders"]["water"]), 1)
        self.assertEqual(built["orders"]["apple"], [])
        self.assertIn("component_reserved_for_water", built["inventory"][0]["pool_exclusions"])

    def test_overlap_markers_dimensions_and_size(self):
        rows = [row(1, categories=["Apples", "Glasses of water"]), row(2, width=511),
                row(3, size=p.MAX_FILE_BYTES + 1),
                row(4, extmetadata={"Categories": {"value": "AI-generated images of apples"}}),
                row(5, extmetadata={"ImageDescription": {"value": "a real apple"}})]
        built = p.build_candidates(rows)
        self.assertEqual([r["pageid"] for r in built["inventory"] if r["metadata_eligible"]], [5])
        self.assertEqual(built["inventory"][3]["marker_hits"][0]["marker"], "ai_generated")

    def test_caps_reserves_blinding_and_determinism(self):
        rows = [row(1_000 * j + i, label) for j, label in enumerate(p.LABELS, 1) for i in range(1, 140)]
        original = copy.deepcopy(rows)
        built = p.build_candidates(rows)
        again = p.build_candidates(list(reversed(rows)))
        self.assertEqual(rows, original)
        self.assertEqual(built, again)
        self.assertEqual([len(built["orders"][x]) for x in p.LABELS], [42, 42, 42, 126])
        self.assertEqual(sum(n["role"] == "primary" for arr in built["orders"].values() for n in arr), 168)
        self.assertEqual(built["summary"]["nominated_files"], 252)
        self.assertTrue(all(set(r) == {"review_id", "preview_file"} for r in built["reviewer_manifest"]))
        self.assertFalse(built["summary"]["material_sets_assigned"])


if __name__ == "__main__":
    unittest.main()
