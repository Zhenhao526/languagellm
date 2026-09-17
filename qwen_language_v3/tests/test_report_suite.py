"""Small disk fixtures for final-score, action and interrupted-time reporting."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from qwen_language_v3.report_suite import build_suite, render_report, summarize_run, write_suite


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_fixture(parent, name="symbol_run", phase="pilot", conditions=None):
    path = Path(parent) / name
    path.mkdir()
    conditions = conditions or ["immediate", "delayed"]
    write_json(path / "manifest.json", {
        "phase": phase, "conditions": conditions, "groups": [17],
        "scenarios": [{"episode": 1, "seed": 6101, "variant": 0}],
    })
    write_json(path / "status.json", {"status": "running"})
    common = {"phase": phase, "condition": conditions[0], "group": 17, "episode": 1, "seed": 6101}
    before = {"inventory": {"A": {"tool": None, "cargo": None}}, "goals": [{"quantity": 2, "delivered": 0}]}
    holding = deepcopy(before)
    holding["inventory"]["A"]["tool"] = "axe"
    delivered = deepcopy(holding)
    delivered["goals"][0]["delivered"] = 1
    steps = [
        {**common, "step": 1, "score": 0, "actions": {"A": {"kind": "pickup"}, "B": {"kind": "cut"}, "C": {"kind": "wait"}},
         "feedback": {"A": {"action_succeeded": True}, "B": {"action_succeeded": False}, "C": {"action_succeeded": False}},
         "state_before": before, "state_after": holding},
        {**common, "step": 2, "score": 0.5, "actions": {"A": {"kind": "deliver_together"}, "B": {"kind": "deliver_together"}, "C": {"kind": "wait"}},
         "feedback": {agent: {"action_succeeded": True} for agent in "ABC"},
         "state_before": holding, "state_after": delivered},
    ]
    if len(conditions) > 1:
        steps.append({**common, "condition": conditions[1], "step": 1, "score": 0.75,
                      "actions": {"A": {"kind": "wait"}}, "feedback": {}, "state_before": before, "state_after": before})
    write_jsonl(path / "steps.jsonl", steps)
    write_jsonl(path / "episodes.jsonl", [{**common, "variant": 0, "score": 0.5, "steps": 2, "success": False, "seconds": 100}])
    return path


class SuiteTests(unittest.TestCase):
    def test_unfinished_episode_progress_is_not_a_final_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = summarize_run(make_fixture(tmp))
            rows = {row["condition"]: row for row in run["episodes"]}
            self.assertEqual(rows["immediate"]["final_score"], 0.5)
            self.assertIsNone(rows["delayed"]["final_score"])
            self.assertIsNone(rows["delayed"]["final_steps"])
            self.assertEqual(rows["delayed"]["last_progress_score_not_final"], 0.75)
            self.assertFalse(rows["delayed"]["completion_recorded"])

    def test_wait_is_separate_and_joint_delivery_units_are_not_doubled(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = summarize_run(make_fixture(tmp))
            row = next(row for row in run["episodes"] if row["condition"] == "immediate")
            actions = row["actions"]
            self.assertEqual(actions["wait_actions"], 2)
            self.assertEqual(actions["failed_nonwait_actions"], 1)
            self.assertEqual(actions["wait_actions_with_false_feedback"], 1)
            self.assertTrue(actions["holding_observed"])
            self.assertEqual(actions["successful_pickups"], 1)
            self.assertEqual(actions["pickup_attempts"], 1)
            self.assertFalse(actions["processing_observed"])
            self.assertEqual(actions["successful_processing_actions"], 0)
            self.assertEqual(actions["processing_attempts"], 1)
            self.assertTrue(actions["accepted_delivery_observed"])
            self.assertEqual(actions["accepted_delivery_actor_actions"], 2)
            self.assertEqual(actions["delivery_actor_attempts"], 2)
            self.assertEqual(actions["newly_delivered_units_from_states"], 1)

    def test_interrupted_call_is_excluded_without_subtracting_epoch_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_fixture(tmp, "calibration_natural_20260914", "calibration", ["natural"])
            write_jsonl(path / "inference.jsonl", [
                {"call": 248, "seconds": 2, "peak_memory_gb": 10.1, "mode": "action", "generation_tokens": 2, "valid": True},
                {"call": 249, "seconds": 1433.6, "peak_memory_gb": 11.0, "mode": "private_analysis", "generation_tokens": 256, "valid": True},
                {"call": 250, "seconds": 3, "peak_memory_gb": 10.5, "mode": "message", "generation_tokens": 5, "valid": True},
            ])
            write_json(path / "power_pause.json", {
                "paused_at_epoch": 10000, "resumed_at_epoch": 14282.9, "pause_seconds": 4282.9,
                "interrupted_calls": [{"call": 249, "seconds": 1433.6}],
            })
            write_json(path / "backend_stats.json", {"calls": 250, "inference_seconds": 1438.6})
            run = summarize_run(path)
            timing = run["timing"]
            self.assertEqual(timing["inference_records"], 3)
            self.assertEqual(timing["uninterrupted_call_records"], 2)
            self.assertEqual(timing["uninterrupted_seconds_sum"], 5)
            self.assertEqual(timing["interrupted_call_ids"], ["249"])
            self.assertIsNone(timing["interrupted_active_seconds"])
            self.assertEqual(timing["peak_mlx_memory_gb"], 11)
            report = render_report(build_suite([path]))
            self.assertIn("不可直接相减", report)
            self.assertIn("1438.60", report)

    def test_existing_symbol_analysis_is_referenced_and_natural_is_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_fixture(tmp)
            write_json(path / "analysis.json", {"groups": [{"condition": "immediate", "group": "17", "phases": {
                "pilot": {"messages": {"message_rows": 24, "length_all": {"mean": 4.25}, "nonempty_ratio": 0.5,
                                         "over_32_count": 0, "agent_steps_over_64": 0, "invalid_character_counts": {}}}}}]})
            run = summarize_run(path)
            ref = next(ref for ref in run["symbol_statistics_references"] if ref["condition"] == "immediate")
            self.assertEqual(ref["statistics_from_existing_analysis"]["length_all"]["mean"], 4.25)
            natural = make_fixture(tmp, "natural", "calibration", ["natural"])
            self.assertEqual(summarize_run(natural)["symbol_statistics_references"], [])

    def test_multi_run_file_output_is_read_only_and_separates_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            natural = make_fixture(tmp, "natural", "calibration", ["natural"])
            symbol = make_fixture(tmp, "symbol", "pilot")
            originals = {path: path.read_bytes() for directory in (natural, symbol) for path in directory.iterdir()}
            output = Path(tmp) / "final" / "总报告.md"
            written = write_suite([natural, symbol], output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("## 能力检查（calibration）", text)
            self.assertIn("## 符号工程先导", text)
            self.assertEqual(written["runs"], 2)
            self.assertTrue(output.with_suffix(".summary.json").exists())
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)

    def test_incomplete_jsonl_line_is_reported_without_inventing_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_fixture(tmp)
            with (path / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write('{"phase": "pilot",')
            run = summarize_run(path)
            self.assertTrue(any("第2行" in warning for warning in run["warnings"]))
            self.assertFalse(next(row for row in run["episodes"] if row["condition"] == "delayed")["completion_recorded"])

    def test_manual_stop_and_unstarted_control_are_not_completed_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_fixture(tmp, "clarified", "calibration", ["natural", "full_information"])
            write_jsonl(path / "episodes.jsonl", [])
            rows = [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]
            write_jsonl(path / "steps.jsonl", [row for row in rows if row["condition"] == "natural"])
            write_json(path / "status.json", {"status": "stopped_for_rule_clarity", "stop_reason": "clarify rules"})
            run = summarize_run(path)
            cases = {row["condition"]: row for row in run["episodes"]}
            self.assertIn("已中止", cases["natural"]["record_status"])
            self.assertIn("尚无", cases["full_information"]["record_status"])
            self.assertIsNone(cases["natural"]["final_score"])
            self.assertIsNone(cases["full_information"]["final_score"])
            self.assertTrue(run["timing"]["stopped_with_unfinished_episode"])
            self.assertIn("手动中止", render_report(build_suite([path])))

    def test_one_completed_control_does_not_complete_the_next_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_fixture(tmp, "complete_rules", "calibration", ["natural", "full_information"])
            run = summarize_run(path)
            cases = {row["condition"]: row for row in run["episodes"]}
            self.assertTrue(cases["natural"]["completion_recorded"])
            self.assertEqual(cases["natural"]["final_score"], 0.5)
            self.assertFalse(cases["full_information"]["completion_recorded"])
            self.assertIsNone(cases["full_information"]["final_score"])
            self.assertEqual(cases["full_information"]["last_progress_score_not_final"], 0.75)

    def test_frozen_core_source_versions_are_identified_without_engine_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = make_fixture(tmp, "original", "calibration", ["natural"])
            second = make_fixture(tmp, "updated_rules", "calibration", ["natural"])
            for path, rules in ((first, "rules_v0"), (second, "rules_v1")):
                source = path / "code_snapshot"
                source.mkdir()
                (source / "environment.py").write_text(rules)
                (source / "agents.py").write_text("same agent code")
            suite = build_suite([first, second])
            self.assertNotEqual(suite["runs"][0]["source_version"]["core_snapshot_sha256"],
                                suite["runs"][1]["source_version"]["core_snapshot_sha256"])
            self.assertEqual(suite["runs"][1]["source_difference_from_previous"]["changed_files"], ["environment.py"])
            self.assertIn("文本差异不自动意味着物理引擎行为不同", render_report(suite))


if __name__ == "__main__":
    unittest.main()
