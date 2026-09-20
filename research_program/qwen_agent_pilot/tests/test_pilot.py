from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

from research_program.qwen_agent_pilot.environment import (
    AGENTS, EPISODES_PER_BLOCK, MEANING_POOL, PILOT_EPISODES, make_episode, score_episode,
)
from research_program.qwen_agent_pilot.analyze import analyze_result
from research_program.qwen_agent_pilot.calibration import (
    CALIBRATION_SYSTEM, CODEWORDS, CONDITIONS, DEFAULT_SEEDS, _condition_helper_prompt,
    _condition_order, run_condition, run_matrix,
)
from research_program.qwen_agent_pilot.calibration_dev import DEVELOPMENT_SEED, evaluate_gate
from research_program.qwen_agent_pilot.pilot import parse_output, run_pilot, valid_message


def test_balanced_episode_generation():
    episodes = [make_episode(i, 9) for i in range(PILOT_EPISODES)]
    assert len(episodes) == 36
    assert EPISODES_PER_BLOCK == 18
    for block in range(2):
        chunk = episodes[block * EPISODES_PER_BLOCK:(block + 1) * EPISODES_PER_BLOCK]
        assert {(e["owner"], e["meaning_id"]) for e in chunk} == {
            (owner, meaning_id) for owner in AGENTS
            for meaning_id in range(len(MEANING_POOL))
        }
        assert [sum(e["owner"] == owner for e in chunk) for owner in AGENTS] == [6, 6, 6]
    assert make_episode(0, 9) == make_episode(0, 9)
    assert [
        sum(e["owner"] == owner and e["meaning_id"] == meaning_id for e in episodes)
        for owner in AGENTS for meaning_id in range(len(MEANING_POOL))
    ] == [2] * (len(AGENTS) * len(MEANING_POOL))
    try:
        make_episode(PILOT_EPISODES, 9)
    except ValueError:
        pass
    else:
        raise AssertionError("episode generator must reject indices outside the frozen pilot")
    for episode in episodes:
        assert episode["owner"] in AGENTS
        assert episode["goal"]["partner"] in episode["helpers"]
        assert sum((x["object"], x["attribute"]) ==
                   (episode["goal"]["object"], episode["goal"]["attribute"])
                   for x in episode["scene"]) == 1


def test_symbol_channel_and_invalid_suppression():
    assert valid_message("@#%&")
    assert valid_message("@#%&+=?~")
    assert not valid_message("abcde")
    assert not valid_message("@#%")
    assert not valid_message("@#%&+?~^")
    parsed = parse_output('{"message":"river camp @#%&"}', "owner")
    assert parsed["message"] == ""
    assert parsed["message_valid"] is False
    assert parse_output('["not", "an object"]', "owner")["json_valid"] is False
    assert parse_output('not json', "helper")["action_valid"] is False


def test_joint_reward_requires_assignment_and_destination():
    episode = make_episode(0, 9)
    goal = episode["goal"]
    correct = {goal["partner"]: {"item_id": episode["target_item_id"],
                                 "destination": goal["destination"]}}
    correct_outcome = score_episode(episode, correct)
    assert correct_outcome["success"]
    assert correct_outcome["designated_item_correct"]
    assert correct_outcome["designated_destination_correct"]
    wrong_destination_value = next(
        destination for destination in episode["destinations"]
        if destination != goal["destination"]
    )
    wrong_destination = {goal["partner"]: {"item_id": episode["target_item_id"],
                                           "destination": wrong_destination_value}}
    wrong_outcome = score_episode(episode, wrong_destination)
    assert not wrong_outcome["success"]
    assert wrong_outcome["designated_item_correct"]
    assert not wrong_outcome["designated_destination_correct"]
    other = next(a for a in episode["helpers"] if a != goal["partner"])
    duplicated = dict(correct)
    duplicated[other] = {"item_id": episode["target_item_id"],
                         "destination": goal["destination"]}
    assert not score_episode(episode, duplicated)["success"]


def test_mocked_runner_completes_balanced_schedule_and_metrics():
    calls = []

    def fake_post(base_url, model, messages, seed, temperature, max_tokens, timeout):
        prompt = messages[-1]["content"]
        calls.append((model, prompt))
        if "Private order:" in prompt:
            return '{"message":"@#%&"}', {"prompt_tokens": 10, "completion_tokens": 3}
        return '{"action":null}', {"prompt_tokens": 10, "completion_tokens": 3}

    health = Mock()
    with patch("research_program.qwen_agent_pilot.pilot.requests.get", return_value=health), \
            patch("research_program.qwen_agent_pilot.pilot._post", side_effect=fake_post):
        result = run_pilot("http://127.0.0.1:8080/v1", "mock", 9)

    assert len(calls) == 108
    assert result["episodes"] == PILOT_EPISODES
    assert result["owner_meaning_repeat_pairs"] == 18
    assert result["valid_message_rate"] == 1.0
    assert result["exact_message_repeat_rate_all_pairs"] == 1.0
    assert result["cross_sender_message_agreement_rate_all_meanings"] == 1.0
    assert result["prompt_tokens_reported_by_server"] == 1080
    assert result["team_success_rate_by_block"] == {"0": 0.0, "1": 0.0}
    assert result["designated_item_accuracy_by_block"] == {"0": 0.0, "1": 0.0}
    assert result["records"][0]["target_item_id"] in {f"I{i}" for i in range(4)}
    audit = analyze_result(result)
    assert audit["episodes"] == PILOT_EPISODES
    assert audit["stable_sender_meaning_pairs"] == 18
    assert audit["cross_sender_meaning_agreement"] == 6


def test_calibration_conditions_are_matched_and_codebook_is_valid():
    assert len(CODEWORDS) == len(MEANING_POOL) == 6
    assert all(valid_message(word) and len(word) == 6 for word in CODEWORDS)
    assert all(
        sum(left != right for left, right in zip(CODEWORDS[i], CODEWORDS[j])) >= 4
        for i in range(len(CODEWORDS)) for j in range(i + 1, len(CODEWORDS))
    )
    assert _condition_order(0) == list(CONDITIONS)
    assert _condition_order(1) == ["known_codebook", "free_symbols", "blank"]
    assert _condition_order(2) == ["free_symbols", "blank", "known_codebook"]
    assert DEFAULT_SEEDS == (20260925, 20260926, 20260927)
    assert DEVELOPMENT_SEED == 20260929
    assert "Every order has exactly one designated helper" in CALIBRATION_SYSTEM
    episode = make_episode(0, 9)
    helper_prompts = [
        _condition_helper_prompt(episode, episode["helpers"][0], "@#%&", condition)
        for condition in CONDITIONS
    ]
    assert all("exactly one helper is responsible" in prompt for prompt in helper_prompts)
    assert all("wait rather than duplicate the action" in prompt for prompt in helper_prompts)
    assert all("scan every public candidate-board entry" in prompt for prompt in helper_prompts)
    assert all("do not infer the item from its position" in prompt for prompt in helper_prompts)
    codebook_prompt = helper_prompts[CONDITIONS.index("known_codebook")]
    assert "You do not know the private order." not in codebook_prompt
    assert "Do not abstain when the matching row names you." in codebook_prompt
    assert "copy that entry's item_id" in codebook_prompt
    assert "responsible_helper" in codebook_prompt

    schedules = {}
    for condition in CONDITIONS:
        calls = []

        def fake_post(base_url, model, messages, seed, temperature, max_tokens, timeout):
            prompt = messages[-1]["content"]
            calls.append(prompt)
            if "Private order:" in prompt and condition == "blank":
                return '{"message":""}', {"prompt_tokens": 10, "completion_tokens": 2}
            if "Private order:" in prompt:
                return '{"message":"+%~?=&"}', {"prompt_tokens": 10, "completion_tokens": 3}
            return '{"action":null}', {"prompt_tokens": 10, "completion_tokens": 3}

        with patch("research_program.qwen_agent_pilot.calibration.requests.get", return_value=Mock()), \
                patch("research_program.qwen_agent_pilot.calibration._post", side_effect=fake_post):
            result = run_condition("http://127.0.0.1:8080/v1", "mock", 9, condition)

        assert len(calls) == 108
        assert result["episodes"] == PILOT_EPISODES
        assert result["model_calls"] == 108
        assert result["team_success_rate"] == 0.0
        if condition == "blank":
            assert all(row["message"] == "" and row["message_valid"] is None for row in result["records"])
        elif condition == "known_codebook":
            assert result["known_codebook_encoder_accuracy"] == 1 / 6
        else:
            assert result["valid_message_rate_nonblank_channels"] == 1.0
        schedules[condition] = [
            (row["episode"], row["block"], row["owner"], row["meaning_id"], row["goal"])
            for row in result["records"]
        ]
    assert schedules["blank"] == schedules["known_codebook"] == schedules["free_symbols"]


def test_calibration_development_gate():
    passing = {
        "known_codebook_encoder_accuracy": 1.0,
        "designated_helper_both_correct_rate": 30 / 36,
        "unassigned_wait_rate": 30 / 36,
        "team_success_rate": 28 / 36,
    }
    assert evaluate_gate(passing)["passed"]
    passing["unassigned_wait_rate"] = 28 / 36
    assert not evaluate_gate(passing)["passed"]


def test_calibration_matrix_checkpoints_and_resumes():
    calls = []

    def fake_run_condition(base_url, model, seed, condition, temperature, max_tokens, timeout):
        calls.append((seed, condition))
        return {"seed": seed, "condition": condition, "team_success_rate": 0.25,
                "episodes": PILOT_EPISODES, "records": []}

    with TemporaryDirectory() as temp_dir:
        out = Path(temp_dir) / "calibration.json"
        with patch("research_program.qwen_agent_pilot.calibration.run_condition",
                   side_effect=fake_run_condition):
            result = run_matrix("http://localhost/v1", "mock", (11, 12), out)
        assert len(result["runs"]) == 6
        assert len(result["paired_seed_contrasts"]) == 2
        assert len(calls) == 6
        with patch("research_program.qwen_agent_pilot.calibration.run_condition") as skipped:
            resumed = run_matrix("http://localhost/v1", "mock", (11, 12), out, resume=True)
        skipped.assert_not_called()
        assert resumed["runs"] == result["runs"]


if __name__ == "__main__":
    test_balanced_episode_generation()
    test_symbol_channel_and_invalid_suppression()
    test_joint_reward_requires_assignment_and_destination()
    test_mocked_runner_completes_balanced_schedule_and_metrics()
    test_calibration_conditions_are_matched_and_codebook_is_valid()
    test_calibration_development_gate()
    test_calibration_matrix_checkpoints_and_resumes()
    print("qwen_agent_pilot tests passed")
