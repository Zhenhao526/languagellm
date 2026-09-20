from unittest.mock import Mock, patch

from research_program.qwen_agent_pilot.environment import (
    AGENTS, EPISODES_PER_BLOCK, MEANING_POOL, PILOT_EPISODES, make_episode, score_episode,
)
from research_program.qwen_agent_pilot.analyze import analyze_result
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


if __name__ == "__main__":
    test_balanced_episode_generation()
    test_symbol_channel_and_invalid_suppression()
    test_joint_reward_requires_assignment_and_destination()
    test_mocked_runner_completes_balanced_schedule_and_metrics()
    print("qwen_agent_pilot tests passed")
