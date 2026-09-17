from qwen_language_v3.agents import build_prompt, remember


def test_prompt_inputs_are_private_and_not_mutated():
    obs = {'agent':'A','local_items':['local_A_7']}
    history = [{'观察':{'old':'only A'}}]
    prompt = build_prompt('A', rules='共同规则', observation=obs, history=history,
        transcript=[], condition='immediate', window=1, remaining=64)
    assert 'local_A_7' in prompt[1]['content']
    assert 'full_information' not in prompt[1]['content']
    assert obs == {'agent':'A','local_items':['local_A_7']}
    assert history == [{'观察':{'old':'only A'}}]


def test_private_history_records_only_own_observation_action_feedback():
    h = {a:[] for a in 'ABC'}
    remember(h, observations={a:{'secret':a} for a in 'ABC'}, messages=[],
        actions={a:{'own_action':a} for a in 'ABC'}, feedback={a:{'own_result':a} for a in 'ABC'})
    assert h['A'][0]['观察'] == {'secret':'A'}
    assert h['A'][0]['自己执行的动作'] == {'own_action':'A'}
    h['A'][0]['公开广播'].append('mutation')
    assert h['B'][0]['公开广播'] == []
    assert set(h['A'][0]) == {'观察','公开广播','自己执行的动作','自己可见结果'}
