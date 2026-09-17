from qwen_language_v3.backend import allowed_choice_tokens


def test_action_prefixes_do_not_accept_outside_menu():
    assert allowed_choice_tokens('', [0, 1, 12, 23]) == (['0','1','2'], False)
    assert allowed_choice_tokens('1', [0, 1, 12, 23]) == (['2'], True)
    assert allowed_choice_tokens('2', [0, 1, 12, 23]) == (['3'], False)
    assert allowed_choice_tokens('23', [0, 1, 12, 23]) == ([], True)
    assert allowed_choice_tokens('21', [0, 1, 12, 23]) == ([], False)
