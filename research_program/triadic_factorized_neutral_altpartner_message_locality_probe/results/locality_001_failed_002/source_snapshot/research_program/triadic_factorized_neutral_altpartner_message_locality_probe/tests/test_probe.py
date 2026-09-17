import numpy as np

from research_program.triadic_factorized_neutral_altpartner_message_locality_probe import runner


def test_route_shape_and_live_visibility():
    tokens = np.arange(12, dtype=np.int8).reshape(1, 3, 4) % 8
    route = runner.routed_window(tokens, "live", "natural")
    assert route.shape == (1, 3, 99)
    assert np.all(route[:, :, 96:] == 1)
    assert np.isclose(route[:, 0, :96].sum(), 12)


def test_symbol_and_position_controls_preserve_token_multiset():
    tokens = np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [0, 7, 1, 6]]], dtype=np.int8)
    transformed = runner.transform_tokens(tokens, "symbol_permutation")
    assert np.array_equal(transformed, runner.SYMBOL_PERMUTATION[tokens])
    assert sorted(np.unique(runner.SYMBOL_PERMUTATION).tolist()) == list(range(8))
    for mode in ("position_rotation", "position_reverse"):
        transformed = runner.transform_tokens(tokens, mode)
        assert transformed.shape == tokens.shape
        assert np.all((transformed >= 0) & (transformed < 8))
        assert all(np.array_equal(np.sort(transformed[0, a]), np.sort(tokens[0, a]))
                   for a in range(3))


def test_sender_mask_keeps_self_payload_but_hides_cross_edges():
    tokens = np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [0, 7, 1, 6]]], dtype=np.int8)
    route = runner.routed_window(tokens, "live", "mask_sender_A")
    # Viewer A still sees own payload; viewers B/C see no A payload and bit.
    assert route[0, 0, :32].sum() == 4
    assert route[0, 1, :32].sum() == 0
    assert route[0, 2, :32].sum() == 0
    assert route[0, 1, 96] == 0 and route[0, 2, 96] == 0


def test_closed_channel_removes_all_visibility_bits_but_keeps_self_payload():
    tokens = np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [0, 7, 1, 6]]], dtype=np.int8)
    route = runner.routed_window(tokens, "live", "cross_closed")
    assert np.all(route[0, :, 96:] == 0)
    assert route[0, 0, :32].sum() == 4
    assert route[0, 0, 32:64].sum() == 0
