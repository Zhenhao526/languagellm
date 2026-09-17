import numpy as np

from research_program.triadic_factorized_neutral_altpartner_direction_probe import design, intervention


def test_neighbor_and_case_counts():
    prepared = design.make_prepared()
    assert prepared['partition']['world_count'] == 1560 * 6 * 6
    assert prepared['cases']['case_count'] == 2112 * 6 * 6
    assert len(prepared['cases']['receiver_state_indices']) == len(prepared['cases']['donor_state_indices'])
    assert set(prepared['cases']['axis']) == {'kind', 'length', 'destination'}


def test_packet_replacement_keeps_sender_private_view():
    native = np.arange(2 * 3 * 4, dtype=np.int8).reshape(2, 3, 4) % 8
    persons = np.array([0, 2], dtype=np.int64)
    donor = np.array([[7, 6, 5, 4], [3, 2, 1, 0]], dtype=np.int8)
    out, routes = intervention._replace(native, persons, donor)
    assert out.shape == (2, 3, 3, 4)
    for row, sender in enumerate(persons):
        assert np.array_equal(out[row, sender, sender], native[row, sender])
        for viewer in range(3):
            if viewer != sender:
                assert np.array_equal(out[row, viewer, sender], donor[row])
    assert routes.shape == (2, 3, 99)
