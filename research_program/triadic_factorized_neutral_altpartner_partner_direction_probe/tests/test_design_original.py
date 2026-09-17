import numpy as np

from research_program.triadic_factorized_neutral_altpartner_partner_direction_probe import design, intervention


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
