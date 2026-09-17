import numpy as np

from research_program.triadic_packet_identity_control_study import metrics


def test_four_choice_margin_keeps_all_action_probability_scale():
    n = 16
    probabilities = np.zeros((n, 3, 17), dtype=np.float64)
    row = dict(group_index=np.zeros(n, dtype=np.int64), background_index=np.zeros(n, dtype=np.int64),
               host_endpoint=np.repeat(np.arange(4), 4), donor_endpoint=np.tile(np.arange(4), 4),
               listeners=np.zeros(n, dtype=np.int8), candidate_receiver_actions=np.tile(np.arange(4), (n, 1)))
    probabilities[:, 0, :4] = 0.10
    probabilities[np.arange(n), 0, row["donor_endpoint"]] = 0.50
    result = metrics.margins(probabilities, row["group_index"], row["background_index"], row["host_endpoint"],
                             row["donor_endpoint"], row["listeners"], row["candidate_receiver_actions"])
    assert np.isclose(result["M"], 0.40)
    assert np.isclose(result["off_diagonal_margin_mean"], 0.40)
