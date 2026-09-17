import numpy as np

from research_program.triadic_content_response_study import dataset, metrics


def test_uniform_probabilities_have_zero_content_margin():
    _, arrays = dataset.make_static()
    row = dataset.flatten_rows(arrays)
    p = np.full((len(row["group_index"]), 3, 17), 1 / 17, dtype=np.float64)
    out = metrics.probe_metrics_with_layers(p, row["group_index"], row["background_index"],
        row["host_endpoint"], row["donor_endpoint"], row["listeners"],
        row["candidate_receiver_actions"], arrays["group_layer_index"])
    assert out["M"] == 0.0
    assert np.allclose(out["layer_means"], 0)
    assert len(out["margins"]) == len(row["group_index"])


def test_fixed_primary_requires_all_sixteen_seeds():
    assert metrics.statistics(np.arange(16, dtype=float))["n"] == 16
