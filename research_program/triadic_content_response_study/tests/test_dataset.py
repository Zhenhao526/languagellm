import numpy as np

from research_program.triadic_content_response_study import dataset


def test_static_case_geometry_is_complete():
    metadata, arrays = dataset.make_static()
    assert metadata["group_count"] == 48
    assert metadata["background_count"] == 36
    assert arrays["world_indices"].shape == (48, 36, 4)
    assert arrays["candidate_actions"].shape == (48, 36, 4, 3)
    assert arrays["candidate_receiver_actions"].shape == (48, 36, 4)
    assert sorted(np.unique(arrays["group_layer_index"]).tolist()) == list(range(12))
    assert all(np.unique(arrays["candidate_receiver_actions"][g, b]).size == 4
               for g in range(48) for b in range(36))


def test_row_expansion_has_all_sixteen_cells():
    _, arrays = dataset.make_static()
    row = dataset.flatten_rows(arrays)
    assert len(row["group_index"]) == 48 * 36 * 16
    assert np.all(np.bincount(row["host_endpoint"] * 4 + row["donor_endpoint"], minlength=16) == 48 * 36)
    assert np.all(row["host_indices"] == arrays["world_indices"][row["group_index"], row["background_index"], row["host_endpoint"]])
