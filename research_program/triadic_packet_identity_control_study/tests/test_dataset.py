import numpy as np

from research_program.triadic_packet_identity_control_study import dataset


def test_control_maps_preserve_declared_relations():
    metadata, maps = dataset.make_maps()
    _, _, arrays = dataset.source_arrays()
    rows = __import__("research_program.triadic_content_response_study.dataset", fromlist=["flatten_rows"]).flatten_rows(arrays)
    group = rows["group_index"]
    donor = rows["donor_endpoint"]
    assert metadata["groups"] == 48
    assert metadata["backgrounds"] == 36
    assert np.array_equal(maps["endpoint_cycle"]["source_group_index"], group)
    assert np.array_equal(maps["endpoint_cycle"]["source_packet_endpoint"], (donor + 1) % 4)
    assert np.array_equal(maps["cross_group_cycle"]["source_packet_endpoint"], donor)
    layers = arrays["group_layer_index"]
    cross_group = maps["cross_group_cycle"]["source_group_index"]
    assert np.all(layers[cross_group] == layers[group])
    assert np.all(cross_group != group)

