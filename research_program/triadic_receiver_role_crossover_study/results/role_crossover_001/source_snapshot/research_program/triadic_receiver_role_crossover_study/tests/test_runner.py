from research_program.triadic_receiver_role_crossover_study import runner


def test_zero_frozen_gradients_updates_only_selected_role():
    import numpy as np
    networks = [{"w": np.ones((2, 2))} for _ in range(9)]
    gradients = [{"w": np.full((2, 2), index + 1.0)} for index in range(9)]
    output = runner.zero_frozen_gradients(gradients, networks, 1)
    assert all(np.array_equal(output[index]["w"], 0) for index in (0, 1, 2, 6, 7, 8))
    assert np.array_equal(output[3]["w"], gradients[3]["w"])
    assert np.array_equal(output[4]["w"], gradients[4]["w"])
    assert np.array_equal(output[5]["w"], gradients[5]["w"])
