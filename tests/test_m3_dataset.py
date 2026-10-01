import numpy as np

from scripts.build_fundidora_m3_dataset import aggregate_to_grid


def test_aggregate_to_grid_mean_and_valid_fraction():
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False

    means, fractions = aggregate_to_grid(values, valid, out_shape=(2, 2))

    np.testing.assert_allclose(means[0, 0], np.mean([1.0, 4.0, 5.0]))
    assert fractions[0, 0] == 0.75
    assert fractions[1, 1] == 1.0
