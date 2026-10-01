import numpy as np

from scripts.run_m3_human_benchmark import aggregate_16_to_8


def test_aggregate_16_to_8_mean_and_labels():
    rows, cols = np.indices((16, 16))
    rows = rows.reshape(-1)
    cols = cols.reshape(-1)
    years = np.asarray(["2025"] * 256)

    X_jepa = np.stack([rows, cols], axis=1).astype(np.float32)
    X_ndvi = (rows + cols).reshape(-1, 1).astype(np.float32)

    labels = {
        ("2025", r, c): ("vegetation" if (r + c) % 2 == 0 else "non_vegetation")
        for r in range(8)
        for c in range(8)
    }

    out_jepa, out_ndvi, y, out_years, out_rows, out_cols = aggregate_16_to_8(
        X_jepa, X_ndvi, years, rows, cols, labels
    )

    assert out_jepa.shape == (64, 2)
    assert out_ndvi.shape == (64, 1)
    np.testing.assert_allclose(out_jepa[0], [0.5, 0.5])
    np.testing.assert_allclose(out_ndvi[0], [1.0])
    assert bool(y[0]) is True
    assert out_years[0] == "2025"
    assert out_rows[0] == 0
    assert out_cols[0] == 0
