import numpy as np

from scripts.analyze_2007_paired_effects import compare, exact_bootstrap_ci


def test_exact_bootstrap_ci_constant_difference():
    values = np.asarray([0.1, 0.1, 0.1, 0.1])
    lo, hi = exact_bootstrap_ci(values)
    assert np.isclose(lo, 0.1)
    assert np.isclose(hi, 0.1)


def test_compare_reports_paired_direction():
    a = np.asarray([0.8, 0.7, 0.6, 0.5])
    b = np.asarray([0.7, 0.7, 0.65, 0.4])
    result = compare(a, b)
    assert np.isclose(result["mean_difference"], 0.0375)
    assert result["ijepa_wins"] == 2
    assert result["ties"] == 1
    assert result["comparator_wins"] == 1
