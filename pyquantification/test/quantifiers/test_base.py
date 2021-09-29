import numpy as np

from pyquantification.quantifiers.base import (
    Interval,
    find_index_in_sorted,
    interval_for_cdf
)


def test_find_index_in_sorted() -> None:
    xs = np.array([1, 2, 3])
    assert find_index_in_sorted(xs, 0) == (-1, 0)
    assert find_index_in_sorted(xs, 1) == (0, 0)
    assert find_index_in_sorted(xs, 2) == (1, 1)
    assert find_index_in_sorted(xs, 2.5) == (1, 2)
    assert find_index_in_sorted(xs, 3) == (2, 2)
    assert find_index_in_sorted(xs, 4) == (2, 3)


def test_interval_for_cdf() -> None:

    def cdf_for_pdf(pdf):
        return np.cumsum(np.array(pdf))

    # Uniform
    assert interval_for_cdf(cdf_for_pdf([0.2, 0.2, 0.2, 0.2, 0.2]), 0.2) == Interval(2, 2)
    assert interval_for_cdf(cdf_for_pdf([0.2, 0.2, 0.2, 0.2, 0.2]), 0.3) == Interval(1, 3)
    assert interval_for_cdf(cdf_for_pdf([0.2, 0.2, 0.2, 0.2, 0.2]), 0.6) == Interval(1, 3)
    assert interval_for_cdf(cdf_for_pdf([0.2, 0.2, 0.2, 0.2, 0.2]), 0.7) == Interval(0, 4)
    assert interval_for_cdf(cdf_for_pdf([0.2, 0.2, 0.2, 0.2, 0.2]), 1.0) == Interval(0, 4)

    # Left-skewed
    assert interval_for_cdf(cdf_for_pdf([0.8, 0.1, 0.1, 0.0, 0.0]), 0.6) == Interval(0, 0)
    assert interval_for_cdf(cdf_for_pdf([0.8, 0.1, 0.1, 0.0, 0.0]), 0.7) == Interval(0, 1)
    assert interval_for_cdf(cdf_for_pdf([0.8, 0.1, 0.1, 0.0, 0.0]), 0.8) == Interval(0, 1)
    assert interval_for_cdf(cdf_for_pdf([0.8, 0.1, 0.1, 0.0, 0.0]), 0.9) == Interval(0, 2)
    assert interval_for_cdf(cdf_for_pdf([0.8, 0.1, 0.1, 0.0, 0.0]), 1.0) == Interval(0, 2)

    # Right-skewed
    assert interval_for_cdf(cdf_for_pdf([0.0, 0.0, 0.1, 0.1, 0.8]), 0.6) == Interval(4, 4)
    assert interval_for_cdf(cdf_for_pdf([0.0, 0.0, 0.1, 0.1, 0.8]), 0.7) == Interval(3, 4)
    # This point is not symmetrical with left-skew, due to floating
    # point precision of lower_alpha determined from 0.8 (0.0999...)
    assert interval_for_cdf(cdf_for_pdf([0.0, 0.0, 0.1, 0.1, 0.8]), 0.8) == Interval(2, 4)
    assert interval_for_cdf(cdf_for_pdf([0.0, 0.0, 0.1, 0.1, 0.8]), 0.9) == Interval(2, 4)
    assert interval_for_cdf(cdf_for_pdf([0.0, 0.0, 0.1, 0.1, 0.8]), 1.0) == Interval(2, 4)
