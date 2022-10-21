import numpy as np

from pyquantification.shift_tests.dists import (
    Hist,
    EqualSpaceHistGrid,
    EqualProbHistGrid,
    get_hists,
    mix_per_class_hists,
    ks_distance,
    hellinger_distance,
    SampledClassConditionalMixture,
)


def test_equal_space_hist_grid():
    calib_probs = np.zeros((100, 3))
    target_probs = calib_probs

    bin_edges = EqualSpaceHistGrid(4).get_bin_edges(calib_probs=calib_probs, target_probs=target_probs)

    assert len(bin_edges) == 3
    for class_bin_edges in bin_edges:
        np.testing.assert_array_almost_equal(
            class_bin_edges,
            np.array([0, 0.25, 0.5, 0.75, 1]),
        )


def test_equal_prob_hist_grid():
    calib_probs = np.array([
        [0.2, 0.0, 0.8],
        [0.1, 0.2, 0.7],
        [0.2, 0.4, 0.4],
        [0.1, 0.6, 0.3],
        [0.2, 0.8, 0.0],
        [0.1, 0.8, 0.1],
    ])
    # Min row count of 2 should result in target of 3 bins (4 edges).
    target_probs = np.zeros((2, 3))
    bin_edges = EqualProbHistGrid().get_bin_edges(calib_probs=calib_probs, target_probs=target_probs)

    assert len(bin_edges) == 3
    np.testing.assert_array_almost_equal(bin_edges[0], np.array([0, 1]))
    np.testing.assert_array_almost_equal(bin_edges[1], np.array([0, 1/3, 2/3, 1]))
    np.testing.assert_array_almost_equal(bin_edges[2], np.array([0, 0.233333, 0.5, 1]))


def test_get_hists():
    probs = np.array([
        [0.25, 0.75],
        [0.25, 0.75],
        [0.25, 0.75],
        [0.25, 0.75],
        [0.25, 0.75],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.8, 0.2],
        [0.8, 0.2],
    ])
    hists = get_hists(probs, bin_edges=[
        np.array([0, 0.25, 0.5, 0.75, 1]),
        np.array([0, 0.7, 0.8, 1]),
    ])
    assert len(hists) == 2
    np.testing.assert_array_equal(hists[0].pdf, np.array([0, 0.5, 0.3, 0.2]))
    np.testing.assert_array_equal(hists[0].cdf, np.array([0, 0.5, 0.8, 1.0]))
    np.testing.assert_array_equal(hists[1].pdf, np.array([0.5, 0.5, 0.0]))
    np.testing.assert_array_equal(hists[1].cdf, np.array([0.5, 1.0, 1.0]))


def test_mix_per_class_hists():
    priors = np.array([0.4, 0.6])
    hists = [
        [
            Hist(np.array([0, 0.5, 0.3, 0.2])),
            Hist(np.array([0.5, 0.5, 0.0])),
        ],
        [
            Hist(np.array([0, 0.2, 0.8, 0.0])),
            Hist(np.array([0.2, 0.5, 0.3])),
        ],
    ]
    mix_hists = mix_per_class_hists(
        class_priors=priors,
        per_class_hists=hists,
    )
    assert len(mix_hists) == 2
    np.testing.assert_array_almost_equal(
        mix_hists[0].pdf,
        np.array([0, 0.32, 0.6, 0.08])
    )
    np.testing.assert_array_almost_equal(
        mix_hists[1].pdf,
        np.array([0.32, 0.5, 0.18])
    )


def test_ks_distance():
    test_cases = [
        (Hist(np.array([0, 0.5, 0.3, 0.2])),
         Hist(np.array([0, 0.5, 0.5, 0.0])),
         0.2),
        (Hist(np.array([0, 0.2, 0.3, 0.5, 0.0])),
         Hist(np.array([0, 0.5, 0.3, 0.0, 0.2])),
         0.3),
    ]
    for hist_a, hist_b, distance in test_cases:
        assert (ks_distance(hist_a, hist_b) - distance) < 0.00001


def test_hellinger_distance():
    test_cases = [
        (Hist(np.array([0.0, 0.5, 0.3, 0.0, 0.2])),
         Hist(np.array([0.0, 0.5, 0.3, 0.0, 0.2])),
         0.0),
        (Hist(np.array([0.0, 0.5, 0.3, 0.0, 0.2])),
         Hist(np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
         0.69111976),
        (Hist(np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
         Hist(np.array([0.0, 0.5, 0.3, 0.0, 0.2])),
         0.69111976),
        (Hist(np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
         Hist(np.array([0.2, 0.2, 0.2, 0.2, 0.2])),
         0.0),
    ]
    for hist_a, hist_b, distance in test_cases:
        assert (hellinger_distance(hist_a, hist_b) - distance) < 0.00001


def test_sampled_class_conditional_mixture():
    mixture = SampledClassConditionalMixture(
        per_class_probs=[
            np.array([
                [0.35, 0.35, 0.3],
            ]),
            np.array([
                [0.2, 0.3, 0.5],
            ]),
            np.array([
                [0.1, 0.5, 0.4],
            ]),
        ]
    )
    probs = mixture.sample_probs(
        class_priors=np.array([0.4, 0.2, 0.4]),
        target_size=5,
        rng=np.random.RandomState(0),
    )
    np.testing.assert_array_equal(probs, np.array([
        [0.35, 0.35, 0.3],
        [0.35, 0.35, 0.3],
        [0.2, 0.3, 0.5],
        [0.1, 0.5, 0.4],
        [0.1, 0.5, 0.4],
    ]))
