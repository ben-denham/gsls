from functools import partial
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import check_grad
import pytest

from pyquantification.quantifiers.gsls.fitters import WeightSumFitter


def test_get_seed_hists() -> None:
    seed_hists = WeightSumFitter.get_seed_hists(
        calib_hist=np.array([0.1, 0.5, 0.4]),
        target_hist=np.array([0.4, 0.5, 0.1]),
        random_state=1,
    )
    assert np.array(seed_hists).shape == (10, 3)
    assert_allclose(np.sum(np.array(seed_hists), axis=1), np.ones(shape=10))


def test_target_func() -> None:
    assert WeightSumFitter.target_func(
        remain_hist=np.array([0.2, 0.3, 0.5]),
        calib_hist=np.array([0.2, 0.3, 0.5]),
        target_hist=np.array([0.2, 0.3, 0.5]),
    ) == -2.0
    assert WeightSumFitter.target_func(
        remain_hist=np.array([0.0, 0.4, 0.6]),
        calib_hist=np.array([0.2, 0.3, 0.5]),
        target_hist=np.array([0.2, 0.5, 0.3]),
    ) == -((0.3 / 0.4) + (0.3 / 0.6))


def test_target_jacobian() -> None:
    calib1 = np.array([0.2, 0.3, 0.5])
    target1 = np.array([0.2, 0.3, 0.5])
    fun1 = partial(WeightSumFitter.target_func, calib_hist=calib1, target_hist=target1)
    jac1 = partial(WeightSumFitter.target_jacobian, calib_hist=calib1, target_hist=target1)
    assert check_grad(fun1, jac1, np.array([0.2, 0.3, 0.5])) < 1e-06
    assert check_grad(fun1, jac1, np.array([0.0, 0.4, 0.6])) < 1e-06

    calib2 = np.array([0.2, 0.3, 0.5])
    target2 = np.array([0.2, 0.5, 0.3])
    fun2 = partial(WeightSumFitter.target_func, calib_hist=calib2, target_hist=target2)
    jac2 = partial(WeightSumFitter.target_jacobian, calib_hist=calib2, target_hist=target2)
    assert check_grad(fun2, jac2, np.array([0.2, 0.3, 0.5])) < 1e-06
    assert check_grad(fun2, jac2, np.array([0.0, 0.4, 0.6])) < 1e-06

    calib3 = np.array([0.2, 0.4, 0.4])
    target3 = np.array([0.2, 0.4, 0.4])
    fun3 = partial(WeightSumFitter.target_func, calib_hist=calib3, target_hist=target3)
    jac3 = partial(WeightSumFitter.target_jacobian, calib_hist=calib3, target_hist=target3)
    assert check_grad(fun3, jac3, np.array([0.2, 0.3, 0.5])) < 1e-06
    assert check_grad(fun3, jac3, np.array([0.4, 0.3, 0.3])) < 1e-06
    assert check_grad(fun3, jac3, np.array([0.0, 0.4, 0.6])) < 1e-06


def test_equality_constraint_func() -> None:
    assert WeightSumFitter.equality_constraint_func(np.array([1.0, 0.0, 0.0])) == 0.0
    assert WeightSumFitter.equality_constraint_func(np.array([0.2, 0.3, 0.5])) == 0.0
    assert WeightSumFitter.equality_constraint_func(np.array([0.0, 0.0, 0.0])) == 1.0
    assert WeightSumFitter.equality_constraint_func(np.array([0.1, 0.2, 0.3])) == pytest.approx(0.4, 1e-6)
    assert WeightSumFitter.equality_constraint_func(np.array([0.3, 0.5, 0.8])) == pytest.approx(-0.6, 1e-6)


def test_equality_constraint_jacobian() -> None:
    test_hists = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.2, 0.3, 0.5]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.1, 0.2, 0.3]),
        np.array([0.3, 0.5, 0.8]),
    ]
    for test_hist in test_hists:
        assert check_grad(WeightSumFitter.equality_constraint_func,
                          WeightSumFitter.equality_constraint_jacobian,
                          test_hist) < 1e-06


def test_find_remain_hist() -> None:
    assert_allclose(WeightSumFitter.find_remain_hist(
        calib_hist=np.array([0.2, 0.3, 0.5]),
        target_hist=np.array([0.2, 0.3, 0.5]),
        random_state=1,
    ), np.array([0.2, 0.3, 0.5]), atol=1e-06)
    assert_allclose(WeightSumFitter.find_remain_hist(
        calib_hist=np.array([0.2, 0.3, 0.5]),
        target_hist=np.array([0.5, 0.3, 0.2]),
        random_state=1,
    ), np.array([0.285714, 0.428571, 0.285714]), atol=1e-06)
    assert_allclose(WeightSumFitter.find_remain_hist(
        calib_hist=np.array([0.0, 0.3, 0.5]),
        target_hist=np.array([0.5, 0.0, 0.2]),
        random_state=1,
    ), np.array([0.0, 0.0, 1.0]), atol=1e-06)
    assert_allclose(WeightSumFitter.find_remain_hist(
        calib_hist=np.array([0.0, 0.3, 0.0]),
        target_hist=np.array([0.5, 0.0, 0.2]),
        random_state=1,
    ), np.array([0.0, 0.0, 0.0]), atol=1e-06)
