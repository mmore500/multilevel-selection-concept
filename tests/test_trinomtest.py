import typing

import numpy as np
import pytest

from pylib._trinomtest import trinomtest_fast, trinomtest_naive


@pytest.mark.parametrize("trinomtest", [trinomtest_fast, trinomtest_naive])
@pytest.mark.parametrize("mu", [0, 1])
def test_empty(trinomtest: typing.Callable, mu: int):
    assert np.isnan(trinomtest([], mu=0))


def test_all_ties():
    assert trinomtest_fast([0, 0, 0, 0], mu=0) == pytest.approx(1.0)
    assert trinomtest_naive([0, 0, 0, 0], mu=0) == pytest.approx(1.0)


def test_symmetric_no_ties():
    data = [1, -2, 3, -4]

    fast, naive = trinomtest_fast(data, mu=0), trinomtest_naive(data, mu=0)
    assert fast > 0.5 and naive > 0.5
    assert pytest.approx(fast) == naive


def test_few_all_positive():
    data = [1.0, 2.5]

    fast, naive = trinomtest_fast(data, mu=0), trinomtest_naive(data, mu=0)
    assert 0.2 < fast < 0.5 and 0.2 < naive < 0.5
    assert pytest.approx(fast) == naive


def test_extreme_all_positive():
    data = [1.0, 2.5, 3.7, 1.2, 4.5] * 1000
    fast, naive = trinomtest_fast(data, mu=0), trinomtest_naive(data, mu=0)
    assert pytest.approx(0.0) == fast and pytest.approx(0.0) == naive
    assert pytest.approx(fast) == naive


@pytest.mark.parametrize("trinomtest", [trinomtest_fast, trinomtest_naive])
def test_list_vs_numpy_array(trinomtest: typing.Callable):
    lst = [1, -1, 0]
    arr = np.array([1, -1, 0])
    p_lst = trinomtest(lst, mu=0)
    p_arr = trinomtest(arr, mu=0)
    assert p_lst == pytest.approx(p_arr)


@pytest.mark.parametrize("trinomtest", [trinomtest_fast, trinomtest_naive])
def test_median_shift_invariance(trinomtest: typing.Callable):
    orig = [2, 3, 1, 4, -1]
    p0 = trinomtest(orig, mu=0)
    shifted = [x + 2 for x in orig]
    p_shifted = trinomtest(shifted, mu=2)
    assert p0 == pytest.approx(p_shifted)


@pytest.mark.parametrize("trinomtest", [trinomtest_fast, trinomtest_naive])
def test_sample_size(trinomtest: typing.Callable):
    orig = [2, 3, 1, 4, -1]
    assert trinomtest(orig, mu=0) > trinomtest(orig * 10, mu=0)


def test_empirical_type_I_error():
    N_SIM = 5000
    SAMPLE_SIZE = 20
    ALPHA = 0.05

    p_values = []
    for _ in range(N_SIM):
        data = np.random.normal(loc=0.0, scale=1.0, size=SAMPLE_SIZE)
        p = trinomtest_fast(data, mu=0.0)
        assert pytest.approx(p) == trinomtest_naive(data, mu=0.0)
        p_values.append(p)
    p_values = np.array(p_values)

    empirical_alpha = np.mean(p_values < ALPHA)
    assert empirical_alpha <= 0.06


@pytest.mark.parametrize("loc", [-10.0, -0.3, 0.3, 10.0])
def test_empirical_sensitivity(loc: float):
    N_SIM = 1000
    SAMPLE_SIZE = 100
    ALPHA = 0.05

    p_values = []
    for _ in range(N_SIM):
        data = np.random.normal(loc=loc, scale=1.0, size=SAMPLE_SIZE)
        p = trinomtest_fast(data, mu=0.0)
        p_values.append(p)
    p_values = np.array(p_values)

    empirical_alpha = np.mean(p_values < ALPHA)
    assert empirical_alpha > 0.5


@pytest.mark.parametrize("loc", [-10.0, -0.3, 0.0, 0.3, 10.0])
@pytest.mark.parametrize("sample_size", range(20))
def test_fuzz_naive_vs_fast(loc: float, sample_size: int):
    for _ in range(20):
        data = np.random.normal(loc=loc, scale=1.0, size=sample_size)
        p_fast = trinomtest_fast(data, mu=0.0)
        p_naive = trinomtest_naive(data, mu=0.0)
        assert p_fast == pytest.approx(p_naive, nan_ok=True)


@pytest.mark.parametrize("loc", [-10.0, -0.3, 0.0, 0.3, 10.0])
@pytest.mark.parametrize("sample_size", range(20))
def test_fuzz_omit_nans(loc: float, sample_size: int):
    data = np.random.normal(loc=loc, scale=1.0, size=sample_size)
    expected = pytest.approx(trinomtest_fast(data, mu=0.0), nan_ok=True)

    data1 = [*data, np.nan]
    assert trinomtest_fast(data1, nan_policy="omit", mu=0.0) == expected
    assert trinomtest_naive(data1, nan_policy="omit", mu=0.0) == expected

    data2 = [np.nan, *data]
    assert trinomtest_fast(data2, nan_policy="omit", mu=0.0) == expected
    assert trinomtest_naive(data2, nan_policy="omit", mu=0.0) == expected


@pytest.mark.parametrize("loc", [-10.0, -0.3, 0.0, 0.3, 10.0])
@pytest.mark.parametrize("sample_size", range(20))
def test_fuzz_propagate_nans(loc: float, sample_size: int):
    data = np.random.normal(loc=loc, scale=1.0, size=sample_size)
    expected = pytest.approx(np.nan, nan_ok=True)

    data1 = [*data, np.nan]
    assert trinomtest_fast(data1, nan_policy="propagate", mu=0.0) == expected
    assert trinomtest_naive(data1, nan_policy="propagate", mu=0.0) == expected

    data2 = [np.nan, *data]
    assert trinomtest_fast(data2, nan_policy="propagate", mu=0.0) == expected
    assert trinomtest_naive(data2, nan_policy="propagate", mu=0.0) == expected
