import numpy as np

from statsplotly.plot_specifiers.data.statistics import (
    affine_func,
    compute_ssquares,
    exponential_regress,
    get_iqr,
    inverse_func,
    kde_1d,
    kde_2d,
    logarithmic_func,
    range_normalize,
    regress,
    sem,
)


def test_compute_ssquares():
    y = np.array([3, -0.5, 2, 7])
    yhat = np.array([2.5, 0.0, 2, 8])

    expected_TSS = 29.1875
    expected_ESS = 35.4375
    expected_RSS = 1.5

    TSS, ESS, RSS = compute_ssquares(y, yhat)

    assert np.isclose(TSS, expected_TSS), f"Expected {expected_TSS}, but got {TSS}"
    assert np.isclose(ESS, expected_ESS), f"Expected {expected_ESS}, but got {ESS}"
    assert np.isclose(RSS, expected_RSS), f"Expected {expected_RSS}, but got {RSS}"


def test_inverse_func():
    x = np.array([1, 2, 3])
    a = 4
    b = 5

    result = inverse_func(x, a, b)
    expected = np.array([9.0, 7.0, 6.333333])

    assert np.allclose(result, expected), f"Expected {expected_output} but got {result}"


def test_affine_func():
    """Test for affine function"""
    x = np.array([1, 2, 3])
    a = 0.5
    b = 2
    expected_output = np.array([2.5, 3, 3.5])

    result = affine_func(x, a, b)
    assert np.allclose(result, expected_output), f"Expected {expected_output} but got {result}"


def test_logarithmic_func():
    x = np.array([1, 2, 3])
    a = 0.5
    b = 2
    result = logarithmic_func(x, a, b)
    expected_result = np.array([2.0, 2.34657359, 2.54930614])

    assert np.allclose(result, expected_result), f"Expected {expected_result} but got {result}"


def test_regress():
    np.random.seed(0)
    x = np.linspace(-10, 10, 50)
    y = 2 * x + 3 + np.random.normal(0, 2, len(x))

    p_expected = [1.85883775, 3.28111855]
    r2_expected = 0.9647825387004532
    x_grid_expected = np.linspace(x.min(), x.max(), 100)
    y_fit_expected = affine_func(x_grid_expected, *p_expected)

    p, r2, (x_grid, y_fit) = regress(x, y, affine_func)

    assert np.allclose(p, p_expected), f"Expected {p_expected} but got {p}"
    assert np.isclose(r2, r2_expected), f"Expected {r2_expected} but got {r2}"

    assert np.allclose(x_grid, x_grid_expected), f"Expected {x_grid_expected} but got {x_grid}"
    assert np.allclose(y_fit, y_fit_expected), f"Expected {y_fit_expected} but got {y_fit}"


def test_exponential_regress():
    np.random.seed(0)
    x = np.linspace(5, 50, 50)
    y = 2 * x + 3 + np.random.normal(0, 2, len(x))

    p_expected = [0.03420418, 3.02920043]
    r2_expected = 0.955300828835159
    x_grid_expected = np.linspace(x.min(), x.max(), 100)
    y_fit_expected = np.exp(p_expected[1]) * np.exp(p_expected[0] * x_grid_expected)

    p, r2, (x_grid, y_fit) = exponential_regress(x, y)

    assert np.allclose(p, p_expected), f"Expected {p_expected} but got {p}"
    assert np.isclose(r2, r2_expected), f"Expected {r2_expected} but got {r2}"

    assert np.allclose(x_grid, x_grid_expected), f"Expected {x_grid_expected} but got {x_grid}"
    assert np.allclose(y_fit, y_fit_expected), f"Expected {y_fit_expected} but got {y_fit}"


def test_kde_1d():
    x = np.random.normal(0, 2, size=100)
    x_grid = np.linspace(-5, 5, num=1000)
    result = kde_1d(x, x_grid)

    assert isinstance(result, np.ndarray), "The output must be an instance of NDArray"
    assert result.shape == x_grid.shape, "Output shape must match input grid shape"


def test_kde_2d():
    x = np.random.rand(10)
    y = np.random.rand(10)

    x_grid = np.linspace(-5, 5, num=1000)
    y_grid = np.linspace(-5, 5, num=1000)
    result = kde_2d(x, y, x_grid, y_grid)

    assert isinstance(result, np.ndarray), "The output must be an instance of NDArray"  # noqa: S101
    assert result.shape == (len(x_grid), len(y_grid)), "Output shape must match input grid shape"


def test_sem():
    data = np.arange(10)
    expected_result = 0.05695625953322525
    result = sem(data)

    assert np.isclose(expected_result, result), f"Expected {expected_result} but got {result}"


def test_get_iqr():
    data = np.arange(10)
    expected_result = 4.5
    result = get_iqr(data)

    assert np.isclose(expected_result, result), f"Expected {expected_result} but got {result}"


def test_range_normalize():
    arr = np.array([-1, -1, 2])
    expected_result = [0.0, 0.0, 1.0]
    result = range_normalize(arr, 0, 1)

    assert np.allclose(expected_result, result), f"Expected {expected_result} but got {result}"
