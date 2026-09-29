import pytest
import numpy as np
from mlfromscratch.utils import data_manipulation as target


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64])
def test_integer_standardization(dtype):
    x = np.array([[1, 10], [2, 20], [3, 30]], dtype=dtype)
    expected = (x.astype(float) - x.mean(axis=0)) / x.std(axis=0)
    np.testing.assert_allclose(target.standardize(x), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_standardization(dtype):
    x = np.array([[1, 10], [2, 20], [3, 30]], dtype=dtype)
    expected = (x.copy() - x.mean(axis=0)) / x.std(axis=0)
    np.testing.assert_allclose(target.standardize(x), expected, rtol=1e-6)


def test_constant_column_existing_behavior():
    x = np.array([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])
    np.testing.assert_array_equal(target.standardize(x)[:, 0], [3.0, 3.0, 3.0])


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_input_is_not_modified(dtype):
    x = np.array([[1, 2], [2, 4], [3, 6]], dtype=dtype)
    original = x.copy()
    result = target.standardize(x)
    np.testing.assert_array_equal(x, original)
    assert not np.shares_memory(result, x)
