import numpy as np
import pytest
import sparse

from zarr_sparse.buffer import SparseNDBuffer
from zarr_sparse.chunk_grid import ChunkGrid


class TestNDBuffer:
    def test_init(self):
        chunk_grid = ChunkGrid(shape=(4, 4), dtype=np.float64)

        buffer = SparseNDBuffer(chunk_grid)
        assert buffer._data is chunk_grid

    def test_create(self):
        shape = (4, 4)
        order = "F"
        fill_value = 0
        dtype = np.uint8
        buffer = SparseNDBuffer.create(
            shape=shape, dtype=dtype, order=order, fill_value=fill_value
        )

        assert buffer._data.shape == shape
        assert buffer._data.dtype == dtype
        assert buffer._data.order == order
        assert buffer._data.fill_value == fill_value

    @pytest.mark.parametrize(
        "array",
        (
            sparse.full(shape=(2, 2, 2), dtype="uint8", fill_value=255),
            sparse.full(shape=(4, 4), dtype="int64", fill_value=0),
            sparse.full(shape=(8,), dtype="float64", fill_value=np.nan),
        ),
    )
    def test_from_ndarray_like(self, array):
        actual = SparseNDBuffer.from_ndarray_like(array)

        assert actual._data.shape == array.shape
        assert actual._data.dtype == array.dtype
        assert actual._data.fill_value == array.fill_value or (
            np.isnan(actual._data.fill_value) and np.isnan(array.fill_value)
        )

        chunk_key = (0,) * array.ndim
        assert actual._data.bounds == {
            chunk_key: tuple(range(0, n, 1) for n in array.shape)
        }
        assert (
            list(actual._data.data.keys()) == [chunk_key]
            and actual._data.data[chunk_key] is array
        )

    def test_as_ndarray_like(self):
        array = np.arange(10)
        chunk_grid = ChunkGrid(shape=array.shape, dtype=array.dtype, fill_value=0)
        chunk_grid[0:5] = array[0:5]
        chunk_grid[5:10] = array[5:10]

        buffer_ = SparseNDBuffer(chunk_grid)

        actual = buffer_.as_ndarray_like()
        np.testing.assert_equal(actual, array)
