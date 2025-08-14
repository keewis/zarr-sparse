import numpy as np
import pytest

from zarr_sparse.buffer import ChunkGrid
from zarr_sparse.comparison import assert_sparse_equal
from zarr_sparse.tests.generate import create_chunk_slices, create_pydata_coo_array


class TestChunkGrid:
    @pytest.mark.parametrize(
        "params",
        (
            {"shape": (2, 3), "dtype": "int64", "order": "C", "fill_value": 0},
            {"shape": (3,), "dtype": "int8", "order": "F", "fill_value": 15},
            {"shape": (10, 1, 15), "dtype": "float32", "order": "C", "fill_value": 0.0},
        ),
    )
    def test_init(self, params):
        obj = ChunkGrid(**params)

        assert {k: getattr(obj, f"_{k}") for k in params} == params
        np.testing.assert_equal(
            obj._data,
            np.full((1,) * len(obj._shape), dtype=object, fill_value=None),
        )
        expected_chunks = tuple((size,) for size in params["shape"])
        assert obj._chunks == expected_chunks

    @pytest.mark.parametrize("shape", ((3, 4), (1, 5, 10)))
    def test_shape(self, shape):
        obj = ChunkGrid(shape=shape, dtype="float32", order="C", fill_value=0.0)

        assert obj.shape == shape

    @pytest.mark.parametrize("dtype", ("float32", "int64"))
    def test_dtype(self, dtype):
        obj = ChunkGrid(shape=(4, 3), dtype=dtype, order="C", fill_value=0.0)

        assert obj.dtype == dtype

    @pytest.mark.parametrize("order", ("C", "F"))
    def test_order(self, order):
        obj = ChunkGrid(shape=(4, 3), dtype="int32", order=order, fill_value=0.0)

        assert obj.order == order

    @pytest.mark.parametrize("fill_value", (0, 1))
    def test_fill_value(self, fill_value):
        obj = ChunkGrid(shape=(4, 3), dtype="int32", order="C", fill_value=fill_value)

        assert obj.fill_value == fill_value

    @pytest.mark.parametrize(
        ["chunks", "expanded"],
        (
            ((2, 1), ((2, 2), (1, 1, 1))),
            ((4, 2), ((4,), (2, 1))),
        ),
    )
    def test_chunks(self, chunks, expanded):
        obj = ChunkGrid(
            shape=(4, 3),
            dtype="int32",
            order="C",
            fill_value=0,
            chunks=chunks,
        )

        assert obj.chunks == expanded

    @pytest.mark.parametrize(
        ["dtype", "fill_value"],
        (
            ("int64", 0),
            ("int32", 3),
            ("float32", 0),
            ("float64", np.nan),
        ),
    )
    @pytest.mark.parametrize(
        ["shape", "chunks"],
        (
            ((60, 3), (10, 3)),
            ((100, 100, 100), (20, 50, 50)),
            ((500,), (5,)),
        ),
    )
    @pytest.mark.parametrize("nnz", (1, 38, 70, 150))
    def test_setitem_full(self, nnz, shape, chunks, dtype, fill_value):
        order = "C"
        sparse_array = create_pydata_coo_array(
            nnz=nnz, shape=shape, dtype=np.dtype(dtype), fill_value=fill_value
        )

        print(f"input: {sparse_array=}, {chunks=}")

        actual = ChunkGrid(
            shape=sparse_array.shape,
            order=order,
            dtype=sparse_array.dtype,
            fill_value=sparse_array.fill_value,
            chunks=chunks,
        )
        print(f"ChunkGrid (before assignment):\n{actual}")
        actual[(slice(None),) * sparse_array.ndim] = sparse_array
        print(f"ChunkGrid (after assignment):\n{actual}")

        assert_sparse_equal(
            actual.get_value(),
            sparse_array,
        )

    @pytest.mark.parametrize(
        ["dtype", "fill_value"],
        (
            ("int64", 0),
            ("int32", 3),
            ("float32", 0),
            ("float64", np.nan),
        ),
    )
    @pytest.mark.parametrize(
        ["shape", "chunks"],
        (
            ((60, 3), (10, 3)),
            ((100, 100, 100), (20, 50, 50)),
            ((500,), (5,)),
        ),
    )
    @pytest.mark.parametrize("nnz", (1, 38, 70, 150))
    def test_setitem_chunks(self, nnz, shape, chunks, dtype, fill_value):
        order = "C"
        sparse_array = create_pydata_coo_array(
            nnz=nnz, shape=shape, dtype=np.dtype(dtype), fill_value=fill_value
        )

        chunk_slices = create_chunk_slices(shape, chunks)

        actual = ChunkGrid(
            shape=sparse_array.shape,
            order=order,
            dtype=sparse_array.dtype,
            fill_value=sparse_array.fill_value,
            chunks=chunks,
        )

        for indexer in chunk_slices:
            actual[indexer] = sparse_array[indexer]

        assert_sparse_equal(
            actual.get_value(),
            sparse_array,
        )

    @pytest.mark.parametrize(
        ["dtype", "fill_value"],
        (
            ("int64", 0),
            ("int32", 3),
            ("float32", 0),
            ("float64", np.nan),
        ),
    )
    @pytest.mark.parametrize(
        ["shape", "chunks"],
        (
            ((60, 3), (10, 3)),
            ((100, 100, 100), (20, 50, 50)),
            ((500,), (5,)),
        ),
    )
    @pytest.mark.parametrize("nnz", (1, 38, 70, 150))
    @pytest.mark.parametrize("chunk_index", (0, 5, -1))
    def test_getitem(self, nnz, shape, chunks, dtype, fill_value, chunk_index):
        order = "C"
        sparse_array = create_pydata_coo_array(
            nnz=nnz, shape=shape, dtype=np.dtype(dtype), fill_value=fill_value
        )

        chunk_slices = create_chunk_slices(shape, chunks)

        grid = ChunkGrid(
            shape=sparse_array.shape,
            order=order,
            dtype=sparse_array.dtype,
            fill_value=sparse_array.fill_value,
            chunks=chunks,
        )
        for index, indexer in enumerate(chunk_slices):
            chunk_loc = np.unravel_index(index, grid._data.shape)

            grid._data[chunk_loc] = sparse_array[indexer]

        chunk_indexer = chunk_slices[chunk_index]
        actual = grid[chunk_indexer].get_value()
        expected = sparse_array[chunk_indexer]

        assert_sparse_equal(actual, expected)
