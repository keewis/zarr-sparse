import numpy as np
import pytest

from zarr_sparse.buffer import (
    ChunkGrid,
    decompose_by_chunks,
    normalize_slice,
    slice_size,
)


@pytest.mark.parametrize(
    ["slice_", "size", "expected"],
    (
        (slice(None), 10, 10),
        (slice(None, 4), 10, 4),
        (slice(1, None), 3, 2),
    ),
)
def test_slice_size(slice_, size, expected):
    actual = slice_size(slice_, size)

    assert actual == expected


@pytest.mark.parametrize(
    ["slice_", "size", "expected"],
    (
        (slice(None), 10, slice(0, 10, 1)),
        (slice(None, 4), 10, slice(0, 4, 1)),
        (slice(1, None), 3, slice(1, 3, 1)),
    ),
)
def test_normalize_slice(slice_, size, expected):
    actual = normalize_slice(slice_, size)

    assert actual == expected


@pytest.mark.parametrize(
    ["indexer", "chunks", "expected"],
    (
        (
            slice(None),
            (5, 5, 5, 3),
            {
                0: slice(0, 5, 1),
                1: slice(0, 5, 1),
                2: slice(0, 5, 1),
                3: slice(0, 3, 1),
            },
        ),
        (slice(1, 5, 2), (10, 10, 1), {0: slice(1, 5, 2)}),
        (4, (3, 3), {1: 1}),
        (
            np.array([1, 2, 4, 6]),
            (3, 3, 2),
            {0: np.array([1, 2]), 1: np.array([1]), 2: np.array([0])},
        ),
    ),
)
def test_decompose_by_chunks(indexer, chunks, expected):
    def compare(indexer1, indexer2):
        if type(indexer1) is not type(indexer2):
            return False

        if isinstance(indexer1, (int, slice)):
            return indexer1 == indexer2

        return np.array_equal(indexer1, indexer2)

    actual = decompose_by_chunks(indexer, chunks)

    assert actual.keys() == expected.keys(), "selected chunks are different"
    assert all(
        compare(actual[k], expected[k]) for k in actual.keys()
    ), "chunk indexers are different"


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
        assert obj._chunks == params["shape"]

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

    @pytest.mark.parametrize("chunks", ((2, 1), (4, 2)))
    def test_chunks(self, chunks):
        obj = ChunkGrid(
            shape=(4, 3),
            dtype="int32",
            order="C",
            fill_value=0,
            chunks=chunks,
        )

        assert obj.chunks == chunks
