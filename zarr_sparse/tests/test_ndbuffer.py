import numpy as np
import pytest

from zarr_sparse.buffer import ChunkGrid


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
