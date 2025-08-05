from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import sparse
from zarr.core.buffer.core import Buffer, BufferPrototype, NDBuffer
from zarr.registry import register_ndbuffer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self


def slice_size(slice_, size):
    return len(range(*slice_.indices(size)))


def normalize_slice(slice_, size):
    return slice(*slice_.indices(size))


def _decompose_slice_by_chunks(slice_, offsets, chunksizes):
    def _decompose_slice(slice_, offset, size):
        if (slice_.start < offset and slice_.stop <= offset) or (
            slice_.start >= offset + size
        ):
            # outside chunk
            return slice(0)

        start = offset if slice_.start < offset else slice_.start
        stop = offset + size if slice_.stop >= offset + size else slice_.stop
        step = slice_.step

        return slice(start, stop, step)

    decomposed = {
        index: _decompose_slice(slice_, offset, size)
        for index, (offset, size) in enumerate(zip(offsets, chunksizes))
    }

    return {
        index: slice_ for index, slice_ in decomposed.items() if slice_size(slice_) > 0
    }


def _decompose_int_by_chunks(indexer, offsets, chunksizes):
    index = bisect.bisect_right(offsets, indexer)
    new_indexer = indexer - offsets[index]

    if index == offsets.size - 1 and new_indexer >= chunksizes[index]:
        return {}
    else:
        return {index: new_indexer}


def _decompose_array_by_chunks(indexer, offsets, chunksizes):
    pass


def decompose_by_chunks(indexer, chunks):
    chunksizes = np.array(chunks, dtype="uint64")
    offsets = np.cumsum(chunksizes)

    if isinstance(indexer, slice):
        return _decompose_slice_by_chunks(indexer, offsets, chunksizes)
    elif isinstance(indexer, (int, np.integer)):
        return _decompose_int_by_chunks(indexer, offsets, chunksizes)
    else:
        return _decompose_array_by_chunks(indexer, offsets, chunksizes)


class ChunkGrid:
    """Chunk grid that records slice assignment"""

    def __init__(self, *, shape, dtype, order, fill_value):
        self._shape = shape
        self._dtype = dtype
        self._order = order  # unused, physical arrays are always 1-d for sparse
        self._fill_value = fill_value
        self._data = {}

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def order(self):
        return self._order

    @property
    def fill_value(self):
        return self._fill_value

    def __getitem__(self, key):
        normalized_key = tuple(normalize_slice(k, s) for k, s in zip(key, self.shape))
        return self._data[normalized_key]

    def __setitem__(self, key, value):
        normalized_key = tuple(normalize_slice(k, s) for k, s in zip(key, self.shape))
        self._data[normalized_key] = value

    def get_value(self):
        chunk_sizes = [
            tuple(slice_size(k, s) for k, s in zip(key, self._shape))
            for key in self._data.keys()
        ]
        return chunk_sizes


@register_ndbuffer
class SparseNDBuffer(NDBuffer):
    def __init__(self, chunk_grid) -> None:
        self._data = chunk_grid

    @classmethod
    def create(
        cls,
        *,
        shape: Iterable[int],
        dtype: npt.DTypeLike,
        order: Literal["C", "F"] = "C",
        fill_value: Any | None = None,
    ) -> Self:
        return cls(
            ChunkGrid(
                shape=tuple(shape), dtype=dtype, order=order, fill_value=fill_value
            )
        )

    @classmethod
    def from_numpy_array(cls, array_like: npt.ArrayLike) -> Self:
        return cls.from_ndarray_like(array_like)

    @classmethod
    def from_ndarray_like(cls, ndarray_like) -> Self:
        buffer = cls.create(
            shape=ndarray_like.shape,
            dtype=ndarray_like.dtype,
            order="C",
            fill_value=ndarray_like.fill_value,
        )
        buffer[(slice(None),) * ndarray_like.ndim] = ndarray_like

        return buffer

    def as_numpy_array(self) -> npt.NDArray[Any]:
        """Returns the buffer as a NumPy array (host memory).

        Warnings
        --------
        Might have to copy data, consider using `.as_ndarray_like()` instead.

        Returns
        -------
            NumPy array of this buffer (might be a data copy)
        """
        raise NotImplementedError("can't convert to `numpy`")

    def as_ndarray_like(self):
        return self._data.get_value()

    def __getitem__(self, key: Any) -> Self:
        return self.__class__(self._data.__getitem__(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, NDBuffer):
            value = value._data

        slice_sizes = tuple(
            slice_size(slice_, size) for slice_, size in zip(key, self._data.shape)
        )
        if value.ndim == 0:
            # fill value
            value = sparse.full(slice_sizes, fill_value=value, dtype=value.dtype)

        self._data.__setitem__(key, value)

    def all_equal(self, other: Any, equal_nan: bool = True) -> bool:
        """Compare to `other` using np.array_equal."""
        if other is None:
            # Handle None fill_value for Zarr V2
            return False

        equal_nan = (
            equal_nan
            if self._data.dtype.kind not in ("U", "S", "T", "O", "V")
            else False
        )
        if other.ndim == 0:
            return np.array_equal(
                self._data.fill_value, other, equal_nan=equal_nan
            ) and (
                self._data.nnz == 0
                or np.equal(
                    self._data.data,
                    np.broadcast_to(other, self._data.data.shape),
                    equal_nan,
                )
            )

        # use array_equal to obtain equal_nan=True functionality
        # Since fill-value is a scalar, isn't there a faster path than allocating a new array for fill value
        # every single time we have to write data?
        _data, other = sparse.broadcast_arrays(self._data, other)

        return sparse.equal(
            self._data,
            other,
            equal_nan=(
                equal_nan
                if self._data.dtype.kind not in ("U", "S", "T", "O", "V")
                else False
            ),
        )


buffer_prototype = BufferPrototype(buffer=Buffer, nd_buffer=SparseNDBuffer)


def sparse_buffer_prototype() -> BufferPrototype:
    return BufferPrototype(buffer=Buffer, nd_buffer=NDBuffer)
