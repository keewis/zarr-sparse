from __future__ import annotations

import bisect
import math
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
        if slice_ == slice(None):
            return normalize_slice(slice_, size)

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
        index: slice_
        for index, slice_ in decomposed.items()
        if slice_size(slice_, chunksizes[index]) > 0
    }


def _decompose_int_by_chunks(indexer, offsets, chunksizes):
    if indexer >= offsets[-1] + chunksizes[-1]:
        return {}

    index = bisect.bisect_right(offsets, indexer) - 1
    new_indexer = indexer - int(offsets[index])

    return {index: new_indexer}


def _decompose_array_by_chunks(indexer, offsets, chunksizes):
    raise NotImplementedError


def decompose_by_chunks(indexer, chunks):
    chunksizes = np.array(chunks, dtype="uint64")
    offsets = np.cumulative_sum(chunksizes, include_initial=True)[:-1]

    if isinstance(indexer, slice):
        return _decompose_slice_by_chunks(indexer, offsets, chunksizes)
    elif isinstance(indexer, (int, np.integer)):
        return _decompose_int_by_chunks(indexer, offsets, chunksizes)
    else:
        return _decompose_array_by_chunks(indexer, offsets, chunksizes)


def expand_chunks(chunks, shape):
    def _expand(chunkspec, size):
        if chunkspec == -1:
            return (size,)
        elif isinstance(chunkspec, int):
            n_full_chunks, remainder = divmod(size, chunkspec)
            chunks = (chunkspec,) * n_full_chunks

            if remainder > 0:
                chunks += (remainder,)

            return chunks

        chunkspec = tuple(chunkspec)
        if sum(chunkspec) != size:
            raise ValueError(f"chunks don't add up to the full size: {chunkspec}")

        return chunkspec

    return tuple(_expand(chunkspec, size) for chunkspec, size in zip(chunks, shape))


def slice_to_chunk_index(slice_, offsets):
    if slice_ == slice(None):
        return 0

    chunk_index = np.flatnonzero(offsets == slice_.start)

    if chunk_index.size == 0:
        raise ValueError(f"Selected chunk out of bounds: {slice_}")

    return chunk_index[0]


def sparse_equal(a, b, equal_nan: bool) -> bool:
    equal_nan = equal_nan if a.dtype.kind not in ("U", "S", "T", "O", "V") else False

    if b.ndim == 0:
        return np.array_equal(
            a.fill_value, getattr(b, "fill_value", b), equal_nan=equal_nan
        ) and (
            a.nnz == 0
            or np.equal(
                a.data,
                np.broadcast_to(b.data, a.data.shape),
                equal_nan,
            )
        )

    # use array_equal to obtain equal_nan=True functionality
    # Since fill-value is a scalar, isn't there a faster path than allocating a new array for fill value
    # every single time we have to write data?
    _data, other = sparse.broadcast_arrays(a, b)

    return sparse.equal(a, other, equal_nan=equal_nan)


class ChunkGrid:
    """Chunk grid that records slice assignment"""

    def __init__(self, *, shape, dtype, order, fill_value, chunks=None):
        self._shape = shape
        self._dtype = dtype
        self._order = order  # unused, physical arrays are always 1-d for sparse
        self._fill_value = fill_value

        if chunks is None:
            chunks = shape

        self._chunks = expand_chunks(chunks, shape)
        self._offsets = tuple(
            np.cumulative_sum(c, include_initial=True)[:-1] for c in self._chunks
        )
        grid_shape = tuple(math.ceil(s / max(c)) for s, c in zip(shape, self._chunks))
        self._data = np.full(grid_shape, dtype=object, fill_value=None)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        return self._dtype

    @property
    def order(self):
        return self._order

    @property
    def fill_value(self):
        return self._fill_value

    @property
    def chunks(self):
        return self._chunks

    def __getitem__(self, key):
        chunk_indices = tuple(
            slice_to_chunk_index(k, o) for k, o in zip(key, self._offsets)
        )
        data = self._data[chunk_indices]
        if data is None:
            chunk_shape = tuple(
                chunksizes[index]
                for index, chunksizes in zip(chunk_indices, self.chunks)
            )
            data = sparse.full(
                chunk_shape, fill_value=self.fill_value, dtype=self.dtype
            )

        result = type(self)(
            shape=data.shape,
            dtype=data.dtype,
            order=self.order,
            fill_value=data.fill_value,
            chunks=None,
        )
        result._data[0, 0] = data
        return result

    def __setitem__(self, key, value):
        if any(
            s == slice(None) and size != total_size
            for s, size, total_size in zip(key, value.shape, self.shape)
        ):
            raise ValueError("size mismatch")

        chunk_indices = tuple(
            slice_to_chunk_index(k, o) for k, o in zip(key, self._offsets)
        )
        if isinstance(value, ChunkGrid):
            value = value._data.item()
        self._data[chunk_indices] = value

    def get_value(self):
        chunk_sizes = [
            tuple(slice_size(k, s) for k, s in zip(key, self._shape))
            for key in self._data.keys()
        ]
        return chunk_sizes

    def all_equal(self, other: Any, equal_nan: bool) -> bool:
        if other.ndim != 0 and (
            self.shape != other.shape
            or self.dtype != other.dtype
            or self.order != other.order
            or self.fill_value != other.fill_value
            or self.chunks != other.chunks
        ):
            return False

        if other.ndim == 0:
            to_compare = (
                (c, other._data.item() if isinstance(other, ChunkGrid) else other)
                for c in self._data.ravel().tolist()
            )
        else:
            to_compare = zip(self._data.ravel().tolist(), other._data.ravel().tolist())

        return all(sparse_equal(a, b, equal_nan=equal_nan) for a, b in to_compare)


@register_ndbuffer
class SparseNDBuffer(NDBuffer):
    def __init__(self, chunk_grid) -> None:
        if chunk_grid is None:
            raise ValueError("chunk grid is `None`")
        self._data = chunk_grid

    @classmethod
    def create(
        cls,
        *,
        shape: Iterable[int],
        dtype: npt.DTypeLike,
        order: Literal["C", "F"] = "C",
        fill_value: Any | None = None,
        chunks=None,
    ) -> Self:
        return cls(
            ChunkGrid(
                shape=tuple(shape),
                dtype=dtype,
                order=order,
                fill_value=fill_value,
                chunks=chunks,
            )
        )

    @classmethod
    def from_numpy_array(cls, array_like: npt.ArrayLike) -> Self:
        return cls.from_ndarray_like(array_like)

    @classmethod
    def from_ndarray_like(cls, ndarray_like, chunks=None) -> Self:
        buffer = cls.create(
            shape=ndarray_like.shape,
            dtype=ndarray_like.dtype,
            order="C",
            fill_value=ndarray_like.fill_value,
            chunks=chunks or getattr(ndarray_like, "chunks", None),
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

        return self._data.all_equal(other, equal_nan=equal_nan)


buffer_prototype = BufferPrototype(buffer=Buffer, nd_buffer=SparseNDBuffer)


def sparse_buffer_prototype() -> BufferPrototype:
    return BufferPrototype(buffer=Buffer, nd_buffer=NDBuffer)
