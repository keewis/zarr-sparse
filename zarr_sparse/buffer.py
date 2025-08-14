from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import sparse
from zarr.core.buffer.core import Buffer, BufferPrototype, NDBuffer
from zarr.registry import register_ndbuffer

from zarr_sparse.chunks import expand_chunks
from zarr_sparse.combine import combine_nd
from zarr_sparse.slices import decompose_slices, normalize_slice, slice_size

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self


def sparse_equal(a, b, equal_nan: bool) -> bool:
    equal_nan = equal_nan if a.dtype.kind not in ("U", "S", "T", "O", "V") else False

    if b.ndim == 0:
        if not np.array_equal(
            a.fill_value, getattr(b, "fill_value", b), equal_nan=equal_nan
        ):
            return False

        if a.nnz == 0:
            return True

        return np.array_equal(
            a.data,
            np.broadcast_to(b.data, a.data.shape),
            equal_nan=equal_nan,
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
        self._bounds = tuple(
            np.cumulative_sum(c, include_initial=True) for c in self._chunks
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

    def __repr__(self):
        shape = self.shape
        fill_value = self.fill_value
        chunks = self.chunks

        grid = self._data

        repr_ = f"<ChunkGrid({shape=}, {fill_value=}, {chunks=}>"

        return "\n".join(
            [
                repr_,
                repr(grid),
            ]
        )

    def __getitem__(self, key):
        if any(not isinstance(k, slice) for k in key):
            return ValueError("indexing is only supported for slices")

        normalized_key = tuple(normalize_slice(k, s) for k, s in zip(key, self.shape))
        decomposed_slices = decompose_slices(normalized_key, self._bounds)
        chunk_indices_ = list(c for c, _, _ in decomposed_slices)
        by_dim = [sorted(set(c)) for c in zip(*chunk_indices_)]
        new_grid_shape = tuple(len(dim) for dim in by_dim)
        new_grid = np.full(new_grid_shape, fill_value=None, dtype=object)

        for chunk_indices, local_indexer, _ in decomposed_slices:
            new_indices = tuple(
                dim.index(indexer) for indexer, dim in zip(chunk_indices, by_dim)
            )
            selected = self._data[chunk_indices]

            new_grid[new_indices] = selected[local_indexer]

        new_shape = tuple(
            sum(chunksizes)
            for chunksizes in zip(*(chunk.shape for chunk in new_grid.ravel()))
        )

        result = type(self)(
            shape=new_shape,
            dtype=self.dtype,
            order=self.order,
            fill_value=self.fill_value,
            chunks=None,
        )
        result._data = new_grid
        return result

    def __setitem__(self, key, value):
        if any(not isinstance(k, slice) for k in key):
            return ValueError("indexing is only supported for slices")

        selection_sizes = tuple(slice_size(k, size) for k, size in zip(key, self.shape))
        if selection_sizes != value.shape:
            raise ValueError("inconsistent assignment")

        normalized_key = tuple(
            normalize_slice(k, size) for k, size in zip(key, self.shape)
        )

        if isinstance(value, ChunkGrid):
            value = value._data.item()

        # decompose into selected chunks and slices into the value
        # iterate over selected chunks and assign the value subset
        decomposed_slices = decompose_slices(normalized_key, self._bounds)

        for chunk_indices, _, global_indexer in decomposed_slices:
            # TODO: fix the bug in `decompose_slice`
            chunk_shape = tuple(
                chunks[index] for index, chunks in zip(chunk_indices, self.chunks)
            )
            if value.shape != chunk_shape:
                sliced = value[global_indexer]
            else:
                sliced = value
            self._data[chunk_indices] = sliced

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

    def get_chunk(self, indexer=None):
        if indexer is None:
            if self._data.size != 1:
                raise ValueError("need to select a chunk for multiple chunk arrays")
            return self._data.item()

        return self._data[indexer]

    def get_value(self):
        return combine_nd(self._data)


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
        return self._data

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
