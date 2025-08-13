from __future__ import annotations

import itertools
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


def tiles_by_id(parts):
    return {
        np.unravel_index(index, parts.shape): array
        for index, array in enumerate(parts.flatten())
    }


def until_nth(index):
    def indexer(val):
        return val[:index]

    return indexer


def as_item_key(key):
    def wrapper(it):
        return key(it[0])

    return wrapper


def groupby_mapping(mapping, key):
    wrapped_key = as_item_key(key)
    raw_groups = itertools.groupby(
        sorted(mapping.items(), key=wrapped_key), key=wrapped_key
    )
    return ((key, (el for _, el in group)) for key, group in raw_groups)


def combine_nd(parts):
    tiles = tiles_by_id(parts)
    xp = parts.flat[0].__array_namespace__()

    # innermost to outermost
    for axis in range(parts.ndim - 1, -1, -1):
        tiles = {
            key: xp.concat(list(arrays), axis=axis)
            for key, arrays in groupby_mapping(tiles, key=until_nth(axis))
        }

    return next(iter(tiles.values()))


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


def slice_to_chunk_indices(slice_, offsets, chunks):
    condition = (slice_.start <= offsets) & (slice_.stop >= offsets + chunks)
    selected = np.flatnonzero(condition)

    if selected.size == 0:
        raise ValueError(f"Selected chunk out of bounds: {slice_}")

    return tuple(int(_) for _ in selected)


def decompose_slice(slice_, offsets, chunks, local=False):
    chunk_indices = slice_to_chunk_indices(slice_, offsets, chunks)
    total_size = sum(chunks)

    decomposed = {
        index: (
            slice_slice(
                slice_,
                slice(int(offsets[index]), int(offsets[index]) + chunks[index], 1),
                total_size,
            )
            if not local
            else slice(
                slice_.start - int(offsets[index]),
                slice_.stop - int(offsets[index]),
                slice_.step,
            )
        )
        for index in chunk_indices
    }
    return list(decomposed.items())


def decompose_slices(slices, all_offsets, all_chunks, local=False):
    decomposed = (
        decompose_slice(slice_, offsets, chunks, local=local)
        for slice_, offsets, chunks in zip(slices, all_offsets, all_chunks)
    )
    combined = [tuple(zip(*elements)) for elements in itertools.product(*decomposed)]
    return combined


def slice_slice(old_slice: slice, applied_slice: slice, size: int) -> slice:
    """Given a slice and the size of the dimension to which it will be applied,
    index it with another slice to return a new slice equivalent to applying
    the slices sequentially
    """
    old_slice = normalize_slice(old_slice, size)

    size_after_old_slice = len(range(old_slice.start, old_slice.stop, old_slice.step))
    if size_after_old_slice == 0:
        # nothing left after applying first slice
        return slice(0)

    applied_slice = normalize_slice(applied_slice, size_after_old_slice)

    start = old_slice.start + applied_slice.start * old_slice.step
    if start < 0:
        # nothing left after applying second slice
        # (can only happen for old_slice.step < 0, e.g. [10::-1], [20:])
        return slice(0)

    stop = old_slice.start + applied_slice.stop * old_slice.step
    if stop < 0:
        stop = None

    step = old_slice.step * applied_slice.step

    return slice(start, stop, step)


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
        decomposed_slices = decompose_slices(
            normalized_key, self._offsets, self.chunks, local=True
        )
        chunk_indices_ = list(c for c, _ in decomposed_slices)
        by_dim = [sorted(set(c)) for c in zip(*chunk_indices_)]
        new_grid_shape = tuple(len(dim) for dim in by_dim)
        new_grid = np.full(new_grid_shape, fill_value=None, dtype=object)

        for chunk_indices, value_slices in decomposed_slices:
            new_indices = tuple(
                dim.index(indexer) for indexer, dim in zip(chunk_indices, by_dim)
            )
            selected = self._data[chunk_indices]

            new_grid[new_indices] = selected[value_slices]

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
        decomposed_slices = decompose_slices(
            normalized_key, self._offsets, self.chunks, local=False
        )

        for chunk_indices, value_slice in decomposed_slices:
            # TODO: fix the bug in `decompose_slice`
            chunk_shape = tuple(
                chunks[index] for index, chunks in zip(chunk_indices, self.chunks)
            )
            if value.shape != chunk_shape:
                sliced = value[value_slice]
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
