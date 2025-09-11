from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import sparse
from zarr.core.buffer.core import BufferPrototype, NDBuffer
from zarr.core.buffer.cpu import Buffer
from zarr.registry import register_ndbuffer

from zarr_sparse.chunk_grid import ChunkGrid
from zarr_sparse.combine import combine_nd
from zarr_sparse.slices import slice_size

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Literal, Self


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
    def from_ndarray_like(cls, ndarray_like, chunks=None) -> Self:
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
        return combine_nd(self._data.data)

    def __getitem__(self, key: Any) -> Self:
        return self.__class__(self._data[key])

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, NDBuffer):
            value = value._data

        slice_sizes = tuple(
            slice_size(slice_, size) for slice_, size in zip(key, self._data.shape)
        )
        if value.ndim == 0:
            # fill value
            value = sparse.full(slice_sizes, fill_value=value, dtype=value.dtype)

        self._data[key] = value

    def all_equal(self, other: Any, equal_nan: bool = True) -> bool:
        """Compare to `other` using np.array_equal."""
        if other is None:
            # Handle None fill_value for Zarr V2
            return False

        # - other is multi-dim: return false if:
        #   - shape
        #   - dtype
        #   - order
        #   - fill_value
        #   - chunk_shape
        #   - bounds
        #   don't match
        # - if other is 0d:
        #   - get a scalar
        #   - compare every value in the chunk grid to the scalar
        # - if other is multi-dim and matches, compare every chunk

        if other.ndim != 0 and (
            self.shape != other.shape
            or self.dtype != other.dtype
            or self.order != other.order
            or self.fill_value != other.fill_value
            or self.chunk_shape != other.chunk_shape
            or self.bounds != other.bounds
        ):
            return False
        if other.ndim == 0:
            if isinstance(other, ChunkGrid):
                # extract a single value from the chunk grid
                raise NotImplementedError(
                    "scalar wrapped by a chunk grid not supported"
                )
            else:
                scalar = other
            to_compare = ((c, scalar) for c in self._data.data.values())
        else:
            to_compare = zip(
                self._data.data.values(), other._data.data.values().ravel().tolist()
            )

        return all(sparse_equal(a, b, equal_nan=equal_nan) for a, b in to_compare)


buffer_prototype = BufferPrototype(buffer=Buffer, nd_buffer=SparseNDBuffer)


def sparse_buffer_prototype() -> BufferPrototype:
    return BufferPrototype(buffer=Buffer, nd_buffer=SparseNDBuffer)
