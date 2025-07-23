from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import sparse
from zarr.core.buffer.core import Buffer, BufferPrototype, NDBuffer
from zarr.registry import register_ndbuffer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self


@register_ndbuffer
class SparseNDBuffer(NDBuffer):
    def __init__(self, sparse_array) -> None:
        super().__init__(sparse_array)

    @classmethod
    def create(
        cls,
        *,
        shape: Iterable[int],
        dtype: npt.DTypeLike,
        order: Literal["C", "F"] = "C",
        fill_value: Any | None = None,
    ) -> Self:
        # np.zeros is much faster than np.full, and therefore using it when possible is better.
        if fill_value is None or (isinstance(fill_value, int) and fill_value == 0):
            return cls(sparse.zeros(shape=tuple(shape), dtype=dtype, order=order))
        else:
            return cls(
                sparse.full(
                    shape=tuple(shape), fill_value=fill_value, dtype=dtype, order=order
                )
            )

    @classmethod
    def from_numpy_array(cls, array_like: npt.ArrayLike) -> Self:
        return cls.from_ndarray_like(array_like)

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

    def __getitem__(self, key: Any) -> Self:
        return self.__class__(self._data.__getitem__(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, NDBuffer):
            value = value._data
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
