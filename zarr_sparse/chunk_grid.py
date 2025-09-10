from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from zarr_sparse.slices import slice_size

if TYPE_CHECKING:
    from typing import Any, Literal, Self

    import numpy.typing as npt

    ChunkKeyType = tuple[int, ...]
    BoundsType = dict[tuple[int, ...], tuple[range, ...]]


def readjust_chunk_keys(bounds: dict[ChunkKeyType, Any]) -> dict[ChunkKeyType, Any]:
    mins = tuple(min(x) for x in zip(*bounds.keys()))
    return {
        tuple(part - offset for part, offset in zip(chunk_key, mins)): bound
        for chunk_key, bound in bounds.items()
    }


def readjust_bounds(bounds: BoundsType) -> BoundsType:
    min_chunk_key = tuple(min(x) for x in zip(*bounds.keys()))

    bound_starts = (tuple(b.start for b in bounds_) for bounds_ in bounds.values())
    min_bound = tuple(min(x) for x in zip(*bound_starts))

    return {
        tuple(part - offset for part, offset in zip(chunk_key, min_chunk_key)): tuple(
            range(b.start - offset, b.stop - offset, b.step)
            for b, offset in zip(bound, min_bound)
        )
        for chunk_key, bound in bounds.items()
    }


@dataclass
class ChunkGrid:

    dtype: npt.DTypeLike
    order: Literal["C", "F"] = "C"
    fill_value: Any | None = None
    shape: tuple[int, ...]

    chunk_shape: tuple[int, ...] = ()

    bounds: BoundsType = field(default_factory=dict, init=False)
    data: dict[tuple[int, ...], Any] = field(default_factory=dict, init=False)

    def __setitem__(self, indexers: tuple[slice, ...], value: Any) -> None:
        offsets = tuple(s.start for s in indexers)
        c_shape = tuple(s.stop - s.start for s in indexers)

        if not self.data and offsets != (0,) * len(indexers):
            # first chunk, must not have an offset
            raise ValueError("First write must write to the first chunk")

        if not self.chunk_shape:
            self.chunk_shape = c_shape

        position = tuple(
            offset // size for offset, size in zip(offsets, self.chunk_shape)
        )

        self.data[position] = value
        self.bounds[position] = tuple(
            range(*indexer.indices(size)) for indexer, size in zip(indexers, self.shape)
        )

    def _select_keys(
        self, selected_keys: list[tuple[int, ...]], shape: tuple[int, ...]
    ) -> Self:
        new = type(self)(shape=shape, chunk_shape=self.chunk_shape)

        new.bounds = readjust_bounds({k: self.bounds[k] for k in selected_keys})
        new.data = readjust_chunk_keys({k: self.data[k] for k in selected_keys})

        return new

    def __getitem__(self, indexers: tuple[slice, ...]) -> Self:
        # find all keys that intersect with the indexers
        selected_keys = [
            key
            for key, bounds in self.bounds.items()
            if all(
                bound.start < indexer.stop and bound.stop > indexer.start
                for bound, indexer in zip(bounds, indexers)
            )
        ]
        region_size = tuple(
            slice_size(slice_, size) for slice_, size in zip(indexers, self.shape)
        )
        shape = tuple(
            math.ceil(s / c) * c for s, c in zip(region_size, self.chunk_shape)
        )

        return self._select_keys(selected_keys, shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def offsets(self) -> dict[tuple[int, ...], tuple[int, ...]]:
        return {k: tuple(b.start for b in bounds) for k, bounds in self.bounds.items()}

    def __repr__(self) -> str:
        shape = self.shape
        chunk_shape = self.chunk_shape
        fill_value = self.fill_value
        order = self.order
        dtype = self.dtype

        return (
            f"<ChunkGrid {shape=}, {chunk_shape=}, {order=}, {dtype=}, {fill_value=}>"
        )
