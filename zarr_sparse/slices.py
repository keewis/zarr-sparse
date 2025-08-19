from __future__ import annotations

import itertools

import numpy as np


def slice_next(offset: int, size: int) -> slice[int]:
    return slice(offset, offset + size)


def slice_size(slice_: slice[int | None], size: int) -> int:
    return len(range(*slice_.indices(size)))


def normalize_slice(slice_: slice[int | None], size: int) -> slice[int]:
    return slice(*slice_.indices(size))


def slice_to_chunk_indices(slice_, bounds):
    fully_contained = (slice_.start <= bounds[:-1]) & (slice_.stop >= bounds[1:])
    partially_contained = (
        (slice_.start >= bounds[:-1]) & (slice_.start < bounds[1:])
    ) | ((slice_.stop > bounds[:-1]) & (slice_.stop <= bounds[1:]))
    condition = fully_contained | partially_contained

    selected = np.flatnonzero(condition)

    if selected.size == 0:
        raise ValueError(f"Selected chunk out of bounds: {slice_}")

    return tuple(int(_) for _ in selected)


def intersect_slice(a, b):
    start = a.start if a.start >= b.start else b.start
    stop = a.stop if a.stop <= b.stop else b.stop

    return slice(start, stop, a.step)


def decompose_slice(slice_, bounds):
    def _decompose(slice_, lower, upper):
        chunk_slice = slice(lower, upper, 1)

        global_slice = intersect_slice(slice_, chunk_slice)

        start = global_slice.start - lower
        stop = global_slice.stop - lower
        local_slice = slice(start, stop, slice_.step)

        return local_slice, global_slice

    total_size = bounds[-1].item()

    slice_ = normalize_slice(slice_, total_size)
    chunk_indices = slice_to_chunk_indices(slice_, bounds)

    decomposed = {
        index: _decompose(slice_, int(bounds[index]), int(bounds[index + 1]))
        for index in chunk_indices
    }
    return [(index, local, global_) for index, (local, global_) in decomposed.items()]


def decompose_slices(slices, all_bounds):
    decomposed = (
        decompose_slice(slice_, bounds) for slice_, bounds in zip(slices, all_bounds)
    )
    combined = [tuple(zip(*elements)) for elements in itertools.product(*decomposed)]
    return combined
