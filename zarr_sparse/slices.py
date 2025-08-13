import itertools

import numpy as np


def slice_size(slice_: slice[int | None], size: int) -> int:
    return len(range(*slice_.indices(size)))


def normalize_slice(slice_: slice[int | None], size: int) -> slice[int]:
    return slice(*slice_.indices(size))


def slice_to_chunk_indices(slice_, offsets, chunks):
    condition = (slice_.start <= offsets) & (slice_.stop <= offsets + chunks)
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
