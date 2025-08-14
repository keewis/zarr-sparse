import itertools

import numpy as np
import pytest

from zarr_sparse.slices import (
    decompose_slice,
    decompose_slices,
    normalize_slice,
    slice_size,
    slice_to_chunk_indices,
)


@pytest.mark.parametrize(
    ["slice_", "size", "expected"],
    (
        (slice(None), 10, 10),
        (slice(None, 4), 10, 4),
        (slice(1, None), 3, 2),
    ),
)
def test_slice_size(slice_, size, expected):
    actual = slice_size(slice_, size)

    assert actual == expected


@pytest.mark.parametrize(
    ["slice_", "size", "expected"],
    (
        (slice(None), 10, slice(0, 10, 1)),
        (slice(None, 4), 10, slice(0, 4, 1)),
        (slice(1, None), 3, slice(1, 3, 1)),
    ),
)
def test_normalize_slice(slice_, size, expected):
    actual = normalize_slice(slice_, size)

    assert actual == expected


@pytest.mark.parametrize(
    ["size", "n_chunks", "slice_", "expected"],
    (
        (4, 2, slice(0, 3), (0, 1)),
        (4, 2, slice(1, 2), (0,)),
        (4, 2, slice(0, 2), (0,)),
        (7, 3, slice(1, 7), (0, 1, 2, 3)),
        (11, 5, slice(5, 11), (2, 3, 4, 5)),
    ),
)
def test_slice_to_chunk_indices(size, n_chunks, slice_, expected):
    chunksize, remainder = divmod(size, n_chunks)
    chunks = tuple(c for c in (chunksize,) * n_chunks + (remainder,) if c > 0)

    bounds = np.cumulative_sum(np.asarray(chunks, dtype="uint64"), include_initial=True)

    print(f"input: {size=}, {n_chunks=}")
    print("chunks:", chunks)
    print("bounds:", bounds)

    actual = slice_to_chunk_indices(slice_, bounds)

    assert actual == expected


@pytest.mark.parametrize(
    ["size", "n_chunks", "slice_", "expected"],
    (
        (
            10,
            2,
            slice(None),
            [(0, slice(0, 5, 1), slice(0, 5, 1)), (1, slice(0, 5, 1), slice(5, 10, 1))],
        ),
        (10, 2, slice(0, 5), [(0, slice(0, 5, 1), slice(0, 5, 1))]),
        (
            10,
            2,
            slice(1, 7),
            [(0, slice(1, 5, 1), slice(1, 5, 1)), (1, slice(0, 2, 1), slice(5, 7, 1))],
        ),
        (
            10,
            5,
            slice(5, 10),
            [
                (2, slice(1, 2, 1), slice(5, 6, 1)),
                (3, slice(0, 2, 1), slice(6, 8, 1)),
                (4, slice(0, 2, 1), slice(8, 10, 1)),
            ],
        ),
        (
            50,
            10,
            slice(36, 48),
            [
                (7, slice(1, 5, 1), slice(36, 40, 1)),
                (8, slice(0, 5, 1), slice(40, 45, 1)),
                (9, slice(0, 3, 1), slice(45, 48, 1)),
            ],
        ),
    ),
)
def test_decompose_slice(size, n_chunks, slice_, expected):
    chunksize, remainder = divmod(size, n_chunks)
    chunks = tuple(c for c in (chunksize,) * n_chunks + (remainder,) if c > 0)

    bounds = np.cumulative_sum(np.asarray(chunks, dtype="uint64"), include_initial=True)

    print(f"input: {size=}, {n_chunks=}")
    print("chunks:", chunks)
    print("bounds:", bounds)

    actual = decompose_slice(slice_, bounds)

    assert actual == expected


def test_decompose_slices():
    # shape = (10, 9)  # for reference
    chunks = ((3, 3, 3, 1), (2, 2, 2, 2, 1))

    bounds = tuple(
        np.cumulative_sum(np.asarray(c, dtype="uint64"), include_initial=True)
        for c in chunks
    )

    slices = (slice(1, 6), slice(3, 8))

    actual = decompose_slices(slices, bounds)

    decomposed = [
        ((0, slice(1, 3, 1), slice(1, 3, 1)), (1, slice(0, 3, 1), slice(3, 6, 1))),
        (
            (1, slice(1, 2, 1), slice(3, 4, 1)),
            (2, slice(0, 2, 1), slice(4, 6, 1)),
            (3, slice(0, 2, 1), slice(6, 8, 1)),
        ),
    ]
    expected = [tuple(zip(*el)) for el in itertools.product(*decomposed)]

    assert actual == expected
