import pytest

from zarr_sparse.slices import normalize_slice, slice_size


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
