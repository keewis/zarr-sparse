import numpy as np
import pytest

from zarr_sparse import combine


@pytest.mark.parametrize(
    ["parts", "expected_ids"],
    (
        (
            np.array(
                [[{"x": 1}, {"x": 2}, {"x": 3}], [{"x": 4}, {"x": 5}, {"x": 6}]],
                dtype=object,
            ),
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        ),
        (
            np.array(
                [[[{"x": 1}], [{"x": 2}]], [[{"x": 3}], [{"x": 4}]]], dtype=object
            ),
            [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)],
        ),
    ),
)
def test_tiles_by_id(parts, expected_ids):
    expected = {id_: {"x": index} for index, id_ in enumerate(expected_ids, start=1)}

    actual = combine.tiles_by_id(parts)
    assert actual == expected


@pytest.mark.parametrize("index", [1, 4, 8, -1])
def test_until_nth(index):
    selector = combine.until_nth(index)

    sequence = range(20)

    actual = selector(sequence)

    assert actual == sequence[:index]


@pytest.mark.parametrize("key", [lambda x: x**2, lambda x: x * 2])
def test_as_item_key(key):
    mapping = dict.fromkeys(range(20))

    wrapped = combine.as_item_key(key)

    expected = [key(x) for x in mapping.keys()]
    actual = list(map(wrapped, mapping.items()))

    assert actual == expected


@pytest.mark.parametrize(
    ["axis", "expected_keys"],
    (
        (
            2,
            [
                [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],
                [(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3)],
                [(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3)],
                [(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3)],
            ],
        ),
        (1, [[(0, 0), (0, 1)], [(1, 0), (1, 1)]]),
        (0, [[(0,), (1,)]]),
    ),
)
def test_groupby_mapping(axis, expected_keys):
    keys = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 1, 3),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 0, 3),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 3),
    ]
    filtered = list(dict.fromkeys(k[: axis + 1] for k in keys))
    mapping = {k: v for v, k in enumerate(filtered)}

    expected = {
        group[0][:axis] if axis > 0 else (): [mapping[k] for k in group]
        for group in expected_keys
    }

    actual = {
        k: list(g)
        for k, g in combine.groupby_mapping(mapping, key=combine.until_nth(axis))
    }

    assert actual == expected


def create_grid(shapes, dtype):
    shapes = np.asarray(shapes, dtype="uint64")
    *grid_shape, ndim = shapes.shape

    grid = np.full(shape=grid_shape, fill_value=None, dtype=object)
    for index in range(np.prod(shapes.shape[:-1])):
        indices = np.unravel_index(index, grid_shape)
        shape = shapes[*indices, :]

        grid[indices] = np.zeros(shape=shape, dtype=dtype)

    return grid


@pytest.mark.parametrize(
    ["shapes", "expected"],
    (
        (
            [
                [(2, 3), (2, 3), (2, 2)],
                [(2, 3), (2, 3), (2, 2)],
                [(1, 3), (1, 3), (1, 2)],
            ],
            np.zeros(shape=(5, 8), dtype="float32"),
        ),
        (
            [
                [
                    [(5, 3, 4), (5, 3, 4)],
                    [(5, 3, 4), (5, 3, 4)],
                    [(5, 1, 4), (5, 1, 4)],
                ],
                [
                    [(5, 3, 4), (5, 3, 4)],
                    [(5, 3, 4), (5, 3, 4)],
                    [(5, 1, 4), (5, 1, 4)],
                ],
                [
                    [(5, 3, 4), (5, 3, 4)],
                    [(5, 3, 4), (5, 3, 4)],
                    [(5, 1, 4), (5, 1, 4)],
                ],
                [
                    [(2, 3, 4), (2, 3, 4)],
                    [(2, 3, 4), (2, 3, 4)],
                    [(2, 1, 4), (2, 1, 4)],
                ],
            ],
            np.zeros(shape=(17, 7, 8), dtype="int16"),
        ),
    ),
)
def test_combine_nd(shapes, expected):
    parts = create_grid(shapes, expected.dtype)

    actual = combine.combine_nd(parts)

    np.testing.assert_equal(actual, expected)
