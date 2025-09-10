import numpy as np
import pytest

from zarr_sparse import chunk_grid


@pytest.mark.parametrize(
    "chunk_keys",
    (
        pytest.param(dict.fromkeys((v,) for v in range(1, 3)), id="1d"),
        pytest.param(dict.fromkeys([(0, 0), (0, 1), (1, 0), (1, 1)]), id="no_change"),
        pytest.param(dict.fromkeys([(1, 0), (1, 1)]), id="row"),
        pytest.param(dict.fromkeys([(0, 1), (1, 1)]), id="column"),
    ),
)
def test_readjust_chunk_keys(chunk_keys):
    actual = chunk_grid.readjust_chunk_keys(chunk_keys)

    min_values = tuple(min(v) for v in zip(*actual.keys()))
    assert min_values == (0,) * len(min_values)


def test_init():
    shape = (10, 10)
    chunk_shape = (5, 5)
    grid = chunk_grid.ChunkGrid(shape=shape, chunk_shape=chunk_shape)

    assert grid.shape == shape
    assert grid.chunk_shape == chunk_shape

    assert grid.bounds == {}
    assert grid.data == {}

    assert grid.ndim == 2
    assert grid.offsets == {}


def test_repr():
    shape = (10, 10)
    chunk_shape = (5, 5)
    grid = chunk_grid.ChunkGrid(shape=shape, chunk_shape=chunk_shape)

    actual = repr(grid)

    assert "ChunkGrid" in actual
    assert f"shape={shape}" in actual
    assert f"chunk_shape={chunk_shape}" in actual


@pytest.mark.parametrize("keys", ([(1,), (2,)], [(0,), (1,)]))
def test_select_keys(keys):
    data = np.arange(10)

    grid = chunk_grid.ChunkGrid(shape=(10,), chunk_shape=(2,))

    grid.bounds = {
        (index,): (range(index * 2, (index + 1) * 2, 1),) for index in range(5)
    }
    grid.data = {key: data[indexer] for key, indexer in grid.bounds.items()}

    actual = grid._select_keys(keys, shape=(4,))

    indexers = [(slice(index * 2, (index + 1) * 2, 1),) for index, in keys]

    assert len(actual.bounds) == len(keys)
    assert list(actual.bounds.keys()) == [(0,), (1,)]
    assert list(actual.bounds.values()) == [(range(*s.indices(10)),) for s, in indexers]
    assert [d.tolist() for d in actual.data.values()] == [
        data[s].tolist() for s in indexers
    ]


def test_setitem_implicit():
    data = np.arange(10 * 8).reshape(10, 8)

    grid = chunk_grid.ChunkGrid(shape=(10, 8))
    assert grid.chunk_shape == ()

    grid[0:5, 0:5] = data[0:5, 0:5]
    step1_bounds = {(0, 0): (range(0, 5, 1), range(0, 5, 1))}
    assert grid.chunk_shape == (5, 5)
    assert grid.bounds == step1_bounds
    np.testing.assert_equal(grid.data[(0, 0)], data[0:5, 0:5])

    grid[5:10, 0:5] = data[5:10, 0:5]
    step2_bounds = {(1, 0): (range(5, 10, 1), range(0, 5, 1))}
    assert grid.bounds == step1_bounds | step2_bounds
    np.testing.assert_equal(grid.data[(1, 0)], data[5:10, 0:5])

    grid[0:5, 5:8] = data[0:5, 5:8]
    step3_bounds = {(0, 1): (range(0, 5, 1), range(5, 8, 1))}
    assert grid.bounds == step1_bounds | step2_bounds | step3_bounds
    np.testing.assert_equal(grid.data[(0, 1)], data[0:5, 5:8])

    grid[5:10, 5:8] = data[5:10, 5:8]
    step4_bounds = {(1, 1): (range(5, 10, 1), range(5, 8, 1))}
    assert grid.bounds == step1_bounds | step2_bounds | step3_bounds | step4_bounds
    np.testing.assert_equal(grid.data[(1, 1)], data[5:10, 5:8])


def test_setitem_explicit():
    data = np.arange(10 * 8).reshape(10, 8)

    grid = chunk_grid.ChunkGrid(shape=(10, 8), chunk_shape=(5, 4))
    assert grid.chunk_shape == (5, 4)

    grid[0:5, 0:4] = data[0:5, 0:4]
    assert grid.bounds == {(0, 0): (range(0, 5, 1), range(0, 4, 1))}
