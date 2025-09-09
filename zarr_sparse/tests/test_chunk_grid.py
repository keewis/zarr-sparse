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
