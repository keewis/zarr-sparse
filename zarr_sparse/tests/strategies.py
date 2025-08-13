import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import sparse

from zarr_sparse.utils import expand_chunks

nnz = st.integers(min_value=1, max_value=30)


@st.composite
def _chunks(draw, shapes):
    shape = draw(shapes)

    n_chunks = tuple(draw(st.integers(min_value=1, max_value=size)) for size in shape)

    return tuple(size // n for size, n in zip(shape, n_chunks))


@st.composite
def _slices(draw, shapes, chunks):
    shape = draw(shapes)
    chunks_ = draw(chunks)

    expanded_chunks = expand_chunks(chunks_, shape)

    offsets = tuple(
        np.cumulative_sum(np.asarray(c, dtype="uint64"), include_initial=True)
        for c in expanded_chunks
    )

    chunk_bounds = tuple(
        tuple(slice(lower, upper) for lower, upper in zip(o[:-1], o[1:]))
        for o in offsets
    )

    return tuple(draw(st.sampled_from(c)) for c in chunk_bounds)


_dtypes = (
    npst.boolean_dtypes()
    | npst.integer_dtypes(endianness="=")
    | npst.floating_dtypes(sizes=(32, 64), endianness="=")
)
_shapes = npst.array_shapes(min_dims=1, max_dims=4, min_side=3, max_side=1000)

dtypes = st.shared(_dtypes, key="dtypes")
fill_values = npst.arrays(st.shared(_dtypes, key="dtypes"), shape=(1,)).map(
    lambda x: x.item()
)
shapes = st.shared(_shapes, key="shapes")
chunks = st.shared(_chunks(st.shared(_shapes, key="shapes")), key="chunks")
slices = _slices(shapes, chunks)


def sparse_coo_arrays(
    *, nnz=nnz, shapes=shapes, dtypes=dtypes, fill_values=fill_values
):
    def _create_coo_array(
        nnz: int,
        shape: tuple[int, ...],
        dtype: np.dtype[np.bool | np.integer | np.floating],
        fill_value: bool | int | float,
    ):
        rng = np.random.default_rng(seed=0)

        if np.isdtype(dtype, "bool"):
            data = rng.integers(low=0, high=1, size=nnz).astype(dtype)
            fill_value = rng.choice([True, False]).item()
        elif np.isdtype(dtype, "integral"):
            iinfo = np.iinfo(dtype)
            data = rng.integers(
                low=iinfo.min, high=iinfo.max, size=nnz, dtype=dtype, endpoint=True
            )
            fill_value = rng.integers(
                low=iinfo.min, high=iinfo.max, size=1, endpoint=True
            ).item()
        else:
            data = rng.random(size=nnz, dtype=dtype)
            fill_value = rng.random(size=1, dtype=dtype).item()

        coords = np.stack(
            [rng.integers(dim_size, size=nnz) for dim_size in shape], axis=0
        )

        return sparse.COO(data=data, coords=coords, shape=shape, fill_value=fill_value)

    if isinstance(nnz, int):
        nnz = st.just(nnz)
    if isinstance(shapes, tuple):
        shapes = st.just(shapes)
    if isinstance(dtypes, np.dtype):
        dtypes = st.just(dtypes)
    if isinstance(fill_values, (bool, int, float)):
        fill_values = st.just(fill_values)

    return st.builds(_create_coo_array, nnz, shapes, dtypes, fill_values)
