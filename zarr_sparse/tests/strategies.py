import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np

from zarr_sparse.tests.generate import create_chunk_slices, create_pydata_coo_array

nnz = st.integers(min_value=1, max_value=30)


@st.composite
def _chunks(draw, shapes):
    shape = draw(shapes)

    n_chunks = tuple(
        draw(st.integers(min_value=1, max_value=size // 2 or 1)) for size in shape
    )

    return tuple(size // n for size, n in zip(shape, n_chunks))


_dtypes = (
    npst.boolean_dtypes()
    | npst.integer_dtypes(endianness="=")
    | npst.floating_dtypes(sizes=(32, 64), endianness="=")
)
_shapes = npst.array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=100)

dtypes = st.shared(_dtypes, key="dtypes")
fill_values = npst.arrays(st.shared(_dtypes, key="dtypes"), shape=(1,)).map(
    lambda x: x.item()
)
shapes = st.shared(_shapes, key="shapes")
__chunks = _chunks(st.shared(_shapes, key="shapes"))
chunks = st.shared(__chunks, key="chunks")
chunk_slices = st.builds(
    create_chunk_slices,
    st.shared(_shapes, key="shapes"),
    st.shared(__chunks, key="chunks"),
)


def sparse_coo_arrays(
    *, nnz=nnz, shapes=shapes, dtypes=dtypes, fill_values=fill_values
):
    if isinstance(nnz, int):
        nnz = st.just(nnz)
    if isinstance(shapes, tuple):
        shapes = st.just(shapes)
    if isinstance(dtypes, np.dtype):
        dtypes = st.just(dtypes)
    if isinstance(fill_values, (bool, int, float)):
        fill_values = st.just(fill_values)

    return st.builds(create_pydata_coo_array, nnz, shapes, dtypes, fill_values)
