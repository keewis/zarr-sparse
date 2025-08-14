import itertools

import numpy as np
import sparse

from zarr_sparse.chunks import expand_chunks


def create_pydata_coo_array(
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

    coords = np.stack([rng.integers(dim_size, size=nnz) for dim_size in shape], axis=0)

    return sparse.COO(data=data, coords=coords, shape=shape, fill_value=fill_value)


def create_chunk_slices(shape, chunks):
    if not isinstance(chunks[0], tuple):
        expanded_chunks = expand_chunks(chunks, shape)
    else:
        expanded_chunks = chunks

    offsets = tuple(
        np.cumulative_sum(np.asarray(c, dtype="uint64"), include_initial=True)
        for c in expanded_chunks
    )

    chunk_bounds = tuple(
        tuple(slice(int(lower), int(upper)) for lower, upper in zip(o[:-1], o[1:]))
        for o in offsets
    )

    if len(shape) == 1:
        return [(slice_,) for slice_ in chunk_bounds[0]]
    else:
        return list(itertools.product(*chunk_bounds))
