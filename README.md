# zarr-sparse

Serialization of sparse arrays to `zarr`, based on a codec.

Unlike [binsparse-python](https://github.com/ivirshup/binsparse-python), the different 1D arrays (data, coordinate arrays, compressed coordinate arrays) are stored in a shard-like structure, per chunk.

This does make reading specific parts (e.g. the coordinates) in a single request a bit harder, but having a single logical array map to a on-disk zarr array does have its advantages.

## Installation

`zarr-sparse` currently requires a special version of zarr. To install it, use:

```sh
pip install \
    "zarr @ git+https://github.com/keewis/zarr-python.git@sparse-array-patch" \
    "zarr-sparse @ git+https://github.com/keewis/zarr-sparse.git@main"
```

## Usage

```python
from zarr_sparse import SparseArrayCodec
import numpy as np
import sparse


def generate_random_coo(nnz, shape, dtype, fill_value):
    rng = np.random.default_rng(seed=0)
    data = rng.random(size=nnz).astype(dtype)
    coords = np.stack([rng.integers(dim_size, size=nnz) for dim_size in shape], axis=0)

    return sparse.COO(data=data, coords=coords, shape=shape, fill_value=fill_value)


x = generate_random_coo(
    nnz=4500, shape=(4500, 6500), dtype="float64", fill_value=np.nan
)
chunks = (500, 500)

with zarr.storage.MemoryStore() as store:
    root = zarr.api.synchronous.create_group(store=store, zarr_format=3)

    z = root.create_array(
        "a",
        data=x,
        write_data=True,
        chunks=chunks,
        serializer=SparseArrayCodec(),
        filters=None,
        compressors=None,
        dimension_names=["x", "y"],
    )

    print(z[:])
```
