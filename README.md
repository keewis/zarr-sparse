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
