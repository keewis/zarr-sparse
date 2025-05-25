from functools import singledispatch


@singledispatch
def extract_arrays(x):
    """convert a sparse array into metadata and arrays

    Parameters
    ----------
    x
        The sparse array.

    Returns
    -------
    metadata : dict
        A description of the sparse array, including the kind.
    arrays : tuple of array-like
        The arrays the sparse array consists of.
    """
    raise NotImplementedError(f"unknown array type: {type(x)}")


try:
    import sparse

    @extract_arrays.register(sparse.COO)
    def _(x):
        metadata = {
            "sparse-kind": x.format,
            "fill_value": x.fill_value,
            "shape": x.shape,
        }
        arrays = (x.data, *x.coords)

        return metadata, arrays

    @extract_arrays.register(sparse.GCXS)
    def _(x):
        metadata = {
            "sparse-kind": x.format,
            "fill_value": x.fill_value,
            "compressed_axes": x.compressed_axes,
            "shape": x.shape,
        }
        arrays = (x.data, x.indices, x.indptr)
        return metadata, arrays

except ImportError:
    pass


try:
    import scipy.sparse

    @extract_arrays.register(scipy.sparse.coo_array)
    def _(x):
        metadata = {"sparse-kind": x.format, "fill_value": 0, "shape": x.shape}
        arrays = x.data, *x.coords

        return metadata, arrays

    @extract_arrays.register(scipy.sparse.coo_matrix)
    def _(x):
        metadata = {"sparse-kind": x.format, "fill_value": 0, "shape": x.shape}
        arrays = x.data, *x.coords

        return metadata, arrays

except ImportError:
    pass
