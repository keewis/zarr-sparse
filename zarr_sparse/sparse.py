from functools import singledispatch

import numpy as np


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


converters = {}


def register_converter(library_name, sparse_kind):
    def wrapper(func):
        library_converters = converters.setdefault(library_name, {})
        library_converters[sparse_kind] = func

        return func

    return wrapper


def assemble_array(metadata, arrays, library):
    """create a sparse array from metadata, arrays, and a given library

    Parameters
    ----------
    metadata : dict
        A description of the sparse array, including the kind.
    arrays : tuple of array-like
        The arrays the sparse array consists of.
    library : str
        The sparse array library to use.

    Returns
    -------
    x
        The sparse array.
    """
    converter_library = converters.get(library)
    if converter_library is None:
        raise ValueError(
            f"unknown array library: {library}."
            f"Choose one of {', '.join(repr(x) for x in converters)}"
        )

    sparse_format = metadata.pop("sparse-kind")
    converter = converter_library.get(sparse_format)
    if converter is None:
        raise ValueError(f"array library {library} does not implement {sparse_format}.")

    return converter(metadata, arrays)


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

    @register_converter("pydata-sparse", "coo")
    def _(metadata, arrays):
        data, *coords = arrays
        return sparse.COO(
            coords=coords,
            data=data,
            fill_value=metadata["fill_value"],
            shape=metadata["shape"],
        )

    @register_converter("pydata-sparse", "gcxs")
    def _(metadata, arrays):
        return sparse.GCXS(
            tuple(arrays),
            fill_value=metadata["fill_value"],
            shape=metadata["shape"],
            compressed_axes=metadata["compressed_axes"],
        )

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

    @register_converter("scipy-sparse", "coo")
    def _(metadata, arrays):
        if metadata["fill_value"] != 0:
            raise ValueError(f"fill_value not supported: {metadata['fill_value']}")

        data, *coords = arrays

        return scipy.sparse.coo_array(
            (data, np.stack(coords, axis=0)), shape=metadata["shape"]
        )

except ImportError:
    pass
