import numpy as np

from zarr_sparse.sparse import extract_arrays


def compare_fill_value(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.isnan(a) and np.isnan(b)

    return a == b


def sparse_equal(a, b):
    if type(a) is not type(b):
        return False

    if a.shape != b.shape or a.dtype != b.dtype or a.nnz != b.nnz:
        return False

    metadata_a, (data_a, *coords_a) = extract_arrays(a)
    metadata_b, (data_b, *coords_b) = extract_arrays(b)

    if not compare_fill_value(metadata_a["fill_value"], metadata_b["fill_value"]):
        return False

    if not np.array_equal(a.coords, b.coords):
        return False

    return np.array_equal(a.data, b.data)


def format_sparse_diff(a, b):
    lines = [
        "Sparse arrays differ:",
    ]

    equal_types = type(a) is type(b)
    equal_shapes = a.shape == b.shape
    equal_dtypes = a.dtype == b.dtype
    equal_nnz = a.nnz == b.nnz

    metadata_a, (data_a, *coords_a) = extract_arrays(a)
    metadata_b, (data_b, *coords_b) = extract_arrays(b)

    equal_fill_values = compare_fill_value(
        metadata_a["fill_value"], metadata_b["fill_value"]
    )

    if not (
        equal_types
        and equal_shapes
        and equal_dtypes
        and equal_nnz
        and equal_fill_values
    ):
        lines.extend(
            [
                "",
                f"L  {repr(a)}",
                f"R  {repr(a)}",
            ]
        )

    lines.extend(["", "Comparing data:"])

    if any(not np.array_equal(c_a, c_b) for c_a, c_b in zip(coords_a, coords_b)):
        lines.append("- Coords are different")

    if not np.array_equal(data_a, data_b):
        lines.append("- Data is different")

    return "\n".join(lines)


def assert_sparse_equal(a, b):
    __tracebackhide__ = True

    assert sparse_equal(a, b), format_sparse_diff(a, b)
