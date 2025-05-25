import numpy as np

from zarr_sparse.sparse import extract_arrays
from zarr_sparse.tests import requires_pydata_sparse, requires_scipy


@requires_pydata_sparse
@requires_scipy
def test_roundtrip():
    import sparse

    x = sparse.COO(
        np.array([[0, 4, 6], [35, 53, 61]]), np.arange(3), shape=(10, 100), fill_value=0
    )

    y = x.to_scipy_sparse()

    metadata_x, (data_x, rows_x, cols_x) = extract_arrays(x)
    metadata_y, (data_y, rows_y, cols_y) = extract_arrays(y)

    assert metadata_x == metadata_y
    np.testing.assert_equal(data_x, data_y)
    np.testing.assert_equal(rows_x, rows_y)
    np.testing.assert_equal(cols_x, cols_y)
