import pytest

try:
    import sparse  # noqa: F401

    has_pydata_sparse = True
except ImportError:
    has_pydata_sparse = False

requires_pydata_sparse = pytest.mark.skipif(
    not has_pydata_sparse, reason="pydata sparse is not available"
)

try:
    import scipy  # noqa: F401

    has_scipy = True
except ImportError:
    has_scipy = False

requires_scipy = pytest.mark.skipif(not has_scipy, reason="scipy is not available")

try:
    import torch.sparse  # noqa: F401

    has_pytorch = True
except ImportError:
    has_pytorch = False

requires_pytorch = pytest.mark.skipif(
    not has_pytorch, reason="pytorch is not available"
)
