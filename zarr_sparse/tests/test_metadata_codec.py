import pytest

from zarr_sparse.codec import metadata


@pytest.mark.parametrize(
    ["ints", "format", "expected"],
    (
        ([160, 1081], "H", b"\x02\x00\xa0\x00\x39\x04"),
        (
            [233, 233, 233],
            "L",
            b"\x03\x00\xe9\x00\x00\x00\xe9\x00\x00\x00\xe9\x00\x00\x00",
        ),
    ),
)
def test_encode_ints(ints, format, expected):
    actual = metadata.encode_ints(ints, format=format)

    assert actual == expected


@pytest.mark.parametrize(
    ["bytes_", "format", "expected"],
    (
        (b"\x02\x00\xa0\x00\x39\x04", "H", [160, 1081]),
        (
            b"\x03\x00\xe9\x00\x00\x00\xe9\x00\x00\x00\xe9\x00\x00\x00",
            "L",
            [233, 233, 233],
        ),
    ),
)
def test_decode_ints(bytes_, format, expected):
    actual = metadata.decode_ints(bytes_, format=format)

    assert actual == expected


@pytest.mark.parametrize(
    ["ints", "format"],
    (
        (list(range(20)), "H"),
        (list(range(300000, 300020)), "L"),
    ),
)
def test_ints_roundtrip(ints, format):
    encoded = metadata.encode_ints(ints, format=format)

    decoded = metadata.decode_ints(encoded, format=format)

    assert decoded == ints


@pytest.mark.parametrize(
    ["strings", "format", "expected"],
    (
        (["a", "bc", "def"], "H", b"\x03\x00\x01\x00\x02\x00\x03\x00abcdef"),
        (
            ["abc", "d", "ef", "ghijkl"],
            "H",
            b"\x04\x00\x03\x00\x01\x00\x02\x00\x06\x00abcdefghijkl",
        ),
        (["C", "C"], "B", b"\x02\x00\x01\x01CC"),
    ),
)
def test_encode_strings(strings, format, expected) -> None:
    actual = metadata.encode_strings(strings, size_format=format)

    assert actual == expected


@pytest.mark.parametrize(
    ["bytes_", "format", "expected"],
    (
        (b"\x03\x00\x01\x00\x02\x00\x03\x00abcdef", "H", ["a", "bc", "def"]),
        (
            b"\x04\x00\x03\x00\x01\x00\x02\x00\x06\x00abcdefghijkl",
            "H",
            ["abc", "d", "ef", "ghijkl"],
        ),
        (b"\x02\x00\x01\x01CC", "B", ["C", "C"]),
    ),
)
def test_decode_strings(bytes_, format, expected):
    actual = metadata.decode_strings(bytes_, size_format=format)

    assert actual == expected


@pytest.mark.parametrize(
    "sparse_metadata",
    (
        {
            "sparse-kind": "coo",
            "nbytes": [10274, 1081, 742],
            "sizes": [233, 233, 233],
            "order": ["C", "C", "C"],
        },
        {
            "sparse-kind": "gcxs",
            "nbytes": [10274, 1081, 742],
            "sizes": [233, 233, 35],
            "order": ["C", "C", "C"],
            "compressed_axes": [0],
        },
    ),
)
def test_metadata_roundtrip(sparse_metadata):
    encoded = metadata._encode_metadata_table(sparse_metadata)
    decoded = metadata._decode_metadata_table(encoded)

    assert isinstance(encoded, bytes)
    assert sparse_metadata == decoded
