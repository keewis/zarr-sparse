import struct
from functools import partial

from zarr_sparse.slices import slice_next


def calculate_offset_size(offset, sizes):
    _offset = offset
    for size in sizes:
        yield offset, size

        offset += size


def encode_ints(ints, format="L"):
    length = len(ints)

    format_str = f"<H{length}{format}"
    return struct.pack(format_str, length, *ints)


def decode_ints(bytes_, format="L"):
    [length] = struct.unpack("<H", bytes_[:2])

    format_ = struct.Struct(f"<{length}{format}")
    nbytes = format_.size
    return list(format_.unpack(bytes_[2 : nbytes + 2]))


def encode_string(s):
    return s.encode("utf-8")


def decode_string(bytes_):
    return bytes_.decode("utf-8")


def encode_strings(strings, size_format="H"):
    encoded = [encode_string(s) for s in strings]
    nbytes_per_entry = [len(b) for b in encoded]

    table = encode_ints(nbytes_per_entry, format=size_format)

    return table + b"".join(encoded)


def decode_strings(bytes_, size_format="H"):
    nbytes_per_entry = decode_ints(bytes_, format=size_format)

    format_str = f"<H{len(nbytes_per_entry)}{size_format}"
    offset = struct.calcsize(format_str)
    encoded = (
        bytes_[slice_next(o, s)]
        for o, s in calculate_offset_size(offset, nbytes_per_entry)
    )

    return [decode_string(b) for b in encoded]


encoders = {
    "sparse-kind": encode_string,
    "order": partial(encode_strings, size_format="B"),
    "nbytes": partial(encode_ints, format="L"),
    "sizes": partial(encode_ints, format="L"),
    "compressed_axes": partial(encode_ints, format="H"),
}
decoders = {
    "sparse-kind": decode_string,
    "order": partial(decode_strings, size_format="B"),
    "nbytes": partial(decode_ints, format="L"),
    "sizes": partial(decode_ints, format="L"),
    "compressed_axes": partial(decode_ints, format="H"),
}


def _encode_offset_table(nbytes):
    # format = "H" takes 2 bytes per entry, plus another two for the size
    initial_offset = (len(nbytes) + 1) * 2
    offsets, _ = zip(*calculate_offset_size(initial_offset, nbytes))

    offset_bytes = encode_ints(offsets, format="H")

    return offset_bytes


def _decode_offset_table(bytes_):
    offsets = decode_ints(bytes_, format="H")
    offsets_ = offsets + [len(bytes_)]
    sizes = [upper - lower for lower, upper in zip(offsets_[:-1], offsets_[1:])]

    return offsets, sizes


def _encode_metadata_table(metadata):
    """encode the metadata table to bytes

    The format is:
    1. offset table (offset + size for each entry)
    2. array of column names
    3. data arrays

    The new format is:
    1. offset table (offset + n_entries for each entry)
    2. list of column names
    3. data values

    Each array is encoded as
    1. 2 bytes for length
    2. array data bytes
    """
    metadata_ = dict(metadata)

    # ignore the dtypes – these can be reconstructed from the array metadata
    metadata_.pop("dtypes", None)

    column_names = encode_strings(list(metadata_), size_format="B")
    values = [encoders.get(name)(column) for name, column in metadata_.items()]

    metadata_table_bytes = [column_names] + values

    offset_bytes = _encode_offset_table([len(part) for part in metadata_table_bytes])

    all_bytes = [offset_bytes] + metadata_table_bytes

    return b"".join(all_bytes)


def _decode_metadata_table(bytes_):
    offsets, sizes = _decode_offset_table(bytes_)

    column_name_bytes, *column_bytes = [
        bytes_[slice_next(offset, size)] for offset, size in zip(offsets, sizes)
    ]
    column_names = decode_strings(column_name_bytes, size_format="B")

    decoded = {
        name: decoders.get(name)(encoded_column)
        for name, encoded_column in zip(column_names, column_bytes, strict=False)
    }
    return decoded
