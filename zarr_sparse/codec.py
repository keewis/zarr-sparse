from __future__ import annotations

import struct
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Self

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.buffer.cpu import numpy_buffer_prototype
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON
from zarr.core.dtype import data_type_registry
from zarr.core.dtype.npy.int import BaseInt, Int64, UInt64
from zarr.core.dtype.npy.string import FixedLengthUTF32, VariableLengthUTF8
from zarr.registry import get_pipeline_class, register_codec

from zarr_sparse.sparse import assemble_array, extract_arrays, sparse_keys

MAX_INT_64 = np.iinfo(np.int64).max


if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.codec import Codec


# TODO: use vlen-str instead of fixed-len
column_dtypes = {
    "sparse-kind": "<U5",
    "order": "<U1",
    "nbytes": "uint64",
    "sizes": "uint64",
    "compressed_axes": "uint64",
    "column_names": "<U20",
}


@dataclass
class ArrayMetadata:
    nbytes: list[int] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    dtypes: list[np.dtype] = field(default_factory=list)
    order: list[str] = field(default_factory=list)

    def __add__(self, other: Self) -> Self:
        return type(self)(
            nbytes=self.nbytes + other.nbytes,
            sizes=self.sizes + other.sizes,
            dtypes=self.dtypes + other.dtypes,
            order=self.order + other.order,
        )

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]):
        translations = {
            "size": "sizes",
            "dtype": "dtypes",
        }
        kwargs = {
            translations.get(k, k): v if isinstance(v, list) else [v]
            for k, v in mapping.items()
        }
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def encode_array(
    array: np.ndarray, chunk_spec: ArraySpec, codecs: tuple[Codec, ...]
) -> tuple[ArrayMetadata, Buffer]:
    pipeline = get_pipeline_class().from_codecs(codecs)

    ndbuffer = numpy_buffer_prototype().nd_buffer.from_numpy_array(array)
    array_bytes = next(iter(await pipeline.encode([(ndbuffer, chunk_spec)])))
    metadata = ArrayMetadata.from_dict(
        {
            "size": array.size,
            "nbytes": len(array_bytes),
            "dtype": array.dtype,
            "order": "C",
        }
    )

    return metadata, array_bytes


def create_table_chunk_spec(*, shape=None, nbytes=None, dtype=None):
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)

    if dtype is None:
        dtype = UInt64(endianness="little")
    elif dtype.kind == "U":
        dtype = FixedLengthUTF32.from_native_dtype(dtype)
    else:
        dtype = data_type_registry.match_dtype(dtype)

    if (shape is None and nbytes is None) or (shape is not None and nbytes is not None):
        raise ValueError("need to pass either shape or nbytes")
    elif shape is None:
        size = nbytes // dtype.item_size
        if size * dtype.item_size != nbytes:
            raise ValueError("mismatching dtype: does not map onto number of bytes")

        shape = (size,)

    if isinstance(dtype, BaseInt):
        fill_value = np.iinfo(dtype.to_native_dtype()).max
    elif isinstance(dtype, (FixedLengthUTF32, VariableLengthUTF8)):
        fill_value = ""

    return ArraySpec(
        shape=shape,
        dtype=dtype,
        fill_value=fill_value,
        config=ArrayConfig(order="C", write_empty_chunks=False),
        prototype=numpy_buffer_prototype(),
    )


async def create_offset_table(buffers, codecs):
    """pack buffer lengths into a small table

    The format is (with n the number of indexed buffers):
    1. 8 bytes for the size of the table
    2. n*8 bytes for the offsets
    3. n*8 bytes for the lengths of each array

    (no compression, so the two arrays are actually the same length)
    """
    prototype = numpy_buffer_prototype()
    pipeline = get_pipeline_class().from_codecs(codecs)

    byte_lengths = np.array(
        [len(buffer_) for buffer_ in buffers],
        dtype="uint64",
    )
    offsets = np.cumulative_sum(byte_lengths, include_initial=True)[:-1]

    chunk_spec = create_table_chunk_spec(shape=offsets.shape, dtype=offsets.dtype)
    data_buffers = await pipeline.encode(
        [
            (prototype.nd_buffer.from_numpy_array(offsets), chunk_spec),
            (prototype.nd_buffer.from_numpy_array(byte_lengths), chunk_spec),
        ]
    )
    data_bytes = sum(data_buffers[1:], start=data_buffers[0])

    n_bytes = len(data_bytes)
    length_bytes = prototype.buffer.from_bytes(struct.pack("<L", n_bytes))

    return length_bytes + data_bytes


def slice_next(offset, size):
    return slice(offset, offset + size)


async def decode_offset_table(
    chunk_data: Buffer, codecs: list[Codec]
) -> np.ndarray[Any, np.dtype[np.unsignedinteger]]:
    pipeline = get_pipeline_class().from_codecs(codecs)

    n_prefix_bytes = 4
    [n_buffer_bytes] = struct.unpack("<L", chunk_data[:n_prefix_bytes].to_bytes())
    bytes_read = n_prefix_bytes

    dtype = np.dtype("uint64")
    n_bytes_per_buffer = n_buffer_bytes // 2
    n_elements = n_bytes_per_buffer // dtype.itemsize

    chunk_spec = create_table_chunk_spec(shape=(n_elements,), dtype=dtype)

    offsets_bytes = chunk_data[slice_next(bytes_read, n_bytes_per_buffer)]
    bytes_read += n_bytes_per_buffer
    sizes_bytes = chunk_data[slice_next(bytes_read, n_bytes_per_buffer)]
    bytes_read += n_bytes_per_buffer

    offset_buffer, sizes_buffer = await pipeline.decode(
        [
            (offsets_bytes, chunk_spec),
            (sizes_bytes, chunk_spec),
        ]
    )

    return bytes_read, offset_buffer.as_numpy_array(), sizes_buffer.as_numpy_array()


async def encode_metadata_table(metadata, codecs):
    """encode the metadata table to bytes

    The format is:
    1. offset table (offset + size for each entry)
    2. array of column names
    3. data arrays

    Each array is encoded as
    1. 2 bytes for length
    2. array data bytes
    """
    prototype = numpy_buffer_prototype()
    pipeline = get_pipeline_class().from_codecs(codecs)

    # dtypes can be reconstructed from the array spec
    metadata.pop("dtypes", None)

    column_names = np.asarray(
        list(metadata.keys()), dtype=column_dtypes["column_names"]
    )
    values = [np.asarray(v, dtype=column_dtypes[k]) for k, v in metadata.items()]
    arrays = [column_names] + values

    to_encode = [
        (
            prototype.nd_buffer.from_numpy_array(array.copy()),
            create_table_chunk_spec(shape=array.shape, dtype=array.dtype),
        )
        for array in arrays
    ]
    column_bytes = await pipeline.encode(to_encode)

    column_size_bytes = [
        prototype.buffer.from_bytes(struct.pack("<H", array.size)) for array in arrays
    ]

    table_bytes = [size + data for size, data in zip(column_size_bytes, column_bytes)]

    offset_bytes = await create_offset_table(table_bytes, pipeline)
    full_table = offset_bytes + sum(table_bytes[1:], start=table_bytes[0])
    return full_table


async def decode_table(chunk_data: Buffer, codecs: list[Codec]) -> list[dict[str, Any]]:
    pipeline = get_pipeline_class().from_codecs(codecs)

    bytes_read, offsets, sizes = await decode_offset_table(chunk_data, codecs)

    header_bytes, *column_bytes = [
        chunk_data[slice_next(offset, size)]
        for offset, size in zip(offsets + bytes_read, sizes)
    ]
    header_chunk_spec = create_table_chunk_spec(
        shape=tuple(struct.unpack("<H", header_bytes[:2].to_bytes())),
        dtype=column_dtypes["column_names"],
    )
    column_names = next(
        iter(
            await pipeline.decode(
                [
                    (header_bytes[2:], header_chunk_spec),
                ]
            )
        )
    ).as_numpy_array()

    chunk_specs = [
        create_table_chunk_spec(
            shape=tuple(struct.unpack("<H", buffer_[:2].to_bytes())),
            dtype=column_dtypes[name],
        )
        for name, buffer_ in zip(column_names, column_bytes)
    ]

    to_decode = [
        (buffer_[2:], chunk_spec)
        for buffer_, chunk_spec in zip(column_bytes, chunk_specs)
    ]
    columns = [col.as_numpy_array() for col in await pipeline.decode(to_decode)]

    return {
        str(name): tuple(col.tolist()) if name != "sparse-kind" else col.item()
        for name, col in zip(column_names, columns)
    }


class SparseArrayCodec(ArrayBytesCodec):
    def __init__(self):
        self.array_codecs = (BytesCodec(), ZstdCodec())
        self.table_codecs = (BytesCodec(),)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "sparse"}

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        bytes_read, offsets, sizes = await decode_offset_table(
            chunk_data, self.table_codecs
        )

        table_bytes, array_bytes = [
            chunk_data[slice_next(offset, size)]
            for offset, size in zip(offsets + bytes_read, sizes)
        ]

        metadata = await decode_table(table_bytes, self.table_codecs)

        array_offsets = np.cumulative_sum(
            np.asarray(metadata["nbytes"], dtype="uint64"), include_initial=True
        )[:-1]

        sparse_metadata = {n: v for n, v in metadata.items() if n in sparse_keys} | {
            "fill_value": chunk_spec.fill_value,
            "shape": chunk_spec.shape,
        }
        array_metadata_columns = {
            n: v for n, v in metadata.items() if n not in sparse_keys
        } | {"offsets": tuple(array_offsets)}
        array_metadata = [
            dict(zip(array_metadata_columns, v))
            for v in zip(*array_metadata_columns.values())
        ]

        to_decode = [
            (
                array_bytes[slice_next(m["offsets"], m["nbytes"])],
                ArraySpec(
                    shape=m["sizes"],
                    dtype=(
                        chunk_spec.dtype if index == 0 else Int64(endianness="little")
                    ),
                    fill_value=chunk_spec.fill_value if index == 0 else MAX_INT_64,
                    config=ArrayConfig(
                        order=m["order"],
                        write_empty_chunks=chunk_spec.config.write_empty_chunks,
                    ),
                    prototype=numpy_buffer_prototype(),
                ),
            )
            for index, m in enumerate(array_metadata)
        ]

        pipeline = get_pipeline_class().from_codecs(self.array_codecs)
        decoded = await pipeline.decode(to_decode)
        arrays = [buf.as_numpy_array() for buf in decoded]

        constructed = assemble_array(sparse_metadata, arrays, library="pydata-sparse")
        return constructed

    # async def decode(
    #     self,
    #     chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    # ) -> Iterable[NDBuffer | None]:
    #     print(list(chunks_and_specs))
    #     raise NotImplementedError

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        data = chunk_array._data.get_chunk()
        if data.nnz == 0 and not chunk_spec.config.write_empty_chunks:
            return None

        sparse_metadata, arrays = extract_arrays(data)
        if sparse_metadata.pop("fill_value") != chunk_spec.fill_value:
            raise ValueError(
                "sparse array fill_value doesn't match the chunk fill value"
            )
        if sparse_metadata.pop("shape") != chunk_spec.shape:
            raise ValueError("sparse array shape doesn't match the chunk shape")

        prototype = numpy_buffer_prototype()

        encoded = []
        array_metadata = []
        for array in arrays:
            spec = ArraySpec(
                shape=array.shape,
                dtype=array.dtype,
                fill_value=0,
                config=chunk_spec.config,
                prototype=prototype,
            )
            metadata, encoded_array = await encode_array(array, spec, self.array_codecs)
            encoded.append(encoded_array)
            array_metadata.append(metadata)

        metadata = (
            sum(array_metadata, start=ArrayMetadata()).to_dict() | sparse_metadata
        )

        array_buffer = sum(encoded[1:], start=encoded[0])

        table_buffer = await encode_metadata_table(metadata, self.table_codecs)

        offset_buffer = await create_offset_table(
            [table_buffer, array_buffer], self.table_codecs
        )

        return offset_buffer + table_buffer + array_buffer

    # async def encode(
    #     self,
    #     chunks_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    # ) -> Iterable[Buffer | None]:
    #     raise NotImplementedError


register_codec("sparse", SparseArrayCodec)
