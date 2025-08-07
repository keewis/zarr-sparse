from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.buffer.cpu import numpy_buffer_prototype
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.codecs.sharding import MAX_UINT_64
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON
from zarr.core.dtype.npy.int import Int64, UInt64
from zarr.registry import get_pipeline_class, register_codec

from zarr_sparse.sparse import extract_arrays

MAX_INT_64 = np.iinfo(np.int64).max


if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.codec import Codec


async def encode_array(array, chunk_spec, codecs):
    pipeline = get_pipeline_class().from_codecs(codecs)

    ndbuffer = numpy_buffer_prototype().nd_buffer.from_numpy_array(array)
    array_bytes = next(iter(await pipeline.encode([(ndbuffer, chunk_spec)])))
    metadata = {
        "size": array.size,
        "nbytes": len(array_bytes),
        "dtype": array.dtype,
        "order": "C",
    }

    return metadata, array_bytes


def create_table_chunk_spec(*, shape=None, nbytes=None):
    dtype = UInt64(endianness="little")

    if (shape is None and nbytes is None) or (shape is not None and nbytes is not None):
        raise ValueError("need to pass either shape or nbytes")
    elif shape is None:
        size = nbytes // dtype.item_size
        if size * dtype.item_size != nbytes:
            raise ValueError("mismatching dtype: does not map onto number of bytes")

        shape = (size,)

    return ArraySpec(
        shape=shape,
        dtype=dtype,
        fill_value=MAX_UINT_64,
        config=ArrayConfig(order="C", write_empty_chunks=False),
        prototype=numpy_buffer_prototype(),
    )


async def encode_table(metadata, codecs):
    prototype = numpy_buffer_prototype()
    pipeline = get_pipeline_class().from_codecs(codecs)

    byte_lengths = np.array([m["nbytes"] for m in metadata], dtype="uint64")
    sizes = np.array([m["size"] for m in metadata], dtype="uint64")
    size_bytes, byte_length_bytes = await pipeline.encode(
        [
            (
                prototype.nd_buffer.from_numpy_array(sizes),
                create_table_chunk_spec(shape=sizes.shape),
            ),
            (
                prototype.nd_buffer.from_numpy_array(byte_lengths),
                create_table_chunk_spec(shape=byte_lengths.shape),
            ),
        ]
    )
    table_bytes = size_bytes + byte_length_bytes

    table_length = np.array([table_bytes._data.size], dtype="uint64")
    length_bytes = next(
        iter(
            await pipeline.encode(
                [
                    (
                        prototype.nd_buffer.from_numpy_array(table_length),
                        create_table_chunk_spec(shape=table_length.shape),
                    )
                ]
            )
        )
    )

    full_table = length_bytes + table_bytes
    return full_table


async def decode_table(
    chunk_data: Buffer, chunk_spec: ArraySpec, codecs: list[Codec]
) -> list[dict[str, Any]]:
    pipeline = get_pipeline_class().from_codecs(codecs)
    nbytes_size = 8

    nbytes_table = (
        next(
            iter(
                await pipeline.decode(
                    [(chunk_data[:nbytes_size], create_table_chunk_spec(shape=(1,)))]
                )
            )
        )
        .as_numpy_array()
        .item()
    )
    nbytes_column = nbytes_table // 2  # two columns

    sizes_offset = nbytes_size
    byte_length_offset = nbytes_size + nbytes_column

    sizes, byte_lengths = map(
        lambda x: x.as_numpy_array(),
        await pipeline.decode(
            [
                (
                    chunk_data[sizes_offset : sizes_offset + nbytes_column],
                    create_table_chunk_spec(nbytes=nbytes_column),
                ),
                (
                    chunk_data[byte_length_offset : byte_length_offset + nbytes_column],
                    create_table_chunk_spec(nbytes=nbytes_column),
                ),
            ]
        ),
    )

    metadata = [
        {
            "size": int(size),
            "nbytes": int(nbytes),
            "dtype": chunk_spec.dtype if index == 0 else Int64(endianness="little"),
            "order": "C",
        }
        for index, (size, nbytes) in enumerate(zip(sizes, byte_lengths))
    ]
    return metadata, chunk_data[nbytes_size + nbytes_table :]


class SparseArrayCodec(ArrayBytesCodec):
    def __init__(self):
        self.array_codecs = (BytesCodec(), ZstdCodec())
        self.table_codecs = (BytesCodec(),)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "sparse"}

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        table, chunk_bytes = await decode_table(
            chunk_data, chunk_spec, self.table_codecs
        )

        offset = 0
        to_decode = []
        for index, metadata in enumerate(table):
            dtype = metadata["dtype"]
            fill_value = (chunk_spec.fill_value if index == 0 else MAX_INT_64,)
            chunk_spec = ArraySpec(
                shape=(metadata["size"],),
                dtype=dtype,
                fill_value=fill_value,
                config=ArrayConfig(order=metadata["order"], write_empty_chunks=False),
                prototype=numpy_buffer_prototype(),
            )
            nbytes = metadata["nbytes"]
            to_decode.append((chunk_bytes[offset : offset + nbytes], chunk_spec))

            offset += nbytes

        pipeline = get_pipeline_class().from_codecs(self.array_codecs)
        decoded = await pipeline.decode(to_decode)

        arrays = [buffer.as_numpy_array() for buffer in decoded]
        print(arrays)

        raise NotImplementedError(f"chunk data: {chunk_data}, chunk spec: {chunk_spec}")

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

        metadata, arrays = extract_arrays(data)

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

        merged_buffer = sum(encoded[1:], start=encoded[0])

        table_buffer = await encode_table(array_metadata, self.table_codecs)

        return table_buffer + merged_buffer

    # async def encode(
    #     self,
    #     chunks_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    # ) -> Iterable[Buffer | None]:
    #     raise NotImplementedError


register_codec("sparse", SparseArrayCodec)
