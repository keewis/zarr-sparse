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
from zarr.core.dtype.npy.int import UInt64
from zarr.registry import get_pipeline_class, register_codec

from zarr_sparse.sparse import extract_arrays

if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.codec import Codec


async def encode_array(array, chunk_spec, codecs):
    pipeline = get_pipeline_class().from_codecs(codecs)

    ndbuffer = numpy_buffer_prototype().nd_buffer.from_numpy_array(array)
    array_bytes = next(iter(await pipeline.encode([(ndbuffer, chunk_spec)])))
    metadata = {
        "size": array.size,
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

    sizes = np.array([m["size"] for m in metadata], dtype="uint64")
    table_bytes = next(
        iter(
            await pipeline.encode(
                [
                    (
                        prototype.nd_buffer.from_numpy_array(sizes),
                        create_table_chunk_spec(shape=sizes.shape),
                    )
                ]
            )
        )
    )

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

    table_size = (
        next(
            iter(
                await pipeline.decode(
                    [(chunk_data[:8], create_table_chunk_spec(shape=(1,)))]
                )
            )
        )
        .as_numpy_array()
        .item()
    )

    sizes = next(
        iter(
            await pipeline.decode(
                [
                    (
                        chunk_data[8 : 8 + table_size],
                        create_table_chunk_spec(nbytes=table_size),
                    )
                ]
            )
        )
    ).as_numpy_array()

    metadata = [
        {
            "size": int(size),
            "dtype": (
                chunk_spec.dtype.to_native_dtype() if index == 0 else np.dtype("<i8")
            ),
            "order": "C",
        }
        for index, size in enumerate(sizes)
    ]
    return metadata


class SparseArrayCodec(ArrayBytesCodec):
    def __init__(self):
        self.array_codecs = (BytesCodec(), ZstdCodec())
        self.table_codecs = (BytesCodec(),)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "sparse"}

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        table = await decode_table(chunk_data, chunk_spec, self.table_codecs)
        print(table)
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
