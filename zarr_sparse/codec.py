from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.buffer.cpu import numpy_buffer_prototype
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.codecs.sharding import MAX_UINT_64
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, NDBuffer, default_buffer_prototype
from zarr.core.common import JSON
from zarr.core.dtype.npy.int import UInt64
from zarr.registry import get_pipeline_class, register_codec

from zarr_sparse.sparse import extract_arrays

if TYPE_CHECKING:
    from zarr.core.array_spec import ArraySpec


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


async def encode_table(metadata, codecs):
    def create_chunk_spec(array):
        return ArraySpec(
            shape=sizes.shape,
            dtype=UInt64(endianness="little"),
            fill_value=MAX_UINT_64,
            config=ArrayConfig(order="C", write_empty_chunks=False),
            prototype=default_buffer_prototype(),
        )

    prototype = numpy_buffer_prototype()
    pipeline = get_pipeline_class().from_codecs(codecs)

    sizes = np.array([m["size"] for m in metadata], dtype="uint64")
    table_bytes = next(
        iter(
            await pipeline.encode(
                [
                    (
                        prototype.nd_buffer.from_numpy_array(sizes),
                        create_chunk_spec(sizes),
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
                        create_chunk_spec(table_length),
                    )
                ]
            )
        )
    )

    full_table = length_bytes + table_bytes
    return full_table


class SparseArrayCodec(ArrayBytesCodec):
    def __init__(self):
        self.array_codecs = (BytesCodec(), ZstdCodec())
        self.table_codecs = (BytesCodec(),)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "sparse"}

    async def _decode_single(
        self, chunk_data: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
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
