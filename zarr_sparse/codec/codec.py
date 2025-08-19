from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.buffer.cpu import numpy_buffer_prototype
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.core.dtype.npy.int import Int64
from zarr.registry import get_pipeline_class, register_codec

from zarr_sparse.codec import metadata
from zarr_sparse.comparison import compare_fill_value
from zarr_sparse.slices import slice_next
from zarr_sparse.sparse import assemble_array, extract_arrays, sparse_keys

MAX_INT_64 = np.iinfo(np.int64).max


if TYPE_CHECKING:
    from typing import Any

    from zarr.abc.codec import Codec


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


async def encode_offset_table(nbytes) -> Buffer:
    prototype = numpy_buffer_prototype()

    raw_bytes = metadata._encode_offset_table(nbytes)

    return prototype.buffer.from_bytes(raw_bytes)


async def decode_offset_table(chunk_data: Buffer) -> list[int]:
    return metadata._decode_offset_table(chunk_data.to_bytes())


async def encode_metadata_table(mapping: dict[str, Any]) -> Buffer:
    prototype = numpy_buffer_prototype()

    raw_bytes = metadata._encode_metadata_table(mapping)

    return prototype.buffer.from_bytes(raw_bytes)


async def decode_metadata_table(table_data: Buffer) -> dict[str, Any]:
    return metadata._decode_metadata_table(table_data.to_bytes())


class SparseArrayCodec(ArrayBytesCodec):
    def __init__(self):
        self.array_codecs = (BytesCodec(), ZstdCodec())
        self.table_codecs = (BytesCodec(),)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "sparse", require_configuration=False
        )
        configuration_parsed = configuration_parsed or {}

        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        return {"name": "sparse"}

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        offsets, sizes = await decode_offset_table(chunk_data)

        table_bytes, array_bytes = [
            chunk_data[slice_next(offset, size)] for offset, size in zip(offsets, sizes)
        ]

        metadata = await decode_metadata_table(table_bytes)

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

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        data = chunk_array._data.get_chunk()
        if data.nnz == 0 and not chunk_spec.config.write_empty_chunks:
            return None

        sparse_metadata, arrays = extract_arrays(data)
        if not compare_fill_value(
            sparse_metadata.pop("fill_value"), chunk_spec.fill_value
        ):
            raise ValueError(
                "sparse array fill_value doesn't match the chunk fill value"
            )
        sparse_metadata.pop("shape")

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

        table_buffer = await encode_metadata_table(metadata)

        offset_buffer = await encode_offset_table(
            [len(table_buffer), len(array_buffer)]
        )

        return offset_buffer + table_buffer + array_buffer


register_codec("sparse", SparseArrayCodec)
