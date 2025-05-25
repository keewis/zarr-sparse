from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON
from zarr.registry import register_codec

if TYPE_CHECKING:
    from collections.abc import Iterable

    from zarr.core.array_spec import ArraySpec


class SparseArrayCodec(ArrayBytesCodec):
    def to_dict(self) -> dict[str, JSON]:
        return {"name": "sparse"}

    async def _decode_single(
        self, chunk_data: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        raise NotImplementedError

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Buffer | None, ArraySpec]],
    ) -> Iterable[NDBuffer | None]:
        raise NotImplementedError

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        raise NotImplementedError

    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[NDBuffer | None, ArraySpec]],
    ) -> Iterable[Buffer | None]:
        raise NotImplementedError


register_codec("sparse", SparseArrayCodec)
