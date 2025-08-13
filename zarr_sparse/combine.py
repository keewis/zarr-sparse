import itertools

import numpy as np


def tiles_by_id(parts):
    return {
        np.unravel_index(index, parts.shape): array
        for index, array in enumerate(parts.flatten())
    }


def until_nth(index):
    def indexer(val):
        return val[:index]

    return indexer


def as_item_key(key):
    def wrapper(it):
        return key(it[0])

    return wrapper


def groupby_mapping(mapping, key):
    wrapped_key = as_item_key(key)
    raw_groups = itertools.groupby(
        sorted(mapping.items(), key=wrapped_key), key=wrapped_key
    )
    return ((key, (el for _, el in group)) for key, group in raw_groups)


def combine_nd(parts):
    tiles = tiles_by_id(parts)
    xp = parts.flat[0].__array_namespace__()

    # innermost to outermost
    for axis in range(parts.ndim - 1, -1, -1):
        tiles = {
            key: xp.concat(list(arrays), axis=axis)
            for key, arrays in groupby_mapping(tiles, key=until_nth(axis))
        }

    return next(iter(tiles.values()))


def expand_chunks(chunks, shape):
    def _expand(chunkspec, size):
        if chunkspec == -1:
            return (size,)
        elif isinstance(chunkspec, int):
            n_full_chunks, remainder = divmod(size, chunkspec)
            chunks = (chunkspec,) * n_full_chunks

            if remainder > 0:
                chunks += (remainder,)

            return chunks

        chunkspec = tuple(chunkspec)
        if sum(chunkspec) != size:
            raise ValueError(f"chunks don't add up to the full size: {chunkspec}")

        return chunkspec

    return tuple(_expand(chunkspec, size) for chunkspec, size in zip(chunks, shape))
