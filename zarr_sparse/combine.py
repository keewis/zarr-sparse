import itertools


def first_key(mapping):
    return next(iter(mapping.keys()))


def first_value(mapping):
    return next(iter(mapping.values()))


def until_nth(index):
    def indexer(val):
        return val[:index]

    return indexer


def as_item_key(key):
    def wrapper(it):
        return key(it[0])

    return wrapper


def by_key(it):
    return it[0]


def groupby_mapping(mapping, key):
    wrapped_key = as_item_key(key)
    raw_groups = itertools.groupby(sorted(mapping.items(), key=by_key), key=wrapped_key)
    return ((key, (el for _, el in group)) for key, group in raw_groups)


def combine_nd(tiles):
    xp = first_value(tiles).__array_namespace__()

    ndim = len(first_key(tiles))

    # innermost to outermost
    for axis in range(ndim - 1, -1, -1):
        tiles = {
            key: xp.concat(list(arrays), axis=axis)
            for key, arrays in groupby_mapping(tiles, key=until_nth(axis))
        }

    return first_value(tiles)
