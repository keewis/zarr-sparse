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
