import functools


def as_decorator(func):
    @functools.wraps(func)
    def wrapper(obj):
        return func(obj) or obj

    return wrapper
