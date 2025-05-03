import functools
import typing
import warnings


def filter_warnings(*args_, **kwargs_) -> typing.Callable:
    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(*args_, **kwargs_)
                return func(*args, **kwargs)

        return wrapper

    return decorator
