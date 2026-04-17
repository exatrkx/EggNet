import time
from contextlib import contextmanager
from functools import wraps


PROFILE_ATTR = "profiling"
PROFILE_METADATA_ATTR = "profile_metadata"


def get_profile_store(batch, attr_name=PROFILE_ATTR):
    store = getattr(batch, attr_name, None)
    if store is None or not isinstance(store, dict):
        store = {}
        setattr(batch, attr_name, store)
    return store


def add_profile_time(batch, name, duration, attr_name=PROFILE_ATTR):
    if batch is None:
        return None

    store = get_profile_store(batch, attr_name=attr_name)
    store[name] = store.get(name, 0.0) + duration
    return store


def get_profile_metadata_store(batch, attr_name=PROFILE_METADATA_ATTR):
    store = getattr(batch, attr_name, None)
    if store is None or not isinstance(store, dict):
        store = {}
        setattr(batch, attr_name, store)
    return store


def set_profile_metadata(batch, name, value, attr_name=PROFILE_METADATA_ATTR):
    if batch is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass

    store = get_profile_metadata_store(batch, attr_name=attr_name)
    store[name] = value
    return store


@contextmanager
def profile_section(batch, name, enabled=True, attr_name=PROFILE_ATTR):
    if not enabled or batch is None:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        add_profile_time(
            batch,
            name,
            time.perf_counter() - start,
            attr_name=attr_name,
        )


def profile_function(name=None, enabled=lambda *args, **kwargs: True, attr_name=PROFILE_ATTR):
    def decorator(f):
        @wraps(f)
        def out_f(*args, **kwargs):
            batch = kwargs.get("profile_target")
            if batch is None and args:
                batch = args[0]

            use_profile = enabled(*args, **kwargs) if callable(enabled) else enabled
            section_name = name or f.__qualname__

            with profile_section(
                batch,
                section_name,
                enabled=use_profile,
                attr_name=attr_name,
            ):
                return f(*args, **kwargs)

        return out_f

    return decorator


def time_function(f):
    """
    Decoration function to evaluate timing of a function.
    The function to decorate is required to take either 'batch', 'event' or 'graph' as a positional argument.
    """

    if "batch" in f.__code__.co_varnames:
        index = f.__code__.co_varnames.index("batch")
    elif "event" in f.__code__.co_varnames:
        index = f.__code__.co_varnames.index("event")
    elif "graph" in f.__code__.co_varnames:
        index = f.__code__.co_varnames.index("graph")
    else:
        raise AttributeError("Function requires a position argument as either 'batch', 'event' or 'graph'!!")

    def out_f(*args, time_yes=False, **kwargs):

        batch = args[index]
        if "time_yes" in f.__code__.co_varnames:
            kwargs["time_yes"] = time_yes
        if time_yes:
            start = time.time()
        try:
            res = f(*args, **kwargs)
        except TypeError as e:
            print(f"{f.__code__.co_varnames=}")
            print(f"{args=}")
            print(f"{kwargs=}")
            raise e
        if time_yes:
            end = time.time()
            if f.__qualname__ in batch.keys():
                batch[f.__qualname__] += end - start
            else:
                batch[f.__qualname__] = end - start
        return res

    return out_f
