import time


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

        res = f(*args, **kwargs)
        if time_yes:
            end = time.time()
            if f.__qualname__ in batch.keys():
                batch[f.__qualname__] += end - start
            else:
                batch[f.__qualname__] = end - start
        return res

    return out_f
