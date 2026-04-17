__all__ = [
    "EggNet",
    "Walkthrough",
    "FastWalkthrough",
]


def __getattr__(name):
    # Lazy loading might prevent long import times per worker thread
    if name == "EggNet":
        from .eggnet import EggNet

        return EggNet
    if name == "Walkthrough":
        from .walkthrough import Walkthrough

        return Walkthrough
    if name == "FastWalkthrough":
        from .fast_walkthrough import FastWalkthrough

        return FastWalkthrough
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
