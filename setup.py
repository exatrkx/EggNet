import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="eggnet",
    version="1.0.0",
    description="EggNet tracking pipeline",
    author="Jay Chan",
    packages=find_packages(include=["eggnet", "eggnet.*"]),
    entry_points={
        "console_scripts": [
            "eggnet = eggnet.core.cli:cli",
        ]
    },
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    keywords=[
        "track reconstruction",
        "graph networks",
        "machine learning",
    ],
    # package_dir={'': 'src'},
    # packages=['eggnet'],
    # url="https://gitlab.cern.ch/gnn4itkteam/acorn",
    # scripts=["scripts/train.py", "scripts/infer.py", "scripts/eval.py"],
)
