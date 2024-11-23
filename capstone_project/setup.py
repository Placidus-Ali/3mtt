from setuptools import setup, find_packages

setup(
    name="sort",
    version="1.0",
    description="Simple Online and Realtime Tracking (SORT)",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "filterpy",
    ],
)
