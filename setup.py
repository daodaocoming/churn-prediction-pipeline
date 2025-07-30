from setuptools import setup, find_packages

setup(
    name="churnpipe",
    version="0.1",
    packages=["src"] + ["src." + p for p in find_packages("src")],
    package_dir={"src": "src"},  
    include_package_data=True,
)
