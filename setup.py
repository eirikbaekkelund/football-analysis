from setuptools import setup, find_packages
import _version

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='football_analysis',
    version=_version.__version__,
    packages=find_packages(where="./"),
    package_dir={'': './'},
    include_package_data=True,
    install_requires=requirements,
)
# NOTE: to install via pip run
# run (uv) pip install -e . --no-deps
