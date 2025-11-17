from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import sys

ext_modules = [
    Pybind11Extension(
        "ctc_forced_aligner.ctc_forced_aligner",
        ["ctc_forced_aligner/forced_align_impl.cpp"],
        extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3"],
    )
]

setup(
    # The list of packages to include, using find_packages()
    # to automatically find all directories with an __init__.py file.
    packages=find_packages(exclude=["tests", "models", "venv"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
