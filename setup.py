# from setuptools import setup, find_packages

# setup(
#     name="kulibrat",
#     version="1.0.0",
#     author="Pablo Rocamora-García",
#     author_email="pablorocg10@gmail.com",
#     description="Implementation of the Kulibrat board game",
#     long_description=open("README.md").read(),
#     long_description_content_type="text/markdown",
#     url="https://github.com/pablorocg/kulibrat",
#     packages=find_packages(),
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "Programming Language :: Python :: 3.8",
#         "Programming Language :: Python :: 3.9",
#         "Programming Language :: Python :: 3.10",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#         "Topic :: Games/Entertainment :: Board Games",
#         "Intended Audience :: End Users/Desktop",
#         "Development Status :: 4 - Beta",
#     ],
#     python_requires=">=3.8",
#     install_requires=[
#         "numpy>=1.20.0",
#         "typing_extensions>=4.0.0",
#     ],
#     entry_points={
#         "console_scripts": [
#             "kulibrat=src.main:main",
#         ],
#     },
#     include_package_data=True,
#     package_data={
#         "src.ui": ["assets/*.png"],
#     },
# )

"""
Setup script to compile the Cython implementation of GameState.
Run this with:
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "src.core.game_state_cy",  # Output module name
        ["src/core/game_state.pyx"],  # Source file
        include_dirs=[np.get_include()],  # Include NumPy header files
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],  # Compiler optimizations
        language="c++"  # Use C++ compilation
    )
]

setup(
    name="kulibrat",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,  # Turn off bounds checking
            "wraparound": False,   # Turn off negative indexing
            "initializedcheck": False,  # Turn off memory initialization checks
            "cdivision": True,     # Turn off Python division behavior
        }
    ),
)