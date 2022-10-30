# Multiprocessing Analysis

This project is an analysis the performance of multiprocessing in `C++` using `OpenMP` and `CUDA` for the following two problems:

- The calculation of Pi using the Monte Carlo method.
- The calculation of a vector/matrix/vector product.

## Analysis

Different multiprocessing approaches are used for testing purpose, like using critical sections, atomics, reductions, etc. The results of the analysis are well explained, commented and justified in the following Jupiter Notebooks:

- [OpenMP Analysis](openmp/analysis.ipynb)
- [CUDA Analysis](cuda/analysis.ipynb)

## Requirements

- [Clang](https://clang.llvm.org/)
- [OpenMP](https://www.openmp.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [Python 3](https://www.python.org/)
- [Jupyter Notebook](https://jupyter.org/)