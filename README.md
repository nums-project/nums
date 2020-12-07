<p align="center">
<img alt="NumS" width="256" height="256" src="https://user-images.githubusercontent.com/66851991/84823802-ca95da00-afd3-11ea-8275-789a7274adf1.jpg">
</p>

[![PyPI version](https://badge.fury.io/py/nums.svg)](https://badge.fury.io/py/nums)
[![Build Status](https://travis-ci.com/nums-project/nums.svg?branch=master)](https://travis-ci.com/nums-project/nums)
[![codecov](https://codecov.io/gh/nums-project/nums/branch/master/graph/badge.svg)](https://codecov.io/gh/nums-project/nums)

# What is NumS?

NumS is a **Num**erical computing library for Python that **S**cales your workload to the cloud. 
It is an array abstraction layer on top of distributed memory systems that implements the NumPy API, 
extending NumPy to scale horizontally, as well as provide inter-operation parallelism 
(e.g. automatic parallelization of Python loops).
NumS differentiates itself from related solutions by implementing the NumPy API,
and providing tighter integration with the Python programming language by supporting
loop parallelism and branching.
Currently, NumS implements a
[Ray](https://github.com/ray-project/ray) system interface, 
S3 and distributed filesystems for storage,
and [NumPy](https://github.com/numpy/numpy) as a backend for CPU-based array operations.

# Installation
NumS is currently supported on Linux-based systems running Python 3.6, 3.7, and 3.8.
Currently, only CPU-based workloads are supported; we are working on providing GPU support.

#### pip installation
To install NumS on Ray with CPU support, simply run the following command.
```shell script
pip install nums
```

#### conda installation
We are working on providing support for conda installations, but in the meantime,
run the following with your conda environment activated. 

```shell script
pip install nums
# Run below to have NumPy use MKL.
conda install -fy mkl
conda install -fy numpy scipy
```

#### S3 Configuration
To run NumS with S3, 
configure credentials for access by following instructions here: 
https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html

# Contributing
To contribute to NumS on Ray, 
we recommend cloning the repository and installing the project in developer mode 
using the following set of commands:

```shell script
conda create --name nums python=3.7 -y
source activate nums
pip install -e .[testing]
```

#### Contributing NumPy Functionality

To make basic contributions to the NumPy API, follow these steps:

1. Replicate the function signature in `nums.numpy.api`. If it's a `np.ndarray` method,
    add the function signature to `nums.core.array.blockarray.BlockArray`.
2. If possible,  implement the function using existing methods 
in `nums.core.array.application.ArrayApplication` 
or `nums.core.array.blockarray.BlockArray`.
3. Write a new implementation `ArrayApplication` or `BlockArray` 
if it's not possible to implement using existing methods, 
or the implementation's execution speed can be 
improved beyond what is achievable using existing methods.
4. Add kernel interfaces to `nums.core.systems.interfaces.ComputeInterface`, and implement
    the interface methods for all existing compute implementations.
    Currently, the only compute interface is `nums.core.systems.numpy_compute`.
5. Write tests covering all branches of your implementation in the corresponding test module
in the project's `tests/` directory.
6. Do your best to implement the API in its entirety. It's generally better to have a partial
implementation than no implementation, so if for whatever reason certain arguments
are difficult to support, follow the convention we use to raise errors for unsupported
arguments in functions like `nums.numpy.api.min`.
7. If you run into any issues and need help with your implementation, open an issue describing
the issue you're experiencing.

We encourage you to follow the `nums.numpy.api.arange` implementation as a reference.