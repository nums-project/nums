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

# Overview

The NumS team takes inspiration from early innovations in programming language design,
such as Fortran, Matlab, and NumPy. Thus, our goal is to simultaneously
provide a simple and performant API by implementing the NumPy API as closely as possible,
and maintaining a system architecture that enables runtime optimizations based on 
state-of-the-art discrete optimization techniques.
Below are a few key features that separate NumS from related libraries.
1. NumS is an eager evaluation distributed array library, which allows for efficient and seamless integration with branch and loop commands in Python.
2. The NumS BlockArray is a mutable data structure, just like NumPy arrays.
3. The NumS optimizer eliminates the need to learn a new programming model to achieve 
data (and model) parallel training. 

# Usage

First, obtain the latest release of NumS by simply running `pip install nums`.

NumS provides concrete implementations of the NumPy API,
providing a clear API with code hinting when used in conjunction with
IDEs (e.g. PyCharm) and interpreters (e.g. iPython, Jupyter Notebook) 
that provide such functionality.

## Basics

Below is a quick snippet that simply samples a few large arrays and 
performs basic array operations.

```python
import nums.numpy as nps

# Compute some products.
x = nps.random.rand(10**8)
# Note below the use of `get`, which blocks the executing process until
# an operation is completed, and constructs a numpy array
# from the blocks that comprise the output of the operation.
print((x.T @ x).get())
x = nps.random.rand(10**4, 10**4)
y = nps.random.rand(10**4)
print((x @ y).shape)
print((x.T @ x).shape)

# NumS also provides a speedup on basic array operations,
# such array search.
x = nps.random.permutation(10**8)
idx = nps.where(x == 10**8 // 2)

# Whenever possible, NumS automatically evaluates boolean operations
# to support Python branching.
if x[idx] == 10**8 // 2:
    print("The numbers are equal.")
else:
    raise Exception("This is impossible.")
```

## I/O

NumS provides an optimized I/O interface for fast persistence of block arrays.
See below for a basic example.

```python
import nums
import nums.numpy as nps

# Write an 800MB object in parallel, utilizing all available cores and
# write speeds available to the OS file system.
x1 = nps.random.rand(10**8)
# We invoke `get` to block until the object is written.
# The result of the write operation provides status of the write
# for each block as a numpy array.
print(nums.write("x.nps", x1).get())

# Read the object back into memory in parallel, utilizing all available cores.
x2 = nums.read("x.nps")
assert nps.allclose(x1, x2)
```

NumS automatically loads CSV files in parallel as distinct arrays, 
and intelligently constructs a partitioned array for block-parallel linear algebra operations.


```python
# Specifying has_header=True discards the first line of the CSV.
dataset = nums.read_csv("path/to/csv", has_header=True)
```

##  Logistic Regression

In this example, we'll run logistic regression on a 
bimodal Gaussian. We'll begin by importing the necessary modules.

```python
import nums.numpy as nps
from nums.models.glms import LogisticRegression
```

NumS initializes its system dependencies automatically as soon as an operation is performed. 
Thus, importing modules triggers no systems-related initializations.

#### Parallel RNG

NumS is based on NumPy's parallel random number generators.
You can sample billions of random numbers in parallel, which are automatically 
block-partitioned for parallel linear algebra operations.

Below, we sample an 800MB bimodal Gaussian, which is asynchronously generated and stored
by the implemented system's workers.

```python
size = 10**8
X_train = nps.concatenate([nps.random.randn(size // 2, 2), 
                           nps.random.randn(size // 2, 2) + 2.0], axis=0)
y_train = nps.concatenate([nps.zeros(shape=(size // 2,), dtype=nps.int), 
                           nps.ones(shape=(size // 2,), dtype=nps.int)], axis=0)
```

#### Training

NumS's logistic regression API follows the scikit-learn API, a
familiar API to the majority of the Python scientific computing community.

```python
model = LogisticRegression(solver="newton-cg", penalty="l2", C=10)
model.fit(X_train, y_train)
```

We train our logistic regression model using the Newton method.
NumS's optimizer automatically optimizes scheduling of 
operations using a mixture of block-cyclic heuristics, and a fast, 
tree-based optimizer to minimize memory and network load across distributed memory devices.
For tall-skinny design matrices, NumS will automatically perform data-parallel
distributed training, a near optimal solution to our optimizer's objective.

#### Evaluation

We evaluate our dataset by computing the accuracy on a sampled test set.

```python
X_test = nps.concatenate([nps.random.randn(10**3, 2), 
                          nps.random.randn(10**3, 2) + 2.0], axis=0)
y_test = nps.concatenate([nps.zeros(shape=(10**3,), dtype=nps.int), 
                          nps.ones(shape=(10**3,), dtype=nps.int)], axis=0)
print("train accuracy", (nps.sum(y_train == model.predict(X_train)) / X_train.shape[0]).get())
print("test accuracy", (nps.sum(y_test == model.predict(X_test)) / X_test.shape[0]).get())
```

We perform the `get` operation to transmit 
the computed accuracy from distributed memory to "driver" (the locally running process) memory.

#### Training on HIGGS

Below is an example of loading the HIGGS dataset
(download [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/)), 
partitioning it for training, and running logistic regression.

```python
import nums
import nums.numpy as nps
from nums.models.glms import LogisticRegression

higgs_dataset = nums.read_csv("HIGGS.csv")
y, X = higgs_dataset[:, 0].astype(nps.int), higgs_dataset[:, 1:]
model = LogisticRegression(solver="newton-cg")
model.fit(X, y)
y_pred = model.predict(X)
print("accuracy", (nps.sum(y == y_pred) / X.shape[0]).get())
```

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
