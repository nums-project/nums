<p align="center">
<img alt="NumS" width="256" height="256" src="https://user-images.githubusercontent.com/66851991/84823802-ca95da00-afd3-11ea-8275-789a7274adf1.jpg">
</p>

[![PyPI version](https://badge.fury.io/py/nums.svg)](https://badge.fury.io/py/nums)
[![Build Status](https://travis-ci.com/nums-project/nums.svg?branch=master)](https://travis-ci.com/nums-project/nums)
[![codecov](https://codecov.io/gh/nums-project/nums/branch/master/graph/badge.svg)](https://codecov.io/gh/nums-project/nums)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nums-project/nums-binder-env/master?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fnums-project%252Fnums%26urlpath%3Dtree%252Fnums%252Fexamples%252Fnotebooks%26branch%3Dmaster)

[//]: # (See this link to generate binder links https://jupyterhub.github.io/nbgitpuller/link?tab=binder)

# What is NumS?

**NumS** is a Numerical cloud computing library that translates Python and NumPy to distributed systems code at runtime. 
NumS scales NumPy operations horizontally, and provides inter-operation (task) parallelism for those operations.
NumS remains faithful to the NumPy API, and provides tight integration with the Python programming language 
by supporting loop parallelism and branching.
NumS' system-level operations are written against the [Ray](https://github.com/ray-project/ray) API;
it supports S3 and basic distributed filesystem operations for storage
and uses [NumPy](https://github.com/numpy/numpy) as a backend for CPU-based array operations.

# Usage

Obtain the latest release of NumS using `pip install nums`.

NumS provides explicit implementations of the NumPy API,
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
y_train = nps.concatenate([nps.zeros(shape=(size // 2,), dtype=nps.int64), 
                           nps.ones(shape=(size // 2,), dtype=nps.int64)], axis=0)
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
y_test = nps.concatenate([nps.zeros(shape=(10**3,), dtype=nps.int64), 
                          nps.ones(shape=(10**3,), dtype=nps.int64)], axis=0)
print("train accuracy", (nps.sum(y_train == model.predict(X_train)) / X_train.shape[0]).get())
print("test accuracy", (nps.sum(y_test == model.predict(X_test)) / X_test.shape[0]).get())
```

We perform the `get` operation to transmit 
the computed accuracy from distributed memory to "driver" (the locally running process) memory.

You can run this example in your browser [here](https://mybinder.org/v2/gh/nums-project/nums-binder-env/master?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fnums-project%252Fnums%26urlpath%3Dtree%252Fnums%252Fexamples%252Fnotebooks%252Flogistic_regression.ipynb%26branch%3Dmaster).

#### Training on HIGGS

Below is an example of loading the HIGGS dataset
(download [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/)), 
partitioning it for training, and running logistic regression.

```python
import nums
import nums.numpy as nps
from nums.models.glms import LogisticRegression

higgs_dataset = nums.read_csv("HIGGS.csv")
y, X = higgs_dataset[:, 0].astype(int), higgs_dataset[:, 1:]
model = LogisticRegression(solver="newton-cg")
model.fit(X, y)
y_pred = model.predict(X)
print("accuracy", (nps.sum(y == y_pred) / X.shape[0]).get())
```

# Running NumS on Dask or MPI

NumS can be configured to run on Dask and MPI.

#### Dask Backend
Install Dask using `pip install dask[complete]`.
The following snippet runs a basic computation using the Dask backend.

```python
import nums.numpy as nps
from nums.core import settings
settings.system_name = "dask"

x = nps.array([1, 2, 3])
y = nps.array([4, 5, 6])
z = x + y
print(z.get())
```

#### MPI Backend
NumS also supports cross-platform execution via it's MPI backend, which can be used to run NumS on HPC clusters.
The following dependencies need to be installed (on Ubuntu or related Linux machine) in order to use the MPI backend: An MPI implementation like `MPICH` and the MPI for Python package `mpi4py`.

```sh
sudo apt update
sudo apt-get install mpich
pip install mpi4py
```

The following snippet runs a basic computation using MPI.

```python
import nums.numpy as nps
from nums.core import settings
settings.system_name = "mpi"

x = nps.array([1, 2, 3])
y = nps.array([4, 5, 6])
z = x + y
print(z.get())
```

Finally, to execute the above script on MPI using two processes, run the following command:

```sh
mpiexec -n 2 python example.py
```

# Installation
NumS releases are tested on Linux-based systems running Python 3.7, 3.8, and 3.9.

NumS runs on Windows, but not all features are tested. We recommend using Anaconda on Windows. Download and install Anaconda for Windows [here](https://docs.anaconda.com/anaconda/install/windows/). Make sure to add Anaconda to your PATH environment variable during installation.

#### pip installation
To install NumS on Ray with CPU support, simply run the following command.
```sh
pip install nums
```

#### conda installation
We are working on providing support for conda installations, but in the meantime,
run the following with your conda environment activated. 

```sh
pip install nums
# Run below to have NumPy use MKL.
conda install -fy mkl
conda install -fy numpy scipy
```

#### S3 Configuration
To run NumS with S3, 
configure credentials for access by following instructions here: 
https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html

#### Cluster Setup
NumS programs can run on a single machine, and can also seamlessly scale to large clusters. \
Read more about [launching clusters](https://github.com/nums-project/nums/tree/master/cluster-setup).
