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
Currently, NumS only supports CPU-based workloads; 
we are working on providing GPU support.

#### pip installation
To install NumS on Ray with CPU support, simply run the following command.
```shell script
pip install nums
```

#### conda installation
We are working on providing support for conda installations, but in the meantime,
run the following with your conda environment activated. 

```shell script
pip install --upgrade nums
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
conda create --name nums python=3.6 -y
source activate nums
pip install ray==0.8.7
python setup.py develop
```

#### Contributing NumPy Functionality
...