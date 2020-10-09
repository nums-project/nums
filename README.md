<p align="center">
<img alt="NumS" width="512" height="512" src="https://user-images.githubusercontent.com/66851991/84823802-ca95da00-afd3-11ea-8275-789a7274adf1.jpg">
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

