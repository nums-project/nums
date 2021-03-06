.. NumS documentation master file, created by
   sphinx-quickstart on Sat Mar  6 00:28:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NumS Reference Documentation
============================

What is NumS?
-------------

NumS is a **Num**\ erical computing library for Python that **S**\ cales your workload to the cloud. It is an array abstraction layer on top of distributed memory systems that implements the NumPy API, extending NumPy to scale horizontally, as well as provide inter-operation parallelism (e.g. automatic parallelization of Python loops). NumS differentiates itself from related solutions by implementing the NumPy API, and providing tighter integration with the Python programming language by supporting loop parallelism and branching. Currently, NumS implements a `Ray <https://github.com/ray-project/ray>`_ system interface, S3 and distributed filesystems for storage, and `NumPy <https://github.com/numpy/numpy>`_ as a backend for CPU-based array operations.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   hello_world.rst


.. automodule:: nums.api
   :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
