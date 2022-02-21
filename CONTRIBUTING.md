# Contributing
To contribute to NumS on Ray, 
we recommend cloning the repository and installing the project in developer mode 
using the following set of commands:

```sh
cd nums
conda create --name nums python=3.7 -y
conda activate nums
pip install -e ".[testing]"
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
4. Add kernel interfaces to `nums.core.backends.interfaces.ComputeInterface`, and implement
the interface methods for all existing compute implementations.
Currently, the only compute interface is `nums.core.backends.numpy_compute`.
5. Write tests covering all branches of your implementation in the corresponding test module
in the project's `tests/` directory.
6. Do your best to implement the API in its entirety. It's generally better to have a partial
implementation than no implementation, so if for whatever reason certain arguments
are difficult to support, follow the convention we use to raise errors for unsupported
arguments in functions like `nums.numpy.api.min`.
7. If you run into any issues and need help with your implementation, open an issue describing
the issue you're experiencing.

We encourage you to follow the `nums.numpy.api.arange` implementation as a reference.
