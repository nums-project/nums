# Contributing As a Mac User

## Installation

For M1 Macs (Apple Silicon), various [online sources](https://github.com/scipy/scipy/issues/13409) recommend using OpenBLAS as a native (not Rosetta) linear algebra backend for dependencies like NumPy.
```sh
brew install openblas
export OPENBLAS=$(/opt/homebrew/bin/brew --prefix openblas)
```

As [documented by Ray](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support), one must install miniforge, a community-driven Conda installer, for packages that support `arm64`. In particular, the GRPCIO package from pip will not work with Ray, so it must be uninstalled (`pip uninstall grpcio`) and replaced with the one from miniforge (`conda install grpcio`).


## Testing

`tests/core/backend/test_backend_init.py::test_head_detection`
As [documented by Ray](https://github.com/ray-project/ray/issues/24130), there is different default behavior for fetching node IP addresses. On Linux, a private address (e.g. `192.168.0.1`) is returned, while on Windows and Mac, `127.0.0.1` is returned. As NumS is primarily released for Linux, this test will fail for Mac contributors, but it shouldn't affect development.
