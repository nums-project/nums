# pylint: disable = wrong-import-position

# Set numpy to single thread.
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from nums.api import init, read, write, delete, read_csv, from_modin
from nums.core.version import __version__

__all__ = ["numpy", "init", "read", "write", "delete", "read_csv", "from_modin"]
