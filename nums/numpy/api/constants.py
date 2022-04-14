import numpy as np

__all__ = [
    "pi",
    "e",
    "euler_gamma",
    "NINF",
    "PZERO",
    "NZERO",
    "nan",
    "bool_",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]

# Distributed memory access of these values will be optimized downstream.
pi = np.pi
e = np.e
euler_gamma = np.euler_gamma
inf = infty = Inf = Infinity = PINF = np.inf
NINF = np.NINF
PZERO = np.PZERO
NZERO = np.NZERO
nan = NAN = NaN = np.nan

bool_ = np.bool_

uint = np.uint
uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64

int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64

float16 = np.float16
float32 = np.float32
float64 = np.float64

complex64 = np.complex64
complex128 = np.complex128
