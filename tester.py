import nums.numpy as nps
import numpy as np
import time

x1 = nps.random.randn_sparse(10, 10)
print(x1.nonzero())
print(x1.get())

# x2 = nps.random.randn_sparse(10 ** 3, 10)

# t0 = time.time()

# x3 = x1 @ x2

# t1 = time.time()

# total = t1-t0

# y1 = x1.get()
# y2 = x2.get()
# y3 = x3.get()

# print(total, "seconds")

# from scipy.sparse import random
# import numpy as np
# from scipy import sparse

# z1 = sparse.csr_matrix(y1) 
# z2 = sparse.csr_matrix(y2) 

# z3 = z1.dot(z2).A

# print(np.allclose(y3, z3))

