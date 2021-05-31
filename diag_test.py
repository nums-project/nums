import nums.numpy as nps
import numpy as np
from nums.numpy import BlockArray
# ba: BlockArray = nps.array(np.full((4, 4), 1.0))
# ba = ba.reshape(block_shape=(2, 3))
# np_arr = ba.get()
# print("Before")
# ba = nps.diag(ba)
# print(ba)
# print(ba.get())
# #np_arr = np.diag(np_arr)
# #print(ba)
# #assert np.allclose(ba.get(), np_arr)

for i in range(1, 11):
        for j in range(1, 11):
            for k in range(1, i + 1):
                for l in range(1, j + 1):
                    ba: BlockArray = nps.array(np.full((i, j), 1.0))
                    ba = ba.reshape(block_shape=(k, l))
                    np_arr = ba.get()
                    ba = nps.diag(ba)
                    np_arr = np.diag(np_arr)
                    print("i = {}, j = {}, k = {}, l = {}".format(i, j, k, l), ba.get())
                    assert not np.allclose(ba.get(), np_arr)