import nums.numpy as nps

a = nps.array([[1,2,3], [4,5,6]])
b = a.blocks[0][0]
c = b * b
print(c.get())