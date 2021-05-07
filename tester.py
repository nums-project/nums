import nums.numpy as nps

x1 = nps.random.randn_sparse(3, 3)
x2 = nps.random.randn_sparse(3, 3)

x3 = x1 @ x2

print(x1.get())
print(x2.get())
print(x3.get())

