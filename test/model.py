import itertools
from syngular.core.model import Model, Output
from syngular.core.model import Linear
from syngular.tensor.tensor_train import MatrixProductDensityOperator, MatrixProductState, MatrixProductOperator

import numpy as np

X = np.arange(1,17).reshape((2,2,2,2))
Y = np.arange(18,18+16).reshape((2,2,2,2))
Z = X + Y
print(X)
print()
print(Y)
print()
print(Z)


Xmp = MatrixProductOperator(X, bond_shape=(3,)).decompose()
Ymp = MatrixProductOperator(Y, bond_shape=(3,)).decompose()
Zmp = Xmp + Ymp
print(Zmp)


print(X[1,1,1,1])
print(Xmp.retrieve((1,1),(1,1)))

print(Y[1,1,1,1])
print(Ymp.retrieve((1,1),(1,1)))

print(Z[1,1,1,1])
print(Zmp.retrieve((1,1),(1,1)))

for a,b,c,d in itertools.product(range(2), range(2), range(2), range(2)):
    print(Z[a,b,c,d], Zmp.retrieve((a,b),(c,d)))

# model = Model([
#     Linear(4,4,core=2, bond=4),
#     Linear(4,1,core=2,bond=2),
#     Output((1,))
# ])
# model.build()

# print("---")
# print(model.draw())
# print("---")

# X = np.random.normal(size=(2,2))
# X = MatrixProductState(X, bond_shape=(2,)).decompose(mode='left')

# y = model.predict(X)
# # model.predict(X)

# print("Result")
# print(y)
# # print(y.reconstruct())

# y_r = np.zeros_like(y.tensor)
# print(y_r)
# print(y.tensor)

# for k in itertools.product(range(1), range(1)):
#     y_r[k] = y.retrieve(k)

# print(y_r)