import itertools
from syngular.core import Model, Output
from syngular.core import Linear
from syngular.tensor import MatrixProductState, MatrixProductOperator

import numpy as np

model = Model([
    Linear(4,4,core=2, bond=4),
    Linear(4,1,core=2,bond=2),
    Output((1,))
])
model.build()

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