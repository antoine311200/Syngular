import itertools
from syngular.core import Model, Output
from syngular.core import Linear
from syngular.tensor import MatrixProductState, MatrixProductOperator

import numpy as np

w = np.arange(2**4).reshape(2,2,2,2)
W = MatrixProductOperator(w, bond_shape=(2,))
W.decompose()

model = Model([
    Linear(4,4,core=2, bond=4, weights_initializer=W),
    Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,4,core=2, bond=4, weights_initializer=W),
    # Linear(4,8,core=2,bond=2),
    # Output((2,))
])
model.build()

print("---")
print(model.draw())
print("---")

# X = np.random.normal(size=(2,2))
# X = MatrixProductState(X, bond_shape=(2,)).decompose(mode='left')

x = np.arange(2**2).reshape(2,2)
X = MatrixProductState(x, bond_shape=(2,)).decompose()
Y = MatrixProductState(np.arange(4).reshape(2,2), bond_shape=(2,)).decompose()
# X = MatrixProductState.random((2,2), (2,))
y = model.predict(X).to_tensor()
print(y)
print(Y.to_tensor())

model.train(X, Y, epochs=2)
print("X", X.to_tensor())
y = (model.predict(X))
# print(y)

# print(model.layers[0].trainable_tensor_weights[0]["weight"].sites)

# print(x.reshape((4,)) @ w.reshape(4,4))

# print("Result")
# print(y)
# # print(y.reconstruct())

# y_r = np.zeros_like(y.tensor)
# print(y_r)
# print(y.tensor)

# for k in itertools.product(range(1), range(1)):
#     y_r[k] = y.retrieve(k)

# print(y_r)