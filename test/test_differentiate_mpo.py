from syngular.tensor import MatrixProductState
from syngular.tensor import MatrixProductOperator
from syngular.tensor import DifferentialMatrixProductOperator

import numpy as np

n = 8
tensor_mpo = np.arange(2**(2*n)).reshape(*((2,)*(2*n))).astype('float64')
tensor_mps = np.arange(2**(n)).reshape(*(2,)*n).astype('float64')

mpo = DifferentialMatrixProductOperator(tensor_mpo, bond_shape=(2,)*(n-1)).decompose()
mps = MatrixProductState(tensor_mps, bond_shape=(2,)*(n-1)).decompose()


print(mpo)
print(mps)

mpo.project(3, mps)