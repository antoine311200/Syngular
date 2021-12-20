

from matplotlib.pyplot import isinteractive
from syngular.tensor.matrix_product_operator import MatrixProductOperator
from syngular.tensor.matrix_product_state import MatrixProductState
from syngular.quantum import gate

from typing import Tuple, List, Union
import numpy as np

class Qbit:

    def __init__(self, size, init=True):
        self.size = size
        self.dim = 2**self.size

        self.state = None
        
        if init:
            self.state = MatrixProductState.zeros((2,)*size, (2,)*(size-1)).decompose()
            self.state.real_parameters_number = self.dim

            for idx in range(self.state.sites_number-1):
                self.state.sites[idx][0] = gate.I
            self.state.sites[-1][0][0] = 1.

    def __matmul__(self, operator: Union[Tuple, Union[List, MatrixProductOperator]]):
        if isinstance(operator, tuple):
            return Qbit.from_mps(self.state.apply(*operator))
        else:
            operator = MatrixProductOperator(operator, bond_shape=()).decompose()
            return Qbit.from_mps(operator @ self.state)

    def apply(self, gate):
        gate = MatrixProductOperator(gate, bond_shape=()).decompose()
        return Qbit.from_mps(gate @ self.state)

    def to_tensor(self):
        return self.state.to_tensor().reshape(self.dim)

    def to_binary(self):
        tensor = self.to_tensor().astype(int)
        # print(tensor)
        # pos = 0
        # value = self.state[(0,) * self.size]
        # while value != 1 and pos < 2**self.size-1:
        #     pos += 1
        #     print(pos, 2**self.size-1)
        #     print((0,) * pos + (1,) + (0,) * (self.size-pos-1))
        #     value = self.state[(0,) * pos + (1,) + (0,) * (self.size-pos-1)]
            # print((0,) * pos + (1,) + (0,) * (self.size-pos-1))
            # print(value)
        return bin(np.where(tensor == 1)[0][0])[2:].zfill(self.size)

    def from_mps(mps):
        # print(mps.sites_number)
        qbit = Qbit(mps.sites_number, init=False)
        qbit.state = mps
        return qbit