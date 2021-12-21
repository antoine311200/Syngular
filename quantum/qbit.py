

from matplotlib.pyplot import isinteractive
from syngular.tensor.matrix_product_operator import MatrixProductOperator
from syngular.tensor.matrix_product_state import MatrixProductState
from syngular.quantum import gate

from typing import Tuple, List, Union
import numpy as np

class Qbit:

    VERBOSE = 0
    LSB = True

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
        if Qbit.VERBOSE: print(self.to_tensor().astype(float))
        
        if isinstance(operator, tuple):
            if len(operator) == 2:
                return Qbit.from_mps(self.state.apply(*operator))
            else:
                qbit = self.swap_in(operator[1], operator[2]-2)
                print('->', qbit.to_binary())
                op = (operator[0], operator[2]-1)
                print(op[1])
                qbit = Qbit.from_mps(qbit.state.apply(*op))
                print('->', qbit.to_binary())
                qbit = qbit.swap_out(operator[1], operator[2]-2)
                return qbit
        else:
            operator = MatrixProductOperator(operator, bond_shape=()).decompose()
            return Qbit.from_mps(operator @ self.state)

    def __imatmul__(self, operator: Union[Tuple, Union[List, MatrixProductOperator]]):
        return self @ operator

    def swap(self, idx1, idx2):
        self = self.swap_in(idx1, idx2)
        self = self.swap_out(idx1, idx2)
        return self

    def swap_in(self, idx1, idx2):
        _idx1 = min(idx1, idx2)
        _idx2 = max(idx1, idx2)

        # for site in self.state.sites:
            # print(site)
        for i in range(_idx1, _idx2):
            # print(i)    

            self @= (gate.SWAP, i)
            # print(self.to_tensor())
            # for site in self.state.sites:
                # print(site)
            # print(self.to_binary())
            # print(">", self.to_tensor(), self.to_binary())
        # print("=>", self.to_tensor(), self.to_binary())
        return self
    
    def swap_out(self, idx1, idx2):
        _idx1 = min(idx1, idx2)
        _idx2 = max(idx1, idx2)

        # self @= (gate.SWAP, _idx2)
        # print("ok", self.to_binary())
        # print(list(range(_idx2, _idx1-1, -1)))
        for i in range(_idx2-2, _idx1-1, -1):
            # print(i)
            # print(self.to_binary())
            self @= (gate.SWAP, i)
            # print(">", self.to_tensor(), self.to_binary())

        return self

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
        # print(tensor)
        if Qbit.LSB:
            return bin(np.where(tensor == 1)[0][0])[2:].zfill(self.size)[::-1]
        else:
            return bin(np.where(tensor == 1)[0][0])[2:].zfill(self.size)

    def from_mps(mps):
        # print(mps.sites_number)
        qbit = Qbit(mps.sites_number, init=False)
        qbit.state = mps
        return qbit