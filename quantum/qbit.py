

from syngular.tensor.matrix_product_state import MatrixProductState
from syngular.quantum import gate

class Qbit:

    def __init__(self, size):
        self.size = size
        self.dim = 2**self.size

        self.state = MatrixProductState.zeros((2,)*size, (2,)*(size-1)).decompose()
        self.state.real_parameters_number = self.dim
        
        for idx in range(self.state.sites_number-1):
            self.state.sites[idx][0] = gate.I
        self.state.sites[-1][0][0] = 1.

    def to_tensor(self):
        return self.state.to_tensor().reshape(self.dim)