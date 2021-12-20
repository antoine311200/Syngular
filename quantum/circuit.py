

from syngular.tensor.matrix_product_state import MatrixProductState


class Circuit:
    
    def __init__(self, size, bond=2, initializer="ground", structure=[]):
        self.initializer = initializer
        self.size = size

        self.structure = structure

        self.current_step = 0
        self.current_state = None
        self.states = []

    def run(self):
        pass

    def reset(self):
        if self.initializer == 'ground':
            self.current_state = MatrixProductState.zeros(
                input_shape=(2,)*self.size,
                bond_shape=(2,)*(self.size-1)
            )
            self.states.append(self.current_state)

    def step(self):
        pass