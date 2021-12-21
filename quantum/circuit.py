from typing import Tuple

from syngular.quantum.qbit import Qbit
from syngular.tensor import MatrixProductState


class Circuit:
    
    def __init__(self, size, bond=2, initializer="ground", structure=[]):
        self.initializer = initializer
        self.size = size

        self.structure = structure

        self.current_step = 0
        self.current_state = None
        self.states = []

        self.reset()

    def run(self):
        length = len(self.structure)
        for step in range(self.current_step, length):
            self.step()

    def reset(self):
        if self.initializer == 'ground':
            self.current_state = Qbit(self.size)
            # self.current_state = MatrixProductState.zeros(
            #     input_shape=(2,)*self.size,
            #     bond_shape=(2,)*(self.size-1)
            # )
            self.states.append(self.current_state)

    def step(self):
        self.current_state @= self.structure[self.current_step]
        self.states.append(self.current_state)
        self.current_step += 1

    def add(self, gate: Tuple):
        self.structure.append(gate)

    def get(self, index=-1):
        return self.states[index]