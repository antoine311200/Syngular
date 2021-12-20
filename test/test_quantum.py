from syngular.quantum import Circuit, Qbit
import syngular.quantum.gate as gate

circ = Circuit(initializer="ground", size=2, structure=[
    (gate.X, 0)
])

size = 30

ground = Qbit(size)

# print(ground.to_tensor())
print(ground.state[(1,)*size])
print(ground.state[(0,)*size])
# print(ground.state.real_parameters_number, ground.state.parameters_number)
# import numpy as np
# from syngular.tensor import MatrixProductState
# ground = np.zeros(shape=(2**size))
# ground[0] = 1
# print(ground)
# qbit2 = MatrixProductState(ground.reshape((2,)*size), (3,)*(size-1)).decompose()
# print(qbit2)
# for site in qbit2.sites:
#     print('---')
#     print(site)
# print(qbit2.to_tensor().reshape(2**size))