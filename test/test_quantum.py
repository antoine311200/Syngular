import itertools
from syngular.quantum import Circuit, Qbit
import syngular.quantum.gate as gate

import time

import warnings
warnings.filterwarnings('ignore')

circ = Circuit(initializer="ground", size=2, structure=[
    (gate.X, 0)
])

size = 1

# ground = Qbit(size)
# print("> Ground state")
# print(ground.to_tensor())

# terminal = ground @ (1.j * gate.X) @ (-1.j * gate.X)
# print("> Terminal state")
# print(terminal.to_tensor())


def fourt_qbit():
    size = 4

    ground = Qbit(size)

    print("> Ground state")
    print(ground.to_tensor())

    # ground = ground @ gate.X

    print("> Intermediate state : X-gate:0 & X-gate:1")
    intermediate = ground @ (gate.X, 1) @ (gate.X, 0) @ (gate.X, 2) @ (gate.X, 3)
    # print(intermediate.to_tensor())
    print(intermediate.to_binary())
    print("> Intermediate state : X-gate:1")
    intermediate = intermediate @ (gate.X, 1)
    # print(intermediate.to_tensor())
    print(intermediate.to_binary())
    print("> Intermediate state : X-gate:1")
    intermediate = intermediate @ (gate.X, 1)
    # print(intermediate.to_tensor())
    print(intermediate.to_binary())
    print("> Intermediate state : X-gate:0")
    intermediate = intermediate @ (gate.X, 0)
    # print(intermediate.to_tensor())
    print(intermediate.to_binary())
    print("> Intermediate state : X-gate:1")
    intermediate = intermediate @ (gate.X, 1)
    # print(intermediate.to_tensor())
    print(intermediate.to_binary())
    # ground.state.apply(gate.X, 1)
    # print(ground.to_tensor())
    print("> Intermediate state : X-gate:3")
    intermediate = intermediate @ (gate.X, 3)
    # print(intermediate.to_tensor())
    print(intermediate.to_binary())

    print("> Terminal state : CX-gate")
    terminal = intermediate @ (gate.CX, 2)
    print(terminal.to_binary())
# ground @ (gate.X, 3)

# start = time.time()
# print(ground.state[(0,)*size])
# print(ground.to_binary())
# end = time.time()
# print(f"> Execution time : {end-start:.8f}sec")

def two_qbit():
    qbit = Qbit(2)

    print(qbit.to_binary())

    qbit = qbit @ (gate.X, 0)
    print(qbit.to_binary())
    # qbit = qbit @ (gate.X, 1)
    # print(qbit.to_binary())
    qbit = qbit @ (gate.CX, 0)
    print(qbit.to_binary())

# two_qbit()

# Qbit.VERBOSE = 1
Qbit.LSB = False

def three_qbit():
    qbit = Qbit(3)
    print(qbit.to_binary())

    qbit @= (gate.X, 1)
    print(qbit.to_binary())
    qbit @= (gate.X, 2)
    print(qbit.to_binary())
    # qbit = qbit @ (gate.X, 1)
    # print(qbit.to_binary())
    qbit @= (gate.CX, 1)
    print(qbit.to_binary())

    qbit @= (gate.X, 0)
    print(qbit.to_binary())

    qbit @= (gate.TOFFOLI, 0)
    print(qbit.to_binary())

def verity_table(g):
    print(g.shape, len(g.shape))
    size = int(len(g.shape)**(1/2))
    print(size)
    print('------------- Vertity Table --------------')
    print(f' > Gate : X')
    n = [range(c) for c in g.shape[:size]]
    print(n)
    for b in itertools.product([0, 1], repeat=size):
        qbit = Qbit(size)
        # print(b)
        for i in range(len(b)):
        #     # print(i)
            if b[i] == 1:
        #         # print("ok")
                qbit @= (gate.X, i)
        # print(qbit.to_tensor())
        # print(qbit.to_binary())
        output = qbit @ (g, 0)
        
        print(qbit.to_binary(), ' > ', output.to_binary())
        # print("".join(map(str,b)), ' > ', qbit.to_binary())


# three_qbit()
verity_table(gate.CX)

def timing():
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    times = []
    curve = []
    mx = 100
    for s in [100*i for i in range(1,mx)]:
        print(s)
        ground = Qbit(s)
        start = time.time()
        print(ground.state[(0,)*s])
        end = time.time()
        print(f"> Execution time : {end-start:.8f}sec")
        times.append(end-start)
        # curve.append(np.exp(s*np.log(5.13302064)/10000)-(1-0.005))

    b = 1-times[0]
    a = np.log((times[-1]+b))/(mx*100)
    for s in [100*i for i in range(1,mx)]:
        curve.append(np.exp(a*s)-b)

    print(times[-1], times[0])

    with open("times.txt", "wb") as fp:
        pickle.dump(times, fp)

    plt.plot([100*i for i in range(1,mx)], curve)
    plt.plot([100*i for i in range(1,mx)], times)
    plt.show()
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