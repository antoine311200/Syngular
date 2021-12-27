import itertools
from syngular.quantum import Circuit, Qbit
import syngular.quantum.gate as gate

import time

import warnings

warnings.filterwarnings('ignore')


def test_circuit():
    Qbit.LSB = False

    circ = Circuit(size=30, structure=[
        (gate.X, 0),
        (gate.X, 2),
        (gate.H, 0),
        (gate.H, 2),
    ])

    circ.run()
    circ.add((gate.H, 1))
    circ.run()

    print(circ.current_state)
    # for state in circ.states:
    #     print(state.to_tensor())

# test_circuit()
def test_swap():
    qbit = Qbit(4)
    print(qbit.to_binary())
    qbit @= (gate.X, 0)
    print(qbit.to_binary())

    # qbit @= (gate.SWAP, 0)
    # print(qbit.to_binary())
    # qbit @= (gate.SWAP, 1)
    # print(qbit.to_binary())
    # qbit @= (gate.SWAP, 2)
    # print(qbit.to_binary())
    # qbit = qbit.swap_in(0,2)
    # print(qbit.to_binary())
    # qbit = qbit.swap_out(0,2)
    # print(qbit.to_binary())

    qbit @= (gate.X, 3)
    print(qbit.to_binary())
    qbit @= (gate.X, 2)
    # qbit @= (gate.X, 1)
    # qbit @= (gate.X, 2)
    # qbit @= (gate.X, 3)
    print(qbit.to_binary())
    # print(qbit.to_tensor())

    # qbit = qbit.swap_in(0, 3)
    # print(qbit.to_tensor())

    # print(",",qbit.to_binary())
    # qbit = qbit.swap_out(0, 3)
    qbit = qbit.swap(0,3)
    # qbit @= (gate.CX, 0, 2)
    print(qbit.to_binary())
    qbit = qbit.swap(0,2)
    print(qbit.to_binary())
    qbit = qbit.swap(3,2)
    print(qbit.to_binary())
    qbit @= (gate.X, 3)
    print(qbit.to_binary())
    qbit = qbit.swap(1,2)
    print(qbit.to_binary())

    # qbit @= (gate.SWAP, 0)
    # circ = Circuit(2)
    # circ.add((gate.X, 0))
    # circ.add((gate.SWAP, 0))
# test_swap()


def test_bernstein_vazirani(token):
    size = len(token)
    # token = token.zfill(size)[::-1]
    # print(token)

    circ = Circuit(size=size+1)
    circ.add((gate.H, size))
    circ.add((gate.Z, size))


    for i in range(size):
        circ.add((gate.H, i))
    # circ.add((gate.X, 0))
    # circ.add((gate.CX, size, 0))
    # circ.add(BlackBox())
    for i in range(size):
        if token[i] == '1':
            print("ok", i)
            circ.add((gate.CX, size, i))

    for i in range(size):
        circ.add((gate.H, i))

    # circ.run()
    for stp in range(len(circ.structure)):
        print('------------ step -------------')
        print(circ.current_state.to_tensor())
        circ.step()

    print(circ.current_state.to_tensor())
    # print("ok")

    import numpy as np
    print(np.argmax(circ.current_state.to_tensor()))


    # qbit_token = Qbit.from_binary(token)
    # print(circ.current_state.state | Qbit(size).state)
    # print(circ.current_state.state | qbit_token.state)


# test_bernstein_vazirani("11")

def test_hadamard(token):

    token = token[::-1]
    size  = len(token)

    qbit = Qbit(size+1)

    qbit @= (gate.H, size)
    qbit @= (gate.Z, size)

    for i in range(size): qbit @= (gate.H, i)

    for i in range(len(token)):
        if token[i] == "1": qbit @= (gate.CX, 2, i)
        else: qbit @= (gate.I, i)
    
    for i in range(size): qbit @= (gate.H, i)

    print(qbit.to_tensor())

    import numpy as np
    print(np.argmax(qbit.to_tensor()))
    print("Probability : ")
    print(
        (qbit.state | Qbit.from_binary(token[::-1]+"1").state)**2  + 
        (qbit.state | Qbit.from_binary(token[::-1]+"0").state)**2
    )


# test_hadamard("111")

def test_bell_state():
    start = time.time()
    qbit = Qbit(2)
    qbit @= (gate.H, 0)
    qbit @= (gate.CX, 0, 1) ## Control 1-th qbit with 0-th qbit 
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")

    print(qbit.to_tensor())

# test_bell_state()

def test_hzh_x():
    qbit = Qbit(1)
    qbit @= gate.X
    print(qbit.to_tensor())

    qbit = Qbit(1)
    qbit @= gate.H @ gate.Z @ gate.H
    print(qbit.to_tensor())

def test_ss_z():
    qbit = Qbit(1)
    qbit @= gate.X
    qbit @= gate.Z
    print(qbit.to_tensor())

    qbit = Qbit(1)
    qbit @= gate.X
    qbit @= gate.S @ gate.S
    print(qbit.to_tensor())


test_hzh_x()
test_ss_z()
# for i in range(1):
#     test_bernstein_vazirani(4)
# size = 1

# ground = Qbit(size)
# print("> Ground state")
# print(ground.to_tensor())

# terminal = ground @ (1.j * gate.X) @ (-1.j * gate.X)
# print("> Terminal state")
# print(terminal.to_tensor())


def four_qbit():
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

def verity_table(g, name):
    # print(g.shape, len(g.shape))
    size = int(len(g.shape) // 2)
    # print(size)
    print('------------- Vertity Table --------------')
    print(f' > Gate : {name}')

    print("="*((3+size)*2+1))
    print("| inp > out |")
    print("="*((3+size)*2+1))
    for b in itertools.product([0, 1], repeat=size):
        qbit = Qbit(size)
        for i in range(len(b)):
            if b[i] == 1:
                qbit @= (gate.X, i)
        output = qbit @ (g, 0)
        print("|",qbit.to_binary(), '|', output.to_binary(), "|")
        print("-"*((3+size)*2+1))
    # print("="*((3+size)*2+1))


# three_qbit()
# verity_table(gate.X, "Not")
# verity_table(gate.CX, "CNot")
# verity_table(gate.TOFFOLI, "Toffoli")

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