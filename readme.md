<h1 align="center">Syngular</h1>
<h4 align="center">Tensor Network in Python</h4>

<h1 align="center"> </h1>

![version](https://img.shields.io/badge/version-0.0.1-blueviolet)
![development](https://img.shields.io/badge/development-in%20progress-orange)
![maintenance](https://img.shields.io/badge/maintained-yes-brightgreen.svg)
![launched](https://img.shields.io/badge/launched-no-red.svg)


<br>

## Introduction

**Syngular** emerged in the context of my quantum computing project at CentraleSuépelec. I was working on the efficient simulation of quantum circuits on classical computers. The last work on the subject that tackled this issue was using Matrix Product State as a base to represent the 2^N parameters quantum state of a quantum system (_What limits the simulation of quantum computers?_ Yiqing Zhou, E. Miles Stoudenmire, Xavier Waintal) with only a fraction of parameters at the price of approximation on the singular values. 
I dived into this world of Matrix Product State and Operator leading me to the theorisation of Tensor Network with Yvan Osedelets works and I created a simple simulator (too simple). Then, in my second year at CentraleSupélec in the same team project, I had the opportunity to work on how to use tensor networks that came from the quantum world to machine learning as to compress efficiently neural network.

As so, I developed this Python package to create easily Tensor Network and simulate as well **Quantum Circuit** as **Neural Networks** and optimization of function.


<br>

## Getting Started

### Installation 

First, you will have to install the package

```bash
pip install syngular
```

### MatrixProductState & MatrixProductOperator

```python
from syngular.tensor import MatrixProductState
from syngular.tensor import MatrixProductOperator

import numpy as np

tensor_W = np.arange(16**6).reshape((16,16,16, 16,16,16))
tensor_X = np.arange(16**3).reshape((16,16,16))

W = MatrixProductOperator(tensor_W, bond_shape=(16,16,))
W.decompose()

X = MatrixProductState(tensor_X, bond_shape=(4,4,))
X.decompose()

T = MatrixProductOperator.random((16,16,16), (16,16,16), (8,8,))
O = MatrixProductOperator.zeros((16,16,16), (16,16,16), (4,4,))
U = MatrixProductState.random((16,16,16), (8,8,))

W = W >> 4
T = T >> 2

Z = ((T + W) @ T) @ X

print(X | U)
print(Z | X)

Z = Z >> 16
Z.left_orthonormalization()

print(np.diag(Z.left_orthogonality(0)))
print(np.diag(Z.left_orthogonality(1)))
```

### Quantum Simulation

```python
from syngular.quantum import Circuit, Qbit
import syngular.quantum.gate as gate

Qbit.LSB = False

circ = Circuit(size=15, structure=[
    (gate.X, 0),
    (gate.X, 2),
    (gate.H, 0),
    (gate.H, 2),
])

circ.run()
circ.add((gate.H, 1))
circ.run()

#########################################################

def verity_table(g, name):
    import itertools

    size = int(len(g.shape) // 2)
    length = ((3+size)*2+1)
    
    print('------------- Vertity Table --------------')
    print(f' > Gate : {name}')
    print("="*length)
    print("| inp > out |")
    print("="*length)

    for b in itertools.product([0, 1], repeat=size):
        qbit = Qbit(size)
        for i in range(len(b)): if b[i] == 1: qbit @= (gate.X, i)
        output = qbit @ (g, 0)

        print("|",qbit.to_binary(), '|', output.to_binary(), "|")
        print("-"*length)

verity_table(gate.TOFFOLI, "Toffoli")

######################################################

Qbit.LSB = True

qbit = Qbit(4)
qbit @= (gate.X, 0)
qbit @= (gate.X, 2)
print(qbit.to_binary())

qbit = qbit.swap(0,3)
print(qbit.to_binary())

qbit = qbit.swap(0,2)
print(qbit.to_binary())

qbit = qbit.swap(3,2)
print(qbit.to_binary())
```