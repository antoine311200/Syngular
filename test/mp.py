from syngular.tensor import MatrixProductState
from syngular.tensor import MatrixProductOperator

import numpy as np
import time





def test_matmul():
    w = np.arange(2**4).reshape((2,2,2,2))
    x = np.arange(2**2).reshape((2,2))
    W = MatrixProductOperator(w, bond_shape=(2,)).decompose()
    X = MatrixProductState(x, bond_shape=(2,)).decompose()

    print("---------- MatrixProductOperator.__matmul__(mp) ----------")
    start = time.time()
    Z = W @ W @ X
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

    x = x.reshape(4)
    w = w.reshape((4,4))

    print(x @ w @ w)
    print(np.matmul(x, np.matmul(w,w)))
    print("----")
    print(Z.to_tensor().reshape(4))

def test_from_sites():
    mp = MatrixProductOperator.random((4,4,4), (2,2,2), (3,3,))
    sites = mp.sites

    print("Shape", mp.shape)

    print("---------- MatrixProductOperator.from_sites(sites) ----------")
    start = time.time()
    MatrixProductOperator.from_sites(sites)
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

def test_mul():
    x = np.arange(2**6).reshape((2,2,2,2,2,2))
    y = np.arange(2**6).reshape((2,2,2,2,2,2))
    X = MatrixProductOperator(x, bond_shape=(3,3,)).decompose()
    Y = MatrixProductOperator(y, bond_shape=(3,3,)).decompose()

    print("---------- MatrixProductOperator.__mul__(mp) ----------")
    start = time.time()
    Z = X * Y
    z = x * y
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

    
    print(x[1,1,1,1,1,1])
    print(X[(1,1,1),(1,1,1)])

    print(y[1,1,1,1,1,1])
    print(Y[(1,1,1),(1,1,1)])

    print(Z)
    print(z[1,1,1,1,1,1])
    print(Z[(1,1,1),(1,1,1)])

def test_random():
    
    print("---------- MatrixProductOperator.random(input_shape, output_shape, bond_shape) ----------")
    start = time.time()
    MatrixProductOperator.random((4,4,4), (2,2,2), (3,3,))
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")


def test_add():
    x = np.arange(1,17).reshape((2,2,2,2))
    y = np.arange(18,34).reshape((2,2,2,2))
    X = MatrixProductOperator(x, bond_shape=(3,)).decompose()
    Y = MatrixProductOperator(y, bond_shape=(3,)).decompose()
    
    print("---------- MatrixProductOperator.__add__(a, b) ----------")
    start = time.time()
    Z = X + Y + Y + Y + Y + Y + Y + Y + Y + Y
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")
    
    z = x+y+y+y+y+y+y+y+y+y

    print(x[1,1,1,1])
    print(X[(1,1),(1,1)])

    print(y[1,1,1,1])
    print(Y[(1,1),(1,1)])

    print(z[1,1,1,1])
    print(Z[(1,1),(1,1)])
    print(X)
    print(Z)
    print(Z.to_tensor())
    # print(Z.retrieve((1,1),(1,1)))

def test_to_tensor():
    tensor = np.arange((2*2*2*2*2*2*2*2)*(2*2*2*2*2*2*2*2)).reshape(2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2)
    mpo = MatrixProductOperator(tensor, bond_shape=(2,2,2,2,2,2,2,)).decompose()
    print(mpo)
    print("---------- MatrixProductOperator.to_tensor() ----------")
    start = time.time()
    mpo.to_tensor()
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

# test_from_sites()
# test_random()
# test_to_tensor()
# test_add()
# test_mul()
test_matmul()