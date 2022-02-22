from syngular.tensor import MatrixProductState
from syngular.tensor import MatrixProductOperator

import numpy as np
import time

np.set_printoptions(suppress=True)




def test_apply():
    # w = np.arange(3**6).reshape((3,3,3,3,3,3))
    # W = MatrixProductOperator(w, bond_shape=(4,4,)).decompose()
    
    # z = np.arange(4**4*3**4).reshape((4,4,4,4,3,3,3,3))
    # Z = MatrixProductOperator(z, bond_shape=(2,2,2,)).decompose()

    # print(Z)
    # Z.apply(W, [1,2,3])
    # print(Z)

    w = np.arange(3**4).reshape((3,3,3,3))
    W = MatrixProductOperator(w, bond_shape=(2,)).decompose()
    z = np.arange(3**4).reshape((3,3,3,3))
    Z = MatrixProductOperator(w, bond_shape=(2,)).decompose()
    # z = np.arange(4**4*3**4).reshape((4,4,4,4,3,3,3,3))
    # Z = MatrixProductOperator(z, bond_shape=(2,2,2,)).decompose()
    print(Z)
    print(W)
    X = Z @ W
    print(X[(1,1),(1,1)])
    print((z * w)[1,1,1,1])

    Z.apply(W, [0,1])

    print(Z[(1,1),(1,1)])
    # print(X.to_tensor())
    # print(Z.to_tensor())

    # q = np.array([1,0,0,0,0,0,0,0]).reshape((2,2,2))
    # Q = MatrixProductOperator(q, (2,2))

def test_decompose():
    w = np.arange(2**4).reshape((2,2,2,2))
    W = MatrixProductOperator(w, bond_shape=(4,))

    
    z = np.arange(4**3*3**3).reshape((4,4,4,3,3,3))
    Z = MatrixProductOperator(z, bond_shape=(3,3,))
    
    print("---------- MatrixProductOperator.decompose() ----------")
    start = time.time()
    W.decompose()
    Z.decompose()
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

    print(w[1,1,1,1])
    print(W[(1,1),(1,1)])

    print(z[3,1,0,1,2,1])
    print(Z[(3,1,0),(1,2,1)])

def test_orthogonality():
    # w = np.arange(16**6).reshape((16,16,16,16,16,16))
    # W = MatrixProductOperator(w, bond_shape=(8,8,)).decompose()
    start = time.time()
    w = np.arange(4**6).reshape((4,4,4,4,4,4))
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    start = time.time()
    W = MatrixProductOperator(w, bond_shape=(2,2,))
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")

    start = time.time()
    W.decompose()
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")

    # print(w[1,0,0,1,0,0])
    # print(W[(1,0,0),(1,0,0)])

    W.left_orthonormalization()
    # print(W)
    # print(w[1,0,0,1,0,0])
    # print(W[(1,0,0),(1,0,0)])
    print("Parameters", W.parameters_number, "v.s True Parameters", W.real_parameters_number)
    print("---------- MatrixProductOperator.left_orthogonality() ----------")
    start = time.time()
    print(np.diag(W.left_orthogonality(0)))
    print(np.diag(W.left_orthogonality(1)))
    # print(np.diag(W.left_orthogonality(2)))
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")
    
    W.right_orthonormalization()
    print()
    print("---------- MatrixProductOperator.right_orthogonality() ----------")
    start = time.time()
    # print(np.diag(W.right_orthogonality(0)))
    print(np.diag(W.right_orthogonality(1)))
    print(np.diag(W.right_orthogonality(2)))
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

    print(w[1,0,0,1,0,0])
    print(W[(1,0,0),(1,0,0)])


def test_compress():
    w = np.arange(16**4).reshape((16,16,16,16))
    W = MatrixProductOperator(w, bond_shape=(8,)).decompose()
    # print(W)
    # print(W.to_tensor())
    print("Parameters", W.parameters_number, "v.s True Parameters", W.real_parameters_number)
    
    print("---------- MatrixProductOperator.compress(mp) ----------")
    print(W.shape)
    start = time.time()
    W.compress(4, mode='left')
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")
    print("Parameters", W.parameters_number, "v.s True Parameters", W.real_parameters_number)
    
    print("---------- MatrixProductOperator.compress(mp) ----------")
    print(W.shape)
    start = time.time()
    # W = W.compress(2, mode='left', strict=True)
    Z = W >> 2
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")
    print("Parameters", W.parameters_number, "v.s True Parameters", W.real_parameters_number)
    print(W.shape)
    print(Z.shape)

    # W = MatrixProductOperator(w, bond_shape=(4,)).decompose()
    # # print(W)
    # # print(W.to_tensor())
    # print("Parameters", W.parameters_number, "v.s True Parameters", W.real_parameters_number)
    
    # print("---------- MatrixProductOperator.compress(mp) ----------")
    # print(W.shape)
    # start = time.time()
    # W.compress(2, mode='right')
    # end = time.time()
    # print(f"> Execution time : {end-start:.8f}sec")
    # print("-------------------------------------------------------")
    # print("Parameters", W.parameters_number, "v.s True Parameters", W.real_parameters_number)
    
    # print(w)
    # print(W.to_tensor())

def test_dot():
    x = np.arange(2**2).reshape((2,2))
    X = MatrixProductState(x, bond_shape=(2,)).decompose()

    print("---------- MatrixProductOperator.__or__(mp) ----------")
    start = time.time()
    Z = X | X
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

    x = x.reshape(4)
    print(np.dot(x,x))
    print(Z)

    print(X.norm())

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
    # x = np.arange(2**6).reshape((2,2,2,2,2,2))
    # y = np.arange(2**6).reshape((2,2,2,2,2,2))

    # X = MatrixProductOperator(x, bond_shape=(3,3,)).decompose()
    # Y = MatrixProductOperator(y, bond_shape=(3,3,)).decompose()

    
    x = np.arange(2**4).reshape((2,2,2,2))
    y = np.arange(2**4).reshape((2,2,2,2))
    X = MatrixProductOperator(x, bond_shape=(2,)).decompose()
    Y = MatrixProductOperator(y, bond_shape=(2,)).decompose()

    print("---------- MatrixProductOperator.__mul__(mp) ----------")
    start = time.time()
    Z = X * Y
    z = x * y
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

    print(z)
    print(Z.to_tensor())
    # print(x[1,1,1,1,1,1])
    # print(X[(1,1,1),(1,1,1)])

    # print(y[1,1,1,1,1,1])
    # print(Y[(1,1,1),(1,1,1)])

    # print(Z)
    # print(z[1,1,1,1,1,1])
    # print(Z[(1,1,1),(1,1,1)])

    # print(z)

def test_random():
    
    print("---------- MatrixProductOperator.random(input_shape, output_shape, bond_shape) ----------")
    start = time.time()
    MatrixProductOperator.random((4,4,4), (2,2,2), (8,8,))
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
    print(mpo.real_parameters_number, mpo.parameters_number)
    print("---------- MatrixProductOperator.to_tensor() ----------")
    start = time.time()
    mpo.to_tensor()
    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")

def test_compare():
    y = np.ones(2**4).reshape((4,4))
    np.fill_diagonal(y, list(range(1,5)))
    y = y.reshape((2,2,2,2))
    Y = MatrixProductOperator(y, bond_shape=(2,)).decompose().right_orthonormalization()

    print(y)

    print(np.around(Y.to_tensor(), decimals=2))

    print("---------- Compare with to_tensor() ----------")
    start = time.time()
    
    diff = np.abs(y - Y.to_tensor())
    print('Min', np.min(diff))
    print('Max', np.max(diff))

    end = time.time()
    print(f"> Execution time : {end-start:.8f}sec")
    print("-------------------------------------------------------")


# test_decompose()
# test_from_sites()
# test_random()
# # test_to_tensor() -- very expensive ~1min
# test_add()
# test_mul()
# test_matmul()
# test_dot()
# test_compress()
# test_orthogonality()
# test_apply()
test_compare()