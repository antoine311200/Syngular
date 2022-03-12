from syngular.tensor.matrix_product_state import MatrixProductState
from syngular.variational import DMRG
from syngular.tensor import MatrixProductOperator
from syngular.variational import Lanczos

import numpy as np

def dmrg_lanczos_1():
    
    input_shape = (3,3,3,3) #,3,3)
    output_shape = (3,3,3,3) #,3,3)
    bond_shape = (9,9,9) #,2,2)

    n = 3*3*3
    m = 4
    A = np.zeros(n**2).reshape((n,n))
    np.fill_diagonal(A, list(range(n)))
    B = A.reshape((3,3,3,3,3,3))
    # print('B', B)

    mpo = MatrixProductOperator.random(input_shape=input_shape, output_shape=output_shape, bond_shape=bond_shape) # MatrixProductOperator(B, bond_shape=(6,6)).decompose()#

    diff = np.abs(B - mpo.to_tensor())
    print('Min', np.min(diff))
    print('Max', np.max(diff))
    # print('C', np.around(mpo.to_tensor(), decimals=2))

    eigenstate = DMRG.solve(mpo, optimizer=Lanczos)
    eigentensor = eigenstate.to_tensor()
    eigentensor /= np.linalg.norm(eigentensor)


    print("Eigenstate", eigenstate.shape)
    print("Eigentensor", eigentensor.shape)
    
    print("MPO Shape", mpo.to_tensor().shape)
    eigenvalues, eigenvectors = np.linalg.eig(mpo.to_tensor().reshape((3**4, 3**4)))
    # eigenvalues2, eigenvectors2 = np.linalg.eig(A)

    eigenvalue = eigenvectors[0]
    eigenvalue /= np.linalg.norm(eigenvalue)
    

    print(eigenvalues)
    # print(eigenvalues2)

    print(eigenvalue - eigentensor.reshape((81,)))
    print(np.abs(eigenvalues[0] - eigenstate.to_tensor()))

def dmrg_lanczos_random():

    k = 3
    input_shape = (k,k,k,k)
    output_shape = (k,k,k,k)
    bond_shape = (k**2,k**2,k**2)

    print((*input_shape,*output_shape,))

    tensor = np.random.rand(*(*input_shape,*output_shape,))
    tensor = (tensor+tensor.T)/2
    # n = k**4
    # A = np.zeros(n**2).reshape((n,n))
    # np.fill_diagonal(A, list(range(n)))
    # tensor = A.reshape((*input_shape,*output_shape,))
    operator = MatrixProductOperator.random(input_shape=input_shape, output_shape=output_shape, bond_shape=bond_shape).decompose()#(tensor, bond_shape=bond_shape).decompose()
    
    diff = np.abs(tensor - operator.to_tensor())
    print('Min', np.min(diff))
    print('Max', np.max(diff))
    print("Param", operator.parameters_number, "True param", operator.real_parameters_number)


    eigenstate = DMRG.solve(operator, optimizer=Lanczos)
    eigenstate.left_orthonormalization()
    eigenstate.normalize()

    eigentensor = eigenstate.to_tensor()
    eigentensor_reshaped = eigentensor.reshape(81)

    print("Norm eigentensor", np.linalg.norm(eigentensor), eigenstate.dot())

    eig = np.linalg.eig(tensor.reshape((3**4, 3**4)))
    ground_eigenstate = eig[1][0]
    # print(eig[0])
    ground_eigenstate /= np.linalg.norm(ground_eigenstate)
    # print(np.linalg.norm(ground_eigenstate))

    print("Eigenstate DMRG Shape", eigentensor.shape)
    print("Eigenstate Ground Shape", ground_eigenstate.shape)
    print("Norm", np.linalg.norm(ground_eigenstate), np.linalg.norm(eigentensor_reshaped))

    print(ground_eigenstate)
    print("\n")
    print(eigentensor_reshaped)
    print("\n")

    for gst in eig[1]:
        gst /= np.linalg.norm(gst)
        print(np.sum(np.abs(gst - eigentensor_reshaped)))

    print((operator @ eigenstate).dot(), eigenstate.dot())


if __name__ == "__main__":
    dmrg_lanczos_random()

