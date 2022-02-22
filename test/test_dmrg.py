from syngular.variational import DMRG
from syngular.tensor import MatrixProductOperator
from syngular.variational import Lanczos

import numpy as np

if __name__ == "__main__":

    input_shape = (3,3,3,3) #,3,3)
    output_shape = (3,3,3,3) #,3,3)
    bond_shape = (2,2,2) #,2,2)

    n = 3*3*3
    m = 4
    A = np.zeros(n**2).reshape((n,n))
    np.fill_diagonal(A, list(range(n)))
    B = A.reshape((3,3,3,3,3,3))
    print('B', B)

    mpo = MatrixProductOperator.random(input_shape=input_shape, output_shape=output_shape, bond_shape=bond_shape) # MatrixProductOperator(B, bond_shape=(3,3)).decompose()#

    print('C', np.around(mpo.to_tensor(), decimals=2))

    eigenstate = DMRG.solve(mpo, optimizer=Lanczos)
    eigentensor = eigenstate.to_tensor()
    eigentensor /= np.linalg.norm(eigentensor)

    print(eigenstate.shape)
    print(eigentensor.shape)
    
    print(mpo.to_tensor().shape)
    eigenvalues, eigenvectors = np.linalg.eig(mpo.to_tensor().reshape((3**4, 3**4)))
    eigenvalues2, eigenvectors2 = np.linalg.eig(A)

    eigenvalue = eigenvectors[0]
    eigenvalue /= np.linalg.norm(eigenvalue)

    print(eigenvalues)
    print(eigenvalues2)

    print(eigenvalue - eigentensor.reshape((81,)))
    # print(eigenvalues[0] - eigenstate.to_tensor())