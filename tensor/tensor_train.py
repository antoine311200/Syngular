import itertools
import numpy as np
from syngular.tensor.mpo import MatrixProductOperator

import matplotlib.pyplot as plt

class MatrixProductState:

    def __init__(self, tensor, bond_shape) -> None:
        self.tensor = tensor
        self.tensor_shape = tensor.shape
        self.bond_shape = bond_shape

        self.sites_number = len(self.bond_shape)+1
        self.sites = []

        self.shape = [(self.tensor_shape[i], self.bond_shape[i]) for i in range(self.sites_number-1)]

        print(self.shape)

    def decompose(self):

        current_matrix = self.tensor
        current_shape = self.shape[0][0]
        current_rank = 1

        for k in range(self.sites_number-1):
            
            unfolding_matrix = np.reshape(current_matrix, newshape=(current_shape*current_rank, -1))
            rank = self.shape[k][1]
            
            Q, R = np.linalg.qr(unfolding_matrix, mode="complete")

            Q = Q[:,:rank]
            Q = np.reshape(Q, newshape=(current_rank, current_shape, -1))
            R = R[:rank, :]

            self.sites.append(Q)

            current_rank = rank
            current_matrix = R

        current_matrix = current_matrix[:, :, np.newaxis]
        self.sites.append(current_matrix)

    def retrieve(self, indices):
        einsum_structure = []
        for idx in range(self.sites_number):
            einsum_structure.append(self.sites[idx][:, indices[idx], :])
            einsum_structure.append([idx, idx+1])

        return np.einsum(*einsum_structure)


class MatrixProductOperator:

    def __init__(self, tensor, bond_shape) -> None:
        self.tensor = tensor
        self.tensor_shape = tensor.shape
        self.bond_shape = bond_shape

        self.sites_number = len(self.bond_shape)+1
        self.sites = []

        self._shape = [(1, self.tensor_shape[0], self.tensor_shape[self.sites_number], self.bond_shape[0])]
        self._shape += [(self.bond_shape[i-1], self.tensor_shape[i], self.tensor_shape[self.sites_number+i], self.bond_shape[i]) for i in range(1, self.sites_number-1)]
        self._shape += [(self.bond_shape[self.sites_number-2], self.tensor_shape[self.sites_number-1], self.tensor_shape[2*self.sites_number-1], 1)]

        print(self._shape)

        self.tensor = np.transpose(self.tensor, axes=sum(list(zip(range(self.sites_number), range(self.sites_number, 2*self.sites_number))), ()))

    def decompose(self):

        current_matrix = self.tensor
        current_input_shape = self._shape[0][1]
        current_output_shape = self._shape[0][2]
        current_rank = 1

        for k in range(self.sites_number-1):
            
            unfolding_matrix = np.reshape(current_matrix, newshape=(current_input_shape*current_output_shape*current_rank, -1))
            rank_right = self._shape[k][3]

            Q, R = np.linalg.qr(unfolding_matrix, mode="complete")

            Q = Q[:,:rank_right]
            Q = np.reshape(Q, newshape=(current_rank, current_input_shape, current_output_shape, rank_right))
            R = R[:rank_right, :]

            self.sites.append(Q)

            current_input_shape = self._shape[k][1]
            current_output_shape = self._shape[k][2]
            current_rank = rank_right
            current_matrix = R

        current_matrix = np.reshape(current_matrix, newshape=(-1, current_input_shape, current_output_shape))[:, :, :, np.newaxis]
        print(current_matrix.shape)
        self.sites.append(current_matrix)

    def retrieve(self, input_indices, output_indices):
        einsum_structure = []
        for idx in range(self.sites_number):
            einsum_structure.append(self.sites[idx][:, input_indices[idx], output_indices[idx], :])
            einsum_structure.append([idx, idx+1])

        return np.einsum(*einsum_structure)

if __name__ == "__main__":

    t = np.arange(5**7).reshape((5,5,5,5,5,5,5))
    mps = MatrixProductState(t, bond_shape=(2,2,2,2,2,2))
    mps.decompose()


    # errors = []
    # for (a,b,c,d,e,f,g) in itertools.product(range(5), range(5), range(5), range(5), range(5), range(5), range(5)):
    #     error = abs(t[a,b,c,d,e,f,g] - mps.retrieve((a,b,c,d,e,f,g))[0][0])
    #     errors.append(error)
    # plt.plot(errors)








    tensor = np.arange(27).reshape((3,3,3))
    print(tensor)
    
    mps = MatrixProductState(tensor, bond_shape=(2,2))
    mps.decompose()

    print("MPS ")
    print([s.shape for s in mps.sites])
    print(mps.sites)

    print(tensor[1,1,1])
    for (i,j,k) in itertools.product(range(3), range(3), range(3)):
        print(mps.retrieve((i,j,k)))


    # errors = []
    # for (a,b,c,) in itertools.product(range(3), range(3), range(3)):
    #     error = abs(tensor[a,b,c] - mps.retrieve((a,b,c))[0][0])
    #     errors.append(error)
    # plt.plot(errors)

    print("-----------")

    operator_tensor = np.arange(3**4 * 4**4).reshape((3,3,3,3,4,4,4,4))
    # print(operator_tensor)

    mpo = MatrixProductOperator(operator_tensor, (2,2,2))
    mpo.decompose()
    print("retrieve >")
    print(operator_tensor[1,1,1,1,1,1,1,1])
    print(mpo.retrieve((1,1,1,1), (1,1,1,1)))
    print(operator_tensor[0,0,0,0,0,0,0,0])
    print(mpo.retrieve((0,0,0,0), (0,0,0,0)))
    print(operator_tensor[1,1,1,0,1,1,1,1])
    print(mpo.retrieve((1,1,1,0), (1,1,1,1)))


    print([s.shape for s in mpo.sites])

    errors = []
    for (a,b,c,d ,e,f,g,h) in itertools.product(range(3), range(3), range(3), range(3)    , range(4), range(4), range(4), range(4)):
        error = abs(operator_tensor[a,b,c,d,e,f,g,h] - mpo.retrieve((a,b,c,d), (e,f,g,h))[0][0])
        errors.append(error)
    plt.plot(errors)
    plt.show()