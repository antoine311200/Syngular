import gc
import itertools
from typing import Tuple, List, Union

import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt




































class MatrixProductDensityOperator:

    def __init__(self, tensor, bond_shape) -> None:
        self.tensor = tensor
        self.bond_shape = bond_shape


        self.state = MatrixProductOperator(tensor, self.bond_shape)
        self.state.decompose()

        self.state_conjugate = MatrixProductOperator(tensor.T, self.bond_shape)
        self.state_conjugate.decompose()

        contract_indices = [self.state._shape_indices[i][2] for i in range(self.state.sites_number)]
        contract_conj_indices = [self.state_conjugate._shape_indices[i][2] for i in range(self.state.sites_number)]
        output_shape = []

        self.state.copy().contract(self.state_conjugate)
        # self.mpdo = #np.einsum(self.state, contract_indices, self.state_conjugate, contract_conj_indices, output_shape)


if __name__ == "__main__":

    # t = np.arange(5**7).reshape((5,5,5,5,5,5,5))
    # mps = MatrixProductState(t, bond_shape=(2,2,2,2,2,2))
    # mps.decompose()


    # errors = []
    # for (a,b,c,d,e,f,g) in itertools.product(range(5), range(5), range(5), range(5), range(5), range(5), range(5)):
    #     error = abs(t[a,b,c,d,e,f,g] - mps.retrieve((a,b,c,d,e,f,g))[0][0])
    #     errors.append(error)
    # plt.plot(errors)








    tensor = np.arange(27).reshape((3,3,3))
    print(tensor)
    
    mps = MatrixProductState(tensor, bond_shape=(2,2))
    mps.decompose(mode="left")

    # update = np.arange(3**4).reshape((3,3,3,3))
    # print(mps.update((1,2), update))

    # print("MPS ")
    # print([s.shape for s in mps.sites])
    # print(mps.sites)

    # print(tensor[1,1,1])
    # for (i,j,k) in itertools.product(range(3), range(3), range(3)):
    #     print(mps.retrieve((i,j,k)))


    # errors = []
    # for (a,b,c,) in itertools.product(range(3), range(3), range(3)):
    #     error = abs(tensor[a,b,c] - mps.retrieve((a,b,c))[0][0])
    #     errors.append(error)
    # plt.plot(errors)

    print("-----------")

    operator_tensor = np.arange(3**4 * 4**4).reshape((3,3,3,3,4,4,4,4))
    # print(operator_tensor)

    mpo = MatrixProductOperator(operator_tensor, (4,4,4))
    mpo.decompose(mode="left")

    # mpo2 = mpo.copy()
    # mpo2.decompose(mode="left")
    # print("retrieve >")
    # print(operator_tensor[1,1,1,1,1,1,1,1])
    # print(mpo.retrieve((1,1,1,1), (1,1,1,1)))
    # print(operator_tensor[0,0,0,0,0,0,0,0])
    # print(mpo.retrieve((0,0,0,0), (0,0,0,0)))
    # print(operator_tensor[1,1,1,0,1,1,1,1])
    # print(mpo.retrieve((1,1,1,0), (1,1,1,1)))
    # print(mpo)

    # print([s.shape for s in mpo.sites])

    errors1, errors2 = [], []
    for (a,b,c,d ,e,f,g,h) in itertools.product(range(3), range(3), range(3), range(3)    , range(4), range(4), range(4), range(4)):
        error1 = abs(operator_tensor[a,b,c,d,e,f,g,h] - mpo.retrieve((a,b,c,d), (e,f,g,h))[0][0])
        errors1.append(error1)

        # error2 = abs(operator_tensor[a,b,c,d,e,f,g,h] - mpo2.retrieve((a,b,c,d), (e,f,g,h))[0][0])
        # errors2.append(error2)
    plt.plot(errors1)
    # plt.plot(errors2)
    plt.show()