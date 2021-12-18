import gc
import itertools
from typing import Tuple, List, Union

import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt

class MatrixProductState:

    def __init__(self, tensor, bond_shape) -> None:
        self.tensor = tensor
        self.tensor_shape = tensor.shape
        self.bond_shape = bond_shape

        self.sites_number = len(self.bond_shape)+1
        self.sites = [None] * self.sites_number

        # print(self.bond_shape, self.sites_number)
        
        self.shape = [(1, self.tensor_shape[0],self.bond_shape[0])]
        self.shape += [(self.bond_shape[i-1], self.tensor_shape[i], self.bond_shape[i]) for i in range(1, self.sites_number-1)]
        self.shape += [(self.bond_shape[self.sites_number-2], self.tensor_shape[self.sites_number-1], 1)]


        self.shape = [(self.tensor_shape[i], self.bond_shape[i]) for i in range(self.sites_number-1)]

    def decompose(self, mode="left"):

        if mode == "left":
            current_matrix = self.tensor
            current_shape = self.shape[0][0]
            current_rank = 1

            # print(self.shape)
            for k in range(self.sites_number-1):
                current_shape = self.shape[k][0]
                
                unfolding_matrix = np.reshape(current_matrix, newshape=(current_shape*current_rank, -1))
                rank = self.shape[k][1]
                
                Q, R = np.linalg.qr(unfolding_matrix, mode="complete")

                Q = Q[:,:rank]
                Q = np.reshape(Q, newshape=(current_rank, current_shape, -1))
                R = R[:rank, :]

                print(f"Core {k} with {current_rank} , {current_shape}")

                self.sites[k] = Q

                current_rank = rank
                current_matrix = R

            current_matrix = current_matrix[:, :, np.newaxis]
            self.sites[-1] = current_matrix

            # print([y.shape for y in self.sites])

        elif mode == "right":
            current_matrix = self.tensor.T
            current_shape = self.shape[0][0]
            current_rank = 1

            self.decompose(mode="left")

        # print("sites", self.shape)
        return self

    def retrieve(self, indices):
        einsum_structure = []
        for idx in range(self.sites_number):
            # print(indices[idx], self.sites[idx].shape)
            einsum_structure.append(self.sites[idx][:, indices[idx], :])
            einsum_structure.append([idx, Ellipsis, idx+1])

        return np.einsum(*einsum_structure)

    def update(self, sites, value):

        print(self.sites[sites[0]].shape)
        print(self.sites[sites[1]].shape)

        merge_site = np.einsum(self.sites[sites[0]], [1,2,3], self.sites[sites[1]], [3,4,5], [1,2,4,5])

        merge_site = np.einsum(merge_site, [1,2,3,4], value, [5,2,3,6], [1,5,6,4])

        print("shape")
        print(merge_site.shape)

        rank_left, rank_1, rank_2, rank_right = merge_site.shape

        unfolding_matrix = np.reshape(merge_site, newshape=(rank_left*rank_1, -1))

        Q, R = np.linalg.qr(unfolding_matrix, mode="complete")

        Q = Q[:,:rank_left]
        Q = np.reshape(Q, newshape=(rank_left, rank_1, -1))
        R = R[:rank_left, :]
        R = np.reshape(R, newshape=(-1, rank_2, rank_right))

        # print(Q.shape, R.shape)

        self.sites[sites[0]] = Q
        self.sites[sites[1]] = R

    
    def __repr__(self) -> str:
        repr = "<Matrix Product State> \n> Sites shape" + str(self.shape) + "\n"
        repr += "\t"+"|   " * self.sites_number + "\n"
        repr += "\t"+("O---" * (self.sites_number-1)) + "O" + "\n"
        return repr


























class MatrixProductOperator:

    def __init__(self, tensor=(), bond_shape: Union[Union[Tuple, List], np.ndarray]=(), verbose=0) -> None:
        if tensor != ():
            self.tensor = tensor
            self.tensor_shape = tensor.shape
            self.bond_shape = tuple(bond_shape)

            self.sites_number = len(self.bond_shape)+1
            self.sites = [None] * self.sites_number

            if self.bond_shape == (): self.bond_shape = (1,)

            self.input_shape    = tuple(self.tensor_shape[:self.sites_number])
            self.output_shape   = tuple(self.tensor_shape[self.sites_number:])

            if len(self.input_shape) != len(self.output_shape):
                raise Exception("input_shape and output_shape of the tensor must have the same length")

            if len(self.input_shape) != len(self.bond_shape)+1:
                raise Exception("dimensions of bond indices do not match input dimension - 1")

            self.shape = [(1, self.tensor_shape[0], self.tensor_shape[self.sites_number], self.bond_shape[0])]
            self.shape += [(self.bond_shape[i-1], self.tensor_shape[i], self.tensor_shape[self.sites_number+i], self.bond_shape[i]) for i in range(1, self.sites_number-1)]
            self.shape += [(self.bond_shape[self.sites_number-2], self.tensor_shape[self.sites_number-1], self.tensor_shape[2*self.sites_number-1], 1)]

            self.shape_indices = [
                (2*self.sites_number+i, i, self.sites_number+i, 2*self.sites_number+i+1) for i in range(self.sites_number)
            ]
            
            self.tensor = np.transpose(self.tensor, axes=sum(list(zip(range(self.sites_number), range(self.sites_number, 2*self.sites_number))), ()))
        
        self.verbose = verbose
        self.decomposed = False

    def __add__(self, mpo):
        print(mpo)
        if self.decomposed and mpo.decomposed:
            sites = []
            for idx in range(self.sites_number):
                wing_left = idx == 0
                wing_right = idx == self.sites_number-1

                site = np.zeros(shape=(
                    self.shape[idx][0] + mpo.shape[idx][0] if not wing_left else self.shape[idx][0],
                    self.shape[idx][1],
                    self.shape[idx][2],
                    self.shape[idx][3] + mpo.shape[idx][3] if not wing_right else self.shape[idx][3]
                ))

                for inp in range(self.shape[idx][1]):
                    for out in range(self.shape[idx][2]):
                        left_matrix = self.sites[idx][:, inp, out, :]
                        right_matrix = mpo.sites[idx][:, inp, out, :]
                        if wing_left:
                            site[:, inp, out, :] = np.block([left_matrix, right_matrix])
                        elif wing_right:
                            site[:, inp, out, :] = np.block([left_matrix.T, right_matrix.T]).T
                        else:
                            site[:, inp, out, :] = linalg.block_diag(left_matrix, right_matrix)

                sites.append(site)
            
            return MatrixProductOperator.from_sites(sites)
        else:
            raise Exception("Both Matrix Product Operator must be in canonical form (use .decompose()")

    def __rshift__(self, dim):
        if isinstance(dim, int):
            return self.compress(dim)
        else:
            raise Exception("dimension should be an integer")

    def __getitem__(self, key):
        key_inp = key[0]
        key_out = key[1]

        if len(key_inp) != self.sites_number:
            raise Exception("input indices do not match the number of sites")
        if len(key_out) != self.sites_number:
            raise Exception("output indices do not match the number of sites")

        return self.retrieve(key_inp, key_out)
    
    def __repr__(self) -> str:
        repr = "<Matrix Product Operator> \n> Sites shape" + str(self.shape) + "\n"
        repr += "\t"+"|   " * self.sites_number + "\n"
        repr += "\t"+("O---" * (self.sites_number-1)) + "O" + "\n"
        repr += "\t"+"|   " * self.sites_number + "\n"
        return repr


    @staticmethod
    def random(input_shape, output_shape, bond_shape):
        tensor = np.random.normal(size=(*input_shape, *output_shape))
        return MatrixProductOperator(tensor, bond_shape=bond_shape).decompose()

    @staticmethod
    def from_sites(sites):
        mp = MatrixProductOperator()
        mp.sites = sites
        mp.sites_number = len(sites)

        mp.decomposed = True

        mp.input_shape, mp.output_shape, mp.bond_shape = (), (), ()
        mp.shape = [None] * mp.sites_number

        for idx in range(mp.sites_number):
            site = sites[idx]
            shape = site.shape
            mp.shape[idx] = shape
            mp.input_shape  += (shape[1],)
            mp.output_shape += (shape[2],)
            if idx != mp.sites_number-1: mp.bond_shape   += (shape[3],)

        return mp

    @staticmethod
    def empty():
        return MatrixProductOperator()

    def to_tensor(self):
        tensor = np.zeros(shape=(*self.input_shape, *self.output_shape))

        range_inp = [range(inp) for inp in self.input_shape]
        range_out = [range(out) for out in self.output_shape]
        
        for inp in itertools.product(*range_inp):
            for out in itertools.product(*range_out):
                tensor[(*inp, *out)] = self[inp, out]
        
        return tensor

    def decompose(self, mode="left"):
        if not self.decomposed:
            if mode == "left":
                current_matrix = self.tensor
                current_input_shape = self.shape[0][1]
                current_output_shape = self.shape[0][2]
                current_rank = 1

                for k in range(self.sites_number-1):
                    
                    unfolding_matrix = np.reshape(current_matrix, newshape=(current_input_shape*current_output_shape*current_rank, -1))
                    rank_right = self.shape[k][3]

                    Q, R = np.linalg.qr(unfolding_matrix, mode="complete")

                    Q = Q[:,:rank_right]
                    Q = np.reshape(Q, newshape=(current_rank, current_input_shape, current_output_shape, rank_right))
                    R = R[:rank_right, :]

                    self.sites[k] = Q

                    current_input_shape = self.shape[k][1]
                    current_output_shape = self.shape[k][2]
                    current_rank = rank_right
                    current_matrix = R

                current_matrix = np.reshape(current_matrix, newshape=(-1, current_input_shape, current_output_shape))[:, :, :, np.newaxis]
                if self.verbose == 1: print(current_matrix.shape)
                self.sites[-1] = current_matrix
            
            elif mode == "right":
                if self.verbose == 1: print(self.sites_number)
                axes = list(range(self.sites_number-1, -1, -1)) + list(range(2*self.sites_number-1, self.sites_number-1, -1))
                if self.verbose == 1: print("axes", axes)
                current_matrix = np.transpose(self.tensor, axes=axes)

                self.decompose(mode="left")

                self.sites = self.sites[::-1]
            
            self.decomposed = True

            del self.tensor
            gc.collect()

        return self

    def compress(self, dim):
        for k in range(self.sites_number-1):
            pass
            # Q, R = np.linalg.qr(unfolding_matrix, mode="complete")


    def retrieve(self, input_indices, output_indices):
        einsum_structure = []

        for idx in range(self.sites_number):
            einsum_structure.append(self.sites[idx][:, input_indices[idx], output_indices[idx], :])
            einsum_structure.append([idx, idx+1])

        return np.einsum(*einsum_structure)

    # def copy(self):
    #     new_mpo = MatrixProductOperator(self.tensor, self.bond_shape)
    #     new_mpo.sites = self.sites 
    #     return new_mpo

    def contract(self, mp, indices=None):
        if isinstance(mp, MatrixProductState):
            for idx in range(self.sites_number):
                self.sites[idx] = np.einsum(self.sites[idx], [1,2,3,4], mp.sites[idx], [5,6,2,7], [5,3,6,7])
        elif isinstance(mp, MatrixProductOperator):
            print("ok")

            '''
            Contraction through the input indices of the current MPO and the output indices of the passed MPO
            > (0, ..., N)

                |   |   |   |
                O---O---O---O
                |   |   |   |
                O---O---O---O
                |   |   |   |
            '''

            for idx in range(self.sites_number):
                self.sites[idx] = np.einsum(self.sites[idx], [1,2,3,4], mp.sites[idx], [5,6,2,7], [5,3,6,7])
        return self

    
    def contract(mp1, mp2, mode="left"):
        sites = [None] * mp1.sites_number
        if isinstance(mp2, MatrixProductState):

            if mode == 'left':
                struct = []
                n = mp2.sites_number

                for idx in range(n):
                    struct += [mp2.sites[idx], [idx+1, n+idx+2, idx+2]]
                    struct += [mp1.sites[idx], [2*n+2+idx, n+idx+2, 3*n+idx+3, 2*n+2+idx+1]]
                struct += [[]]
                # struct += [mp2, list(range(1, self.cores_number+1))]
                # struct += [self.cores[0], [1, self.cores_number+1, 2*self.cores_number+1]]
                
                # for idx in range(2, self.cores_number):
                #     struct += [self.cores[idx-1]]
                #     struct += [[idx, self.cores_number+idx, 2*self.cores_number+1, 2*self.cores_number+2]]

                # struct += [self.cores[-1], [self.cores_number, 2*self.cores_number, 3*self.cores_number-1]]
                # struct += [list(range(self.cores_number+1, 2*self.cores_number+1))]

                
                result = np.einsum(*struct)

            for idx in range(mp1.sites_number):
                # print(mp1._shape[idx], mp2._shape[idx])
                print(mp1.sites[idx].shape, mp2.sites[idx].shape)
                sites[idx] = np.einsum(mp1.sites[idx], [1,2,3,4], mp2.sites[idx], [5,2,6], [5,3,6])
                # p rint(sites[idx].shape)
            
        elif isinstance(mp2, MatrixProductOperator):
            '''
            Contraction through the input indices of the current MPO and the output indices of the passed MPO
            > (0, ..., N)

                |   |   |   |
                O---O---O---O
                |   |   |   |
                O---O---O---O
                |   |   |   |
            '''

            if mode == 'left':
                struct = []
                n = mp2.sites_number

                for idx in range(n):
                    struct += [[idx+1, n+idx+2, 2*n+idx+2, idx+2]]
                    struct += [[3*n+2+idx, 2*n+idx+2, 4*n+idx+3, 3*n+2+idx+1]]
                print("STRUCT", struct)
                result = np.einsum(*struct)

            # for idx in range(mp1.sites_number):
            #     sites[idx] = np.einsum(mp1.sites[idx], [1,2,3,4], mp2.sites[idx], [5,6,2,7])
        else:
            print("Type", type(mp2))
        # print("Sites", sites)
        print("Result", result)
        mp = mp2.copy()
        mp.sites = sites
        return mp

















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