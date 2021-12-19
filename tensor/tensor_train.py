import gc
import itertools
from typing import Tuple, List, Union

import numpy as np
from scipy import linalg

import matplotlib.pyplot as plt

class MatrixProductState:

    """__init__ method of MatrixProductState
    Setting up the main property of the instance of MPS such as its number of cores/sites


    @type tensor: np.ndarray
    @param tensor: A tensor to represent as a tensor-train / MPS
        (default is None)
    @type bond_shapes: Union[Union[Tuple, List], np.ndarray]
    @param bond_shapes: Bond indices shape for the MPS
        (default is ())
    @type verbose: int
    @param verbose: Verbose value (0 or 1)
        (default is 0)
    """
    def __init__(self, tensor=None, bond_shape: Union[Union[Tuple, List], np.ndarray]=(), verbose=0) -> None:
        if tensor is not None:
            self.tensor = tensor
            self.tensor_shape = tensor.shape
            self.bond_shape = bond_shape

            self.sites_number = len(self.bond_shape)+1
            self.sites = [None] * self.sites_number
            
            self.shape = [(1, self.tensor_shape[0],self.bond_shape[0])]
            self.shape += [(self.bond_shape[i-1], self.tensor_shape[i], self.bond_shape[i]) for i in range(1, self.sites_number-1)]
            self.shape += [(self.bond_shape[self.sites_number-2], self.tensor_shape[self.sites_number-1], 1)]
            
        self.verbose = verbose
        self.decomposed = False

    def __or__(self, mp):

        if isinstance(mp, MatrixProductState):
            struct = []
            n = self.sites_number
            for idx in range(n):
                struct += [
                    self.sites[idx], [idx,n+idx+1,idx+1], 
                    mp.sites[idx], [2*n+idx+1, n+idx+1, 2*n+idx+2]
                ]
            r = np.einsum(*struct)
            return r.reshape(1)[0]
        else:
            raise Exception("right-hand site must be a MatrixProductState")

    
    def __getitem__(self, key):

        if len(key) != self.sites_number:
            raise Exception("input indices do not match the number of sites")

        return self.retrieve(key)
    
    def __repr__(self) -> str:
        repr = "<Matrix Product State> \n> Sites shape" + str(self.shape) + "\n"
        repr += "\t"+"|   " * self.sites_number + "\n"
        repr += "\t"+("O---" * (self.sites_number-1)) + "O" + "\n"
        return repr


    @staticmethod
    def from_sites(sites):
        mp = MatrixProductState()
        mp.sites = sites
        mp.sites_number = len(sites)

        mp.decomposed = True

        mp.input_shape, mp.bond_shape = (), ()
        mp.shape = [None] * mp.sites_number

        for idx in range(mp.sites_number):
            site = sites[idx]
            shape = site.shape
            mp.shape[idx] = shape
            mp.input_shape  += (shape[1],)
            if idx != mp.sites_number-1: mp.bond_shape   += (shape[2],)

        return mp

    def norm(self):
        return self | self
    
    def to_tensor(self):
        tensor = np.zeros(shape=self.input_shape)

        range_inp = [range(inp) for inp in self.input_shape]
        
        for inp in itertools.product(*range_inp):
                tensor[inp] = self[inp]
        
        return tensor

    def decompose(self, mode="left"):
        if not self.decomposed:
            if mode == "left":
                current_matrix = self.tensor
                current_shape = self.shape[0][1]
                current_rank = 1

                for k in range(self.sites_number-1):
                    
                    unfolding_matrix = np.reshape(current_matrix, newshape=(current_shape*current_rank, -1))
                    rank = self.shape[k][2]
                    
                    Q, R = np.linalg.qr(unfolding_matrix, mode="complete")

                    Q = Q[:,:rank]
                    Q = np.reshape(Q, newshape=(current_rank, current_shape, -1))
                    R = R[:rank, :]

                    if self.verbose: print(f"Core {k} with {current_rank} , {current_shape}")

                    self.sites[k] = Q

                    current_shape = self.shape[k+1][1]
                    current_rank = rank
                    current_matrix = R

                current_matrix = current_matrix[:, :, np.newaxis]
                self.sites[-1] = current_matrix
                
                del self.tensor
                gc.collect()

            elif mode == "right":
                current_matrix = self.tensor.T
                current_shape = self.shape[0][0]
                current_rank = 1

                self.decompose(mode="left")

            self.decomposed = True
        return self

    

    def retrieve(self, indices):
        einsum_structure = []
        for idx in range(self.sites_number):
            einsum_structure.append(self.sites[idx][:, indices[idx], :])
            einsum_structure.append([idx, Ellipsis, idx+1])

        return np.einsum(*einsum_structure)

























class MatrixProductOperator:

    def __init__(self, tensor=None, bond_shape: Union[Union[Tuple, List], np.ndarray]=(), verbose=0) -> None:
        
        self.parameters_number = 0
        self.real_parameters_number = 0

        if tensor is not None:
            self.tensor = tensor
            self.tensor_shape = tensor.shape
            self.bond_shape = tuple(bond_shape)

            self.real_parameters_number = np.prod(self.tensor_shape)

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
        self.orthonormalized = None

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

    def __mul__(self, mp):
        print(mp)
        if not self.decomposed and mp.decomposed:
            raise Exception("Both Matrix Product Operator must be in canonical form (use .decompose()")

        sites = []

        if isinstance(mp, MatrixProductOperator):
            for idx in range(self.sites_number):
                wing_left = idx == 0
                wing_right = idx == self.sites_number-1

                site = np.zeros(shape=(
                    self.shape[idx][0] * mp.shape[idx][0],
                    self.shape[idx][1],
                    self.shape[idx][2],
                    self.shape[idx][3] * mp.shape[idx][3]
                ))

                for inp in range(self.shape[idx][1]):
                    for out in range(self.shape[idx][2]):
                        left_matrix = self.sites[idx][:, inp, out, :]
                        right_matrix = mp.sites[idx][:, inp, out, :]
                        
                        site[:, inp, out, :] = np.kron(left_matrix, right_matrix)
            
                sites.append(site)
            
            return MatrixProductOperator.from_sites(sites)

        elif isinstance(mp, MatrixProductState):
            pass
        else:
            raise Exception("left hand-side must be either a MatrixProductState or a MatrixProductOperator")

    def __matmul__(self, mp):
        sites = []

        if isinstance(mp, MatrixProductState):
            for idx in range(self.sites_number):
                site = np.einsum(mp.sites[idx], [1,2,3], self.sites[idx], [4,2,6,5], [1,4,6,3,5])

                site = site.reshape((
                    self.shape[idx][0]*mp.shape[idx][0],
                    self.shape[idx][2],
                    self.shape[idx][3]*mp.shape[idx][2]
                ))
                sites.append(site)
            return MatrixProductState.from_sites(sites)
        elif isinstance(mp, MatrixProductOperator):
            for idx in range(self.sites_number):
                site = np.einsum(self.sites[idx], [1,2,3,4], mp.sites[idx], [5,3,6,7], [1,5,2,6,4,7])

                site = site.reshape((
                    self.shape[idx][0]*mp.shape[idx][0],
                    self.shape[idx][1],
                    self.shape[idx][2],
                    self.shape[idx][3]*mp.shape[idx][3]
                ))
                sites.append(site)
            return MatrixProductOperator.from_sites(sites)

    def __mod__(self, mode):
        if mode == 'L' or mode == 'left' or mode == 'l' or mode == 'left-canonical':
            for idx in range(self.sites_number-1):
                site = self.sites[idx]

                # unfolding_site = np.reshape(site, newshape=())
                # unfolding_matrix = np.reshape(current_matrix, newshape=(current_input_shape*current_output_shape*current_rank, -1))
                # rank_right = self.shape[k][3]

                # Q, R = np.linalg.qr(unfolding_matrix, mode="complete")

                # Q = Q[:,:rank_right]
                # Q = np.reshape(Q, newshape=(current_rank, current_input_shape, current_output_shape, rank_right))
                # R = R[:rank_right, :]

                # self.sites[k] = Q

                # current_input_shape = self.shape[k][1]
                # current_output_shape = self.shape[k][2]
                # current_rank = rank_right
                # current_matrix = R
        elif mode == 'R' or mode == 'right' or mode == 'r' or mode == 'right-canonical':
            pass
        else:
            raise Exception("when calling mod (%) mode should be left-canonical or right-canonical alias")

    def __rshift__(self, dim):
        if isinstance(dim, int):
            return self.compress(dim, strict=True)
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
    def from_sites(sites, orthogonality=None, real_parameters_number=None):
        mp = MatrixProductOperator()
        mp.sites = sites
        mp.sites_number = len(sites)

        mp.decomposed = True
        mp.orthonormalized = orthogonality

        mp.parameters_number = np.sum([np.prod(site.shape) for site in sites])
        mp.real_parameters_number = real_parameters_number

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

    def copy(self):
        return MatrixProductOperator.from_sites(self.sites)

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

                n = self.sites_number-1
                T = self.tensor

                for k in range(n):
                    L = self.left_matricization(T, index=k)
                    
                    Q, R = np.linalg.qr(L, mode="complete")

                    rank_right = self.shape[k][3]
                    Q = Q[:,:rank_right]
                    R = R[:rank_right, :]

                    self.sites[k] = self.tensoricization(Q, k)
                    T = R

                    self.parameters_number += np.prod(self.sites[k].shape)

                self.sites[-1] = self.tensoricization(T, n)
                self.parameters_number += np.prod(self.sites[-1].shape)

                del self.tensor
                gc.collect()
            
            elif mode == "right":
                if self.verbose == 1: print(self.sites_number)
                axes = list(range(self.sites_number-1, -1, -1)) + list(range(2*self.sites_number-1, self.sites_number-1, -1))
                if self.verbose == 1: print("axes", axes)
                current_matrix = np.transpose(self.tensor, axes=axes)

                self.decompose(mode="left")

                self.sites = self.sites[::-1]
            
            self.decomposed = True

        return self

    def compress(self, dim, mode='left', strict=False):
        n = self.sites_number-1
        parameters_number = 0

        if not strict:
            if mode == 'left':
                if self.orthonormalized != 'left': self.left_orthonormalization()

                for k in range(n):
                    L = self.left_site_matricization(k)
                    R = self.right_site_matricization(k+1)
                    
                    Q, S = np.linalg.qr(L, mode="complete")

                    Q = Q[:,:dim]
                    S = S[:dim, :]

                    W = S @ R
                    
                    print(Q.shape, L.shape, S.T.shape, R.shape)
                    print(W.shape)

                    l1 = list(self.shape[k])
                    l2 = list(self.shape[k+1])
                    l1[3] = dim
                    l2[0] = dim
                    self.shape[k] = tuple(l1)
                    self.shape[k+1] = tuple(l2)

                    self.sites[k] = self.tensoricization(Q, k)
                    self.sites[k+1] = self.tensoricization(W, k+1)

                    parameters_number += np.prod(self.sites[k].shape)

                parameters_number += np.prod(self.sites[-1].shape)
                self.parameters_number = parameters_number
            elif mode == 'right':
                
                if self.orthonormalized != 'right': self.right_orthonormalization()

                for k in range(n, 0, -1):
                    R = self.right_site_matricization(k)
                    L = self.left_site_matricization(k-1)

                    Q, S = np.linalg.qr(R.T)

                    Q = Q[:,:dim]
                    S = S[:dim, :]


                    W = L @ S.T

                    l1 = list(self.shape[k])
                    l2 = list(self.shape[k-1])
                    l1[0] = dim
                    l2[3] = dim
                    self.shape[k] = tuple(l1)
                    self.shape[k-1] = tuple(l2)

                    self.sites[k] = self.tensoricization(Q.T, k)
                    self.sites[k-1] = self.tensoricization(W, k-1)

                    parameters_number += np.prod(self.sites[k].shape)

                parameters_number += np.prod(self.sites[0].shape)
                self.parameters_number = parameters_number

                # print([site.shape for site in self.sites])
        else:
            that = self.copy()

            print(that)

            if mode == 'left':
                # if self.orthonormalized != 'left': self.left_orthonormalization()
                for k in range(n):
                    L = that.left_site_matricization(k)
                    R = that.right_site_matricization(k+1)
                    
                    Q, S = np.linalg.qr(L, mode="complete")

                    Q = Q[:,:dim]
                    S = S[:dim, :]

                    W = S @ R
                    
                    print(Q.shape, L.shape, S.T.shape, R.shape)
                    print(W.shape)

                    l1 = list(that.shape[k])
                    l2 = list(that.shape[k+1])
                    l1[3] = dim
                    l2[0] = dim
                    that.shape[k] = tuple(l1)
                    that.shape[k+1] = tuple(l2)

                    that.sites[k] = that.tensoricization(Q, k)
                    that.sites[k+1] = that.tensoricization(W, k+1)

                    parameters_number += np.prod(that.sites[k].shape)

                parameters_number += np.prod(that.sites[-1].shape)
                that.parameters_number = parameters_number

            return that

    def apply(self, operator, indices):
        if self.decomposed:
            if isinstance(operator, MatrixProductOperator):
                n = self.sites_number
                m = len(indices)
                struct = []
                for idx in range(m):
                    jdx = indices[idx]
                    
                    struct += [self.sites[jdx], [jdx+1, n+jdx+2, 2*n+jdx+2, jdx+2]]
                    struct += [operator.sites[idx], [3*n+jdx+2, 2*n+jdx+2, 4*n+jdx+3, 3*n+jdx+3]]
                
                output_shape = [
                    [indices[0]+1, 3*n+indices[0]+2] + 
                    list(range(n+indices[0]+2,n+indices[0]+m+2)) + 
                    list(range(4*n+indices[0]+3,4*n+indices[0]+m+3)) + 
                    [indices[-1]+2, 3*n+indices[-1]+3]
                ]
                struct += output_shape
                # print(struct)

                T = np.squeeze(np.einsum(*struct), axis=(1, len(output_shape)-2))

                for idx in range(m-1):
                    jdx = indices[idx]
                    lrank_shape = self.sites[jdx].shape[0]
                    input_shape = self.sites[jdx].shape[1]
                    output_shape = operator.sites[idx].shape[2]

                    L = np.reshape(T, (lrank_shape*input_shape*output_shape, -1))
                    
                    Q, R = np.linalg.qr(L, mode="complete")

                    rank_right = self.sites[jdx].shape[3]
                    Q = Q[:,:rank_right]
                    R = R[:rank_right, :]

                    self.sites[jdx] = np.reshape(Q, (lrank_shape, input_shape, output_shape, rank_right))
                    T = R

                lrank_shape = self.sites[indices[-1]].shape[0]
                input_shape = self.sites[indices[-1]].shape[1]
                output_shape = operator.sites[m-1].shape[2]
                rank_right = self.sites[indices[-1]].shape[3]

                self.sites[indices[-1]] = np.reshape(T, (lrank_shape, input_shape, output_shape, rank_right))
            else:
                pass
        else:
            raise Exception("MatrixProductState not decomposed")

    @staticmethod
    def split():
        pass

    def retrieve(self, input_indices, output_indices):
        einsum_structure = []

        for idx in range(self.sites_number):
            einsum_structure.append(self.sites[idx][:, input_indices[idx], output_indices[idx], :])
            einsum_structure.append([idx, idx+1])

        return np.einsum(*einsum_structure)

    
    def left_orthonormalization(self, bond_shape=()):
        for k in range(self.sites_number-1):
            print(k, k+1)
            L = self.left_site_matricization(k)
            R = self.right_site_matricization(k+1)

            U, B = np.linalg.qr(L)

            W = B @ R

            self.sites[k] = self.tensoricization(U, k)
            self.sites[k+1] = self.tensoricization(W, k+1)
        
        # L = self.left_site_matricization(-1)
        # V, U = np.linalg.qr(L)
        # self.sites[-1] = self.tensoricization(V, -1)

        self.orthonormalized = 'left'
    
    def right_orthonormalization(self, bond_shape=()):
        for k in range(self.sites_number-1, 0, -1):
            
            R = self.right_site_matricization(k)
            L = self.left_site_matricization(k-1)
            
            V, U = np.linalg.qr(R.T)
            # U, S, V = np.linalg.svd(R, full_matrices=False)
            W = L @ U.T
            # W = L @ (U * S)

            self.sites[k] = self.tensoricization(V.T, k)
            self.sites[k-1] = self.tensoricization(W, k-1)

        # R = self.right_site_matricization(0)
        # V, U = np.linalg.qr(R.T)
        # self.sites[0] = self.tensoricization(V.T, 0)

        self.orthonormalized = 'right'

    def left_site_matricization(self, index):
        return self.left_matricization(self.sites[index], index)

    def right_site_matricization(self, index):
        return self.right_matricization(self.sites[index], index)

    def left_matricization(self, matrix: Union[np.ndarray, List] = None, index: int = 0):
        if matrix is None:
            lrank = self.shape[index][0]
            rrank = self.shape[index][3]
                
            input_shape = self.shape[index][1]
            output_shape = self.shape[index][2]

            return np.reshape(self.sites[index], newshape=(input_shape*output_shape*lrank, rrank))
        else:
            lrank = self.shape[index][0]
                
            input_shape = self.shape[index][1]
            output_shape = self.shape[index][2]

            return np.reshape(matrix, newshape=(input_shape*output_shape*lrank, -1))

    def right_matricization(self, matrix: Union[np.ndarray, List] = None, index: int = 0):
        if matrix is None:
            lrank = self.shape[index][0]
            rrank = self.shape[index][3]
                
            input_shape = self.shape[index][1]
            output_shape = self.shape[index][2]

            return np.reshape(self.sites[index], newshape=(lrank, input_shape*output_shape*rrank))
        else:
            rrank = self.shape[index][3]
                
            input_shape = self.shape[index][1]
            output_shape = self.shape[index][2]

            return np.reshape(matrix, newshape=(-1, input_shape*output_shape*rrank))
    
    def tensoricization(self, matrix, index):
        
        lrank = self.shape[index][0]
        rrank = self.shape[index][3]
            
        input_shape = self.shape[index][1]
        output_shape = self.shape[index][2]

        return np.reshape(matrix, newshape=(lrank, input_shape, output_shape, rrank))

    def left_orthogonality(self, index):
        L = self.left_matricization(self.sites[index], index)
        return L.T @ L
    
    def right_orthogonality(self, index):
        R = self.right_matricization(self.sites[index], index)
        return R @ R.T













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