from __future__ import annotations

import gc
import itertools
from typing import Tuple, List, Union

import numpy as np
from scipy import linalg

from opt_einsum import contract

# from syngular.tensor.matrix_product_operator import MatrixProductOperator

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
        
        self.parameters_number = 0
        self.real_parameters_number = 0
        
        if tensor is not None:
            self.tensor = tensor
            self.tensor_shape = tensor.shape
            self.bond_shape = bond_shape

            self.order = len(self.tensor_shape)
            self.real_parameters_number = np.prod(self.tensor_shape)

            self.input_shape = self.tensor_shape

            self.sites_number = len(self.bond_shape)+1
            self.sites = [None] * self.sites_number

            self.norm = None

            if self.order != self.sites_number:
                raise Exception("dimensions of bond indices do not match order - 1")
     
            if self.bond_shape != ():
                self.shape = [(1, self.tensor_shape[0],self.bond_shape[0])]
                self.shape += [(self.bond_shape[i-1], self.tensor_shape[i], self.bond_shape[i]) for i in range(1, self.sites_number-1)]
                self.shape += [(self.bond_shape[self.sites_number-2], self.tensor_shape[self.sites_number-1], 1)]
            else:
                self.shape = [(1, self.tensor_shape[0], 1)]
        
        self.verbose = verbose
        self.decomposed = False


    
    """__add__ method of MatrixProductState
    Compute the addition of two MatrixProductState


    @type mp: MatrixProductState
    @param mp: A matrix product state to sum with

    @rtype: MatrixProductState
    @returns: a float representing the sum of the two MatrixProductState
    """
    def __add__(self, mps):
        if self.decomposed and mps.decomposed:
            sites = []
            for idx in range(self.sites_number):
                wing_left = idx == 0
                wing_right = idx == self.sites_number-1

                site = np.zeros(shape=(
                    self.shape[idx][0] + mps.shape[idx][0] if not wing_left else self.shape[idx][0],
                    self.shape[idx][1],
                    self.shape[idx][2] + mps.shape[idx][2] if not wing_right else self.shape[idx][2]
                ))

                for inp in range(self.shape[idx][1]):
                    left_matrix = self.sites[idx][:, inp, :]
                    right_matrix = mps.sites[idx][:, inp, :]
                    if wing_left:
                        site[:, inp, :] = np.block([left_matrix, right_matrix])
                    elif wing_right:
                        site[:, inp, :] = np.block([left_matrix.T, right_matrix.T]).T
                    else:
                        site[:, inp, :] = linalg.block_diag(left_matrix, right_matrix)

                sites.append(site)
            
            return MatrixProductState.from_sites(sites)
        else:
            raise Exception("Both Matrix Product Operator must be in canonical form (use .decompose()")


    
    """__or__ method of MatrixProductState
    Compute the inner product of two MatrixProductState


    @type mp: MatrixProductState
    @param mp: A matrix product state to compute the inner product with

    @rtype: float
    @returns: a float representing the inner product
    """
    def __or__(self, mp):

        if isinstance(mp, MatrixProductState):
            struct = []
            n = self.sites_number
            for idx in range(n):
                struct += [
                    self.sites[idx], [idx,n+idx+1,idx+1], 
                    mp.sites[idx], [2*n+idx+1, n+idx+1, 2*n+idx+2]
                ]
            r = contract(*struct)
            return r.reshape(1)[0]
        else:
            raise Exception("right-hand site must be a MatrixProductState")


    """
    @type dim: int
    @param dim: the compression dimension

    @rtype: MatrixProductState
    @returns: a compressed MatrixProductState
    """
    def __rshift__(self, dim: int):
        if isinstance(dim, int):
            return self.compress(dim, strict=True)
        else:
            raise Exception("dimension should be an integer")



    """__getitem__ method of MatrixProductState
    Retrieve a tensor component of the MatrixProductState with the indices


    @type key: Union[List, Tuple]
    @param key: a list or tuple containing input indices

    @rtype: float
    @returns: a float corresponding to the corresponding tensor component 
    """
    def __getitem__(self, key: Union[List, Tuple]):
        if len(key) != self.sites_number:
            raise Exception("input indices do not match the number of sites")

        return self.retrieve(key)
    
    """__repr__ method of MatrixProductState
    Representation of the MatrixProductState for printing functions

    @rtype: str
    @returns: the MatrixProductState representation
    """
    def __repr__(self) -> str:
        repr = "<Matrix Product State> \n> Sites shape" + str(self.shape) + "\n"
        repr += "\t"+"|   " * self.sites_number + "\n"
        repr += "\t"+("O---" * (self.sites_number-1)) + "O" + "\n"
        return repr

    @staticmethod
    def random(input_shape, bond_shape):
        tensor = np.random.normal(size=input_shape)
        return MatrixProductState(tensor, bond_shape=bond_shape).decompose()


    """__from_sites__ staticmethod of MatrixProductState
    Create a MatrixProductState from sites


    @type sites: List[np.ndarray]
    @param sites: sites to convert into a MatrixProductState

    @rtype: MatrixProductState
    @returns: the MatrixProductState generated from sites
    """
    @staticmethod
    def from_sites(sites: List[np.ndarray], orthogonality=None, real_parameters_number=None):
        mp = MatrixProductState()
        mp.sites = sites.copy()
        mp.sites_number = len(sites)
        # mp.order = len()

        mp.decomposed = True
        mp.orthonormalized = orthogonality
        
        mp.parameters_number = np.sum([np.prod(site.shape) for site in sites])
        mp.real_parameters_number = real_parameters_number

        mp.input_shape, mp.bond_shape = (), ()
        mp.shape = [None] * mp.sites_number

        for idx in range(mp.sites_number):
            site = sites[idx]
            shape = site.shape
            mp.shape[idx] = shape
            mp.input_shape  += (shape[1],)
            if idx != mp.sites_number-1: mp.bond_shape   += (shape[2],)

        return mp
    
    
    @staticmethod
    def empty():
        return MatrixProductState()

    @staticmethod
    def zeros(input_shape, bond_shape):
        n = len(input_shape)

        if bond_shape != ():
            shape = [(1, input_shape[0],bond_shape[0])]
            shape += [(bond_shape[i-1], input_shape[i], bond_shape[i]) for i in range(1, n-1)]
            shape += [(bond_shape[n-2], input_shape[n-1], 1)]
        else:
            shape = [(1, input_shape[0], 1)]
        # print(shape)
        sites = []
        for idx in range(n):
            sites.append(np.zeros(shape=shape[idx]))
        return MatrixProductState.from_sites(sites)
    
    """dot method of MatrixProductState
    Compute the norm of the MatrixProductState

    @rtype: float
    @returns: the MatrixProductState norm
    """
    def dot(self):
        return np.sqrt(self | self)

    """normalize method of MatrixProductState
    Normalize the MatrixProductState

    @rtype: MatrixProductState
    @returns: self
    """
    def normalize(self):
        norm = np.linalg.norm(self.sites[-1].reshape((self.sites[-1].shape[0], -1)))
        self.sites[-1] /= norm

        return self
    
    """to_tensor method of MatrixProductState
    Retrieve the based tensor of the MatrixProductState
    (computationally expensive)

    @rtype: np.ndarray
    @returns: the corresponding numpy tensor
    """
    def to_tensor(self):
        tensor = np.zeros(shape=self.input_shape, dtype=np.complex)

        range_inp = [range(inp) for inp in self.input_shape]
        
        for inp in itertools.product(*range_inp):
            tensor[inp] = self[inp]
        
        return tensor
    
    
    """copy method of MatrixProductState
    Copy the MatrixProductState

    @rtype: MatrixProductState
    @returns: a copy of the MPS
    """
    def copy(self):
        return MatrixProductState.from_sites(self.sites)

    
    """decompose method of MatrixProductState
    Decompose (left/right normalization)


    @type mode: "left" | "right"
    @param mode: mode of orthonormalization (left/rigth)

    @rtype: MatrixProductState
    @returns: self
    """
    def decompose(self, mode="left"):
        if not self.decomposed:
            if mode == "left":

                n = self.sites_number-1
                T = self.tensor

                
                for k in range(self.sites_number-1):
                    
                    L = self.left_matricization(T, index=k)
                    Q, R = np.linalg.qr(L, mode="complete")

                    rank = self.shape[k][2]
                    Q = Q[:,:rank]
                    R = R[:rank, :]

                    self.sites[k] = self.tensoricization(Q, k)
                    T = R

                    self.parameters_number += np.prod(self.sites[k].shape)

                self.sites[-1] = self.tensoricization(T, n)
                self.parameters_number += np.prod(self.sites[-1].shape)

                del self.tensor
                gc.collect()

            elif mode == "right":

                n = self.sites_number-1
                T = self.tensor

                for k in range(self.sites_number-1,0,-1):

                    L = self.right_matricization(T, index=k).T
                    Q, R = np.linalg.qr(L, mode="complete")
                    
                    rank = self.shape[k][0]
                    Q = Q[:,:rank].T
                    R = R[:rank, :].T

                    self.sites[k] = self.tensoricization(Q, k)
                    T = R

                    self.parameters_number += np.prod(self.sites[k].shape)

                self.sites[0] = self.tensoricization(T, 0)
                self.parameters_number += np.prod(self.sites[0].shape)

                del self.tensor
                gc.collect()

            self.decomposed = True
        return self

    """compress method of MatrixProductState
    Compression of the MatrixProductStates QR algorithm


    @type dim: int
    @param dim: dimension of the compression < current bond dimension

    @rtype: MatrixProductState
    @returns: compressed MatrixProductState
    """
    def compress(self, dim: int, mode="left", strict=False):
        n = self.sites_number-1
        parameters_number = 0
        
        # print(dim, min(self.bond_shape), self.bond_shape)
        if dim >= min(self.bond_shape):
            return self

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
                    
                    # print(Q.shape, L.shape, S.T.shape, R.shape)
                    # print(W.shape)

                    l1 = list(self.shape[k])
                    l2 = list(self.shape[k+1])
                    l1[2] = dim
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
                    l2[2] = dim
                    self.shape[k] = tuple(l1)
                    self.shape[k-1] = tuple(l2)

                    self.sites[k] = self.tensoricization(Q.T, k)
                    self.sites[k-1] = self.tensoricization(W, k-1)

                    parameters_number += np.prod(self.sites[k].shape)

                parameters_number += np.prod(self.sites[0].shape)
                self.parameters_number = parameters_number

        else:
            that = self.copy()

            # print(that)

            if mode == 'left':
                # if self.orthonormalized != 'left': self.left_orthonormalization()
                for k in range(n):
                    L = that.left_site_matricization(k)
                    R = that.right_site_matricization(k+1)
                    
                    Q, S = np.linalg.qr(L, mode="complete")

                    Q = Q[:,:dim]
                    S = S[:dim, :]

                    W = S @ R
                    
                    # print(Q.shape, L.shape, S.T.shape, R.shape)
                    # print(W.shape)

                    l1 = list(that.shape[k])
                    l2 = list(that.shape[k+1])
                    l1[2] = dim
                    l2[0] = dim
                    that.shape[k] = tuple(l1)
                    that.shape[k+1] = tuple(l2)

                    that.sites[k] = that.tensoricization(Q, k)
                    that.sites[k+1] = that.tensoricization(W, k+1)

                    parameters_number += np.prod(that.sites[k].shape)

                parameters_number += np.prod(that.sites[-1].shape)
                that.parameters_number = parameters_number

            return that

    """apply method of MatrixProductState
    Apply an operator (for now MatrixProductOperator) to the MatrixProductState


    @type operator: MatrixProductOperator
    @param operator: the MPO operator to apply
    
    @type index: int
    @param index: the MPO operator to apply

    @type mode: "compress" | "normal" | "density"
    @param mode: Mode of application of the operator by compressing the bond dimensions or not 
                 and using an optimized algorithm

    @rtype: MatrixProductState
    @returns: the transformation of the MatrixProductState by the MatrixProductOperator
    """
    def apply(self, operator: MatrixProductOperator, index, strict=True, mode="compress"):
        if strict: that = self.copy()
        # print("> Apply")
        m = len(operator.shape) // 2
        n = that.sites_number

        # index = n-1-index
        # print(">",n, m, index)

        struct = []
        struct += [operator, [*list(range(3*n+index, 3*n+m+index)), *list(range(2*n+index, 2*n+m+index))]]
        for idx in range(index, m+index):
            struct += [that.sites[idx], [idx, 2*n+idx, idx+1]]
        struct += [[index, *list(range(3*n+index, 3*n+m+index)), m+index]]
        # print(struct)

        T = contract(*struct)
        # print('T', T.shape)
        # print('op', operator.shape)

        # print(index, m, m+index-1, list(range(index, m+index-1)))
        for k in range(index, m+index-1):
            # print("K", k)
            lrank = T.shape[0]
            input_shape = T.shape[1]

            L = T.reshape(input_shape*lrank, -1)
            
            Q, R = np.linalg.qr(L, mode="complete")
            rank = that.shape[k][2]
            Q = Q[:,:rank]
            R = R[:rank, :]
            # print(k, m+k, 'L', L.shape, 'Q', Q.shape, 'R', R.shape)
            # print(that.sites[k].shape)
            # print(m+k-1, operator.shape, rank)
            that.sites[k] = that.tensoricization(Q, k)
            T = R.reshape((rank, -1, that.shape[k+1][2]))
            # print('Tr', T.shape)
        
        # print(that.sites[m+index-1])
        # print('---')
        # print(T)
        # print(that.sites[m+index-1].shape)
        # print(m+index-1)
        that.sites[m+index-1] = that.tensoricization(T, m+index-1)
        # print(that.sites[m+index-1].shape)
        
        return that

    """retrieve method of MatrixProductState
    Fixe physical indices to retrieve the real/complex value of the tensor  


    @type indices: List[int]
    @param indices: physicial indices list to be fixed

    @rtype: float
    @returns: the selected coefficient of the MatrixProductState 
    """
    def retrieve(self, indices):
        einsum_structure = []
        for idx in range(self.sites_number):
            einsum_structure.append(self.sites[idx][:, indices[idx], :])
            einsum_structure.append([idx, Ellipsis, idx+1])
        return contract(*einsum_structure)

    
    def left_orthonormalization(self, bond_shape=()):
        for k in range(self.sites_number-1):
            print(k, k+1)
            L = self.left_site_matricization(k)
            R = self.right_site_matricization(k+1)

            U, B = np.linalg.qr(L)
            W = B @ R

            self.sites[k] = self.tensoricization(U, k)
            self.sites[k+1] = self.tensoricization(W, k+1)
        
        self.orthonormalized = 'left'
    
    def right_orthonormalization(self, bond_shape=()):
        for k in range(self.sites_number-1, 0, -1):
            
            R = self.right_site_matricization(k)
            L = self.left_site_matricization(k-1)
            
            V, U = np.linalg.qr(R.T)
            W = L @ U.T

            self.sites[k] = self.tensoricization(V.T, k)
            self.sites[k-1] = self.tensoricization(W, k-1)

        self.orthonormalized = 'right'


    def left_site_matricization(self, index):
        return self.left_matricization(self.sites[index], index)

    def right_site_matricization(self, index):
        return self.right_matricization(self.sites[index], index)

    def left_matricization(self, matrix: Union[np.ndarray, List] = None, index: int = 0):
        if matrix is None:
            lrank = self.shape[index][0]
            rrank = self.shape[index][2]
                
            input_shape = self.shape[index][1]

            return np.reshape(self.sites[index], newshape=(input_shape*lrank, rrank))
        else:
            lrank = self.shape[index][0]
            input_shape = self.shape[index][1]

            return np.reshape(matrix, newshape=(input_shape*lrank, -1))

    def right_matricization(self, matrix: Union[np.ndarray, List] = None, index: int = 0):
        if matrix is None:
            lrank = self.shape[index][0]
            rrank = self.shape[index][2]
            input_shape = self.shape[index][1]

            return np.reshape(self.sites[index], newshape=(lrank, input_shape*rrank))
        else:
            rrank = self.shape[index][2]
            input_shape = self.shape[index][1]

            return np.reshape(matrix, newshape=(-1, input_shape*rrank))
    
    def tensoricization(self, matrix, index):
        
        lrank = self.shape[index][0]
        rrank = self.shape[index][2]
        input_shape = self.shape[index][1]

        return np.reshape(matrix, newshape=(lrank, input_shape, rrank))

    def left_orthogonality(self, index):
        L = self.left_matricization(self.sites[index], index)
        return L.T @ L
    
    def right_orthogonality(self, index):
        R = self.right_matricization(self.sites[index], index)
        return R @ R.T

    def grad(self, index):
        return self.sites[:index]+self.sites[index+2:]