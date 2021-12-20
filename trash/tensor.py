from __future__ import annotations

from typing import List, Union
from numpy.core.fromnumeric import reshape

import tensorflow as tf
import numpy as np

class TensorTrain:

    def __init__(self, tensor: Union[tf.Tensor, list]) -> None:
        self.tensor = tensor

        self.cores = []

        self.shape = self.tensor.shape
    
    def __add__(self, tensor: tf.Tensor) -> TensorTrain:
        return TensorTrain(self.tensor + tensor)
    
    def factorize(self, index_shape, index_names):
        self.index_names = index_names
        self.index_shape = index_shape

        def mps():
            pass


    def __getitem__(self, index):
        print(index)
        value = self.tensor
        if type(index) == int:
            index = [index]
        while len(index) > 0:
            value = value[index[0]]
            index = index[1:]
        return value


    def decompose_as_mpo(self, input_shape, output_shape, bond_shape):
        self.cores = []

        __current_tt = np.reshape(self.tensor.copy(), (*input_shape, *output_shape), order='F')

        ipt_rank = len(input_shape)
        opt_rank = len(output_shape)
        bnd_rank = len(bond_shape)
        rank = ipt_rank+opt_rank+bnd_rank

        rest_ipt_dim = input_shape
        rest_opt_dim = output_shape
        rest_bnd_dim = bond_shape

        cores_number = ipt_rank

        for idx in range(cores_number-1):
            bnd_dim = bond_shape[idx]
            ipt_dim = input_shape[idx]
            opt_dim = output_shape[idx]

            rest_ipt_dim = rest_ipt_dim[1:]
            rest_opt_dim = rest_opt_dim[1:]
            rest_bnd_dim = rest_bnd_dim[1:]

            print(__current_tt.shape)

            reshape_tt = np.transpose(__current_tt, (0,2,1,3))
            reshape_tt = np.reshape(reshape_tt, newshape=(ipt_dim*opt_dim, -1), order='F')
            U, S, V = np.linalg.svd(reshape_tt, full_matrices=False)
            
            print(U.shape, S.shape, V.shape)

            print((bnd_dim, *rest_ipt_dim, *rest_opt_dim))

            S_round = S[:bnd_dim*2]
            S_round = np.reshape(S_round, newshape=(bnd_dim, bnd_dim))
            U_round = np.reshape(U, (ipt_dim, opt_dim, -1), order='F')[:,:,:bnd_dim]
            V_round = np.reshape(V, (*rest_ipt_dim, *rest_opt_dim, -1), order='F')[:,:,:bnd_dim]
            # V_round = np.transpose(V_round, (2,0,1))
            
            print(U_round.shape)
            print(S_round.shape)
            print(V_round.shape)

            __current_tt = np.einsum(S_round, [0,1], V_round, [Ellipsis,1], [Ellipsis,0])
            core = U_round

            print(__current_tt.shape)

            self.cores.append(core)
        self.cores.append(__current_tt)

        print(self.cores)

        print("Core shape")
        for c in self.cores:
            print(c.shape)

    def retrieve(self, index_list):
        np.einsum()

    @staticmethod
    def empty(shape):
        return TensorTrain(tf.Variable(np.empty(shape), dtype=np.float32))

    ''' Simple implemtation for two tensors for now '''
    @staticmethod
    def contract(a: TensorTrain, b: TensorTrain, index: List[List[float]]):
        pass
        

if __name__ == "__main__":
    tensor = np.random.normal(size=(1024,1024))
    # print(tensor)
    A = TensorTrain(tensor)

    A.decompose_as_mpo((32,32), (32,32), (2,))