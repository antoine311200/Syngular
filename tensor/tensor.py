from __future__ import annotations

from typing import List, Union

from ..utils.benchmark import Timer

import tensorflow as tf
import numpy as np

class Tensor:

    def __init__(self, tensor: Union[tf.Tensor, list]) -> None:
        self.tensor = tensor

        self.shape = self.tensor.get_shape().as_list()
    
    def __add__(self, tensor: tf.Tensor) -> Tensor:
        return Tensor(self.tensor + tensor)
    
    @Timer.wrapper
    def __getitem__(self, index):
        print(index)
        value = self.tensor
        if type(index) == int:
            index = [index]
        while len(index) > 0:
            value = value[index[0]]
            index = index[1:]
        return value


    def decompose(self, cores, bond_dim):
        pass

    @staticmethod
    def empty(shape):
        return Tensor(tf.Variable(np.empty(shape), dtype=np.float32))

    ''' Simple implemtation for two tensors for now '''
    @staticmethod
    def contract(a: Tensor, b: Tensor, index: List[List[float]]):
        c = Tensor.empty((2,2))
        print('A', a)
        print(a.shape, b.shape)

        # case index = [[x,y]]

        shape = a.shape[:index[0][0]-1]+a.shape[index[0][0]:] + b.shape[:index[0][1]-1]+b.shape[index[0][1]:]
        print(shape)

        c = Tensor.empty(shape)
        print(c.tensor)

        

