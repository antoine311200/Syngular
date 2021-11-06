from __future__ import print_function, annotations
from functools import reduce
from re import match

import numpy as np
from numpy.core.einsumfunc import einsum


def unfold_shape(shape):
    return reduce(lambda x, y: x+y, shape)

class TensorTrainLayer():

    def __init__(self) -> None:
        pass

    def build(self):
        pass

    def call(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]
        
        d_layer_d_input = np.eye(num_units)
        
        return np.dot(grad_output, d_layer_d_input)

    def train(self):
        pass

class ReLU(TensorTrainLayer):
    
    def forward(self, input):
        relu_forward = np.maximum(0,input)
        return relu_forward
    
    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad

class Dense(TensorTrainLayer):

    def __init__(self, input_shape, output_shape, bond_dim=2, core_number=None, learning_rate=0.01) -> None:
        
        if len(input_shape) != len(output_shape):
            raise Exception("input shape and output shape should have the same length")

        if core_number != None and core_number != len(input_shape):
            raise Exception("number of cores does not match the size of input_shape")
        
        self.input_shape = input_shape
        self.unfold_input_shape = unfold_shape(self.input_shape)
        self.output_shape = output_shape
        self.unfold_output_shape = unfold_shape(self.output_shape)

        self.cores_number = core_number if core_number != None else len(input_shape)
        self.bond_dim = bond_dim

        self.learning_rate = learning_rate

        self.cores = []
        self.bias = []

    def __get_core_shape(self, index):
        if index == 0 or index == self.cores_number-1:
            return (self.input_shape[index], self.output_shape[index], self.bond_dim,)
        else:
            return (self.input_shape[index], self.output_shape[index], self.bond_dim, self.bond_dim,)

    def __add_core(self, name, type):
        index = len(self.cores)

        shape = self.__get_core_shape(index)
        size = unfold_shape(shape)

        print(shape)

        if type == 'middle' and 0 < index < self.cores_number-1:
            return np.random.normal(
                loc=0.0,
                scale = np.sqrt(2/size), 
                size = shape
            )
        elif type == 'extreme' and (index == 0 or index == self.cores_number-1):
            return np.random.normal(
                loc=0.0,
                scale = np.sqrt(2/size), 
                size = shape
            )
        else:
            raise Exception('the type of core to add does not match the current cores structure')


    def build(self):
        self.cores.append(self.__add_core(name='core_1', type='extreme'))

        for i in range(1, self.cores_number-1):
            self.cores.append(self.__add_core(name = "core_"+str(i), type='middle'))

        self.cores.append(self.__add_core(name='core_'+str(self.cores_number), type='extreme'))

        self.bias = np.zeros(shape=self.output_shape)


    def call(self):
        pass

    def forward(self, input):
        input = np.array(input)
        unfold_input = unfold_shape(input.shape)

        if self.unfold_input_shape != unfold_input:
            exception = f"input of shape {input.shape} cannot be reshaped into {self.input_shape} [{unfold_input} != {self.unfold_input_shape}]"
            raise Exception(exception)
        
        input_tensor = np.reshape(input, newshape=self.input_shape)

        print(input_tensor)

        einsum_structure = []
        input_index = np.arange(self.cores_number)

        einsum_structure.append(input_tensor)
        einsum_structure.append(input_index)

        for idx in range(self.cores_number):
            ipt_index = idx
            opt_index = self.cores_number+idx
            einsum_structure.append(self.cores[idx])
            if idx == 0:
                bnd_index = 2*self.cores_number
                einsum_structure.append([ipt_index, opt_index, bnd_index])
            elif idx == self.cores_number-1:
                bnd_index = 3*self.cores_number-2
                einsum_structure.append([ipt_index, opt_index, bnd_index])
            else:
                bnd_index_1 = 2*self.cores_number+idx-1
                bnd_index_2 = 2*self.cores_number+idx
                einsum_structure.append([ipt_index, opt_index, bnd_index_1, bnd_index_2])

        output_index = np.arange(self.cores_number)+self.cores_number

        einsum_structure.append(output_index)

        print("Structure")
        print(einsum_structure)
        print(len(einsum_structure))

        contraction = np.einsum(*einsum_structure)

        print("Contraction")
        print(contraction)

        result = contraction+self.bias
        print(result)


    def backward(self):
        pass

    def train(self):
        pass


if __name__ == "__main__":

    layer = Dense((2,2), (3,3), bond_dim=2)
    layer.build()

    print("Cores")
    print(layer.cores)
    print("Bias")
    print(layer.bias)

    layer.forward([[1,4],[2,5]])