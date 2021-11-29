from functools import reduce

from keras.layers import Layer

import tensorflow as tf
import numpy as np

def unfold_dim(shape):
    return reduce(lambda x, y: x*y, shape)

class TensorRNNCell(Layer):

    def __init__(self, output_shhape, bond_shape, **kwargs):
        super(TensorRNNCell, self).__init__()

        self.tt_output_shape = output_shhape
        self.tt_bond_shape = bond_shape

        self.tt_output_shape_unfold = unfold_dim(self.tt_output_shape)

        self.cores = []
        self.recurrent_cores = []
        self.cores_number = len(self.tt_output_shape)
    
    def build(self, tt_input_shape):
        if self.tt_input_shape == None:
            roots = int(np.power(tt_input_shape[1:], 1/self.cores_number))
            self.tt_input_shape = tuple([roots] * self.cores_number)

        self.bias = tf.Variable(tf.zeros(shape=self.tt_output_shape), name="bias", trainable=True)
        
        last_idx = self.cores_number-1

        # Creating the first core of the weight MPS
        self.cores.append(self.add_weight(
            shape = (self.tt_input_shape[0], self.tt_output_shape[0], self.tt_bond_shape[0]),
            name = "core1",
            initializer = "random_normal",
            trainable = True
        ))
        self.recurrent_cores.append(self.add_weight(
            shape = (self.tt_input_shape[0], self.tt_output_shape[0], self.tt_bond_shape[0]),
            name = "recurrentcore1",
            initializer = "random_normal",
            trainable = True
        ))

        # Creating the intermediate cores of the weight MPS
        for idx in range(1, self.cores_number-1):
            self.cores.append(self.add_weight(
                shape = (self.tt_input_shape[idx], self.tt_output_shape[idx], self.tt_bond_shape[idx-1], self.tt_bond_shape[idx]),
                name = f"core{idx}",
                initializer="random_normal",
                trainable=True
            ))
            self.recurrent_cores.append(self.add_weight(
                shape = (self.tt_input_shape[idx], self.tt_output_shape[idx], self.tt_bond_shape[idx-1], self.tt_bond_shape[idx]),
                name = f"recurrentcore{idx}",
                initializer="random_normal",
                trainable=True
            ))

        # Creating the last core of the weight MPS
        self.cores.append(self.add_weight(
            shape = (self.tt_input_shape[last_idx], self.tt_output_shape[last_idx], self.tt_bond_shape[last_idx-1]),
            name = f"core{self.cores_number}",
            initializer = "random_normal",
            trainable = True
        ))
        self.recurrent_cores.append(self.add_weight(
            shape = (self.tt_input_shape[last_idx], self.tt_output_shape[last_idx], self.tt_bond_shape[last_idx-1]),
            name = f"recurrentcore{self.cores_number}",
            initializer = "random_normal",
            trainable = True
        ))

    # h(t)​ = f(U x(t)​ + W h(t−1)​)
    def call(self, inputs, states):
        prev_output = states[0]

        def process(input):
            pass

        result = tf.vectorized_map(lambda vec: process(vec, self.bias), inputs)
        return self.activation(tf.reshape(result, (-1, self.tt_output_shape_unfold)))

