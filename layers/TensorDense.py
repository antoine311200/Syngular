from functools import reduce

import tensorflow as tf
from tensorflow.keras.layers import Layer

import tensornetwork as tn
import numpy as np

class TensorDense(Layer):

    def __init__(self, units, cores_number, bond_dim=2, shape=None) -> None:
        super(TensorDense, self).__init__()

        self.units = units
        self.cores_number = cores_number
        self.bond_dim = bond_dim

        if shape == None:
            roots = int(np.power(self.units, 1/self.cores_number))
            self.shape = [roots] * self.cores_number
        else:
            self.shape = shape

        self.cores = []

    def build(self, input_shape):

        self.bias = tf.Variable(tf.zeros(shape=self.shape), name="bias", trainable=True)

        # self.shape_input = []

        self.cores.append(self.add_weight(
            shape = (input_shape[1], self.shape[0], self.bond_dim,),
            name = "core_1",
            initializer = 'random_normal',
            trainable = True
        ))
        # self.shape_input.append(input_shape[1])

        for i in range(1, self.cores_number-1):
            self.cores.append(self.add_weight(
                shape = (input_shape[1], self.shape[i], self.bond_dim, self.bond_dim,),
                name = "core_"+str(i),
                initializer = 'random_normal',
                trainable = True
            ))
            # self.shape_input.append(input_shape[1])

        self.cores.append(self.add_weight(
            shape = (input_shape[1], self.shape[-1], self.bond_dim,),
            name = "core_"+str(self.cores_number),
            initializer = 'random_normal',
            trainable = True
        ))
        # self.shape_input.append(input_shape[1])
        # self.shape_input = tuple(self.shape_input)

    def call(self, inputs):

        def process(input, cores, bias):
            # unfold = tf.reshape(input,[-1])
            # reduction = reduce(lambda x, y: x*y, self.shape_input)
            # padding = tf.convert_to_tensor(np.zeros((reduction-unfold.shape[0]), dtype="float32"))
            # input = tf.reshape(tf.concat(values=[input, padding], axis=0), self.shape_input)
            
            input = [input, input]
            input = tf.reshape(input, (2,2))
            
            mx = self.cores_number

            cores = [tn.Node(core, backend="tensorflow").tensor for core in cores]
            x = tn.Node(input, backend="tensorflow")

            links = [[i, -i, "bond"+str(i-1), "bond"+str(i)] for i in range(2, mx)]
            
            # print([list(range(1,mx+1)), [1, -1, "bond"+str(1)], *links, [mx, -mx, "bond"+str(mx-1)]])

            result = tn.ncon(
                tensors = [x.tensor] + cores, 
                network_structure = [list(range(1,mx+1)), [1, -1, "bond"+str(1)], *links, [mx, -mx, "bond"+str(mx-1)]],
                backend="tensorflow"
            )

            return result + bias

        result = tf.vectorized_map(lambda vec: process(vec, self.cores, self.bias), inputs)

        return tf.nn.relu(tf.reshape(result, (-1, self.units)))