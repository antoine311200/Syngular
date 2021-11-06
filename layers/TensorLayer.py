from functools import reduce

import tensorflow as tf
from tensorflow.keras.layers import Layer

import tensornetwork as tn
import numpy as np


from ..utils.benchmark import Timer

class Core:

    def __init__(self, shape, bond) -> None:
        self.shape = shape
        self.bond = bond

    @staticmethod
    def create(shape, name=np.random.random_sample(), random=True):
        if random:
            normal_var = tf.random.normal(
                shape = shape,
                stddev = 1.0/shape[0]
            )
        return tf.Variable(normal_var, trainable = True, name=name)

class TensorLayer(Layer):

    @Timer.wrapper
    def __init__(self, shape, bond_dim=2, core_number=None) -> None:
        super().__init__()

        self.shape = shape
        self.cores_number = core_number if core_number != None else len(shape)
        self.bond_dim = bond_dim

        self.cores = []


    def build(self, input_shape):

        self.bias = tf.Variable(tf.zeros(shape=self.shape), name="bias", trainable=True)

        print("INPUT SHAPE ", input_shape)

        self.cores.append(self.add_weight(
            shape = self.shape[0:2]+(self.bond_dim,),
            name = "core_1",
            initializer = 'random_normal',
            trainable = True
        ))

        for i in range(1, self.cores_number-1):
            self.cores.append(self.add_weight(
                shape = self.shape[i-1:i+1]+(self.bond_dim,self.bond_dim,),
                name = "core_"+str(i),
                initializer = 'random_normal',
                trainable = True
            ))

        self.cores.append(self.add_weight(
            shape = self.shape[-2:]+(self.bond_dim,),
            name = "core_"+str(self.cores_number),
            initializer = 'random_normal',
            trainable = True
        ))

    def call(self, inputs):

        # print("TRAINABLE ", self.trainable_variables)
        # print("TRAINABLE ", self.trainable_weights)

        # print("NON TRAINABLE", self.non_trainable_variables)
        # print("NON TRAINABLE", self.non_trainable_weights)

        def process(input, cores, bias):
            input = tf.reshape(input, self.shape)
            mx = self.cores_number

            cores = [tn.Node(core, backend="tensorflow").tensor for core in cores]
            x = tn.Node(input, backend="tensorflow")

            links = [[-i, i, "bond"+str(i-1), "bond"+str(i)] for i in range(2, mx)]
            
            # print([list(range(1,mx+1)), [-1, 1, "bond"+str(1)], *links, [-mx, mx, "bond"+str(mx-1)]])

            result = tn.ncon(
                tensors = [x.tensor] + cores, 
                network_structure = [list(range(1,mx+1)), [-1, 1, "bond"+str(1)], *links, [-mx, mx, "bond"+str(mx-1)]],
                backend="tensorflow"
            )

            return result + bias

        result = tf.vectorized_map(lambda vec: process(vec, self.cores, self.bias), inputs)
        reduction = reduce(lambda x, y: x*y, self.shape)

        return tf.nn.relu(tf.reshape(result, (-1, reduction)))
