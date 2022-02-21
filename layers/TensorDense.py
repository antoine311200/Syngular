from functools import reduce
from itertools import zip_longest
from numpy.core.einsumfunc import einsum

import tensorflow as tf
from tensorflow.keras.layers import Layer

from math import floor, ceil, sqrt
import tensornetwork as tn
import numpy as np

from opt_einsum import contract

def unfold_dim(shape):
    return reduce(lambda x, y: x*y, shape)

class TensorDense(Layer):

    def __init__(self, tt_input_shape, tt_output_shape, tt_bond_shape, activation=tf.nn.relu):
        super(TensorDense, self).__init__()

        self.tt_input_shape = tt_input_shape
        self.tt_output_shape = tt_output_shape
        self.tt_bond_shape = tt_bond_shape

        self.tt_input_shape_unfold = unfold_dim(self.tt_input_shape) if self.tt_input_shape != None else None
        self.tt_output_shape_unfold = unfold_dim(self.tt_output_shape)

        if (self.tt_input_shape != None and (len(self.tt_input_shape) != len(self.tt_output_shape) or len(self.tt_input_shape) != len(self.tt_bond_shape)+1)) or len(self.tt_output_shape) != len(self.tt_bond_shape)+1:
            raise Exception(f"Incompatible shapes. Cannot create TensorDense with {len(self.tt_input_shape)} {len(self.tt_output_shape)} and {len(self.tt_bond_shape)} ")

        self.cores = []
        self.cores_number = len(self.tt_input_shape) if self.tt_input_shape != None else len(self.tt_output_shape)

        self.activation = activation

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

        # Creating the intermediate cores of the weight MPS
        for idx in range(1, self.cores_number-1):
            self.cores.append(self.add_weight(
                shape = (self.tt_input_shape[idx], self.tt_output_shape[idx], self.tt_bond_shape[idx-1], self.tt_bond_shape[idx]),
                name = f"core{idx}",
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

    def call(self, inputs):

        def process(input, bias):
            # input_reshaped = tf.reshape(input,self.tt_input_shape)
    
            # last_idx = self.cores_number-1
            # einsum_structure = []

            # cores = [tn.Node(core, backend="tensorflow").tensor for core in self.cores]
            # x = tn.Node(input_reshaped, backend="tensorflow")


            # einsum_structure = []
            # einsum_structure.append(list(range(1, self.cores_number+1)))
            # einsum_structure.append([1, -(self.cores_number+1), 2*self.cores_number+1])

            # for idx in range(1, last_idx):
            #     einsum_structure.append([idx+1, -(self.cores_number+idx+1), 2*self.cores_number+idx+1])
            
            # einsum_structure.append([last_idx+1, -(self.cores_number+last_idx+1), 2*self.cores_number+last_idx-1+1])
            
            # #print(einsum_structure)

            # result = tn.ncon(
            #     tensors = [x.tensor] + cores,
            #     network_structure = einsum_structure,
            #     backend = "tensorflow"
            # )

            x = tf.reshape(input,self.tt_input_shape)#tf.reshape(input, ((self.tt_input_shape[0],)*self.cores_number))

            
            struct = []
            struct += [x, list(range(1, self.cores_number+1))]
            struct += [self.cores[0], [1, self.cores_number+1, 2*self.cores_number+1]]
            
            for idx in range(2, self.cores_number):
                struct += [self.cores[idx-1]]
                struct += [[idx, self.cores_number+idx, 2*self.cores_number+1, 2*self.cores_number+2]]

            struct += [self.cores[-1], [self.cores_number, 2*self.cores_number, 3*self.cores_number-1]]
            struct += [list(range(self.cores_number+1, 2*self.cores_number+1))]


            result = contract(*struct)

            # print(type(self.cores[0].numpy()))

            # cores = list(map(lambda core: tf.convert_to_tensor(core), self.cores))
            # print(type(cores[0]))

            # einsum_structure.append(cores[0])
            # einsum_structure.append([0, self.cores_number, 2*self.cores_number])
            
            # for idx in range(1, last_idx):
            #     einsum_structure.append(cores[idx])
            #     einsum_structure.append([idx, self.cores_number+idx, 2*self.cores_number+idx])

            # einsum_structure.append(cores[self.cores_number-1])
            # einsum_structure.append([last_idx, self.cores_number+last_idx, 2*self.cores_number+last_idx-1])

            # # print(list(range(self.cores_number, 2*self.cores_number)))
            # einsum_structure.append(list(range(self.cores_number+1, 2*self.cores_number)))
            # result = np.einsum(*einsum_structure)

            return result+self.bias

        result = tf.vectorized_map(lambda vec: process(vec, self.bias), inputs)
        return self.activation(tf.reshape(result, (-1, self.tt_output_shape_unfold)))

class TDense(Layer):

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

    def build(self, tt_input_shape):

        self.bias = tf.Variable(tf.zeros(shape=self.shape), name="bias", trainable=True)

        # self.shape_input = []

        self.cores.append(self.add_weight(
            shape = (tt_input_shape[1], self.shape[0], self.bond_dim,),
            name = "core_1",
            initializer = 'random_normal',
            trainable = True
        ))
        # self.shape_input.append(tt_input_shape[1])

        for i in range(1, self.cores_number-1):
            self.cores.append(self.add_weight(
                shape = (tt_input_shape[1], self.shape[i], self.bond_dim, self.bond_dim,),
                name = "core_"+str(i),
                initializer = 'random_normal',
                trainable = True
            ))
            # self.shape_input.append(tt_input_shape[1])

        self.cores.append(self.add_weight(
            shape = (tt_input_shape[1], self.shape[-1], self.bond_dim,),
            name = "core_"+str(self.cores_number),
            initializer = 'random_normal',
            trainable = True
        ))
        # self.shape_input.append(tt_input_shape[1])
        # self.shape_input = tuple(self.shape_input)

    def call(self, inputs):

        def process(input, cores, bias):
            # unfold = tf.reshape(input,[-1])
            # reduction = reduce(lambda x, y: x*y, self.shape_input)
            # padding = tf.convert_to_tensor(np.zeros((reduction-unfold.shape[0]), dtype="float32"))
            # input = tf.reshape(tf.concat(values=[input, padding], axis=0), self.shape_input)
            
            # input = [input, input]
            print("Core shape", self.cores[0].shape)
            print("Input shape", input.shape)
            print("Input reshape", (floor(sqrt(input.shape[0])),ceil(sqrt(input.shape[0]))))
            input = tf.reshape(input, (floor(sqrt(input.shape[0])),ceil(sqrt(input.shape[0]))))

            print(input.shape)
            
            mx = self.cores_number

            cores = [tn.Node(core, backend="tensorflow").tensor for core in cores]
            x = tn.Node(input, backend="tensorflow")

            links = [[i, -i, "bond"+str(i-1), "bond"+str(i)] for i in range(2, mx)]
            
            # print([list(range(1,mx+1)), [1, -1, "bond"+str(1)], *links, [mx, -mx, "bond"+str(mx-1)]])
            print( [list(range(1,mx+1)), [1, -1, "bond"+str(1)], *links, [mx, -mx, "bond"+str(mx)]])

            result = tn.ncon(
                tensors = [x.tensor] + cores, 
                network_structure = [list(range(1,mx+1)), [1, -1, "bond"+str(1)], *links, [mx, -mx, "bond"+str(mx-1)]],
                backend="tensorflow"
            )

            return result + bias

        result = tf.vectorized_map(lambda vec: process(vec, self.cores, self.bias), inputs)

        return tf.nn.relu(tf.reshape(result, (-1, self.units)))