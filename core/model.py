from numpy.linalg.linalg import matrix_rank
from syngular.tensor.tensor_train import MatrixProductOperator, MatrixProductState

import numpy as np
from opt_einsum import contract

class Model:

    def __init__(self, layers):
        self.layers = layers

    def predict(self, inputs):
        values = inputs
        for layer in self.layers:
            values = layer(values)
            # print("Values shape", values._shape)

        return values

    def build(self):
        for layer in self.layers:
            if not layer.is_built:
                layer.build(None)
                layer.is_built = True

    def train(self):
        pass

    def draw(self):
        repr = ''
        for layer in self.layers:
            repr += layer.draw()
        return repr


class Layer:

    def __init__(self):

        self.trainable_tensor_weights = []
        self.trainable_tensor_bias = []

        self.is_built = False

    def __call__(self, inputs):
        input_shape = inputs.shape
        if not self.is_built:
            self.build(input_shape)
            self.is_built = True
        else:
            print("Built")

    def build(self, input_shape):
        pass

    def draw(self):
        repr = ''
        for weight in self.trainable_tensor_weights:
            mp = weight["weight"]
            repr += "\t"+"|   " * mp.sites_number + "\n"
            repr += "\t"+("O---" * (mp.sites_number-1)) + "O" + "\n"
            repr += "\t"+"|   " * mp.sites_number + "\n"
        return repr
            

    def add_weight(self, input_shape, output_shape, bond, name=None, initializer="normal"):
        if name == None:
            name = f'weight_{np.random.randint(0,999999)}'

        if initializer == "normal":
            weight = np.random.normal(size=(*self._input_shape, *self._output_shape))
        else:
            weight = np.zeros(shape=(*self._input_shape, *self._output_shape))

        matrix_product_weight = MatrixProductOperator(weight, bond_shape=bond)
        matrix_product_weight.decompose(mode='left')

        self.trainable_tensor_weights.append({'name': name, 'weight': matrix_product_weight})
    


    def add_bias(self, size, name=None, initializer="normal"):
        if name == None:
            name = f'bias_{np.random.randint(0,999999)}'

        if initializer == "normal":
            bias = np.random.normal(size=size)
        else:
            bias = np.zeros(shape=size)

        self.trainable_tensor_bias.append({name: name, bias: bias})

class Linear(Layer):

    def __init__(self, 
        input_units, output_units,
        core=1, bond=None,
        bias_initializer="normal",
        weights_initializer="normal",
        activation="relu"):

        super(Linear, self).__init__()

        self.input_units = input_units
        self.output_units = output_units

        self.core = core
        self.bond_dimension = bond

        self.input_core_dim = round(self.input_units**(1/self.core))
        self.output_core_dim = round(self.output_units**(1/self.core))

        self._input_shape    = (self.input_core_dim,) * self.core
        self._output_shape   = (self.output_core_dim,) * self.core
        self._bond_shape     = (self.bond_dimension,) * (self.core-1)

        self.bias_initializer = bias_initializer
        self.weights_initializer = weights_initializer

        self.activation = activation

    def build(self, input_shape):
        # self.add_bias(self._output_shape, name="bias", initializer="normal")
        # print(self._input_shape, self._output_shape)
        self.add_weight(self._input_shape, self._output_shape, bond=self._bond_shape, name="bias", initializer="normal")

    def __call__(self, inputs):
        super(Linear, self).__call__(inputs)

        weight = self.trainable_tensor_weights[0]["weight"]

        print("Weight", weight)
        print("input", inputs)
        print("contract", MatrixProductOperator.contract(weight, inputs))

        return MatrixProductOperator.contract(weight, inputs)

class Output(Layer):

    def __init__(self, output_shape):
        super(Output, self).__init__()

        self.output_shape = output_shape

    def __call__(self, inputs):
        # print(">", inputs)

        return inputs#.reshape(self.output_shape)