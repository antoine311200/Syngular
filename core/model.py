from numpy.matrixlib.defmatrix import matrix
from syngular.tensor import MatrixProductOperator, MatrixProductState, matrix_product_operator

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

    def train(self, x, y, batchsize=32, epochs=1, verbose=1):
        if verbose: print("[START] Training ")
        for e in range(epochs):
            if verbose: print(f"Epoch {str(e+1)} : ")
            for layer in self.layers:
                for weight in layer.trainable_tensor_weights:
                    # print(weight)
                    weight["weight"] += MatrixProductOperator.random((2,2), (2,2), (4,))
                    # print(weight["weight"])
        if verbose: print("[END] Training ")

    def draw(self):
        repr = ''
        for layer in self.layers:
            repr += layer.draw()
        return repr

    def describe(self):
        strict_layer = ['output']
        params = 0
        params_fictive = 0
        layers_num = 0
        for layer in self.layers:
            info = layer.describe()
            layers_num += 1
            params += info['param']
            params_fictive += info['fictive-param']
            print('---------------------------------------------------------------------------------------------------------------')
            if info["type"] not in strict_layer:
                print(
                    f"|  Layer [{info['type']}] -> {info['name']} \t " + \
                    f"Parameters: {info['param']} \t " + \
                    f"Saved: {info['fictive-param']-info['param']} " + \
                    f"(compression {round((info['fictive-param']-info['param'])/info['fictive-param'], 4)*100}%)"
                )
            else:
                print(f"|  Layer [{info['type']}] ->  {info['name']} \t ")
            print('---------------------------------------------------------------------------------------------------------------')

        print('===============================================================================================================')
        print(f"| \t Total number of layers : {layers_num}")
        print(f"| \t Total number of parameters : {params} (v.s {params_fictive})")
        print(f"| \t Total compression factor : {round((params_fictive-params)/params_fictive, 4)*100}%")
        print('===============================================================================================================')

class Layer:

    def __init__(self, name):
        self.name = name

        self.trainable_tensor_weights = []
        self.trainable_tensor_bias = []

        self.is_built = False

    def __call__(self, inputs):
        input_shape = inputs.shape
        if not self.is_built:
            self.build(input_shape)
            self.is_built = True
        else:
            pass #print("Built")

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
    
    def describe(self):
        description = {}

        total_parameters = 0
        total_true_parameters = 0

        for w in self.trainable_tensor_weights:
            total_parameters += w['weight'].parameters_number
            total_true_parameters += w['weight'].real_parameters_number
        
        for b in self.trainable_tensor_bias:
            total_parameters += b['bias'].parameters_number
            total_true_parameters += b['bias'].real_parameters_number

        description['param'] = total_parameters
        description['fictive-param'] = total_true_parameters
        description['name'] = self.name
        description['type'] = self.type

        return description

            

    def add_weight(self, input_shape, output_shape, bond_shape, name=None, initializer="normal"):
        if name == None:
            name = f'weight_{np.random.randint(0,999999)}'

        # if initializer == "normal":
        #     weight = np.random.normal(size=(*self._input_shape, *self._output_shape))
        # else:
        #     weight = np.zeros(shape=(*self._input_shape, *self._output_shape))

        # matrix_product_weight = MatrixProductOperator(weight, bond_shape=bond)
        # matrix_product_weight.decompose()

        matrix_product_weight = MatrixProductOperator.random(input_shape, output_shape, bond_shape)

        self.trainable_tensor_weights.append({'name': name, 'weight': matrix_product_weight})
    


    def add_bias(self, size, name=None, initializer="normal"):
        # if name == None:
        #     name = f'bias_{np.random.randint(0,999999)}'

        # if initializer == "normal":
        #     bias = np.random.normal(size=size)
        # else:
        #     bias = np.zeros(shape=size)

        # self.trainable_tensor_bias.append({'name': name, 'bias': bias})
        pass

class Linear(Layer):

    def __init__(self, 
        input_units, output_units,
        core=1, bond=None,
        bias_initializer="normal",
        weights_initializer="normal",
        activation="relu",
        name="linear"):

        super(Linear, self).__init__(name)

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

        self.type = 'linear'

    def build(self, input_shape):
        
        if self.weights_initializer == "normal":
            self.add_weight(self._input_shape, self._output_shape, bond_shape=self._bond_shape, name="weight", initializer="normal")
        else:
            self.trainable_tensor_weights.append({'name': '', 'weight': self.weights_initializer})

    def __call__(self, inputs):
        super(Linear, self).__call__(inputs)

        weight = self.trainable_tensor_weights[0]["weight"]

        return weight @ inputs

class Output(Layer):

    def __init__(self, output_shape, name = "output"):
        super(Output, self).__init__(name)

        self.output_shape = output_shape

        self.type = 'output'

    def __call__(self, inputs):
        return inputs.to_tensor().reshape(self.output_shape)