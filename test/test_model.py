import itertools
from syngular.core import Model, Output
from syngular.core import Linear
from syngular.tensor import MatrixProductState, MatrixProductOperator

import numpy as np

# w = np.arange(2**6).reshape(2,2,2,2,2,2).astype('float64')
# W = MatrixProductOperator(w, bond_shape=(2,2,))
# W.decompose()

def simple_model():

    core = 2
    input_dim = 2
    output_dim = 2
    bond_dim = 2

    input_size = input_dim**core
    output_size = 2**core

    input_shape = (input_dim,) * core
    output_shape = (output_dim,) * core
    bond_shape = (bond_dim,) * (core-1)

    model = Model([
        Linear(input_size, output_size, core=core, bond=bond_dim),
        Linear(input_size, output_size, core=core, bond=bond_dim),
        Output((output_size, ))
    ])
    model.build()

    print(model.describe())


    x = np.arange(input_size).reshape(input_shape)
    X = MatrixProductState(x, bond_shape=bond_shape).decompose()

    y = model.predict(X)
    print("Prediction", y)
    print('\n')

    train_df = [X]

    for epoch in range(5):
        print(f"Epoch {str(epoch+1)} : ")

        for sample in train_df:
            model.feed_forward(sample)
        # for layer in model.layers:
        #     for weight in layer.trainable_tensor_weights:
        #         weight["weight"] += MatrixProductOperator.random(input_shape, output_shape, bond_shape)
        y = model.predict(X)
        print("Prediction", y)


simple_model()