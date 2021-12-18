import itertools
from PIL.Image import new
from numpy.core.fromnumeric import size
from syngular.tensor.tensor_train import MatrixProductOperator, MatrixProductState

import numpy as np


class Model:

    def __init__(self, layers):
        self.layers = layers

    def predict(self, inputs):
        values = inputs
        for layer in self.layers:
            values = layer(values)

        return values

    def train(self):
        pass

class Layer:

    def __init__(self) -> None:
        self.trainable_weights = []

        self.build()

    def build(self):
        pass

    def train(self):
        for trainable in self.trainable_weights:
            pass


class Shaper(Layer):

    def __init__(self, shape, bond_shape=None) -> None:
        super(Shaper, self).__init__()

        self.shape = shape
        self.bond_shape = bond_shape

    def __call__(self, inputs):
        print(inputs.shape)
        outputs = np.reshape(inputs, newshape=self.shape, order="F")

        if self.bond_shape != None:
            outputs = MatrixProductState(outputs, bond_shape=self.bond_shape)
            outputs.decompose(mode='left')

        return outputs


class Linear(Layer):

    def __init__(self, unit_shape, bond_shape) -> None:
        super(Linear, self).__init__()

        self.unit_shape
        self.bond_shape

        self.weights = None
        self.bias = None

    def build(self):
        self.bias = np.zeros(self.bond_shape)
        # self.weights = np.random.normal(size=)

    def __call__(self, inputs):
        pass

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.image as mimage


    tensor = np.arange(4**4).reshape((4,4,4,4))
    image = np.arange(32*32)

    img = mimage.imread('C:\\Antoine\\Coding\\Tensor Network\Project\\syngular\\examples\\resources\images\\paysage2.jpg')
    img = img[:,:,0] / 255.0





    model = Model([
        Shaper((20,20,30,20), (20,20,20,)) # 20800 / 240 000
        #Shaper((64,75,50), (50,50,)) # 193200 / 240 000
    ])

    y = model.predict(img)





    img_reconstruct = y.reconstruct()
    img_reconstruct = img_reconstruct.reshape((400, 600), order="F")


    errors = []
    for i, j in zip(range(400), range(600)):
        e = abs(img[i,j] - img_reconstruct[i,j])
        errors.append(e)

        # print(img_reconstruct[a,b,c,d,e])
        # errors1.append(error1)
    # plt.plot(errors)
    # 
    plt.rcParams['figure.dpi'] = 150

    print("Shape", img_reconstruct.shape)
    plt.subplot(2,1,1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.subplot(2,1,2)
    plt.imshow(img_reconstruct, cmap='gray')
    plt.axis('off')
    plt.show()




