import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

import tensornetwork as tn
import numpy as np


class TensorNetwork(Model):

    def compile(self, optimizer, loss_fn):
        super(TensorNetwork, self).compile()

        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):

        input_tensor, output_tensor = data

        with tf.GradientTape() as tape:
            predictions = self(input_tensor, trainable=True)
            d_loss = self.loss_fn(output_tensor, predictions)