import tensorflow as tf
import numpy as np

from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras.metrics import Mean
from keras.layers import Layer, Dense, Input, Conv2D, Flatten
from keras import Model

from syngular.layers.TensorDense import TensorDense

class Sampling(Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(Layer):

    def __init__(self, latent_dim=(8,8), intermediate_dim=(16,16), name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim

        self.latent_bond_dim = tuple(d//4 for d in self.latent_dim[:-1])
        self.intermediate_bond_dim = tuple(d//4 for d in self.intermediate_dim[:-1])

        self.conv1 = Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv2 = Conv2D(64, 3, activation="relu", strides=2, padding="same")

        self.tt_dense_proj = Dense(64) #TensorDense(None, self.intermediate_dim, self.intermediate_bond_dim)
        self.tt_dense_mean = Dense(32) #TensorDense(None, self.latent_dim, self.latent_bond_dim)
        self.tt_dense_log_var = Dense(32) #TensorDense(None, self.latent_dim, self.latent_bond_dim)

        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = Flatten()(x)

        x = self.tt_dense_proj(x)
        z_mean = self.tt_dense_mean(x)
        z_log_var = self.tt_dense_log_var(x)

        z = self.sampling((z_mean, z_log_var))

        return z_mean, z_log_var, z

class Decoder(Layer):

    def __init__(self, original_dim, intermediate_dim=(16,16), name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim

        self.original_bond_dim = tuple(d//4 for d in self.original_dim[:-1])
        self.intermediate_bond_dim = tuple(d//4 for d in self.intermediate_dim[:-1])

        self.tt_dense_proj = Dense(64)#TensorDense(None, self.intermediate_dim, self.intermediate_bond_dim)
        self.tt_dense_output = Dense(28*28) #TensorDense(None, self.original_dim, self.original_bond_dim) #, activation=tf.nn.sigmoid)
        
    def call(self, inputs):
        x = self.tt_dense_proj(inputs)

        return self.tt_dense_output(x)

class TensorAutoEncoder(Model):

    def __init__(self, original_dim,  intermediate_dim=(16,16), latent_dim=(8,8), name="tensorautoencoder", **kwargs):
        super(TensorAutoEncoder, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)
        
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

        self.add_loss(kl_loss)
        return reconstructed

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inputs)
            
            reconstruction = self.decoder(z)

            print("train shape", inputs.shape, reconstruction.shape)
            tf.reshape(inputs, (tf.shape(inputs)[0],28*28))

            # reconstruction_loss = tf.reduce_mean(tf.reduce_sum(binary_crossentropy(inputs, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = kl_loss #reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        # self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            # "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    (x_train, _), (x_test, _) = mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

    tae = TensorAutoEncoder(original_dim=(28,28), intermediate_dim=(16,16), latent_dim=(8,8))
    tae.compile(optimizer="adam")
    tae.fit(mnist_digits, epochs=30, batch_size=128, verbose=0)

    def plot_latent_space(tae, n=30, figsize=15):
        # display a n*n 2D manifold of digits
        digit_size = 28
        scale = 1.0
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = tae.decoder(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[
                    i * digit_size : (i + 1) * digit_size,
                    j * digit_size : (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        plt.show()


    plot_latent_space(tae)