"""
Authors: Nathan Starliper, Reddy Aravind Karnam, Anushka Gupta
ECE542 Project 6
12/7/2018
Script to build and train a variational autoencoder on the MNIST
dataset. Plots the latent space manifold, scatterplot of codes
and randomly generated images.
"""


from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os


def sample_z(args):
    z_mean, z_log_var = args
    z_batch = K.shape(z_mean)[0]
    z_dim = K.int_shape(z_mean)[1]
    # random vector with mean 0 and std 1
    eps = K.random_normal(shape=(z_batch, z_dim))
    return z_mean + K.exp(0.5 * z_log_var) * eps


def visualizations(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    # function to plot the manifold and encoded images
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "vae_mean.png")
    # display a scatterplot of the encoded data and class clusters
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.title('2-Dimensional Scatter Plot of the Encoded Images')
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits latent space
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # symmetrically spaced gaussian grid
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.title('Visualization of the 2D Latent Space Manifold')
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


def generate_imgs(
    models,
    data,
    batch_size = 128,
    model_name = "vae_mnist",
    dim = 2):
    # function to generate images from random latent space vectors
    # and plot the grid of random reconstructed images
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)
    dim_title = str(dim)
    title = ['generated_imgs_', dim_title, 'd.png']
    itb = ''
    title = itb.join(title)
    # display a grid of randomly generated and decoded latent space vectors
    filename_gen = os.path.join(model_name, title)
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure_gen = np.zeros((digit_size * n, digit_size * n))
    for i in range(n):
        for j in range(n):
            x_rand = norm.rvs(size=dim)
            z_sample = np.array([x_rand])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure_gen[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure_gen, cmap='Greys_r')
    plt.title('Randomly Generated Images with %dD Latent Space' % dim)
    plt.savefig(filename_gen)
    plt.show()


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_size = x_train.shape[1] * x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size])
x_test = np.reshape(x_test, [-1, image_size])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Initialize parameters
input_shape = (image_size, )
hidden_dim = 512
latent_dim = 2
batch_size = 128
epochs = 10

# Encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(hidden_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
# Sample z using reparameterization trick
z = Lambda(sample_z, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
# instantiate encoder
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
# decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(hidden_dim, activation='relu')(latent_inputs)
outputs = Dense(image_size, activation='sigmoid')(x)
# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
models = (encoder, decoder)
data = (x_test, y_test)
# calculate reconstruction loss
reconstruction_loss = binary_crossentropy(inputs, outputs) * image_size
# KL divergence loss
k_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
k_loss = -0.5 * K.sum(k_loss, axis=-1)
# create custom loss layer
vae_loss = K.mean(reconstruction_loss + k_loss)
vae.add_loss(vae_loss)
# compile vae
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae,
           to_file='vae_mlp.png',
           show_shapes=True)

# train the VAE
history = vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_valid, None))
vae.save_weights('vae_mlp_mnist.h5')

# Generate random images with VAE
generate_imgs(models,
               data,
               batch_size=batch_size,
               model_name="vae_mlp",
               dim = latent_dim)

# Visualize the 2D manifold and encoded scatter plot
if latent_dim == 2:
    visualizations(models,
                 data,
                 batch_size=batch_size,
                 model_name="mnist")