from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, MaxPooling2D, Lambda, Conv2DTranspose
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.callbacks import Callback
from keras.utils import plot_model
from keras import backend as K
import random
import glob
import wandb
from wandb.keras import WandbCallback
import argparse
import subprocess
import os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

run = wandb.init()
config = run.config

config.num_epochs = 40
config.batch_size = 64
config.img_dir = "dataset/preprocessed"
config.height = 64
config.width = 64

steps_per_epoch = (109560/config.batch_size)

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch,dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def my_generator(batch_size, img_dir):
    """A generator that returns black and white images and color images"""
    image_filenames = glob.glob(img_dir + "/**/*.jpg")
    counter = 0
    while True:
        images_tensor = np.zeros((batch_size, config.width, config.height, 3))
        random.shuffle(image_filenames)
        if ((counter+1)*batch_size>=len(image_filenames)):
            counter = 0
        for i in range(batch_size):
           img = Image.open(image_filenames[counter + i]).resize((config.width, config.height))
           #import pdb;pdb.set_trace()
           images_tensor[i] = np.array(img)/255.
        yield (images_tensor, None)
        counter += batch_size

def generate_images(generator):
    n = 8 #figure with 8x8 pics
    pic_size = 64
    figure = np.zeros((pic_size * n, pic_size * n, 3)) 
    #sample n points within [-8,8] standarad deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)
    epsilon_std = 1.0

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * epsilon_std
            x_decoded = generator.predict(z_sample)
            pic = x_decoded[0]
            figure[i * pic_size: (i+1) * pic_size,
                   j * pic_size: (j+1) * pic_size] = pic
    #import pdb;pdb.set_trace()
    plt.figure(figsize=(10,10))
    plt.imshow(figure)
    plt.savefig('clouds.png')
    plt.show()


#print(data.shape)

input_shape = (config.height, config.width, 3)
latent_dim = 10 
filters=8
kernel_size=3


inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(3):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

shape = K.int_shape(x)

x = Flatten()(x)
x = Dense(16, activation='relu')(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(3):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='sigmoid',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=3,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action="store_true")
    help_ = "Generate images over the learned space"
    parser.add_argument("-g", "--generate", help=help_, action="store_true")
    args = parser.parse_args()
    models = (encoder, decoder)
    #data = (x_test, y_test)

    if args.mse:
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))
        #Think so..
    reconstruction_loss *= input_shape[0] * input_shape[1] 
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    #plot_model(vae, to_file='vae_cnn.png', show_shapes=True)
    val_data, _  = next(my_generator(config.batch_size, config.img_dir))
    #import pdb;pdb.set_trace()

    if args.weights:
        vae.load_weights(args.weights)
    else:
        vae.fit_generator(my_generator(config.batch_size, config.img_dir),
                steps_per_epoch=steps_per_epoch,
                epochs=config.num_epochs,
                callbacks=[WandbCallback()],
                validation_data=(val_data, None))
        vae.save_weights('vae_cnn_dim10_skynet.h5')

    if args.generate:
        generate_images(vae.get_layer('decoder'))
