import numpy as np
import tensorflow as tf
from sagan.spectral import SpectralConv2D, SpectralConv2DTranspose
from sagan.attention import SelfAttnModel

# From https://github.com/leafinity/SAGAN-tensorflow2.0

def create_generator(image_size=64, z_dim=100, filters=64, kernel_size=4):
        
    input_layers = tf.keras.layers.Input((z_dim,))
    x = tf.keras.layers.Reshape((1, 1, z_dim))(input_layers)

    repeat_num = int(np.log2(image_size)) - 1
    mult = 2 ** (repeat_num - 1)
    curr_filters = filters * mult

    for i in range(3):
        curr_filters = curr_filters // 2
        strides = 4 if i == 0 else 2
        x = SpectralConv2DTranspose(filters=curr_filters,
                                    kernel_size=kernel_size,
                                    strides=strides, 
                                    padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x, attn1 = SelfAttnModel(curr_filters)(x)
        
    for i in range(repeat_num - 4):
        curr_filters = curr_filters // 2
        x = SpectralConv2DTranspose(filters=curr_filters, 
                                    kernel_size=kernel_size,
                                    strides=2,
                                    padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x, attn2 = SelfAttnModel(curr_filters)(x)

    x = SpectralConv2DTranspose(filters=3, 
                                kernel_size=kernel_size,
                                strides=2,
                                padding='same')(x)
    x = tf.keras.layers.Activation('tanh')(x)

    return tf.keras.models.Model(input_layers, [x, attn1, attn2])


def create_discriminator(image_size=64, filters=64, kernel_size=4):
    input_layers = tf.keras.layers.Input((image_size, image_size, 3))
        
    curr_filters = filters
    x = input_layers
    for i in range(3):
        curr_filters = curr_filters * 2
        x = SpectralConv2D(filters=curr_filters,
                           kernel_size=kernel_size,
                           strides=2,
                           padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
    x, attn1 = SelfAttnModel(curr_filters)(x)

    for i in range(int(np.log2(image_size)) - 5):
        curr_filters = curr_filters * 2
        x = SpectralConv2D(filters=curr_filters,
                           kernel_size=kernel_size,
                           strides=2,
                           padding='same')(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        
            
    x, attn2 = SelfAttnModel(curr_filters)(x)

    x = SpectralConv2D(filters=1, kernel_size=4)(x)
    x = tf.keras.layers.Flatten()(x)

    return tf.keras.models.Model(input_layers, [x, attn1, attn2])
