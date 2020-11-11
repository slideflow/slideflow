import numpy as np
import tensorflow as tf

from slideflow.gan.sagan.spectral import SpectralConv2DTranspose, SpectralConv2D
from slideflow.gan.sagan.attention import SelfAttnModel
from slideflow.gan.conditional_batch_norm import ConditionalBatchNorm
from slideflow.gan.sagan.spectral_norm_conv import DenseSN
from slideflow.gan.sagan.spectral_normalization import SpectralNormalization

def dsample(x):
	x = tf.keras.layers.AveragePooling2D((2,2), strides=(2,2), padding='valid')(x)
	return x

def optimized_block(x, out_channels):
	x_0 = x
	x = SpectralConv2D(out_channels, kernel_size=3, strides=1, padding='same')(x)
	x = tf.keras.layers.LeakyReLU()(x)
	x = SpectralConv2D(out_channels, kernel_size=3, strides=1, padding='same')(x)
	x = dsample(x)
	x_0 = dsample(x_0)
	x_0 = SpectralConv2D(out_channels, kernel_size=1, strides=1, padding='same')(x_0)		
	return tf.keras.layers.Add()([x_0, x])

def block(x, out_channels, downsample=True):
	input_channels = x.get_shape().as_list()[-1]
	x_0 = x
	x = tf.keras.layers.LeakyReLU()(x)
	x = SpectralConv2D(out_channels, kernel_size=3, strides=1, padding='same')(x)
	x = tf.keras.layers.LeakyReLU()(x)
	x = SpectralConv2D(out_channels, kernel_size=3, strides=1, padding='same')(x)
	if downsample:
		x = dsample(x)
	if downsample or input_channels != out_channels:
		x_0 = SpectralConv2D(out_channels, kernel_size=1, strides=1, padding='same')(x_0)
		if downsample:
			x_0 = dsample(x_0)
	return tf.keras.layers.Add()([x_0, x])

def create_discriminator(image_size, df_dim=64):
	image = tf.keras.layers.Input((image_size, image_size, 3), name="discriminator_input")
	h0 = optimized_block(image, df_dim)
	h1 = block(h0, df_dim * 2)
	h1, _ = SelfAttnModel(df_dim * 2)(h1)
	h2 = block(h1, df_dim * 4)
	h3 = block(h2, df_dim * 8)
	h4 = block(h3, df_dim * 16)
	h5 = block(h4, df_dim * 16, downsample=False)
	h5_act = tf.keras.layers.LeakyReLU()(h5)
	#h6 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1,2]))(h5_act)
	h6 = tf.keras.layers.Flatten()(h5_act)
	output = DenseSN(1, dtype=tf.float32)(h6)
	return tf.keras.models.Model(image, output)

def create_discriminator_large(image_size, df_dim=64):
	image = tf.keras.layers.Input((image_size, image_size, 3), name="discriminator_input")
	h0 = optimized_block(image, df_dim)
	h1 = block(h0, df_dim * 2)
	h1, _ = SelfAttnModel(df_dim * 2)(h1)
	h2 = block(h1, df_dim * 4)
	h3 = block(h2, df_dim * 8)
	h4 = block(h3, df_dim * 16)
	h5 = block(h4, df_dim * 16)
	h6 = block(h5, df_dim * 32)

	h7 = block(h6, df_dim * 32, downsample=False)
	h7_act = tf.keras.layers.LeakyReLU()(h7)
	#h6 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=[1,2]))(h5_act)
	h8 = tf.keras.layers.Flatten()(h7_act)
	output = DenseSN(1, dtype=tf.float32)(h8)
	return tf.keras.models.Model(image, output)