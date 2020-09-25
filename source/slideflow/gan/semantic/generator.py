import numpy as np
import tensorflow as tf
from slideflow.gan.sagan.spectral import SpectralConv2DTranspose, SpectralConv2D
from slideflow.gan.sagan.attention import SelfAttnModel
from slideflow.gan.conditional_batch_norm import ConditionalBatchNorm

from tensorflow_gan.examples.self_attention_estimator.ops import ConditionalBatchNorm as SACondBN

# Alternative spectral normalization implementations
from slideflow.gan.sagan.spectral_norm_conv import DenseSN #ConvSN2D, ConvSN2DTranspose
from slideflow.gan.sagan.spectral_normalization import SpectralNormalization
#from slideflow.gan.sagan.wzspectral import SNConv2D, SNConv2DTranspose

def block(x, labels, out_channels, num_classes):
	in_channel = x.get_shape().as_list()[-1]
	x_0 = x
	x = ConditionalBatchNorm(in_channel)(x, labels)
	x = tf.keras.layers.LeakyReLU()(x)
	x = SpectralConv2DTranspose(out_channels, kernel_size=4, strides=2, padding='same')(x)
	x = ConditionalBatchNorm(out_channels)(x, labels)
	x = tf.keras.layers.LeakyReLU()(x)
	x = SpectralConv2DTranspose(out_channels, kernel_size=3, strides=1, padding='same')(x)
	x_0 = SpectralConv2DTranspose(out_channels, kernel_size=1, strides=2, padding='same')(x_0)
	return tf.keras.layers.Add()([x_0, x])

def alt_block(x, labels, out_channels, num_classes):
	in_channel = x.get_shape().as_list()[-1]
	x_0 = x
	x = SpectralConv2DTranspose(in_channel, kernel_size=4, strides=2, padding='same')(x)
	x = tf.keras.layers.LeakyReLU()(x)
	x = ConditionalBatchNorm(in_channel)(x, labels)
	x = SpectralConv2DTranspose(out_channels, kernel_size=3, strides=1, padding='same')(x)
	x_0 = SpectralConv2DTranspose(out_channels, kernel_size=1, strides=2, padding='same')(x_0)
	merged = tf.keras.layers.Add()([x_0, x])
	merged = tf.keras.layers.LeakyReLU()(merged)
	merged = ConditionalBatchNorm(out_channels)(merged, labels)
	return merged

def create_generator(
	feature_tensors,
	n_classes,
	z_dim=128,
	gf_dim=64,
	use_alt_block=False
):
	'''Creates a generator.'''
	mask_order = ('mask_conv0', 'mask_conv1', 'mask_conv2', 'mask_conv3', 'mask_conv4')
	dummy_mask_sizes = {m: (1,) for m in mask_order}
	resblock = alt_block if use_alt_block else block

	# Inputs
	noise_input = tf.keras.layers.Input((z_dim,), name='noise_input')
	class_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='class_input')
	c = tf.one_hot(class_input, n_classes, dtype=tf.int32)
	c = tf.squeeze(c, axis=1)

	# Linear layers
	act0 = DenseSN(gf_dim * 16 * 4 * 4)(noise_input)
	act0 = tf.keras.layers.Reshape((4, 4, gf_dim * 16))(act0)

	# Conv layers
	act1 = resblock(act0, c, gf_dim * 16, n_classes)
	act2 = resblock(act1, c, gf_dim * 8, n_classes)
	act3 = resblock(act2, c, gf_dim * 4, n_classes)
	act3, _ = SelfAttnModel(gf_dim * 4)(act3)
	act4 = resblock(act3, c, gf_dim * 2 , n_classes)
	act5 = resblock(act4, c, gf_dim, n_classes)
	act5 = ConditionalBatchNorm(gf_dim)(act5, c)
	act5 = tf.keras.layers.LeakyReLU()(act5)
	act6 = SpectralConv2DTranspose(3, kernel_size=3, strides=1, padding='same')(act5)
	out = tf.keras.layers.Activation('tanh', dtype=tf.float32)(act6)

	return tf.keras.models.Model([noise_input, class_input], out), [noise_input, class_input], dummy_mask_sizes, mask_order