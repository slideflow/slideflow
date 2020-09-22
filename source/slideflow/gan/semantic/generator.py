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

def masked_feature_input(feature_input, input_shape, merge, suffix, out_channels=None):
	'''Creates a feature input (calculated features of a real input image, as calculated from a separate trained model, are provided as input here),
	   multiplies the feature input by a mask (also supplied as input),
	   and adds the the masked feature input with the given model layer.
	   
	   In https://semantic-pyramid.github.io/paper.pdf, this is Figure 3b.
	'''
	mask_size = input_shape[0] if merge == 'dense' else input_shape[1:]
	mask_input = tf.keras.layers.Input(mask_size, dtype=tf.bool, name=f'mask_{suffix}')
	masked_output = tf.keras.layers.Multiply(name=f"mask_mult_{suffix}")([tf.cast(mask_input, dtype=tf.float16), feature_input])
	if merge == 'dense':
		masked_output = tf.keras.layers.Dense(input_shape[0], name=f"mask_dense_{suffix}")(masked_output)
	elif merge == 'conv':
		masked_output = SpectralConv2DTranspose(filters=out_channels,
											   kernel_size=1,
											   strides=(1,1),
											   padding='same',
											   use_bias=False,
											   name=f'mask_conv_{suffix}')(masked_output)
	masked_output = tf.keras.layers.LeakyReLU()(masked_output)
	return mask_input, masked_output, mask_size

def create_generator_old(
	feature_tensors,
	n_classes,
	z_dim=100,
	feature_channels=(64,128,256,512,512,4096,4096),
	channels=(8,8,4,2,2,1),
	blocks='rrrrar',
	padding='vsvsvv',
	channel_multiplier=64,
):
	'''Creates a generator per https://semantic-pyramid.github.io/paper.pdf.'''
	noise_input = tf.keras.layers.Input((z_dim,), name='noise_input')
	#c = tf.keras.layers.Input((n_classes,), dtype=tf.int32, name='class_input')
	c = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='class_input')
	input_layers = [noise_input, c]
	c = tf.one_hot(c, n_classes, dtype=tf.float32)
	c = tf.squeeze(c, axis=1)
	#input_layers = [noise_input, c, feature_tensors['image'], feature_tensors['image_vgg16']]
	
	#features_with_pool = [#tf.cast(feature_tensors['fc8'], dtype=tf.float32),
	#					  #tf.cast(feature_tensors['fc7'], dtype=tf.float32),
	#					  tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv0']), dtype=tf.float32),
	#					  tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv1']), dtype=tf.float32),
	#					  tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv2']), dtype=tf.float32),
	#					  tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv3']), dtype=tf.float32),
	#					  tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv4']), dtype=tf.float32),
	#]
	features_with_pool = []
	#reconstructed_features = []
	x = noise_input
	mask_sizes = {}

	# First linear layer of generator
	#x = tf.keras.layers.Dense(feature_channels[-1])(x)
	#x = tf.keras.layers.ReLU()(x)

	# Feature input to first linear layer
	#mask_fc8, masked_input_fc8, mask_size = masked_feature_input(feature_input=tf.cast(feature_tensors['fc8'], dtype=tf.float16),
	#												  			input_shape=(feature_channels[-1],),
	#												  			merge='dense',
	#												  			suffix='fc8')
	#x = tf.keras.layers.Add()([x, masked_input_fc8])
	#x = ConditionalBatchNorm(feature_channels[-1])(x, c)

	#mask_sizes['mask_fc8'] = mask_size
	#input_layers += [mask_fc8]

	# Second linear layer of generator
	#x = tf.keras.layers.Dense(feature_channels[-2])(x)
	#x = tf.keras.layers.ReLU()(x)

	# Feature input to second linear layer
	#mask_fc7, masked_input_fc7, mask_size = masked_feature_input(feature_input=tf.cast(feature_tensors['fc7'], dtype=tf.float16),
	#												  			input_shape=(feature_channels[-2],),
	#												  			merge='dense',
	#												  			suffix='fc7')
	#x = tf.keras.layers.Add()([x, masked_input_fc7])
	#x = ConditionalBatchNorm(feature_channels[-2])(x, c)

	#mask_sizes['mask_fc7'] = mask_size
	#input_layers += [mask_fc7]

	# Expand to 2D
	x = DenseSN(4 * 4 * feature_channels[-3], dtype=tf.float32)(x)
	x = tf.keras.layers.Reshape((4, 4, feature_channels[-3],))(x)
	#x = tf.keras.layers.ReLU()(x)
	#x = ConditionalBatchNorm(feature_channels[-3])(x, c)

	# Convolutional blocks
	b_id = 0
	in_channel = feature_channels[-3]
	for block, ch, pad in zip(blocks, channels, padding):
		out_channel = ch * channel_multiplier
		if block == 'r':
			# ResNet Block
			resblock = SpectralConv2DTranspose(filters=out_channel,
										 kernel_size=3,
										 strides=2,
										 padding='same' if pad == 's' else 'valid',
										 name=f'spec_block{b_id}_conv0')(x)
			resblock = tf.keras.layers.LeakyReLU(name=f'block{b_id}_relu_0')(resblock)
			resblock = ConditionalBatchNorm(out_channel, name=f'block{b_id}_bn_0')(resblock, c)

			resblock = SpectralConv2DTranspose(filters=out_channel,
										 kernel_size=3,
										 strides=1,
										 padding='same',
										 name=f'spec_block{b_id}_conv1')(resblock)
			resblock = tf.keras.layers.LeakyReLU(name=f'block{b_id}_relu_1')(resblock)

			# Skip / bypass
			skip = SpectralConv2DTranspose(filters=out_channel,
								kernel_size=3,
								strides=2,
								padding='same' if pad == 's' else 'valid',
								name=f'spec_block{b_id}_skip_conv')(x)
			skip = tf.keras.layers.LeakyReLU(name=f'block{b_id}_relu_skip')(skip)
			x = tf.keras.layers.Add()([resblock, skip])
			
			# Add feature inputs
			mask_input, masked_output, mask_size = masked_feature_input(feature_input=tf.cast(feature_tensors[f'conv{b_id}'], dtype=tf.float16),
																		input_shape=x.get_shape().as_list(),
																		merge='conv',
																		out_channels=out_channel,
																		suffix=f'conv{b_id}')
			#x = tf.keras.layers.Add()([masked_output, x])
			x = ConditionalBatchNorm(out_channel, name=f'block{b_id}_bn_end')(x, c)

			mask_sizes[f'mask_conv{b_id}'] = mask_size
			#input_layers += [mask_input]
			in_channel = out_channel
			b_id += 1

		elif block == 'a':
			# SelfAttention block
			x, _ = SelfAttnModel(out_channel)(x)

	# Final colorizing layer
	x = SpectralConv2DTranspose(filters=3, 
								kernel_size=3,
								strides=2,
								padding='valid',
								name='colorizer')(x)

	x = tf.keras.layers.Activation('tanh', dtype=tf.float32)(x)

	#mask_order = ('mask_fc8', 'mask_fc7', 'mask_conv0', 'mask_conv1', 'mask_conv2', 'mask_conv3', 'mask_conv4')
	mask_order = ('mask_conv0', 'mask_conv1', 'mask_conv2', 'mask_conv3', 'mask_conv4')

	return tf.keras.models.Model(input_layers, [x] + features_with_pool), input_layers, mask_sizes, mask_order

def usample(x):
	_, image_height, image_width, n_channels = x.get_shape().as_list()
	expanded_x = tf.expand_dims(tf.expand_dims(x, axis=2), axis=4)
	after_tile = tf.tile(expanded_x, [1, 1, 2, 1, 2, 1])
	return tf.keras.layers.Reshape((image_height * 2, image_width * 2, n_channels))(after_tile)

def block(x, labels, out_channels, num_classes):
	in_channel = x.get_shape().as_list()[-1]
	x_0 = x
	x = ConditionalBatchNorm(in_channel)(x, labels)
	x = tf.keras.layers.LeakyReLU()(x)
	x = usample(x)
	x = SpectralConv2D(out_channels, kernel_size=3, strides=1, padding='same')(x)
	x = ConditionalBatchNorm(out_channels)(x, labels)
	x = tf.keras.layers.LeakyReLU()(x)
	x = SpectralConv2D(out_channels, kernel_size=3, strides=1, padding='same')(x)
	x_0 = usample(x_0)
	x_0 = SpectralConv2D(out_channels, kernel_size=1, strides=1, padding='same')(x_0)
	return tf.keras.layers.Add()([x_0, x])

def create_generator(
	feature_tensors,
	n_classes,
	z_dim=128,
	gf_dim=64
):
	'''Creates a generator.'''
	mask_order = ('mask_conv0', 'mask_conv1', 'mask_conv2', 'mask_conv3', 'mask_conv4')
	dummy_mask_sizes = {m: (1,) for m in mask_order}

	# Inputs
	noise_input = tf.keras.layers.Input((z_dim,), name='noise_input')
	class_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='class_input')
	c = tf.one_hot(class_input, n_classes, dtype=tf.int32)
	c = tf.squeeze(c, axis=1)

	# Linear layers
	act0 = DenseSN(gf_dim * 16 * 4 * 4)(noise_input)
	act0 = tf.keras.layers.Reshape((4, 4, gf_dim * 16))(act0)

	# Conv layers
	act1 = block(act0, c, gf_dim * 16, n_classes)
	act2 = block(act1, c, gf_dim * 16, n_classes)
	act3 = block(act2, c, gf_dim * 8, n_classes)
	act3, _ = SelfAttnModel(gf_dim * 8)(act3)
	act4 = block(act3, c, gf_dim * 4, n_classes)
	act5 = block(act4, c, gf_dim * 2 , n_classes)
	act6 = block(act5, c, gf_dim, n_classes)
	act6 = ConditionalBatchNorm(gf_dim)(act6, c)
	act6 = tf.keras.layers.LeakyReLU()(act6)
	act7 = SpectralConv2D(3, kernel_size=3, strides=1, padding='same')(act6)
	out = tf.keras.layers.Activation('tanh', dtype=tf.float32)(act7)
	
	return tf.keras.models.Model([noise_input, class_input], out), [noise_input, class_input], dummy_mask_sizes, mask_order