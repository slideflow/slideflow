import numpy as np
import tensorflow as tf
from slideflow.gan.sagan.spectral import SpectralConv2DTranspose, SpectralConv2D
from slideflow.gan.sagan.attention import SelfAttnModel
from slideflow.gan.conditional_batch_norm import ConditionalBatchNorm

# Alternative spectral normalization implementations
#from slideflow.gan.sagan.spectral_norm_conv import ConvSN2D, ConvSN2DTranspose
from slideflow.gan.sagan.spectral_normalization import SpectralNormalization
#from slideflow.gan.sagan.wzspectral import SNConv2D, SNConv2DTranspose

# TODO: missing co-normalization of real features and generated features

def masked_feature_input(feature_input, input_shape, merge, suffix, out_channels=None):
	'''Creates a feature input (calculated features of a real input image, as calculated from a separate trained model, are provided as input here),
	   multiplies the feature input by a mask (also supplied as input),
	   and adds the the masked feature input with the given model layer.
	   
	   In https://semantic-pyramid.github.io/paper.pdf, this is Figure 3b.
	'''
	mask_size = input_shape[0] if merge == 'dense' else input_shape[1:]
	mask_input = tf.keras.layers.Input(mask_size, dtype=tf.bool, name=f'mask_{suffix}')
	masked_output = tf.keras.layers.Multiply(name=f"mask_mult_{suffix}")([tf.cast(mask_input, dtype=tf.float16), feature_input])
	masked_output = tf.keras.layers.BatchNormalization()(masked_output)
	masked_output = tf.keras.layers.ReLU()(masked_output)
	if merge == 'dense':
		masked_output = tf.keras.layers.Dense(input_shape[0], name=f"mask_dense_{suffix}")(masked_output)
	elif merge == 'conv':
		masked_output = SpectralConv2DTranspose(filters=out_channels,
											   kernel_size=1,
											   strides=(1,1),
											   padding='same',
											   use_bias=False,
											   name=f'mask_conv_{suffix}')(masked_output)
	
	return mask_input, masked_output, mask_size

def create_generator(
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
	c = tf.keras.layers.Input((n_classes,), dtype=tf.bool, name='class_input')
	input_layers = [noise_input, c, feature_tensors['image'], feature_tensors['image_vgg16']]
	features_with_pool = [tf.cast(feature_tensors['fc8'], dtype=tf.float32),
					 tf.cast(feature_tensors['fc7'], dtype=tf.float32),
					 tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv0']), dtype=tf.float32),
					 tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv1']), dtype=tf.float32),
					 tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv2']), dtype=tf.float32),
					 tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv3']), dtype=tf.float32),
					 tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv4']), dtype=tf.float32),
	]
	#reconstructed_features = []
	x = noise_input
	mask_sizes = {}

	# First linear layer of generator
	x = tf.keras.layers.Dense(feature_channels[-1])(x)

	# Feature input to first linear layer
	mask_fc8, masked_input_fc8, mask_size = masked_feature_input(feature_input=tf.cast(feature_tensors['fc8'], dtype=tf.float16),
													  			input_shape=(feature_channels[-1],),
													  			merge='dense',
													  			suffix='fc8')
	x = tf.keras.layers.Add()([x, masked_input_fc8])

	mask_sizes['mask_fc8'] = mask_size
	input_layers += [mask_fc8]

	# Second linear layer of generator
	x = ConditionalBatchNorm(feature_channels[-1])(x, c)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.Dense(feature_channels[-2])(x)

	# Feature input to second linear layer
	mask_fc7, masked_input_fc7, mask_size = masked_feature_input(feature_input=tf.cast(feature_tensors['fc7'], dtype=tf.float16),
													  			input_shape=(feature_channels[-2],),
													  			merge='dense',
													  			suffix='fc7')
	x = tf.keras.layers.Add()([x, masked_input_fc7])

	mask_sizes['mask_fc7'] = mask_size
	input_layers += [mask_fc7]

	# Expand to 2D
	x = ConditionalBatchNorm(feature_channels[-2])(x, c)
	x = tf.keras.layers.ReLU()(x)
	x = tf.keras.layers.Dense(4 * 4 * feature_channels[-3])(x)
	x = tf.keras.layers.Reshape((4, 4, feature_channels[-3],))(x)

	# Convolutional blocks
	b_id = 0
	in_channel = feature_channels[-3]
	for block, ch, pad in zip(blocks, channels, padding):
		out_channel = ch * channel_multiplier
		if block == 'r':
			# ResNet Block
			resblock = ConditionalBatchNorm(in_channel, name=f'block{b_id}_bn_0')(x, c)
			resblock = tf.keras.layers.ReLU(name=f'block{b_id}_relu_0')(resblock)
			resblock = SpectralConv2DTranspose(filters=out_channel,
										 kernel_size=3,
										 strides=2,
										 padding='same' if pad == 's' else 'valid',
										 name=f'spec_block{b_id}_conv0')(resblock)
			resblock = ConditionalBatchNorm(out_channel, name=f'block{b_id}_bn_1')(resblock, c)
			resblock = tf.keras.layers.ReLU(name=f'block{b_id}_relu_1')(resblock)
			resblock = SpectralConv2DTranspose(filters=out_channel,
										 kernel_size=3,
										 strides=1,
										 padding='same',
										 name=f'spec_block{b_id}_conv1')(resblock)


			# Skip / bypass
			skip = SpectralConv2DTranspose(filters=out_channel,
								kernel_size=3,
								strides=2,
								padding='same' if pad == 's' else 'valid',
								name=f'spec_block{b_id}_skip_conv')(x)
			x = tf.keras.layers.Add()([resblock, skip])

			# Add feature inputs
			mask_input, masked_output, mask_size = masked_feature_input(feature_input=tf.cast(feature_tensors[f'conv{b_id}'], dtype=tf.float16),
																		input_shape=x.get_shape().as_list(),
																		merge='conv',
																		out_channels=out_channel,
																		suffix=f'conv{b_id}')
			x = tf.keras.layers.Add()([masked_output, x])

			mask_sizes[f'mask_conv{b_id}'] = mask_size
			input_layers += [mask_input]
			in_channel = out_channel
			b_id += 1

		elif block == 'a':
			# SelfAttention block
			x, _ = SelfAttnModel(out_channel)(x)

	# Final colorizing layer
	x = ConditionalBatchNorm(in_channel)(x, c) # I think this would ideally be conditional spectral normalization?
	x = tf.keras.layers.ReLU()(x)
	x = SpectralConv2DTranspose(filters=3, 
								kernel_size=3,
								strides=2,
								padding='valid',
								name='colorizer')(x)

	x = tf.keras.layers.Activation('tanh', dtype=tf.float32)(x)

	mask_order = ('mask_fc8', 'mask_fc7', 'mask_conv0', 'mask_conv1', 'mask_conv2', 'mask_conv3', 'mask_conv4')

	return tf.keras.models.Model(input_layers, [x] + features_with_pool), input_layers, mask_sizes, mask_order


def create_discriminator(image_size=64, filters=32, kernel_size=3):
	'''Creates a Self-attentive discriminator, as per https://arxiv.org/abs/1805.08318 

	In the https://semantic-pyramid.github.io/paper.pdf implementation, they provide no specifics about the discriminator
	but reference the above paper.

	This implementation is from https://github.com/leafinity/SAGAN-tensorflow2.0 with minor modifications
		(Added a softmax layer in order to calculate binary crossentropy loss)
	'''
	input_layers = tf.keras.layers.Input((image_size, image_size, 3), name="discriminator_input")
		
	curr_filters = filters
	x = input_layers
	for i in range(3):
		curr_filters = curr_filters * 2
		x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
		x = SpectralConv2D(filters=curr_filters,
						   kernel_size=kernel_size,
						   strides=2,
						   padding='same')(x)
		
	
	x, attn1 = SelfAttnModel(curr_filters)(x)

	for i in range(int(np.log2(image_size)) - 5):
		curr_filters = curr_filters * 2
		x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
		x = SpectralConv2D(filters=curr_filters,
						   kernel_size=kernel_size,
						   strides=2,
						   padding='same')(x)
		
	x, attn2 = SelfAttnModel(curr_filters)(x)

	x = SpectralConv2D(filters=1, kernel_size=4)(x) 
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dense(1, activation='linear', dtype=tf.float32)(x) # Added this

	return tf.keras.models.Model(input_layers, x)
