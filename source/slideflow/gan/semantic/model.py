from slideflow.gan.semantic.discriminator import *
from slideflow.gan.semantic.generator import *

'''Models for self-attention GAN discriminator and generator.'''

def create_discriminator_old(image_size=64, filters=32, kernel_size=3):
	'''Creates a Self-attentive discriminator, as per https://arxiv.org/abs/1805.08318 

	In the https://semantic-pyramid.github.io/paper.pdf implementation, they provide no specifics about the discriminator
	but reference the above paper.

	This implementation is from https://github.com/leafinity/SAGAN-tensorflow2.0 with minor modifications
		(Added a softmax layer in order to calculate binary crossentropy loss)
	'''
	input_layers = tf.keras.layers.Input((image_size, image_size, 3), name="discriminator_input")
		
	curr_filters = filters
	x = input_layers

	x = SpectralConv2D(filters=curr_filters,
						kernel_size=kernel_size,
						strides=1,
						padding='same')(x)

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
		
	#x, attn2 = SelfAttnModel(curr_filters)(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	x = SpectralConv2D(filters=1, kernel_size=4)(x)
	x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
	x = tf.keras.layers.Flatten()(x)
	x = DenseSN(1, activation='linear', dtype=tf.float32)(x) # Added this

	return tf.keras.models.Model(input_layers, x)