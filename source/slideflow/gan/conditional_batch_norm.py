# From https://stackoverflow.com/questions/54101593/conditional-batch-normalization-in-keras
from tensorflow.keras import regularizers, initializers, constraints
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Input, InputSpec
from tensorflow.keras.models import Model
import tensorflow as tf

global c1, c2, c3
c1 = K.variable([0])
c2 = K.variable([0])
c3 = K.variable([0])

class ConditionalBatchNorm(tf.keras.layers.Layer):
	def __init__(self, channel, **kwargs):
		super(ConditionalBatchNorm, self).__init__(**kwargs)
		self.linear_gamma = tf.keras.layers.Dense(channel, 
			kernel_initializer = tf.ones,
			use_bias = False,
			name = 'linear_gamma')
		self.linear_beta = tf.keras.layers.Dense(channel, 
			kernel_initializer = tf.zeros,
			use_bias = False,
			name = 'linear_beta')
		self.batchnorm = tf.keras.layers.BatchNormalization()

	def call(self, x, c, train = True):
		n_dim = len(x.get_shape().as_list())
		x = self.batchnorm(x)
		gamma = self.linear_gamma(c)
		beta = self.linear_beta(c)
		for i in range(n_dim - 2):
			gamma = tf.expand_dims(gamma, axis = 1)
			beta = tf.expand_dims(beta, axis = 1)
		return x * gamma + beta

class ConditionalBatchNormalization(Layer):
	"""Conditional Batch normalization layer.
	"""

	def __init__(self, 
				axis=-1,
				momentum=0.99,
				epsilon=1e-3,
				center=True,
				scale=True,
				beta_initializer='zeros',
				gamma_initializer='ones',
				moving_mean_initializer='zeros',
				moving_variance_initializer='ones',
				beta_regularizer=None,
				gamma_regularizer=None,
				beta_constraint=None,
				gamma_constraint=None,
				**kwargs):
		super(ConditionalBatchNormalization, self).__init__(**kwargs)
		self.axis = axis
		self.momentum = momentum
		self.epsilon = epsilon
		self.center = center
		self.scale = scale
		self.beta_initializer = initializers.get(beta_initializer)
		self.gamma_initializer = initializers.get(gamma_initializer)
		self.moving_mean_initializer = initializers.get(moving_mean_initializer)
		self.moving_variance_initializer = (
			initializers.get(moving_variance_initializer))
		self.beta_regularizer = regularizers.get(beta_regularizer)
		self.gamma_regularizer = regularizers.get(gamma_regularizer)
		self.beta_constraint = constraints.get(beta_constraint)
		self.gamma_constraint = constraints.get(gamma_constraint)


	def build(self, input_shape):

		dim = input_shape[0][self.axis]
		if dim is None:
			raise ValueError('Axis ' + str(self.axis) + ' of '
							'input tensor should have a defined dimension '
							'but the layer received an input with shape ' +
							str(input_shape[0]) + '.')

		shape = (dim,)

		if self.scale:
			self.gamma1 = self.add_weight(shape=shape,
										name='gamma',
										initializer=self.gamma_initializer,
										regularizer=self.gamma_regularizer,
										constraint=self.gamma_constraint)
			self.gamma2 = self.add_weight(shape=shape,
										name='gamma',
										initializer=self.gamma_initializer,
										regularizer=self.gamma_regularizer,
										constraint=self.gamma_constraint)
			self.gamma3 = self.add_weight(shape=shape,
										name='gamma',
										initializer=self.gamma_initializer,
										regularizer=self.gamma_regularizer,
										constraint=self.gamma_constraint)
		else:
			self.gamma1 = None
			self.gamma2 = None
			self.gamma3 = None

		if self.center:
			self.beta1 = self.add_weight(shape=shape,
										name='beta',
										initializer=self.beta_initializer,
										regularizer=self.beta_regularizer,
										constraint=self.beta_constraint)

			self.beta2 = self.add_weight(shape=shape,
										name='beta',
										initializer=self.beta_initializer,
										regularizer=self.beta_regularizer,
										constraint=self.beta_constraint)

			self.beta3 = self.add_weight(shape=shape,
										name='beta',
										initializer=self.beta_initializer,
										regularizer=self.beta_regularizer,
										constraint=self.beta_constraint)
		else:
			self.beta1 = None
			self.beta2 = None
			self.beta3 = None

		self.moving_mean = self.add_weight(
			shape=shape,
			name='moving_mean',
			initializer=self.moving_mean_initializer,
			trainable=False)
		self.moving_variance = self.add_weight(
			shape=shape,
			name='moving_variance',
			initializer=self.moving_variance_initializer,
			trainable=False)

		super(ConditionalBatchNormalization, self).build(input_shape) 

	def call(self, inputs, training=None):

		input_shape = K.int_shape(inputs[0])
		c1 = inputs[1][0]
		c2 = inputs[2][0]

		# Prepare broadcasting shape.
		ndim = len(input_shape)
		reduction_axes = list(range(len(input_shape)))
		del reduction_axes[self.axis]
		broadcast_shape = [1] * len(input_shape)
		broadcast_shape[self.axis] = input_shape[self.axis]

		# Determines whether broadcasting is needed.
		needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

		def normalize_inference():
			if needs_broadcasting:
				# In this case we must explicitly broadcast all parameters.
				broadcast_moving_mean = K.reshape(self.moving_mean,
												broadcast_shape)
				broadcast_moving_variance = K.reshape(self.moving_variance,
													broadcast_shape)
				if self.center:
					broadcast_beta = \
						tf.case({
									c1: lambda: K.reshape(self.beta1,
														broadcast_shape),
									c2: lambda: K.reshape(self.beta2,
														broadcast_shape)
								},
									default=lambda: K.reshape(self.beta3,
															broadcast_shape)
								)

				else:
					broadcast_beta = None

				if self.scale:

					broadcast_gamma = \
						tf.case({
									c1: lambda: K.reshape(self.gamma1,
														broadcast_shape),
									c2: lambda: K.reshape(self.gamma2,
														broadcast_shape)
								},
									default=lambda: K.reshape(self.gamma3,
															broadcast_shape)
								)

				else:
					broadcast_gamma = None

				return K.batch_normalization(
					inputs[0],
					broadcast_moving_mean,
					broadcast_moving_variance,
					broadcast_beta,
					broadcast_gamma,
					axis=self.axis,
					epsilon=self.epsilon)
			else:
				out = \
				tf.case({
						c1: lambda: K.batch_normalization(
											inputs[0],
											self.moving_mean,
											self.moving_variance,
											self.beta1,
											self.gamma1,
											axis=self.axis,
											epsilon=self.epsilon),
						c2: lambda: K.batch_normalization(
											inputs[0],
											self.moving_mean,
											self.moving_variance,
											self.beta2,
											self.gamma2,
											axis=self.axis,
											epsilon=self.epsilon)
					},
						default=lambda: K.batch_normalization(
											inputs[0],
											self.moving_mean,
											self.moving_variance,
											self.beta3,
											self.gamma3,
											axis=self.axis,
											epsilon=self.epsilon)
							)

				return out

		# If the learning phase is *static* and set to inference:
		if training in {0, False}:
			return normalize_inference()


		# If the learning is either dynamic, or set to training:
		normed_training, mean, variance = \
			tf.case({
						c1: lambda: K.normalize_batch_in_training(
								inputs[0], self.gamma1, self.beta1, reduction_axes,
								epsilon=self.epsilon),
						c2: lambda: K.normalize_batch_in_training(
								inputs[0], self.gamma2, self.beta2, reduction_axes,
								epsilon=self.epsilon)
					},
						default=lambda: K.normalize_batch_in_training(
								inputs[0], self.gamma3, self.beta3, reduction_axes,
								epsilon=self.epsilon)
					)

		print(normed_training)

		if K.backend() != 'cntk':
			sample_size = K.prod([K.shape(inputs[0])[axis]
								for axis in reduction_axes])
			sample_size = K.cast(sample_size, dtype=K.dtype(inputs[0]))
			if K.backend() == 'tensorflow' and sample_size.dtype != 'float32':
				sample_size = K.cast(sample_size, dtype='float32')

			# sample variance - unbiased estimator of population variance
			variance *= sample_size / (sample_size - (1.0 + self.epsilon))

		self.add_update([K.moving_average_update(self.moving_mean,
												mean,
												self.momentum),
						K.moving_average_update(self.moving_variance,
												variance,
												self.momentum)],
						inputs[0])

		# Pick the normalized form corresponding to the training phase.

		return K.in_train_phase(normed_training,
								normalize_inference,
								training=training)

	def get_config(self):
		config = {
			'axis': self.axis,
			'momentum': self.momentum,
			'epsilon': self.epsilon,
			'center': self.center,
			'scale': self.scale,
			'beta_initializer': initializers.serialize(self.beta_initializer),
			'gamma_initializer': initializers.serialize(self.gamma_initializer),
			'moving_mean_initializer':
				initializers.serialize(self.moving_mean_initializer),
			'moving_variance_initializer':
				initializers.serialize(self.moving_variance_initializer),
			'beta_regularizer': regularizers.serialize(self.beta_regularizer),
			'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
			'beta_constraint': constraints.serialize(self.beta_constraint),
			'gamma_constraint': constraints.serialize(self.gamma_constraint)
		}
		base_config = super(ConditionalBatchNormalization, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def compute_output_shape(self, input_shape):

		return input_shape[0]

if __name__ == '__main__':
    x = Input((10,))
    c1 = Input(batch_shape=(1,), dtype=tf.bool)
    c2 = Input(batch_shape=(1,), dtype=tf.bool)
    h = ConditionalBatchNormalization()([x, c1, c2])
    model = Model([x, c1, c2], h)
    model.compile(optimizer=Adam(1e-4), loss='mse')

    c1 = K.constant([False]*100, dtype=tf.bool)
    c2 = K.constant([True]*100, dtype=tf.bool)

    X = np.random.rand(100, 10)
    Y = np.random.rand(100, 10)


    model.train_on_batch(x=[X, c1, c2], y=Y)

    c1 = K.constant([False]*100, dtype=tf.bool)
    c2 = K.constant([True]*100, dtype=tf.bool)

    model.train_on_batch(x=[X, c1, c2], y=Y)