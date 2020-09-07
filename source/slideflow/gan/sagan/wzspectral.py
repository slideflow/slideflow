# From https://github.com/WangZesen/Spectral-Normalization-GAN/blob/master/SNLayer.py
import tensorflow as tf
from functools import partial

class SNConv2DTranspose(tf.keras.layers.Layer):
	def __init__(self,
			filters, 
			kernel_size, 
			stride = (1, 1),
			padding = 'same',
			output_padding = None,
			data_format = 'NHWC',
			dilation_rate = (1, 1),
			activation = None,
			use_bias = True,
			kernel_initializer = tf.keras.initializers.Orthogonal,
			bias_initializer = tf.zeros,
			**kwargs):
		super(SNConv2DTranspose, self).__init__(**kwargs)
		self.filters = filters
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding.upper()
		assert self.padding in ['SAME', 'VALID']
		self.output_padding = output_padding
		self.data_format = data_format
		self.dilation_rate = dilation_rate
		self.activation = activation
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		if isinstance(kernel_size, int):
			self.kernel_shape = [kernel_size, kernel_size]
		else:
			self.kernel_shape = kernel_size
		
	def build(self, input_shape):
		self.kernel = self.add_variable("kernel",
										shape = self.kernel_shape + [self.filters, int(input_shape[-1])],
										initializer = self.kernel_initializer)
		if self.use_bias:
			self.bias = self.add_variable("bias",
											shape = [1, self.filters],
											initializer = self.bias_initializer)
		self.u = self.add_variable("u",
									shape = [self.filters, 1],
									initializer = tf.random.normal,
									trainable = False)

		self._output_shape = input_shape.as_list()
		self._output_shape[1] *= self.stride
		self._output_shape[2] *= self.stride
		self._output_shape[3] = self.filters

	def call(self, x, test = False):
		def _power_iteration(w, test = False):
			_v = tf.matmul(tf.transpose(w), self.u)
			v = tf.math.l2_normalize(_v, axis = 0)
			_u = tf.matmul(w, v)
			u = tf.math.l2_normalize(_u, axis = 0)
			if not test:
				self.u.assign(u)

			v = tf.stop_gradient(v)
			u = tf.stop_gradient(u)
			w = w / (tf.matmul(tf.matmul(tf.transpose(u), w), v))
			return w

		_kernel = tf.transpose(self.kernel, perm = [0, 1, 3, 2])
		w = tf.transpose(tf.reshape(_kernel, [-1, self.filters]))
		w = tf.transpose(_power_iteration(w, test = test))
		dims = self.kernel.get_shape().as_list()
		_kernel = tf.reshape(w, [dims[0], dims[1], dims[3], dims[2]])
		_kernel = tf.transpose(_kernel, perm = [0, 1, 3, 2])

		self._output_shape[0] = x.get_shape().as_list()[0]
		x = tf.nn.conv2d_transpose(x, _kernel, self._output_shape, self.stride, self.padding, self.data_format, self.dilation_rate)
		x = x + self.bias
		if callable(self.activation):
			x = self.activation(x)

		return x

class SNConv2D(tf.keras.layers.Layer):
	def __init__(self, filters, 
			kernel_size, 
			stride = (1, 1),
			padding = 'same',
			data_format = 'NHWC',
			dilation_rate = (1, 1),
			activation = None,
			use_bias = True,
			kernel_initializer = tf.keras.initializers.Orthogonal,
			bias_initializer = tf.zeros,
			**kwargs
			):
		super(SNConv2D, self).__init__(**kwargs)

		self.filters = filters
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding.upper()
		assert self.padding in ['SAME', 'VALID']
		self.data_format = data_format
		self.dilation_rate = dilation_rate
		self.activation = activation
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer

		if isinstance(kernel_size, int):
			self.kernel_shape = [kernel_size, kernel_size]
		else:
			self.kernel_shape = kernel_size
		
	def build(self, input_shape):
		self.kernel = self.add_variable("kernel", 
										shape = self.kernel_shape + [int(input_shape[-1]), self.filters],
										initializer = self.kernel_initializer)

		if self.use_bias:
			self.bias = self.add_variable("bias", 
											shape = [1, self.filters],
											initializer = self.bias_initializer)

		self.u = self.add_variable("u",
									shape = [self.filters, 1],
									initializer = tf.random.normal,
									trainable = False)

	def call(self, x, test = False):
		def _power_iteration(w, test = False):
			_v = tf.matmul(tf.transpose(w), self.u)
			v = tf.math.l2_normalize(_v, axis = 0)
			_u = tf.matmul(w, v)
			u = tf.math.l2_normalize(_u, axis = 0)
			if not test:
				self.u.assign(u)

			v = tf.stop_gradient(v)
			u = tf.stop_gradient(u)
			w = w / (tf.matmul(tf.matmul(tf.transpose(u), w), v))
			return w

		w = tf.transpose(tf.reshape(self.kernel, [-1, self.filters]))
		w = tf.transpose(_power_iteration(w, test = test))
		w = tf.reshape(w, self.kernel.get_shape().as_list())

		x = tf.nn.conv2d(x, w, self.stride, self.padding, self.data_format, self.dilation_rate)
		if self.use_bias:
			x = x + self.bias

		if callable(self.activation):
			x = self.activation(x)

		return x

class SNConv1D(tf.keras.layers.Layer):
	def __init__(self, filters, 
			kernel_size, 
			stride = 1,
			padding = 'same',
			data_format = 'NHWC',
			dilation_rate = 1,
			activation = None,
			use_bias = True,
			kernel_initializer = tf.keras.initializers.Orthogonal,
			bias_initializer = tf.zeros,
			**kwargs
			):
		super(SNConv2D, self).__init__(**kwargs)

		self.filters = filters
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding.upper()
		assert self.padding in ['SAME', 'VALID', 'CAUSAL']
		self.data_format = data_format
		self.dilation_rate = dilation_rate
		self.activation = activation
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer

		if isinstance(kernel_size, int):
			self.kernel_shape = [kernel_size, kernel_size]
		else:
			self.kernel_shape = kernel_size
		
	def build(self, input_shape):
		self.kernel = self.add_variable("kernel", 
										shape = self.kernel_shape + [int(input_shape[-1]), self.filters],
										initializer = self.kernel_initializer)

		if self.use_bias:
			self.bias = self.add_variable("bias", 
											shape = [1, self.filters],
											initializer = self.bias_initializer)

		self.u = self.add_variable("u",
									shape = [self.filters, 1],
									initializer = tf.random.normal,
									trainable = False)

	def call(self, x, test = False):
		def _power_iteration(w, test = False):
			_v = tf.matmul(tf.transpose(w), self.u)
			v = tf.math.l2_normalize(_v, axis = 0)
			_u = tf.matmul(w, v)
			u = tf.math.l2_normalize(_u, axis = 0)
			if not test:
				self.u.assign(u)

			v = tf.stop_gradient(v)
			u = tf.stop_gradient(u)
			w = w / (tf.matmul(tf.matmul(tf.transpose(u), w), v))
			return w

		w = tf.transpose(tf.reshape(self.kernel, [-1, self.filters]))
		w = tf.transpose(_power_iteration(w, test = test))
		w = tf.reshape(w, self.kernel.get_shape().as_list())

		x = tf.nn.conv2d(x, w, self.stride, self.padding, self.data_format, self.dilation_rate)
		if self.use_bias:
			x = x + self.bias

		if callable(self.activation):
			x = self.activation(x)

		return x

class SNDense(tf.keras.layers.Layer):
	def __init__(self, 
			units, 
			activation = None,
			use_bias = True,
			kernel_initializer = tf.keras.initializers.Orthogonal,
			bias_initializer = tf.zeros,
			**kwargs):
		
		super(SNDense, self).__init__(**kwargs)
		self.units = units
		self.activation = activation
		self.use_bias = use_bias
		self.kernel_initializer = kernel_initializer
		self.bias_initializer = bias_initializer
		
	def build(self, input_shape):
		self.kernel = self.add_variable("kernel", 
										shape = [int(input_shape[-1]), self.units],
										initializer = self.kernel_initializer)
		if self.use_bias:
			self.bias = self.add_variable("bias", 
											shape = [1, self.units],
											initializer = self.bias_initializer)
		self.u = self.add_variable("u",
									shape = [self.units, 1],
									initializer = tf.random.normal,
									trainable = False)

	def call(self, x, test = False):
		def _power_iteration(w, test = False):
			_v = tf.matmul(tf.transpose(w), self.u)
			v = tf.math.l2_normalize(_v, axis = 0)
			_u = tf.matmul(w, v)
			u = tf.math.l2_normalize(_u, axis = 0)
			if not test:
				self.u.assign(u)

			v = tf.stop_gradient(v)
			u = tf.stop_gradient(u)
			w = w / (tf.matmul(tf.matmul(tf.transpose(u), w), v))
			return w

		w = tf.transpose(self.kernel)
		w = tf.transpose(_power_iteration(w, test = test))

		if self.use_bias:
			x = tf.matmul(x, w) + self.bias

		if callable(self.activation):
			x = self.activation(x)

		return x


class SN(tf.keras.layers.Wrapper):
	def __init__(self, layer, **kwargs):
		super(SN, self).__init__(layer, **kwargs)
		self.layer = layer

	def build(self, input_shape):
		if not self.layer.built:
			self.layer.build(input_shape)

			self.w = self.layer.kernel
			self.w_shape = self.w.get_shape().as_list()
			self.u = self.add_variable("u",
									shape = [self.w_shape[-1], 1],
									initializer = tf.random.normal,
									trainable = False)
		super(SN, self).build()

	def call(self, x, test = False):

		def _power_iteration(w, test = False):
			_v = tf.matmul(w, self.u)
			v = tf.math.l2_normalize(_v, axis = 0)
			_u = tf.matmul(tf.transpose(w), v)
			u = tf.math.l2_normalize(_u, axis = 0)
			if not test:
				self.u.assign(u)

			v = tf.stop_gradient(v)
			u = tf.stop_gradient(u)
			w = w / (tf.matmul(tf.matmul(tf.transpose(v), w), u))
			return w

		w = tf.reshape(self.w, [-1, self.w_shape[-1]])
		self.layer.kernel = tf.reshape(_power_iteration(w), self.w_shape)
		return self.layer(x)