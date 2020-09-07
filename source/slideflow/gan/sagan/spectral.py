# From https://github.com/taki0112/Self-Attention-GAN-Tensorflow
from functools import reduce
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.eager import context


def l2normalize(v, eps=1e-12):
    return v / (tf.norm(v) + eps)


class CopyWeight(tf.initializers.Initializer):
    
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, shape, dtype=tf.dtypes.float32):
        return tf.identity(self.weight)


class L2normalizedRandomNormal(tf.initializers.TruncatedNormal):
    
    def __init__(self, seed=None):
        super(L2normalizedRandomNormal, self).__init__(mean=0.0, stddev=1.0, seed=seed)
    
    def __call__(self, shape, dtype=tf.dtypes.float32):
        initial = super(L2normalizedRandomNormal, self).__call__(shape, dtype=dtype)
        return l2normalize(initial)


class SpectralConv2D(tf.keras.layers.Conv2D):
    """docstring for SpectralConv2D"""
    def __init__(self, filters, kernel_size, power_iterations=1, **kwargs):
        super(SpectralConv2D, self).__init__(filters, kernel_size, kernel_initializer='TruncatedNormal', **kwargs)
        self.power_iterations = power_iterations
    
    def _made_params(self):
        try:
            v = getattr(self, 'v')
            u = getattr(self, 'u')
            w = getattr(self, 'w')
            return True
        except AttributeError:
            return False
    
    def _get_kernel_shape(self, input_dim):
        return self.kernel_size + (input_dim, self.filters)


    def build(self, input_shape):
        super(SpectralConv2D, self).build(input_shape)
        
        if not self._made_params():

            if self.data_format == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = -1

            input_dim = int(input_shape[channel_axis])
            kernel_shape = self._get_kernel_shape(input_dim)

            self.u = self.add_weight(self.name + '_v',
                shape=[self.filters, 1], 
                initializer=L2normalizedRandomNormal,
                trainable=False
            )

            self.v = self.add_weight(self.name + '_u',
                shape=[self.kernel_size[0] * self.kernel_size[1] * input_dim, 1],
                initializer=L2normalizedRandomNormal,
                trainable=False
            )

    def compute_spectral_norm(self):
        for _ in range(self.power_iterations):
            k = tf.reshape(self.kernel, (-1, self.filters))

            new_v = l2normalize(tf.matmul(k, self.u))
            new_u = l2normalize(tf.matmul(tf.transpose(k), self.v))
            
        sigma = tf.matmul(tf.matmul(tf.transpose(new_v), k), new_u)
        new_kernel = tf.divide(self.kernel, sigma)

        return new_v, new_u, self.kernel


    def call(self, inputs):
        new_v, new_u, new_kernel = self.compute_spectral_norm()
        outputs = self._convolution_op(inputs, new_kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')
        if self.activation is not None:
            return self.activation(outputs)

        return outputs + [new_v, new_u, new_kernel]


class SpectralConv2DTranspose(SpectralConv2D, tf.keras.layers.Conv2DTranspose):
    
    def __init__(self, filters, kernel_size, power_iterations=1, **kwargs):
        super(SpectralConv2DTranspose, self).__init__(filters, kernel_size, power_iterations=1, **kwargs)

    def _get_kernel_shape(self, input_dim):
        return self.kernel_size + (self.filters, input_dim)
    
    def call(self, inputs):
        new_v, new_u, new_kernel = self.compute_spectral_norm()
        
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
    
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        out_height = conv_utils.deconv_output_length(height,
                                                     kernel_h,
                                                     padding=self.padding,
                                                     output_padding=out_pad_h,
                                                     stride=stride_h,
                                                     dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = tf.stack(output_shape)
        outputs = tf.keras.backend.conv2d_transpose(
            inputs,
            new_kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)

        return outputs + [new_v, new_u, new_kernel]
