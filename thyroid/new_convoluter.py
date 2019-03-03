# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, March 2019
# ==========================================================================

'''Convolutionally applies a saved Tensorflow model to a larger image, displaying
the result as a heatmap overlay.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import warnings

from PIL import Image
import tensorflow as tf

# TODO: extract_image_patches expands data too unnecessarily, creating large memory pressure.
#  Need to find iterator method to only extract as many patches as are needed to generate a batch

class Convoluter:
	# Model variables
	SIZE = 512
	NUM_CLASSES = 1
	BATCH_SIZE = 32
	USE_FP16 = True

	# Display variables
	stride_divisor = 4
	STRIDES = [1, int(SIZE/stride_divisor), int(SIZE/stride_divisor), 1]
	WINDOW_SIZE = 6000

	WHOLE_IMAGE = '/Users/james/thyroid/images/234781-2.jpg'

	def __init__(self):
		self.DTYPE = tf.float16 if self.USE_FP16 else tf.float32
		self.DTYPE_INT = tf.int16 if self.USE_FP16 else tf.int32

	def scan_image(self):
		warnings.simplefilter('ignore', Image.DecompressionBombWarning)
		with tf.Graph().as_default() as g:

			image_string = tf.read_file(self.WHOLE_IMAGE)
			image = tf.cast(tf.image.decode_jpeg(image_string, channels = 3), tf.int32)

			window_size = [self.SIZE, self.SIZE]
			window_stride = [int(self.SIZE/4), int(self.SIZE/4)]

			with tf.Session() as sess:
				init = (tf.global_variables_initializer(), tf.local_variables_initializer())
				sess.run(init)

				shape = sess.run(tf.shape(image))

				print(shape)

				if (window_size[0] > shape[0]) or (window_size[1] > shape[1]):
					raise IndexError("Window size is too large")

				coord = []

				for x in range(0, (shape[0]+1) - window_size[0], window_stride[0]):
					for y in range(0, (shape[1]+1) - window_size[1], window_stride[1]):
						coord.append([x, y])

				coord_dataset = tf.data.Dataset.from_tensor_slices(coord)
				coord_dataset = coord_dataset.batch(self.BATCH_SIZE, drop_remainder = True)
				iterator = coord_dataset.make_one_shot_iterator()
				next_coord = iterator.get_next()

				next_windows = tf.map_fn(lambda c: image[c[0]:c[0]+window_size[0], c[1]:c[1]+window_size[1],], tf.cast(next_coord, dtype=tf.int32))
				#next_windows = tf.map_fn(tf.image.per_image_standardization, tf.cast(next_window, tf.float32))

				next_jpegs = tf.map_fn(lambda c: tf.image.encode_jpeg(tf.cast(c, tf.uint8)), next_windows, dtype=tf.string)

				x = 0
				while 1:
					try:
						jpegs = sess.run(next_jpegs)
						for j in range(len(jpegs)):
							print("Writing image {} from batch {}".format(j, x))
							write_op = tf.write_file('/Users/james/thyroid/images/sample{}-{}.jpg'.format(x, j), jpegs[j])
							sess.run(write_op)

					except tf.errors.OutOfRangeError:
						break
					x += 1

			sys.exit()

if __name__==('__main__'):
	c = Convoluter()
	c.scan_image()
	#c.scan_image()


