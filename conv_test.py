# conv_test.py

import tensorflow as tf
import sys

raw_image = tf.constant([[[0.],  [1.],  [2.],  [3.],  [4.],  [5.],  [6.],  [7.],  [8.],  [9.]],
						 [[10.], [11.], [12.], [13.], [14.], [15.], [16.], [17.], [18.], [19.]],
						 [[20.], [21.], [22.], [23.], [24.], [25.], [26.], [27.], [28.], [29.]],
						 [[30.], [31.], [32.], [33.], [34.], [35.], [36.], [37.], [38.], [39.]],
						 [[40.], [41.], [42.], [43.], [44.], [45.], [46.], [47.], [48.], [49.]]], dtype=tf.float32)

unshaped_patches = tf.extract_image_patches(images=[raw_image], ksizes=[1, 3, 3, 1],
											strides = [1, 3, 3, 1], rates = [1, 1, 1, 1],
											padding = "VALID")

patches = tf.cast(tf.reshape(unshaped_patches, [-1, 3, 3, 1]), tf.float32)

q = tf.FIFOQueue(capacity = 5, 
				dtypes=[tf.float32],
				shapes=[3, 3, 1])

eq_op = q.enqueue_many(patches)
deq_op = q.dequeue_many(2)

qr = tf.train.QueueRunner(q, [eq_op])
tf.train.add_queue_runner(qr)

#operation = tf.reduce_sum()

#tf.nn.conv2d

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()
	enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

	do = sess.run(deq_op)
	print(do)

	do = sess.run(deq_op)
	print(do)

	do = sess.run(deq_op)
	print(do)

	coord.request_stop()
	coord.join(enqueue_threads)

	sys.exit()