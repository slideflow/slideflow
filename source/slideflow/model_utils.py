import tensorflow as tf

def negative_log_likelihood(y_true, y_pred):
	E = y_pred[:, -1]
	y_pred = y_pred[:, :-1]
	E = tf.reshape(E, [-1])
	y_pred = tf.reshape(y_pred, [-1])
	y_true = tf.reshape(y_true, [-1])
	order = tf.argsort(y_true)
	E = tf.gather(E, order)
	y_pred = tf.gather(y_pred, order)
	gamma = tf.math.reduce_max(y_pred)
	eps = tf.constant(1e-7, dtype=tf.float16)
	log_cumsum_h = tf.math.add(tf.math.log(tf.math.add(tf.math.cumsum(tf.math.exp(tf.math.subtract(y_pred, gamma))), eps)), gamma)
	neg_likelihood = -tf.math.divide(tf.reduce_sum(tf.math.multiply(tf.subtract(y_pred, log_cumsum_h), E)),tf.reduce_sum(E))
	return neg_likelihood
	
def concordance_index(y_true, y_pred):
	E = y_pred[:, -1]
	y_pred = y_pred[:, :-1]
	E = tf.reshape(E, [-1])
	y_pred = tf.reshape(y_pred, [-1])
	y_pred = -y_pred #negative of log hazard ratio to have correct relationship with survival
	g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
	g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)
	f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
	event = tf.multiply(tf.transpose(E), E)
	f = tf.multiply(tf.cast(f, tf.float32), event)
	f = tf.compat.v1.matrix_band_part(tf.cast(f, tf.float32), -1, 0)
	g = tf.reduce_sum(tf.multiply(g, f))
	f = tf.reduce_sum(f)
	return tf.where(tf.equal(f, 0), 0.0, g/f)