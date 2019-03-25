import argparse
import sys
import os
import numpy as np
from numpy import genfromtxt

from tensorflow.contrib.tensorboard.plugins import projector

import tensorflow as tf

FLAGS = None

TRANSPOSE = False

def visualize():
	#Expect data to be set up with column 1 = meta identifier, column 2 = training output

	data = genfromtxt(FLAGS.data_dir + '/' + FLAGS.file, delimiter=',')[1:] # training data
	data_meta = genfromtxt(FLAGS.data_dir + '/' + FLAGS.file, dtype=np.str_, delimiter=',')

	if TRANSPOSE:
		data = data.transpose()
		data_meta = data_meta.transpose()

	meta = [i[0:FLAGS.meta] for i in data_meta]

	def save_metadata(file):
		with open(file, 'w') as f:
			for i in range(len(meta)):
				c = meta[i]
				f.write('\t'.join(map(str, c)) + '\n')
				#f.write('{}/n'.format(c))

	save_metadata(FLAGS.log_dir + '/projector/metadata.tsv')

	rp_values = np.array([ i[FLAGS.meta::] for i in data])

	sess = tf.InteractiveSession()

	# input for Embedded TensorBoard visualization, performed with CPU
	with tf.device("/cpu:0"):
		embedding = tf.Variable(tf.stack(rp_values, axis=0), trainable=False, name='embedding')

	merged = tf.summary.merge_all()
	tf.global_variables_initializer().run()

	saver = tf.train.Saver()
	writer = tf.summary.FileWriter(FLAGS.log_dir + '/projector', sess.graph)

	config = projector.ProjectorConfig()
	embed = config.embeddings.add()
	embed.tensor_name = 'embedding:0'
	embed.metadata_path = os.path.join(FLAGS.log_dir + '/projector/metadata.tsv')

	projector.visualize_embeddings(writer, config)

	saver.save(sess, os.path.join(FLAGS.log_dir, 'projector/a_model.ckpt'))

	print('Finished saving data for tensorboard')


def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir + '/projector')
	visualize()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/home/shawarma/thyroid/images/WSI_50/Test_Set_Heatmaps', help='Directory for storing input data')
	parser.add_argument('--log_dir', type=str, default = '/home/shawarma/histcon/logs', help='Summaries log directory')
	parser.add_argument('--file', required=True, help='Training data filename')
	parser.add_argument('--meta', type=int, default = 1, help = 'Number of metadata columns')

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
