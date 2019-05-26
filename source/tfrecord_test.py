#from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as numpy
from os import listdir
from os.path import isfile, isdir, join
from random import shuffle

import time

def _float_feature(value):
	"""Returns a bytes_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a dictionary with features that may be relevant.
def image_example(category, case, image_string):
	feature = {
		'category': _int64_feature(category),
		'case':     _bytes_feature(case),
		'image_raw':_bytes_feature(image_string),
	}
	return tf.train.Example(features=tf.train.Features(feature=feature))

def write_records():
	image_labels = {}
	train_dir = '/home/shawarma/histcon/train_data'
	cat_dirs = [_dir for _dir in listdir(train_dir) if isdir(join(train_dir, _dir))]
	for cat_dir in cat_dirs:
		case_dirs = [_dir for _dir in listdir(join(train_dir, cat_dir)) if isdir(join(train_dir, cat_dir, _dir))]
		for case_dir in case_dirs:
			files = [file for file in listdir(join(train_dir, cat_dir, case_dir)) 
						if (isfile(join(train_dir, cat_dir, case_dir, file))) and
							(file[-3:] == "jpg")]
			shuffle(files)
			for tile in files:
				image_labels.update({join(train_dir, cat_dir, case_dir, tile): [int(cat_dir), bytes(case_dir, 'utf-8')]})


	with tf.python_io.TFRecordWriter('images.tfrecords') as writer:
		for filename, labels in image_labels.items():
			image_string = open(filename, 'rb').read()
			tf_example = image_example(labels[0], labels[1], image_string)
			writer.write(tf_example.SerializeToString())

def read_records():
	raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
	# Create a dictionary describing the features
	image_feature_description = {
		'category': tf.FixedLenFeature([], tf.int64),
		'case':     tf.FixedLenFeature([], tf.string),
		'image_raw':tf.FixedLenFeature([], tf.string),
	}

	def _parse_image_function(example_proto):
		"""Parses the input tf.Example proto using the above feature dictionary."""
		return tf.parse_single_example(example_proto, image_feature_description)

	parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

	counts = {}
	count = 0

	starttime = time.time()
	for image_features in parsed_image_dataset:
		image_raw = image_features['image_raw'].numpy()
		category = image_features['category']
		count += 1
		
		if count % 5000 == 0:
			current_time = time.time()
			print(f"{5000/(current_time - starttime)} images/sec (total: {count} images)")
			starttime = current_time

#write_records()
read_records()