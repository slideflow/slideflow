import tensorflow as tf

import numpy as numpy
import os
import shutil
from os import listdir
from os.path import isfile, isdir, join
from random import shuffle

import time
import sys
import csv

from util import sfutil
from glob import glob

FEATURE_DESCRIPTION =  {'category': tf.io.FixedLenFeature([], tf.int64),
						'case':     tf.io.FixedLenFeature([], tf.string),
						'image_raw':tf.io.FixedLenFeature([], tf.string)}

def _parse_function(example_proto):
	return tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)

def _float_feature(value):
	"""Returns a bytes_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(category, case, image_string):
	feature = {
		'category': _int64_feature(category),
		'case':     _bytes_feature(case),
		'image_raw':_bytes_feature(image_string),
	}
	return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecords(input_directory, output_directory, label, annotations_file):
	annotations_dict = sfutil.get_annotations_dict(annotations_file, key_name="slide", value_name="category")
	tfrecord_path = join(output_directory, f'{label}.tfrecords')
	image_labels = {}
	case_dirs = [_dir for _dir in listdir(input_directory) if isdir(join(input_directory, _dir))]
	for case_dir in case_dirs:
		files = [file for file in listdir(join(input_directory, case_dir)) 
					if (isfile(join(input_directory, case_dir, file))) and
						(file[-3:] == "jpg")]
		for tile in files:
			# Assign arbitrary category number for now (TEMPORARY), for now assigning value of 0
			try:
				category = annotations_dict[case_dir]
			except KeyError:
				print(f" + [{sfutil.fail('ERROR')}] Case {sfutil.green(case_dir)} not found in annotation file.")
				sys.exit()
			image_labels.update({join(input_directory, case_dir, tile): [category, bytes(case_dir, 'utf-8')]})

	keys = list(image_labels.keys())
	shuffle(keys)
	with tf.io.TFRecordWriter(tfrecord_path) as writer:
		for filename in keys:
			labels = image_labels[filename]
			image_string = open(filename, 'rb').read()
			tf_example = image_example(labels[0], labels[1], image_string)
			writer.write(tf_example.SerializeToString())
	print(f" + Wrote {len(keys)} image tiles to {tfrecord_path}")