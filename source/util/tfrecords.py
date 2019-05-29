import tensorflow as tf

import numpy as numpy
from os import listdir
from os.path import isfile, isdir, join
from random import shuffle

import time
import sys
import csv

from util import sfutil

ANNOTATIONS = None
FEATURE_DESCRIPTION =  {'category': tf.FixedLenFeature([], tf.int64),
						'case':     tf.FixedLenFeature([], tf.string),
						'image_raw':tf.FixedLenFeature([], tf.string)}

def _float_feature(value):
	"""Returns a bytes_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	"""Returns an int64_list from a bool / enum / int / uint."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def load_annotations(annotations_file):
	if ANNOTATIONS:
		return ANNOTATIONS
	with open(annotations_file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		header = next(csv_reader, None)
		use_encode = False
		case_dict_int = {}
		case_dict_str = {}
		try:
			cat_index = header.index('category')
			case_index = header.index('case')
		except:
			print(f" + [{sfutil.fail('ERROR')}] Unable to find category and/or case headers in annotation file")
			sys.exit()
		for row in csv_reader:
			cat = row[cat_index]
			case = row[case_index]
			case_dict_str.update({case: cat})
			try:
				int_cat = int(cat)
				case_dict_int.update({case: int_cat})
			except:
				if not use_encode:
					print(f" + [{sfutil.warn('WARN')}] Non-integer in category header, will encode with integer values")
					use_encode = True
		if use_encode:
			categories = set(case_dict_str.values())
			category_str_to_int = {}
			for i, c in enumerate(categories):
				category_str_to_int.update({c: i})
				print(f" + [{sfutil.info('INFO')}] Category '{c}' assigned to value '{i}'")
			for category_string in case_dict_str.keys():
				case_dict_str[category_string] = category_str_to_int[case_dict_str[category_string]]
			return case_dict_str
		else:
			return case_dict_int

# Create a dictionary with features that may be relevant.
def image_example(category, case, image_string):
	feature = {
		'category': _int64_feature(category),
		'case':     _bytes_feature(case),
		'image_raw':_bytes_feature(image_string),
	}
	return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecords(input_directory, output_directory, label, annotations_file):
	global ANNOTATIONS
	if not ANNOTATIONS:
		ANNOTATIONS = load_annotations(annotations_file)
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
				category = ANNOTATIONS[case_dir]
			except KeyError:
				print(f" + [{sfutil.fail('ERROR')}] Case {sfutil.green(case_dir)} not found in annotation file.")
				sys.exit()
			image_labels.update({join(input_directory, case_dir, tile): [category, bytes(case_dir, 'utf-8')]})

	keys = list(image_labels.keys())
	shuffle(keys)
	with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
		for filename in keys:
			labels = image_labels[filename]
			image_string = open(filename, 'rb').read()
			tf_example = image_example(labels[0], labels[1], image_string)
			writer.write(tf_example.SerializeToString())
	print(f" + Wrote {len(keys)} image tiles to {tfrecord_path}")