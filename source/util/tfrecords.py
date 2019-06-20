import tensorflow as tf

import numpy as numpy
import os
import shutil
from os import listdir
from os.path import isfile, isdir, join
from random import shuffle, randint

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
	
def _get_images_by_dir(directory):
	files = [f for f in listdir(directory) if (isfile(join(directory, f))) and
				(f[-3:] == "jpg")]
	return files

def _try_getting_category(annotations_dict, case):
	try:
		category = annotations_dict[case]
	except KeyError:
		print(f" + [{sfutil.fail('ERROR')}] Case {sfutil.green(case)} not found in annotation file.")
		sys.exit()
	return category

def _parse_tfrecord_function(record):
	features = tf.io.parse_single_example(record, FEATURE_DESCRIPTION)
	return features

def join_tfrecord(input_folder, output_file):
	'''Randomly samples from tfrecords in the input folder with shuffling,
	and combines into a single tfrecord file.'''
	writer = tf.io.TFRecordWriter(output_file)
	tfrecord_files = glob(join(input_folder, "*.tfrecords"))
	datasets = []
	for tfrecord in tfrecord_files:
		dataset = tf.data.TFRecordDataset(tfrecord)
		dataset = dataset.shuffle(1000)
		dataset_iter = iter(dataset)
		datasets += [dataset_iter]
	while len(datasets):
		index = randint(0, len(datasets)-1)
		try:
			record = next(datasets[index])
		except StopIteration:
			del(datasets[index])
			continue
		features = _parse_tfrecord_function(record)
		case = features['case'].numpy()
		category = features['category'].numpy()
		image_raw = features['image_raw'].numpy()
		tf_example = image_example(category, case, image_raw)
		writer.write(tf_example.SerializeToString())

def split_tfrecord(tfrecord_file, output_folder):
	'''Splits records from a single tfrecord file into individual tfrecord files by case.'''
	dataset = tf.data.TFRecordDataset(tfrecord_file)
	writers = {}
	for record in dataset:
		features = _parse_tfrecord_function(record)
		case = features['case'].numpy()
		category = features['category'].numpy()
		image_raw = features['image_raw'].numpy()
		shortname = sfutil._shortname(case.decode('utf-8'))

		if shortname not in writers.keys():
			tfrecord_path = join(output_folder, f"{shortname}.tfrecords")
			writer = tf.io.TFRecordWriter(tfrecord_path)
			writers.update({shortname: writer})
		else:
			writer = writers[shortname]
		tf_example = image_example(category, case, image_raw)
		writer.write(tf_example.SerializeToString())

	for case in writers.keys():
		writers[case].close()

def _print_record(filename):
	v_dataset = tf.data.TFRecordDataset(filename)
	for i, record in enumerate(v_dataset):
		features = _parse_tfrecord_function(record)
		category = str(features['category'].numpy())
		case = str(features['case'].numpy())
		print(f"{sfutil.header(filename)}: Record {i}: Category {sfutil.info(category)} Case: {sfutil.green(case)}")

def print_tfrecord(target):
	'''Prints the case names for records in the given tfrecord file'''
	if isfile(target):
		_print_record(target)
	else:
		tfrecord_files = glob(join(target, "*.tfrecords"))
		for tfr in tfrecord_files:
			_print_record(tfr)		

def write_tfrecords_merge(input_directory, output_directory, filename, annotations_file):
	'''Scans a folder for subfolders, assumes subfolders are case names. Assembles all image tiles within 
	subfolders and labels using the provided annotation_dict, assuming the subfolder is the case name. 
	Collects all image tiles and exports into a single tfrecord file.'''
	annotations_dict = sfutil.get_annotations_dict(annotations_file, key_name="slide", value_name="category")
	tfrecord_path = join(output_directory, filename)
	image_labels = {}
	case_dirs = [_dir for _dir in listdir(input_directory) if isdir(join(input_directory, _dir))]
	for case_dir in case_dirs:
		category = _try_getting_category(annotations_dict, case_dir)
		files = _get_images_by_dir(join(input_directory, case_dir))
		for tile in files:
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

def write_tfrecords_multi(input_directory, output_directory, annotations_file):
	'''Scans a folder for subfolders, assumes subfolders are case names. Assembles all image tiles within 
	subfolders and labels using the provided annotation_dict, assuming the subfolder is the case name. 
	Collects all image tiles and exports into multiple tfrecord files, one for each case.'''
	annotations_dict = sfutil.get_annotations_dict(annotations_file, key_name="slide", value_name="category")
	case_dirs = [_dir for _dir in listdir(input_directory) if isdir(join(input_directory, _dir))]
	for case_dir in case_dirs:
		category = _try_getting_category(annotations_dict, case_dir)
		write_tfrecords_single(join(input_directory, case_dir), output_directory, f'{sfutil._shortname(case_dir)}.tfrecords', category, case_dir)

def write_tfrecords_single(input_directory, output_directory, filename, category, case):
	'''Scans a folder for image tiles, annotates using the provided category and case, exports
	into a single tfrecord file.'''
	tfrecord_path = join(output_directory, filename)
	image_labels = {}
	files = _get_images_by_dir(input_directory)
	for tile in files:
		image_labels.update({join(input_directory, tile): [category, bytes(case, 'utf-8')]})
	keys = list(image_labels.keys())
	shuffle(keys)
	with tf.io.TFRecordWriter(tfrecord_path) as writer:
		for filename in keys:
			labels = image_labels[filename]
			image_string = open(filename, 'rb').read()
			tf_example = image_example(labels[0], labels[1], image_string)
			writer.write(tf_example.SerializeToString())
	print(f" + Wrote {len(keys)} image tiles to {tfrecord_path}")