import tensorflow as tf

import numpy as np
import os
import shutil
from os import listdir
from os.path import isfile, isdir, join, exists
from random import shuffle, randint

import time
import sys
import csv

import slideflow.util as sfutil
from slideflow.util import log
from glob import glob

FEATURE_TYPES = (tf.int64, tf.string, tf.string)

FEATURE_DESCRIPTION =  {'slide':    tf.io.FixedLenFeature([], tf.string),
						'image_raw':tf.io.FixedLenFeature([], tf.string)}

FEATURE_DESCRIPTION_MULTI =  {'slide':    tf.io.FixedLenFeature([], tf.string),
							  'input_1':  tf.io.FixedLenFeature([], tf.string),
							  'input_2':  tf.io.FixedLenFeature([], tf.string)}

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

def _get_images_by_dir(directory):
	files = [f for f in listdir(directory) if (isfile(join(directory, f))) and
				(sfutil.path_to_ext(f) == "jpg")]
	return files

def _parse_tfrecord_function(record):
	features = tf.io.parse_single_example(record, FEATURE_DESCRIPTION)
	return features

def _read_and_return_features(record):
	features = _parse_tfrecord_function(record)
	slide = features['slide'].numpy()
	image_raw = features['image_raw'].numpy()
	return slide, image_raw

def _read_and_return_record(record, assign_slide=None):
	slide, image_raw = _read_and_return_features(record)
	if assign_slide:
		slide = assign_slide
	tf_example = image_example(slide, image_raw)
	return tf_example.SerializeToString()

def _print_record(filename):
	v_dataset = tf.data.TFRecordDataset(filename)
	for i, record in enumerate(v_dataset):
		features = _parse_tfrecord_function(record)
		slide = str(features['slide'].numpy())
		print(f"{sfutil.header(filename)}: Record {i}: Slide: {sfutil.green(slide)}")

def example_to_image(record, size, include_slide=True, image_format='raw', normalizer=None):
	'''TFRecord parsing function, to be used with TFRecordDataset.map(). 
	Parses a TFRecord into its constituent components. For each record, will return a processed and standardized image.

	Args:
		record			The TFRecord being processed (not explicitly supplied if this function is being used with TFRecordDataset.map())
		size			Tile size in pixels
		include_slide	Bool. If true, will return both the processed image as well as the slide name, as a tuple
		format			Either 'raw' (default) or 'dict'. If 'dict', the image will be provided as a dictionary mapping the input layer name to the image.
							This is the syntax used for models in slideflow version 1.9 and greater.
		normalizer		Optional. Normalizer to use on images (e.g. 'macenko', 'reinhard', or 'vahadane')
	'''

	features = tf.io.parse_single_example(record, FEATURE_DESCRIPTION)
	slide = features['slide']
	image_string = features['image_raw']
	raw_image = tf.image.decode_jpeg(image_string, channels=3)

	if normalizer:
		raw_image = tf.py_function(normalizer.tf_to_rgb, [raw_image], tf.int8)

	
	processed_image = tf.image.convert_image_dtype(raw_image, tf.float32)
	processed_image = tf.image.per_image_standardization(processed_image)
	processed_image.set_shape([size, size, 3])

	if image_format=='dict':
		processed_image = {'tile_image': processed_image}

	if include_slide:
		return processed_image, slide
	else:
		return processed_image

def image_example(slide, image_string):
	'''Returns a Tensorflow Data example for TFRecord storage.'''
	feature = {
		'slide':     _bytes_feature(slide),
		'image_raw':_bytes_feature(image_string),
	}
	return tf.train.Example(features=tf.train.Features(feature=feature))

def multi_image_example(slide, image_dict):
	'''Returns a Tensorflow Data example for TFRecord storage with multiple images.'''
	feature = {
		'slide':	_bytes_feature(slide)
	}
	for image_label in image_dict:
		feature.update({
			image_label: _bytes_feature(image_dict[image_label])
		})
	return tf.train.Example(features=tf.train.Features(feature=feature))

def merge_split_tfrecords(source, destination):
	'''Merges TFRecords with the same name in subfolders within the given source folder,
	as may be the case when using split TFRecords for tile-level validation.'''
	tfrecords = {}
	subdirs = [d for d in listdir(source) if isdir(join(source, d))]
	for subdir in subdirs:
		tfrs = [tfr for tfr in listdir(join(source, subdir)) if isfile(join(source, subdir, tfr)) and tfr[-9:] == 'tfrecords']
		for tfr in tfrs:
			name = sfutil.path_to_name(tfr)
			if name not in tfrecords:
				tfrecords.update({name: [join(source, subdir, tfr)] })
			else:
				tfrecords[name] += [join(source, subdir, tfr)]
	for tfrecord_name in tfrecords:
		writer = tf.io.TFRecordWriter(join(destination, f'{tfrecord_name}.tfrecords'))
		datasets = []
		for tfrecord in tfrecords[tfrecord_name]:
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
			writer.write(_read_and_return_record(record, None))

def join_tfrecord(input_folder, output_file, assign_slide=None):
	'''Randomly samples from tfrecords in the input folder with shuffling,
	and combines into a single tfrecord file.'''
	writer = tf.io.TFRecordWriter(output_file)
	tfrecord_files = glob(join(input_folder, "*.tfrecords"))
	datasets = []
	if assign_slide: assign_slide = assign_slide.encode('utf-8')
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
		writer.write(_read_and_return_record(record, assign_slide))

def split_tfrecord(tfrecord_file, output_folder):
	'''Splits records from a single tfrecord file into individual tfrecord files by slide.'''
	dataset = tf.data.TFRecordDataset(tfrecord_file)
	writers = {}
	for record in dataset:
		features = _parse_tfrecord_function(record)
		slide = features['slide'].numpy()
		image_raw = features['image_raw'].numpy()
		shortname = sfutil._shortname(slide.decode('utf-8'))

		if shortname not in writers.keys():
			tfrecord_path = join(output_folder, f"{shortname}.tfrecords")
			writer = tf.io.TFRecordWriter(tfrecord_path)
			writers.update({shortname: writer})
		else:
			writer = writers[shortname]
		tf_example = image_example(slide, image_raw)
		writer.write(tf_example.SerializeToString())

	for slide in writers.keys():
		writers[slide].close()

def print_tfrecord(target):
	'''Prints the slide names for records in the given tfrecord file'''
	if isfile(target):
		_print_record(target)
	else:
		tfrecord_files = glob(join(target, "*.tfrecords"))
		for tfr in tfrecord_files:
			_print_record(tfr)		

def write_tfrecords_merge(input_directory, output_directory, filename):
	'''Scans a folder for subfolders, assumes subfolders are slide names. Assembles all image tiles within 
	subfolders and labels using the provided annotation_dict, assuming the subfolder is the slide name. 
	Collects all image tiles and exports into a single tfrecord file.'''
	tfrecord_path = join(output_directory, filename)
	if not exists(output_directory):
		os.makedirs(output_directory)
	image_labels = {}
	slide_dirs = [_dir for _dir in listdir(input_directory) if isdir(join(input_directory, _dir))]
	for slide_dir in slide_dirs:
		files = _get_images_by_dir(join(input_directory, slide_dir))
		for tile in files:
			image_labels.update({join(input_directory, slide_dir, tile): bytes(slide_dir, 'utf-8')})
	keys = list(image_labels.keys())
	shuffle(keys)
	with tf.io.TFRecordWriter(tfrecord_path) as writer:
		for filename in keys:
			label = image_labels[filename]
			image_string = open(filename, 'rb').read()
			tf_example = image_example(label, image_string)
			writer.write(tf_example.SerializeToString())
	log.empty(f"Wrote {len(keys)} image tiles to {sfutil.green(tfrecord_path)}", 1)
	return len(keys)

def write_tfrecords_multi(input_directory, output_directory):
	'''Scans a folder for subfolders, assumes subfolders are slide names. Assembles all image tiles within 
	subfolders and labels using the provided annotation_dict, assuming the subfolder is the slide name. 
	Collects all image tiles and exports into multiple tfrecord files, one for each slide.'''
	slide_dirs = [_dir for _dir in listdir(input_directory) if isdir(join(input_directory, _dir))]
	total_tiles = 0
	for slide_dir in slide_dirs:
		total_tiles += write_tfrecords_single(join(input_directory, slide_dir), output_directory, f'{slide_dir}.tfrecords', slide_dir)
	log.complete(f"Wrote {sfutil.bold(total_tiles)} image tiles across {sfutil.bold(len(slide_dirs))} tfrecords in {sfutil.green(output_directory)}", 1)

def write_tfrecords_single(input_directory, output_directory, filename, slide):
	'''Scans a folder for image tiles, annotates using the provided slide, exports
	into a single tfrecord file.'''
	if not exists(output_directory):
		os.makedirs(output_directory)
	tfrecord_path = join(output_directory, filename)
	image_labels = {}
	files = _get_images_by_dir(input_directory)
	for tile in files:
		image_labels.update({join(input_directory, tile): bytes(slide, 'utf-8')})
	keys = list(image_labels.keys())
	shuffle(keys)
	with tf.io.TFRecordWriter(tfrecord_path) as writer:
		for filename in keys:
			label = image_labels[filename]
			image_string = open(filename, 'rb').read()
			tf_example = image_example(label, image_string)
			writer.write(tf_example.SerializeToString())
	log.empty(f"Wrote {len(keys)} image tiles to {sfutil.green(tfrecord_path)}", 1)
	return len(keys)

def checkpoint_to_h5(models_dir, model_name):
	checkpoint = join(models_dir, model_name, "cp.ckpt")
	h5 = join(models_dir, model_name, "untrained_model.h5")
	updated_h5 = join(models_dir, model_name, "checkpoint_model.h5")
	model = tf.keras.models.load_model(h5)
	model.load_weights(checkpoint)
	try:
		model.save(updated_h5)
	except KeyError:
		# Not sure why this happens, something to do with the optimizer?
		pass

def split_patients_list(patients_dict, n, balance=None, randomize=True):
	'''Splits a dictionary of patients into n groups, balancing according to key "balance" if provided.'''
	patient_list = list(patients_dict.keys())
	shuffle(patient_list)

	def flatten(l):
		'''Flattens a list'''
		return [y for x in l for y in x]

	def split(a, n):
		'''Function to split a list into n components'''
		k, m = divmod(len(a), n)
		return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

	if balance:
		# Get patient outcomes
		patient_outcomes = [patients_dict[p][balance] for p in patients_dict]

		# Get unique outcomes
		unique_outcomes = list(set(patient_outcomes))

		# Now, split patient_list according to outcomes
		patients_split_by_outcomes = [[p for p in patient_list if patients_dict[p][balance] == uo] for uo in unique_outcomes]

		# Then, for each sublist, split into n components
		patients_split_by_outcomes_split_by_n = [list(split(sub_l, n)) for sub_l in patients_split_by_outcomes]

		# Print splitting as a table
		log.empty(sfutil.bold("Category\t" + "\t".join([str(cat) for cat in range(len(set(unique_outcomes)))])), 2)
		for k in range(n):
			log.empty(f"K-fold-{k}\t" + "\t".join([str(len(clist[k])) for clist in patients_split_by_outcomes_split_by_n]), 2)

		# Join sublists
		return [flatten([item[ni] for item in patients_split_by_outcomes_split_by_n]) for ni in range(n)]
	else:
		return list(split(patient_list, n))

def get_training_and_validation_tfrecords(dataset, validation_log, outcomes, model_type, validation_target, validation_strategy, 
											validation_fraction, validation_k_fold=None, k_fold_iter=None, read_only=False):
	'''From a specified subfolder within the project's main TFRecord folder, prepare a training set and validation set.
	If a validation plan has already been prepared (e.g. K-fold iterations were already determined), the previously generated plan will be used.
	Otherwise, create a new plan and log the result in the TFRecord directory so future models may use the same plan for consistency.

	Args:
		dataset:				A slideflow.datasets.Dataset object
		validation_log:			Path to .log file containing validation plans
		outcomes:				Dictionary mapping slides to outcomes (used for balancing outcomes in training and validation cohorts)
		model_type:				Either 'categorical' or 'linear'
		validation_target:		Either 'per-slide' or 'per-tile'
		validation_strategy:	Either 'k-fold', 'bootstrap', or 'fixed'.
		validation_fraction:	Float, proportion of data for validation. Not used if strategy is k-fold.
		validation_k_fold:		K, if using K-fold validation.
		k_fold_iter:			Which K-fold iteration, if using K-fold validation.

	Returns:
		Two arrays: 	an array of full paths to training tfrecords, and an array of paths to validation tfrecords.''' 

	# Prepare dataset
	tfr_folders = dataset.get_tfrecords_folders()
	subdirs = []
	for folder in tfr_folders:
		try:
			detected_subdirs = [sd for sd in os.listdir(folder) if isdir(join(folder, sd))]
		except:
			log.error(f"Unable to find TFRecord location {sfutil.green(folder)}")
			sys.exit()
		subdirs = detected_subdirs if not subdirs else subdirs
		if detected_subdirs != subdirs:
			log.error(f"Unable to combine TFRecords from datasets; TFRecord subdirectory structures do not match.")
			sys.exit()

	if k_fold_iter: 
		k_fold_index = int(k_fold_iter)-1
	k_fold = validation_k_fold
	training_tfrecords = []
	validation_tfrecords = []
	accepted_plan = None
	slide_list = list(outcomes.keys())

	# If validation is done per-tile, use pre-separated TFRecord files (validation separation done at time of TFRecord creation)
	if validation_target == 'per-tile':
		log.info(f"Attempting to load pre-separated TFRecords", 1)
		if validation_strategy == 'bootstrap':
			log.warn("Validation bootstrapping is not supported when the validation target is per-tile; using tfrecords in 'training' and 'validation' subdirectories", 1)
		if validation_strategy in ('bootstrap', 'fixed'):
			# Load tfrecords from 'validation' and 'training' subdirectories
			if ('validation' not in subdirs) or ('training' not in subdirs):
				log.error(f"{sfutil.bold(validation_strategy)} selected as validation strategy but tfrecords are not organized as such (unable to find 'training' or 'validation' subdirectories)")
				sys.exit()
			training_tfrecords += dataset.get_tfrecords_by_subfolder("training")
			validation_tfrecords += dataset.get_tfrecords_by_subfolder("validation")
		elif validation_strategy == 'k-fold':
			if not k_fold_iter:
				log.warn("No k-fold iteration specified; assuming iteration #1", 1)
				k_fold_iter = 1
			if k_fold_iter > k_fold:
				log.error(f"K-fold iteration supplied ({k_fold_iter}) exceeds the project K-fold setting ({k_fold})", 1)
				sys.exit()
			for k in range(k_fold):
				if k == k_fold_index:
					validation_tfrecords += dataset.get_tfrecords_by_subfolder(f'kfold-{k}')
				else:
					training_tfrecords += dataset.get_tfrecords_by_subfolder(f'kfold-{k}')
		elif validation_strategy == 'none':
			if len(subdirs):
				log.error(f"Validation strategy set as 'none' but the TFRecord directory has been configured for validation (contains subfolders {', '.join(subdirs)})", 1)
				sys.exit()
		# Remove tfrecords not specified in slide_list
		training_tfrecords = [tfr for tfr in training_tfrecords if tfr.split('/')[-1][:-10] in slide_list]
		validation_tfrecords = [tfr for tfr in validation_tfrecords if tfr.split('/')[-1][:-10] in slide_list]

	elif validation_target == 'per-patient':
		# Assemble dictionary of patients linking to list of slides and outcome
		# slideflow.util.get_outcomes_from_annotations() ensures no duplicate outcomes are found in a single patient
		tfrecord_dir_list = dataset.get_tfrecords()
		tfrecord_dir_list_names = [tfr.split('/')[-1][:-10] for tfr in tfrecord_dir_list]
		patients_dict = {}
		num_warned = 0
		warn_threshold = 3
		for slide in slide_list:
			patient = outcomes[slide][sfutil.TCGA.patient]
			print_func = print if num_warned < warn_threshold else None
			# Skip slides not found in directory
			if slide not in tfrecord_dir_list_names:
				log.warn(f"Slide {slide} not found in tfrecord directory, skipping", 1, print_func)
				num_warned += 1
				continue
			if patient not in patients_dict:
				patients_dict[patient] = {
					'outcome': outcomes[slide]['outcome'],
					'slides': [slide]
				}
			elif patients_dict[patient]['outcome'] != outcomes[slide]['outcome']:
				log.error(f"Multiple outcomes found for patient {patient} ({patients_dict[patient]['outcome']}, {outcomes[slide]['outcome']})", 1)
				sys.exit()
			else:
				patients_dict[patient]['slides'] += [slide]
		if num_warned >= warn_threshold:
			log.warn(f"...{num_warned} total warnings, see {sfutil.green(log.logfile)} for details", 1)
		patients = list(patients_dict.keys())
		sorted_patients = [p for p in patients]
		sorted_patients.sort()
		shuffle(patients)

		# Create and log a validation subset
		if len(subdirs):
			log.error(f"Validation target set to 'per-patient', but the TFRecord directory has validation configured per-tile (contains subfolders {', '.join(subdirs)}", 1)
			sys.exit()
		if validation_strategy == 'none':
			log.info(f"Validation strategy set to 'none'; selecting no tfrecords for validation.", 1)
			training_slides = np.concatenate([patients_dict[patient]['slides'] for patient in patients_dict.keys()]).tolist()
			validation_slides = []
		elif validation_strategy == 'bootstrap':
			num_val = int(validation_fraction * len(patients))
			log.info(f"Using boostrap validation: selecting {sfutil.bold(num_val)} patients at random to use for validation testing", 1)
			validation_patients = patients[0:num_val]
			training_patients = patients[num_val:]
			if not len(validation_patients) or not len(training_patients):
				log.error("Insufficient number of patients to generate validation dataset.", 1)
				sys.exit()
			validation_slides = np.concatenate([patients_dict[patient]['slides'] for patient in validation_patients]).tolist()
			training_slides = np.concatenate([patients_dict[patient]['slides'] for patient in training_patients]).tolist()
		else:
			# Try to load validation plan
			validation_plans = [] if not exists(validation_log) else sfutil.load_json(validation_log)
			for plan in validation_plans:
				# First, see if plan type is the same
				if plan['strategy'] != validation_strategy:
					continue
				# If k-fold, check that k-fold length is the same
				if validation_strategy == 'k-fold' and len(list(plan['tfrecords'].keys())) != k_fold:
					continue
				# Then, check if patient lists are the same
				plan_patients = list(plan['patients'].keys())
				plan_patients.sort()
				if plan_patients == sorted_patients:
					# Finally, check if outcome variables are the same
					if [patients_dict[p]['outcome'] for p in plan_patients] == [plan['patients'][p]['outcome'] for p in plan_patients]:
						log.info(f"Using {validation_strategy} validation plan detected at {sfutil.green(validation_log)}", 1)
						accepted_plan = plan
						break
			# If no plan found, create a new one
			if not accepted_plan:
				log.info(f"No suitable validation plan found; will log plan at {sfutil.green(validation_log)}", 1)
				new_plan = {
					'strategy':		validation_strategy,
					'patients':		patients_dict,
					'tfrecords':	{}
				}
				if validation_strategy == 'fixed':
					num_val = int(validation_fraction * len(patients))
					validation_patients = patients[0:num_val]
					training_patients = patients[num_val:]
					if not len(validation_patients) or not len(training_patients):
						log.error("Insufficient number of patients to generate validation dataset.", 1)
						sys.exit()
					validation_slides = np.concatenate([patients_dict[patient]['slides'] for patient in validation_patients]).tolist()
					training_slides = np.concatenate([patients_dict[patient]['slides'] for patient in training_patients]).tolist()
					new_plan['tfrecords']['validation'] = validation_slides
					new_plan['tfrecords']['training'] = training_slides
				elif validation_strategy == 'k-fold':
					k_fold_patients = split_patients_list(patients_dict, k_fold, balance=('outcome' if model_type == 'categorical' else None), randomize=True)
					# Verify at least one patient is in each k_fold group
					if not min([len(patients) for patients in k_fold_patients]):
						log.error("Insufficient number of patients to generate validation dataset.", 1)
						sys.exit()
					training_patients = []
					for k in range(k_fold):
						new_plan['tfrecords'][f'k-fold-{k+1}'] = np.concatenate([patients_dict[patient]['slides'] for patient in k_fold_patients[k]]).tolist()
						if k == k_fold_index:
							validation_patients = k_fold_patients[k]
						else:
							training_patients += k_fold_patients[k]
					validation_slides = np.concatenate([patients_dict[patient]['slides'] for patient in validation_patients]).tolist()
					training_slides = np.concatenate([patients_dict[patient]['slides'] for patient in training_patients]).tolist()
				else:
					log.error(f"Unknown validation strategy {validation_strategy} requested.")
					sys.exit()
				# Write the new plan to log
				validation_plans += [new_plan]
				if not read_only:
					sfutil.write_json(validation_plans, validation_log)
			else:
				# Use existing plan
				if validation_strategy == 'fixed':
					validation_slides = accepted_plan['tfrecords']['validation']
					training_slides = accepted_plan['tfrecords']['training']
				elif validation_strategy == 'k-fold':
					validation_slides = accepted_plan['tfrecords'][f'k-fold-{k_fold_iter}']
					training_slides = np.concatenate([accepted_plan['tfrecords'][f'k-fold-{ki+1}'] for ki in range(k_fold) if ki != k_fold_index]).tolist()
				else:
					log.error(f"Unknown validation strategy {validation_strategy} requested.")
					sys.exit()

		# Perform final integrity check to ensure no patients are in both training and validation slides
		validation_pt = list(set([outcomes[slide][sfutil.TCGA.patient] for slide in validation_slides]))
		training_pt = list(set([outcomes[slide][sfutil.TCGA.patient] for slide in training_slides]))
		if sum([pt in training_pt for pt in validation_pt]):
			log.error(f"At least one patient is in both validation and training sets.")
			sys.exit()

		# Return list of tfrecords
		validation_tfrecords = [tfr for tfr in tfrecord_dir_list if sfutil.path_to_name(tfr) in validation_slides]
		training_tfrecords = [tfr for tfr in tfrecord_dir_list if sfutil.path_to_name(tfr) in training_slides]
	else:
		log.error(f"Invalid validation strategy '{validation_target}' detected; must be either 'per-tile' or 'per-patient'.", 1)
		sys.exit()
	log.info(f"Using {sfutil.bold(len(training_tfrecords))} TFRecords for training, {sfutil.bold(len(validation_tfrecords))} for validation", 1)
	return training_tfrecords, validation_tfrecords

def update_tfrecord_dir(directory, old_feature_description=FEATURE_DESCRIPTION, slide='slide', image_raw='image_raw'):
	'''Updates tfrecords in a directory from an old format to a new format.'''
	if not exists(directory):
		log.error(f"Directory {directory} does not exist; unable to update tfrecords.")
	else:
		tfrecord_files = glob(join(directory, "*.tfrecords"))
		for tfr in tfrecord_files:
			update_tfrecord(tfr, old_feature_description, slide, image_raw)
		return len(tfrecord_files)

def update_tfrecord(tfrecord_file, old_feature_description=FEATURE_DESCRIPTION, slide='slide', image_raw='image_raw'):
	'''Updates a single tfrecord from an old format to a new format.'''
	shutil.move(tfrecord_file, tfrecord_file+".old")
	dataset = tf.data.TFRecordDataset(tfrecord_file+".old")
	writer = tf.io.TFRecordWriter(tfrecord_file)
	for record in dataset:
		features = tf.io.parse_single_example(record, old_feature_description)
		slidename = features[slide].numpy()
		image_raw_data = features[image_raw].numpy()
		tf_example = image_example(slide=slidename, image_string=image_raw_data)
		writer.write(tf_example.SerializeToString())
	writer.close()

def transform_tfrecord(origin, target, assign_slide=None, hue_shift=None, resize=None, silent=False):
	'''Transforms images in a single tfrecord. Can perform hue shifting, resizing, or re-assigning slide label.'''
	print_func = None if silent else print
	log.empty(f"Transforming tiles in tfrecord {sfutil.green(origin)}", 1, print_func)
	log.info(f"Saving to new tfrecord at {sfutil.green(target)}", 2, print_func)
	if assign_slide:
		log.info(f"Assigning slide name {sfutil.bold(assign_slide)}", 2, print_func)
	if hue_shift:
		log.info(f"Shifting hue by {sfutil.bold(str(hue_shift))}", 2, print_func)
	if resize:
		log.info(f"Resizing records to ({resize}, {resize})", 2, print_func)

	dataset = tf.data.TFRecordDataset(origin)
	writer = tf.io.TFRecordWriter(target)

	def process_image(image_string):
		if hue_shift:
			decoded_image = tf.image.decode_jpeg(image_string, channels=3)
			adjusted_image = tf.image.adjust_hue(decoded_image, hue_shift)
			encoded_image = tf.io.encode_jpeg(adjusted_image, quality=80)
			return encoded_image.numpy()
		elif resize:
			decoded_image = tf.image.decode_jpeg(image_string, channels=3)
			resized_image = tf.image.resize(decoded_image, (resize, resize), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			encoded_image = tf.io.encode_jpeg(resized_image, quality=80)
			return encoded_image.numpy()
		else:
			return image_string

	for record in dataset:
		features = tf.io.parse_single_example(record, FEATURE_DESCRIPTION)
		slidename = features['slide'].numpy() if not assign_slide else bytes(assign_slide, 'utf-8')
		image_raw_data = features['image_raw'].numpy()
		image_processed_data = process_image(image_raw_data)
		tf_example = image_example(slide=slidename, image_string=image_processed_data)
		writer.write(tf_example.SerializeToString())
	writer.close()

def shuffle_tfrecord(target):
	'''Shuffles records in a TFRecord, saving the original to a .old file.'''
	old_tfrecord = target+".old"
	shutil.move(target, old_tfrecord)

	dataset = tf.data.TFRecordDataset(old_tfrecord)
	writer = tf.io.TFRecordWriter(target)

	extracted_tfrecord = []
	for record in dataset:
		extracted_tfrecord += [record.numpy()]

	shuffle(extracted_tfrecord)

	for record in extracted_tfrecord:
		writer.write(record)
	
	writer.close()

def shuffle_tfrecords_by_dir(directory):
	'''For each TFRecord in a directory, shuffles records in the TFRecord, saving the original to a .old file.'''
	records = [tfr for tfr in listdir(directory) if tfr[-10:] == ".tfrecords"]
	for record in records:
		log.empty(f'Working on {record}')
		shuffle_tfrecord(join(directory, record))

def get_tfrecord_by_index(tfrecord, index, decode=True):
	'''Reads and returns an individual record from a tfrecord by index, including slide name and JPEG-processed image data.'''

	if type(index) != int:
		try:
			index = int(index)
		except:
			raise IndexError(f"index must be an integer, not {type(index)} (provided {index}).")

	def _decode(record):
		features = _parse_tfrecord_function(record)
		slide = features['slide']
		image_string = features['image_raw']
		if not decode:
			return slide, image_string
		else:
			raw_image = tf.image.decode_jpeg(image_string, channels=3)
			return slide, raw_image

	dataset = tf.data.TFRecordDataset(tfrecord)
	total = 0
	for i, data in enumerate(dataset):
		total += 1
		if i == index:
			return _decode(data)
		else: continue

	log.error(f"Unable to find record at index {index} in {sfutil.green(tfrecord)} ({total} total records)", 1)
	return False, False

def get_tfrecords_from_model_manifest(manifest, dataset='validation'):
	slides = []
	with open(manifest, 'r') as manifest_file:
		reader = csv.reader(manifest_file)
		header = next(reader)
		dataset_index = header.index('dataset')
		slide_index = header.index('slide')
		for row in reader:
			dataset_name = row[dataset_index]
			slide_name = row[slide_index]
			if dataset_name == dataset or not dataset:
				slides += [slide_name]
	return slides

def extract_tiles(tfrecord, destination, description=FEATURE_DESCRIPTION, feature_label='image_raw'):
	'''Reads and saves images from a TFRecord to a destination folder.'''
	if not exists(destination):
		os.makedirs(destination)
	log.empty(f"Extracting tiles from tfrecord {sfutil.green(tfrecord)}", 1)
	log.info(f"Saving tiles to directory {sfutil.green(destination)}", 2)

	dataset = tf.data.TFRecordDataset(tfrecord)
	for i, record in enumerate(dataset):
		features = tf.io.parse_single_example(record, description)
		slidename = features['slide'].numpy().decode('utf-8')
		image_raw_data = features[feature_label].numpy()
		dest_folder = join(destination, slidename)
		if not exists(dest_folder):
			os.makedirs(dest_folder)
		tile_filename = f"tile{i}.jpg"
		image_string = open(join(dest_folder, tile_filename), 'wb')
		image_string.write(image_raw_data)
		image_string.close()