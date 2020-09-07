# Implementation of https://semantic-pyramid.github.io/
# Written by James Dolezal, September 2020
# james.dolezal@uchospitals.edu

import os
import sys
import logging
import random
import numpy as np

from functools import partial

logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, LeakyReLU, Lambda
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import slideflow as sf
from slideflow.io.tfrecords import example_to_image
from slideflow.model import SlideflowModel
from slideflow.gan.sagan.spectral_normalization import SpectralNormalization
from slideflow.gan.semantic import model as semantic_model
from slideflow.gan import semantic
from slideflow.gan.utils import *

def _parse_tfrecord_brs(record, sf_model, n_classes, include_slidenames=False, multi_image=False):
	features = tf.io.parse_single_example(record, sf.io.tfrecords.FEATURE_DESCRIPTION)
	slide = features['slide']
	image_string = features['image_raw']
	image = sf_model._process_image(image_string, augment=True)

	brs = sf_model.ANNOTATIONS_TABLES[0].lookup(slide)
	label = tf.cond(brs < tf.constant(0, dtype=tf.float32), lambda: tf.constant(0), lambda: tf.constant(1))
	label = tf.one_hot(label, n_classes)
	
	return image, label

def gan_test(batch_size=4, mixed_precision=False):
	# Set mixed precision flag; it seems that mixed precision worsens GAN performance so 
	#  I would recommend against its use for now
	if mixed_precision:
		policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
		mixed_precision.set_policy(policy)

	# Setup project-specific details. This will eventually need to be replaced
	#  With a more flexibile solution.
	sample_project = '/home/shawarma/Thyroid-Paper-Final/projects/TCGA'
	SFP = sf.SlideflowProject(sample_project)
	sf_dataset = SFP.get_dataset(tile_px=299, tile_um=302, filters={'brs_class': ['Braf-like', 'Ras-like']})
	tfrecords = sf_dataset.get_tfrecords()
	xception_path = '/home/shawarma/Thyroid-Paper-Final/projects/TCGA/models/brs-BRS_FULL/trained_model_epoch1.h5'
	vgg16_path = '/home/shawarma/Thyroid-Paper-Final/projects/TCGA/models/brs-BRS_VGG16_FULL_NEWT/trained_model_epoch1.h5'

	# Build actual dataset inputs using a slideflow model
	checkpoint_dir = '/home/shawarma/test_log'
	slide_annotations, _ = sf_dataset.get_outcomes_from_annotations('brs', use_float=True)
	train_tfrecords = tfrecords
	validation_tfrecords = None
	manifest = sf_dataset.get_manifest()
	SFM = SlideflowModel(checkpoint_dir, 299, slide_annotations, train_tfrecords, validation_tfrecords, manifest, model_type='linear')
	dataset, dataset_with_slidenames, num_tiles = SFM._build_dataset_inputs(tfrecords, batch_size, 'NO_BALANCE', augment=False,
																												 include_slidenames=False,
																												 parse_fn=partial(_parse_tfrecord_brs, sf_model=SFM,
																												   									   n_classes=2))
	dataset = dataset.prefetch(20)
	
	# Load the external model
	with tf.name_scope('ExternalModel'):
		model = tf.keras.models.load_model(vgg16_path)

	# Legacy model format
	vgg16_activation_layer_names = [	'block1_pool',			# 64 channels  (149x149)
										'block2_pool',			# 128 channels (74x74)
										'block3_pool',			# 256 channels (37x37)
										'block4_pool',			# 512 channels (18x18)
										'block5_pool' ] 		# 512 channels (9x9)

	# Set loaded model as non-trainable
	for layer in model.layers:
		layer.trainable = False

	# Identify the feature activation maps that will be used for the generator input
	feature_tensors = {
		'image': model.input,
		'image_vgg16': model.get_layer('vgg16').input,
		'fc8':   model.get_layer('hidden_1').output,
		'fc7':   model.get_layer('hidden_0').output,
		'conv0': model.get_layer('vgg16').get_layer('block5_pool').output,
		'conv1': model.get_layer('vgg16').get_layer('block4_pool').output,
		'conv2': model.get_layer('vgg16').get_layer('block3_pool').output,
		'conv3': model.get_layer('vgg16').get_layer('block2_pool').output,
		'conv4': model.get_layer('vgg16').get_layer('block1_pool').output,
	}

	# Build the generator and discriminator
	with tf.name_scope('Generator'):
		generator, inputs, mask_sizes = semantic_model.create_generator(feature_tensors, n_classes=2)

	with tf.name_scope('Discriminator'):
		discriminator = semantic_model.create_discriminator(image_size=299)	

	# Setup the dataset which will be supplying the feature masks
	with tf.name_scope('Masking'):
		mask_dataset = semantic.mask_dataset(mask_sizes, valid_masks=('mask_conv0', 'mask_conv1', 'mask_conv2', 'mask_conv3', 'mask_conv4'),
													  image_size=299,
													  batch_size=batch_size)

	# Print model summaries
	print("Model summary")
	model.summary()
	print("Generator summary")
	generator.summary()
	print("Discriminator summary")
	discriminator.summary()

	# Begin training
	semantic.train(dataset, generator, discriminator, mask_dataset, image_size=299, 
																	batch_size=batch_size,
																	steps_per_epoch=round(num_tiles/batch_size))