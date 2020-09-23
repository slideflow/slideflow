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

import tensorflow_gan.examples.self_attention_estimator.discriminator as sagan_discriminator

def _parse_tfgan(record, sf_model, n_classes, include_slidenames=False, multi_image=False, z_dim=128, resize=False):
	features = tf.io.parse_single_example(record, sf.io.tfrecords.FEATURE_DESCRIPTION)
	slide = features['slide']
	image_string = features['image_raw']
	image = sf_model._process_image(image_string, augment=True)

	if resize:
		image = tf.image.resize(image, (resize, resize))

	brs = sf_model.ANNOTATIONS_TABLES[0].lookup(slide)
	label = tf.cond(brs < tf.constant(0, dtype=tf.float32), lambda: tf.constant(0, dtype=tf.int32), lambda: tf.constant(1, dtype=tf.int32))
	label = tf.cast(label, tf.int32)

	return image, label

def _parse_tfrecord_brs(record, sf_model, n_classes, include_slidenames=False, multi_image=False, resize=False):
	features = tf.io.parse_single_example(record, sf.io.tfrecords.FEATURE_DESCRIPTION)
	slide = features['slide']
	image_string = features['image_raw']
	image = sf_model._process_image(image_string, augment=True)

	if resize:
		image = tf.image.resize(image, (resize, resize))

	brs = sf_model.ANNOTATIONS_TABLES[0].lookup(slide)
	label = tf.cond(brs < tf.constant(0, dtype=tf.float32), lambda: tf.constant(0), lambda: tf.constant(1))
	#label = tf.one_hot(label, n_classes)
	label = tf.cast(label, tf.int32)
	
	return image, label

def gan_test(
	project, 
	model,
	checkpoint_dir,
	batch_size=4,
	epochs=10,
	load_checkpoint=None,
	load_checkpoint_prefix=None,
	starting_step=0,
	summary_step=200,
	generator_steps=1,
	discriminator_steps=1,
	z_dim=128,
	image_size=299,
	adversarial_loss_weight=0.5,
	diversity_loss_weight=10.0,
	reconstruction_loss_weight=1e-4,
	use_mixed_precision=False,
	enable_features=True,
	gen_alt_block=False
):
	# Set mixed precision flag; it seems that mixed precision worsens GAN performance so 
	#  I would recommend against its use for now
	if use_mixed_precision:
		policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
		mixed_precision.set_policy(policy)

	keras_strategy = tf.distribute.get_strategy()#tf.distribute.MirroredStrategy()
	with keras_strategy.scope():
		# Setup project-specific details. This will eventually need to be replaced
		#  With a more flexibile solution.
		SFP = sf.SlideflowProject(project, ignore_gpu=True)
		sf_dataset = SFP.get_dataset(tile_px=299, tile_um=302, filters={'brs_class': ['Braf-like', 'Ras-like']})
		tfrecords = sf_dataset.get_tfrecords()
		# Build actual dataset inputs using a slideflow model
		slide_annotations, _ = sf_dataset.get_outcomes_from_annotations('brs', use_float=True)
		train_tfrecords = tfrecords
		validation_tfrecords = None
		manifest = sf_dataset.get_manifest()
		SFM = SlideflowModel(checkpoint_dir, 299, slide_annotations, train_tfrecords, validation_tfrecords, manifest, model_type='linear')
		dataset, _, num_tiles = SFM._build_dataset_inputs(tfrecords, batch_size, 'NO_BALANCE', augment=False,
																							   finite=True,
																							   include_slidenames=False,
																							   parse_fn=partial(_parse_tfrecord_brs, sf_model=SFM, n_classes=2, resize=image_size),
																							   drop_remainder=True)
		dataset = dataset.prefetch(20)
		dataset = keras_strategy.experimental_distribute_dataset(dataset)
		
		# Load the external model
		with tf.name_scope('ExternalModel'):
			model = tf.keras.models.load_model(model)	

		# Set loaded model as non-trainable
		for layer in model.layers:
			layer.trainable = False

		# Identify the feature activation maps that will be used for the generator input
		feature_tensors = {
			'image': model.input,
			'image_vgg16': model.get_layer('vgg16').input,
			#'fc8':   tf.keras.layers.BatchNormalization()(model.get_layer('hidden_1').output),
			#'fc7':   tf.keras.layers.BatchNormalization()(model.get_layer('hidden_0').output),
			'conv0': tf.keras.layers.BatchNormalization()(model.get_layer('vgg16').get_layer('block5_pool').output), # 512 channels (9x9)
			'conv1': tf.keras.layers.BatchNormalization()(model.get_layer('vgg16').get_layer('block4_pool').output), # 512 channels (18x18)
			'conv2': tf.keras.layers.BatchNormalization()(model.get_layer('vgg16').get_layer('block3_pool').output), # 256 channels (37x37)
			'conv3': tf.keras.layers.BatchNormalization()(model.get_layer('vgg16').get_layer('block2_pool').output), # 128 channels (74x74)
			'conv4': tf.keras.layers.BatchNormalization()(model.get_layer('vgg16').get_layer('block1_pool').output), # 64 channels  (149x149)
		}

		# Build the generator and discriminator
		with tf.name_scope('Generator'):
			generator, generator_input_layers, mask_sizes, mask_order = semantic_model.create_generator(feature_tensors, n_classes=2, z_dim=z_dim, use_alt_block=gen_alt_block)

		with tf.name_scope('Discriminator'):
			discriminator = semantic_model.create_discriminator(image_size=image_size)

		# Build a model that will output pooled features from the reference model, to be used for reconstruction loss
		features_with_pool = [#tf.cast(feature_tensors['fc8'], dtype=tf.float32),
							  #tf.cast(feature_tensors['fc7'], dtype=tf.float32),
							  tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv0']), dtype=tf.float32),
							  tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv1']), dtype=tf.float32),
							  tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv2']), dtype=tf.float32),
							  tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv3']), dtype=tf.float32),
							  tf.cast(tf.keras.layers.MaxPool2D((2,2))(feature_tensors['conv4']), dtype=tf.float32)
		]
		input_layers = [feature_tensors['image'], feature_tensors['image_vgg16']]
		reference_features = tf.keras.models.Model(input_layers, features_with_pool)

		# Setup the dataset which will be supplying the feature masks
		with tf.name_scope('Masking'):
			conv_masks = ('mask_conv0', 'mask_conv1', 'mask_conv2', 'mask_conv3', 'mask_conv4')
			mask_dataset = semantic.mask_dataset(mask_sizes, mask_order=mask_order,
															conv_masks=conv_masks,
															image_size=image_size,
															batch_size=batch_size,
															block_all=(not enable_features))
			mask_dataset = keras_strategy.experimental_distribute_dataset(mask_dataset)
			noise_dataset = semantic.noise_dataset(z_dim=z_dim, batch_size=batch_size)
			noise_dataset = keras_strategy.experimental_distribute_dataset(noise_dataset)

		# Print model summaries
		print("Model summary")
		model.summary()
		print("Generator summary")
		generator.summary()
		print("Discriminator summary")
		discriminator.summary()

		# Begin training
		semantic.train(dataset, generator, discriminator, reference_features, mask_dataset=mask_dataset,
																			  mask_order=mask_order,
																			  conv_masks=conv_masks,
																			  noise_dataset=noise_dataset,
																			  image_size=image_size, 
																			  steps_per_epoch=round(num_tiles/batch_size),
																			  keras_strategy=keras_strategy,
																			  checkpoint_dir=checkpoint_dir,
																			  load_checkpoint=load_checkpoint,
																			  load_checkpoint_prefix=load_checkpoint_prefix,
																			  starting_step=starting_step,
																			  generator_steps=generator_steps,
																			  discriminator_steps=discriminator_steps,
																			  batch_size=batch_size,
																			  epochs=epochs,
																			  summary_step=summary_step,
																			  z_dim=z_dim,
																			  reconstruction_loss_weight=reconstruction_loss_weight,
																			  diversity_loss_weight=diversity_loss_weight,
																			  adversarial_loss_weight=adversarial_loss_weight)
