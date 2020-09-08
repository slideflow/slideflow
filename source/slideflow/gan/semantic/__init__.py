import os
import sys
import time
import numpy as np
import collections
import tensorflow as tf
import random

from slideflow.util import ProgressBar

from tensorflow_gan.python.eval import eval_utils

def generator_adversarial_loss(fake_output):
	# Adversarial loss
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_reconstruction_loss(
	real_features,
	reconstructed_features,
	masks,
	feature_type,
	reconstruction_loss_fn=tf.compat.v1.losses.absolute_difference,
	reconstruction_loss_weight=0.1,
	add_summaries=False
):
	'''Calculates reconstruction loss for the generator.'''
	# Semantic reconstruction loss
	reconstruction_losses = []
	for i, (real, reconstructed, is_conv, mask) in enumerate(zip(real_features, reconstructed_features, feature_type, masks)):
		if is_conv:
			reconstruction_losses += [reconstruction_loss_fn(tf.boolean_mask(real, tf.cast(tf.keras.layers.MaxPool2D((2,2))(tf.cast(mask, dtype=tf.uint8)), dtype=tf.bool)), 
									  						 tf.boolean_mask(reconstructed, tf.cast(tf.keras.layers.MaxPool2D((2,2))(tf.cast(mask, dtype=tf.uint8)), dtype=tf.bool)))]
		else:
			reconstruction_losses += [reconstruction_loss_fn(tf.boolean_mask(real, mask),
															 tf.boolean_mask(reconstructed, mask))]

	return tf.math.reduce_sum(reconstruction_losses) * reconstruction_loss_weight

def generator_diversity_loss(
	noise,
	generated_images,
	diversity_loss_fn=tf.compat.v1.losses.absolute_difference,
	diversity_loss_weight=0.1
):
	'''Calculates diversity loss for the generator.'''
	diversity_loss = diversity_loss_fn(*noise) / diversity_loss_fn(*generated_images)
	diversity_loss *= diversity_loss_weight
	return diversity_loss

def discriminator_real_loss(real_output, add_summaries=False):
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	return cross_entropy(tf.ones_like(real_output), real_output)

def discriminator_fake_loss(fake_output, add_summaries=False):
	'''Calculates adversarial loss for the discriminator.'''
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	return cross_entropy(tf.zeros_like(fake_output), fake_output)

def generate_masks(mask_sizes, mask_order, conv_masks, image_size, batch_size, spatial_variation=False):
	'''Generates random masks as described in https://semantic-pyramid.github.io.
	Generated mask crops are only square.'''

	def _mask_helper(x_l, x_h, y_l, y_h, size):
		return np.array([[False if int(x_l*size) <= r <= int(x_h*size) else True for r in range(size)]
				if int(y_l*size) <= c <= int(y_h*size) else [True] * size 
				for c in range(size)], dtype=np.bool)

	mask_dict = {}
	selected_layer = random.choice(range(len(mask_order)))
	crop_x_l = random.uniform(0, 0.9)
	crop_x_h = random.uniform(crop_x_l, 1)
	crop_y_l = random.uniform(0, 0.9)
	crop_y_h = random.uniform(crop_x_l, 1)
	mask = _mask_helper(crop_x_l, crop_x_h, crop_y_l, crop_y_h, image_size)
	image_mask = np.broadcast_to(mask[..., np.newaxis], (image_size, image_size, 3))
	layer_used = []

	for m, mask_label in enumerate(mask_order):
		size = mask_sizes[mask_label]
		size = [size] if not (isinstance(size, list) or isinstance(size, tuple)) else size
		if (m == selected_layer) or (spatial_variation and m > selected_layer and m not in conv_masks):
			mask_dict[mask_label] = np.ones((batch_size, *size), dtype=np.bool)
			layer_used += [True]
		elif not spatial_variation or (spatial_variation and (m < selected_layer)):
			mask_dict[mask_label] = np.zeros((batch_size, *size), dtype=np.bool)
			layer_used += [False]
		elif spatial_variation and (m > selected_layer) and (m in conv_masks):
			mask = _mask_helper(crop_x_l, crop_x_h, crop_y_l, crop_y_h, size[0])
			image_mask = np.broadcast_to(mask[..., np.newaxis], (size[0], size[0], size[-1]))
			batched_image_mask = np.broadcast_to(image_mask[np.newaxis, ...], (batch_size, *image_mask.shape))
			mask_dict[mask_label] = batched_image_mask
			layer_used += [True]

	return mask_dict, image_mask

def mask_dataset(mask_sizes, mask_order, conv_masks, image_size, batch_size, crop_prob=0.3):
	'''Returns a tf.data.Dataset containing generated masks, using generate_masks()'''
	def mask_generator():
		while True:
			# Generate cropped masks at a rate of `crop_prob` probability
			spatial_variation = random.random() < crop_prob
			mask_dict, image_mask = generate_masks(mask_sizes, mask_order, conv_masks, image_size, batch_size, spatial_variation)
			mask_dict['image_mask'] = image_mask
			yield mask_dict
	output_types = {m: tf.bool for m in mask_sizes}
	output_types['image_mask'] = tf.bool
	dataset = tf.data.Dataset.from_generator(mask_generator, output_types=output_types)
	dataset.prefetch(2)
	return dataset	

def train(
	dataset, 
	generator,
	discriminator,
	reference_features,
	mask_dataset,
	mask_order,
	conv_masks,
	image_size,
	steps_per_epoch,
	batch_size=4,
	gen_lr=1e-4,
	disc_lr=1e-4,
	epochs=10,
	checkpoint_dir='/home/shawarma/test_log'
):
	'''Trains a semantic pyramid GAN.'''
	generator_optimizer = tf.keras.optimizers.Adam(gen_lr)
	discriminator_optimizer = tf.keras.optimizers.Adam(disc_lr)

	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									discriminator_optimizer=discriminator_optimizer,
									generator=generator,
									discriminator=discriminator)
	#checkpoint.restore(checkpoint_prefix+'-19')

	writer = tf.summary.create_file_writer(checkpoint_dir)

	is_conv = [True if m in conv_masks else False for m in mask_order]

	def _gen_output_helper(input):
		'''With a given input, uses a generator to calculate generated images, as well as both
		   the input real features (as calculated from the input image),
		   as well as the reconstructed features from the generator.'''
		generator_output = generator(input, training=True)
		generated_images = generator_output[0]
		feature_output = generator_output[1:]
		#reconstructed_feature_output = generator_output[8:]
		return generated_images, feature_output#, reconstructed_feature_output

	@tf.function
	def summary_step(images, labels, masks, step):
		'''Step which saves summary statistics and sample images for display with Tensorboard.'''
		generated_images, gen_loss, disc_loss, gen_adv_loss, rec_loss, div_loss = train_step(images, labels, masks)
		
		with writer.as_default():
			tf.summary.image(
				'real_data',
				eval_utils.image_grid(
					images[:4],
					grid_shape=(2, 2),
					image_shape=(299, 299)),
				max_outputs=1,
				step=step)
			tf.summary.image(
				'generated_data',
				eval_utils.image_grid(
					generated_images[:4],
					grid_shape=(2, 2),
					image_shape=(299, 299)),
				max_outputs=1,
				step=step)
			tf.summary.scalar(
				'generator_total_loss',
				gen_loss,
				step=step)
			tf.summary.scalar(
				'discriminator_loss',
				disc_loss,
				step=step)
			tf.summary.scalar(
				'diversity_loss',
				div_loss,
				step=step)
			tf.summary.scalar(
				'generator_adversarial_loss',
				gen_adv_loss,
				step=step)
			tf.summary.scalar(
				'reconstruction_loss',
				rec_loss,
				step=step)

	@tf.function
	def train_step(images, labels, masks):
		'''Training step.'''
		# Noise inputs. In order to calculate diversity loss, 
		#  Identical pairs of input batches are processed together,
		#  Except with different noise inputs.
		#  Using diversity loss, we are optimizing to increase output diversity
		#  From differing noise inputs.
		noise1 = tf.random.normal([batch_size, 100])
		noise2 = tf.random.normal([batch_size, 100])

		generator_input = {
			'tile_image': images,
			'input_1': images,
			'class_input': labels,
			'noise_input': noise1
		}
		# Supply all masks as input data apart from 'image_mask', which is is the mask
		#  Supposed to be applied to the final generator image, but I'm not sure how to use/implement this.
		for m in masks:	
			if m != 'image_mask': generator_input[m] = masks[m]
	
		# Calculate gradients from losses.
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			# Images are generated in two groups with different noise inputs, 
			#  in order to generate diversity loss

			# Get first half of generated images and real image features
			generated_images_first, real_feat_out_first = _gen_output_helper(generator_input)
			
			# Second half of generated images and real image features, using a different noise vector
			generator_input['noise_input'] = noise2
			generated_images_sec, real_feat_out_sec = _gen_output_helper(generator_input)
			
			# Get reconstructed features from first generated images
			recon_feat_out_first = reference_features({'tile_image': generated_images_first, 'input_1': generated_images_first})

			# Get reconstructed features from second generated images
			recon_feat_out_sec = reference_features({'tile_image': generated_images_sec, 'input_1': generated_images_sec})

			# Get real and generated discriminator output
			real_output = discriminator(images, training=True)
			fake_output_first = discriminator(generated_images_first, training=True)
			fake_output_sec = discriminator(generated_images_sec, training=True)

			# Calculate adversarial generator loss
			gen_adv_loss = generator_adversarial_loss(fake_output_first)
			gen_adv_loss += generator_adversarial_loss(fake_output_sec)

			# Calculate reconstruction generator loss
			rec_loss = generator_reconstruction_loss(real_features=real_feat_out_first,
													 reconstructed_features=recon_feat_out_first,
													 feature_type=is_conv,
													 masks=[masks[m] for m in mask_order])
			rec_loss += generator_reconstruction_loss(real_features=real_feat_out_sec,
													 reconstructed_features=recon_feat_out_sec,
													 feature_type=is_conv,
													 masks=[masks[m] for m in mask_order])

			# Calculate diversity loss
			div_loss = generator_diversity_loss(noise=[noise1, noise2],
												generated_images=[fake_output_first, fake_output_sec])

			# Sum generator loss
			gen_loss = div_loss + rec_loss + gen_adv_loss

			# Calculate discriminator loss
			disc_loss = discriminator_real_loss(real_output)
			disc_loss += discriminator_fake_loss(fake_output_first)
			disc_loss += discriminator_fake_loss(fake_output_sec)

		# Calculate and apply gradients.
		gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
		gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

		generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
		discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

		return generated_images_first, gen_loss, disc_loss, gen_adv_loss, rec_loss, div_loss

	for epoch in range(epochs):
		print(f"Epoch {epoch}")

		pb = ProgressBar(steps_per_epoch*batch_size, show_eta=True, show_counter=True, counter_text='images', leadtext="Step 0")
		for step, ((image_batch, label_batch), mask_batch) in enumerate(zip(dataset, mask_dataset)):
			# Training step
			train_step(image_batch, label_batch, mask_batch)
			pb.increase_bar_value(batch_size)
			pb.leadtext = f"Step {step:>5}"

			# Summary step
			if step % 20 == 0:
				summary_step(image_batch, label_batch, mask_batch, tf.constant(step, dtype=tf.int64))
				writer.flush()

			# Save a checkpoint
			if step % 2000 == 0:
				checkpoint.save(file_prefix=checkpoint_prefix)
		pb.end()