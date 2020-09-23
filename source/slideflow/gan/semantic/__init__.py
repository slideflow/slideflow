import os
import sys
import time
import numpy as np
import collections
import tensorflow as tf
import random

from slideflow.util import ProgressBar

from tensorflow_gan.python.eval import eval_utils

def conormalize(tensors, batchnorm):
	# Stack reconstructed and real feature maps for co-normalization
	stacked_features = tf.stack(tensors, axis=0)
	normalized_features = batchnorm(stacked_features)
	return tf.unstack(normalized_features)

def generator_reconstruction_loss(
	real_features,
	reconstructed_features,
	masks,
	feature_type,
	batchnorm,
	reconstruction_loss_weight=1e-4,
):
	'''Calculates reconstruction loss for the generator.'''
	# Semantic reconstruction loss
	reconstruction_losses = []
	for i, (real, reconstructed, is_conv, mask, bn) in enumerate(zip(real_features, reconstructed_features, feature_type, masks, batchnorm)):

		# First, co-normalize real and reconstructed features
		real_norm, reconstructed_norm = conormalize([real, reconstructed], bn)

		# Then, apply masks
		if is_conv:
			mask = tf.cast(tf.keras.layers.MaxPool2D((2,2))(tf.cast(mask, dtype=tf.uint8)), dtype=tf.bool)
		masked_reconstructed = tf.boolean_mask(reconstructed_norm, mask)
		masked_real = tf.boolean_mask(real_norm, mask)
		
		# Now calculate L1 norm
		l1 = tf.keras.regularizers.L1()(masked_real - masked_reconstructed)
		reconstruction_losses += [l1]

	return tf.math.reduce_sum(reconstruction_losses) * reconstruction_loss_weight

def generator_diversity_loss(
	noise,
	generated_images,
	diversity_loss_weight=10.0,
	epsilon=1e-4
):
	'''Calculates diversity loss for the generator.'''
	diversity_loss = tf.keras.regularizers.L1()(noise[0] - noise[1]) / (tf.keras.regularizers.L1()(generated_images[0] - generated_images[1]) + epsilon)
	diversity_loss *= diversity_loss_weight
	return diversity_loss

def generator_adversarial_loss(fake_output, adversarial_loss_weight=0.5):
	# Adversarial loss
	#loss = tf.keras.losses.MeanSquaredError()
	#return loss(tf.ones_like(fake_output), fake_output) * adversarial_loss_weight
	return tf.math.reduce_sum(tf.math.square(fake_output - tf.ones_like(fake_output))) * adversarial_loss_weight

def discriminator_real_loss(real_output, adversarial_loss_weight=1.0):
	#loss = tf.keras.losses.MeanSquaredError()
	#return loss(tf.ones_like(real_output), real_output) * adversarial_loss_weight
	return tf.math.reduce_sum(tf.math.square(real_output - tf.ones_like(real_output))) * adversarial_loss_weight

def discriminator_fake_loss(fake_output, adversarial_loss_weight=1.0):
	'''Calculates adversarial loss for the discriminator.'''
	#loss = tf.keras.losses.MeanSquaredError()
	#return loss(tf.zeros_like(fake_output), fake_output) * adversarial_loss_weight
	return tf.math.reduce_sum(tf.math.square(fake_output)) * adversarial_loss_weight

def generate_masks(mask_sizes, mask_order, conv_masks, image_size, batch_size, spatial_variation=False, block_all=False):
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
	layer_used = []

	if spatial_variation:
		source_image_mask_flat = _mask_helper(crop_x_l, crop_x_h, crop_y_l, crop_y_h, image_size)
		source_image_mask = np.broadcast_to(source_image_mask_flat[..., np.newaxis], (image_size, image_size, 3))
	else:
		source_image_mask = np.zeros((batch_size, image_size, image_size, 3), dtype=np.bool)

	for m, mask_label in enumerate(mask_order):
		size = mask_sizes[mask_label]
		size = [size] if not (isinstance(size, list) or isinstance(size, tuple)) else size
		if block_all:
			mask_dict[mask_label] = np.zeros((batch_size, *size), dtype=np.bool)
			layer_used += [False]
		elif (m == selected_layer) or (spatial_variation and m > selected_layer and m not in conv_masks):
			mask_dict[mask_label] = np.ones((batch_size, *size), dtype=np.bool)
			layer_used += [True]
		elif not spatial_variation or (spatial_variation and (m < selected_layer)):
			mask_dict[mask_label] = np.zeros((batch_size, *size), dtype=np.bool)
			layer_used += [False]
		elif spatial_variation and (m > selected_layer) and (m in conv_masks):
			mask_flat = _mask_helper(crop_x_l, crop_x_h, crop_y_l, crop_y_h, size[0])
			image_mask = np.broadcast_to(mask_flat[..., np.newaxis], (size[0], size[0], size[-1]))
			batched_image_mask = np.broadcast_to(image_mask[np.newaxis, ...], (batch_size, *image_mask.shape))
			mask_dict[mask_label] = batched_image_mask
			layer_used += [True]

	return mask_dict, source_image_mask

def mask_dataset(mask_sizes, mask_order, conv_masks, image_size, batch_size, crop_prob=0.3, block_all=False):
	'''Returns a tf.data.Dataset containing generated masks, using generate_masks()'''
	def mask_generator():
		while True:
			# Generate cropped masks at a rate of `crop_prob` probability
			spatial_variation = random.random() < crop_prob
			mask_dict, image_mask = generate_masks(mask_sizes, mask_order, conv_masks, image_size, batch_size, spatial_variation, block_all)
			mask_dict['image_mask'] = image_mask
			yield mask_dict
	output_types = {m: tf.bool for m in mask_sizes}
	output_types['image_mask'] = tf.bool
	dataset = tf.data.Dataset.from_generator(mask_generator, output_types=output_types)
	dataset.prefetch(2)
	return dataset

def noise_dataset(z_dim, batch_size):
	def noise_generator():
		while True:
			noise1 = np.random.rand(batch_size, z_dim)
			noise2 = np.random.rand(batch_size, z_dim)
			yield noise1, noise2
	dataset = tf.data.Dataset.from_generator(noise_generator, output_types=(tf.float32, tf.float32))
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
	noise_dataset,
	image_size,
	steps_per_epoch,
	keras_strategy,
	batch_size=4,
	summary_step=200,
	z_dim=128,
	gen_lr=1e-4,
	disc_lr=1e-4,
	epochs=10,
	starting_step=0,
	checkpoint_dir='/home/shawarma/test_log',
	generator_steps=400,
	discriminator_steps=400,
	load_checkpoint_prefix=None,
	load_checkpoint=None,
	reconstruction_loss_weight=1e-4,
	diversity_loss_weight=10.0,
	adversarial_loss_weight=0.5
):
	with keras_strategy.scope():
		'''Trains a semantic pyramid GAN.'''
		generator_optimizer = tf.keras.optimizers.Adam(gen_lr)
		discriminator_optimizer = tf.keras.optimizers.Adam(disc_lr)

		checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
		checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
										discriminator_optimizer=discriminator_optimizer,
										generator=generator,
										discriminator=discriminator)
		if load_checkpoint_prefix:
			checkpoint_to_load = checkpoint_prefix+f'-{load_checkpoint_prefix}'
			print(f"Loading saved checkpoint {checkpoint_to_load}")
			checkpoint.restore(checkpoint_to_load)
		elif load_checkpoint:
			print(f"Loading saved checkpoint {load_checkpoint}")
			checkpoint.restore(load_checkpoint)

		writer = tf.summary.create_file_writer(checkpoint_dir)

		reconstruction_batchnorm = [tf.keras.layers.BatchNormalization(dtype=tf.float32) for m in mask_order]
		
		is_conv = [True if m in conv_masks else False for m in mask_order]

		@tf.function
		def generator_summary_step(images, labels, masks, noise, step):
			'''Step which saves summary statistics and sample images for display with Tensorboard.'''
			generated_images, gen_loss, gen_adv_loss, rec_loss, div_loss = distributed_generator_step(images, labels, masks, noise, False)
			
			images = keras_strategy.experimental_local_results(images)[0]
			generated_images_first = keras_strategy.experimental_local_results(generated_images[0])[0]
			generated_images_second = keras_strategy.experimental_local_results(generated_images[1])[0]
			gen_loss = keras_strategy.experimental_local_results(gen_loss)[0]
			gen_adv_loss = keras_strategy.experimental_local_results(gen_adv_loss)[0]
			rec_loss = keras_strategy.experimental_local_results(rec_loss)[0]
			div_loss = keras_strategy.experimental_local_results(div_loss)[0]

			with writer.as_default():
				tf.summary.image('real0', images, max_outputs=4, step=step)
				tf.summary.image('fake-noise1', generated_images_first, max_outputs=4, step=step)
				tf.summary.image('fake-noise2', generated_images_second, max_outputs=4, step=step)
				tf.summary.scalar('generator/total_loss', gen_loss,	step=step)
				tf.summary.scalar('generator/diversity_loss', div_loss,	step=step)
				tf.summary.scalar('generator/adversarial_loss',	gen_adv_loss, step=step)
				tf.summary.scalar('generator/reconstruction_loss', rec_loss, step=step)

		@tf.function
		def discriminator_summary_step(images, labels, masks, noise, step):
			'''Step which saves summary statistics and sample images for display with Tensorboard.'''
			disc_loss = distributed_discriminator_step(images, labels, masks, noise, False)
			disc_loss = keras_strategy.experimental_local_results(disc_loss)[0]

			with writer.as_default():
				tf.summary.scalar(
					'discriminator/total_loss',
					disc_loss,
					step=step)

		@tf.function
		def distributed_generator_step(dist_images, dist_labels, dist_masks, dist_noise, apply_grads=True):
			gen_images, gen_loss, gen_adv_loss, rec_loss, div_loss = keras_strategy.run(generator_step, args=(dist_images, dist_labels, dist_masks, dist_noise, apply_grads))

			sum_gen_loss = keras_strategy.reduce(tf.distribute.ReduceOp.SUM, gen_loss, axis=None)
			sum_gen_adv_loss = keras_strategy.reduce(tf.distribute.ReduceOp.SUM, gen_adv_loss, axis=None)
			sum_rec_loss = keras_strategy.reduce(tf.distribute.ReduceOp.SUM, rec_loss, axis=None)
			sum_div_loss = keras_strategy.reduce(tf.distribute.ReduceOp.SUM, div_loss, axis=None)
			return gen_images, sum_gen_loss, sum_gen_adv_loss, sum_rec_loss, sum_div_loss

		@tf.function
		def distributed_discriminator_step(dist_images, dist_labels, dist_masks, dist_noise, apply_grads=True):
			disc_loss = keras_strategy.run(discriminator_step, args=(dist_images, dist_labels, dist_masks, dist_noise, apply_grads))
			return keras_strategy.reduce(tf.distribute.ReduceOp.SUM, disc_loss, axis=None)

		def _gen_output_helper(input):
			'''With a given input, uses a generator to calculate generated images, as well as both
			the input real features (as calculated from the input image),
			as well as the reconstructed features from the generator.'''
			#generator_output = generator(input, training=True)
			#generated_images = generator_output[0]
			#feature_output = generator_output[1:]
			#return generated_images, feature_output
			return generator(input, training=True), None

		@tf.function
		def generator_step(images, labels, masks, noise, apply_grads=True):
			# Noise inputs. In order to calculate diversity loss, 
			#  Identical pairs of input batches are processed together,
			#  Except with different noise inputs.
			#  Using diversity loss, we are optimizing to increase output diversity
			#  From differing noise inputs.

			generator_input = {
				'tile_image': images,
				'input_1': images,
				'class_input': labels,
				'noise_input': noise[0]
			}
			# Supply all masks as input data apart from 'image_mask', which is is the mask
			#  Supposed to be applied to the final generator image, but I'm not sure how to use/implement this.
			if type(masks) == tuple: masks = masks[0]
			#for m in masks:	
			#	if m != 'image_mask': generator_input[m] = masks[m]

			# Calculate gradients from losses.
			with tf.GradientTape() as gen_tape:
				# Images are generated in two groups with different noise inputs, 
				#  in order to generate diversity loss

				# Get first half of generated images and real image features
				generated_images_first, real_feat_out_first = _gen_output_helper(generator_input)
				
				# Second half of generated images and real image features, using a different noise vector
				generator_input['noise_input'] = noise[1]
				generated_images_sec, real_feat_out_sec = _gen_output_helper(generator_input)
				
				# Get reconstructed features from first generated images
				#recon_feat_out_first = reference_features({'tile_image': generated_images_first, 'input_1': generated_images_first})

				# Get reconstructed features from second generated images
				#recon_feat_out_sec = reference_features({'tile_image': generated_images_sec, 'input_1': generated_images_sec})

				# Get discriminator output from generated images
				fake_output_first = discriminator(generated_images_first, training=True)
				fake_output_sec = discriminator(generated_images_sec, training=True)

				# Calculate adversarial generator loss
				gen_adv_loss = generator_adversarial_loss(fake_output_first, adversarial_loss_weight=adversarial_loss_weight)
				gen_adv_loss += generator_adversarial_loss(fake_output_sec, adversarial_loss_weight=adversarial_loss_weight)

				# Calculate reconstruction generator loss
				#rec_loss = generator_reconstruction_loss(real_features=real_feat_out_first,
				#										reconstructed_features=recon_feat_out_first,
				#										feature_type=is_conv,
				#										masks=[masks[m] for m in mask_order],
				#										batchnorm=reconstruction_batchnorm,
				#										reconstruction_loss_weight=reconstruction_loss_weight)
				#rec_loss += generator_reconstruction_loss(real_features=real_feat_out_sec,
				#										reconstructed_features=recon_feat_out_sec,
				#										feature_type=is_conv,
				#										masks=[masks[m] for m in mask_order],
				#										batchnorm=reconstruction_batchnorm,
				#										reconstruction_loss_weight=reconstruction_loss_weight)
				rec_loss = 0

				# Calculate diversity loss
				div_loss = generator_diversity_loss(noise=noise,
													generated_images=[generated_images_first, generated_images_sec],
													diversity_loss_weight=diversity_loss_weight)

				# Sum generator loss
				gen_loss = div_loss + rec_loss + gen_adv_loss

			# Calculate and apply gradients.
			if apply_grads:
				gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
				generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

			return [generated_images_first, generated_images_sec], gen_loss, gen_adv_loss, rec_loss, div_loss

		@tf.function
		def discriminator_step(images, labels, masks, noise, apply_grads=True):
			'''Training step.'''
			# Noise inputs. In order to calculate diversity loss, 
			#  Identical pairs of input batches are processed together,
			#  Except with different noise inputs.
			#  Using diversity loss, we are optimizing to increase output diversity
			#  From differing noise inputs.

			generator_input = {
				'tile_image': images,
				'input_1': images,
				'class_input': labels,
				'noise_input': noise[0]
			}
			# Supply all masks as input data apart from 'image_mask', which is is the mask
			#  Supposed to be applied to the final generator image, but I'm not sure how to use/implement this.
			if type(masks) == tuple: masks = masks[0]
			#for m in masks:	
			#	if m != 'image_mask': generator_input[m] = masks[m]

			# Get first half of generated images and real image features
			generated_images, real_feat_out = _gen_output_helper(generator_input)

			# Calculate gradients from losses.
			with tf.GradientTape() as disc_tape:
				# Get real and generated discriminator output
				real_output = discriminator(images, training=True)
				fake_output = discriminator(generated_images, training=True)
				
				# Split output so the real/fake images are non-overlapping
				real_output_first_half, _ = tf.split(real_output, 2)
				_, fake_output_second_half = tf.split(fake_output, 2)

				# Calculate discriminator loss
				disc_loss = discriminator_real_loss(real_output_first_half)
				disc_loss += discriminator_fake_loss(fake_output_second_half)

			# Calculate and apply gradients.
			if apply_grads:
				gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
				discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

			return disc_loss

		disc_step = 0
		gen_step = -1

		for epoch in range(epochs):
			print(f"Epoch {epoch}")

			pb = ProgressBar(steps_per_epoch*batch_size, show_eta=True, show_counter=True, counter_text='images', leadtext="Step 0")
			for step, ((image_batch, label_batch), mask_batch, noise_batch) in enumerate(zip(dataset, mask_dataset, noise_dataset)):
				if step == 0:
					distributed_generator_step(image_batch, label_batch, mask_batch, noise_batch)
					continue
				step += (steps_per_epoch * epoch)
				step += starting_step



				if (step > 0) and step % summary_step == 0:
					# Discriminator summary step
					discriminator_summary_step(image_batch, label_batch, mask_batch, noise_batch, tf.constant(step, dtype=tf.int64))
					writer.flush()

				elif (step > 1) and (step+1) % summary_step == 0:
					# Generator summary step
					generator_summary_step(image_batch, label_batch, mask_batch, noise_batch, tf.constant(step, dtype=tf.int64))
					writer.flush()

				elif disc_step != -1:
					if disc_step < discriminator_steps:
						# Discriminator training step
						distributed_discriminator_step(image_batch, label_batch, mask_batch, noise_batch)
						disc_step += 1
					else:
						disc_step = -1
						gen_step = 0
				elif gen_step != -1:
					if gen_step < generator_steps:
						# Generator training step
						distributed_generator_step(image_batch, label_batch, mask_batch, noise_batch)
						gen_step += 1
					else:
						gen_step = -1
						disc_step = 0
				else:
					pb.print("Error, unknown training step")

				# Increase progress bar
				pb.increase_bar_value(batch_size)
				pb.leadtext = f"Step {step:>5}"
			
				# Save a checkpoint
				if step > 0 and step % 10000 == 0:
					checkpoint.save(file_prefix=checkpoint_prefix)
					pb.print(f"Checkpoint at step {step} saved to {checkpoint_prefix}")
			pb.end()
