	def build_model(self, model_dir):
		self.padded_batch_pl = tf.placeholder(self.DTYPE, shape=[self.BATCH_SIZE, self.SIZE, self.SIZE, 3])
		with tf.Graph().as_default() as g:
			# Generate dataset from coordinates
			with tf.name_scope('dataset_input'):
				tile_dataset = tf.data.Dataset.from_generator(self.get_slice, (self.DTYPE, tf.int64, tf.bool))
				tile_dataset = tile_dataset.batch(self.BATCH_SIZE, drop_remainder = False)
				tile_dataset = tile_dataset.prefetch(1)
				tile_iterator = tile_dataset.make_one_shot_iterator()
				next_batch_images, next_batch_labels, next_batch_unique  = tile_iterator.get_next()

				# Generate ops that will convert batch of coordinates to extracted & processed image patches from whole-slide-image
				image_patches = tf.map_fn(lambda patch: tf.cast(tf.image.per_image_standardization(patch), self.DTYPE), next_batch_images)

				# Pad the batch if necessary to create a batch of minimum size BATCH_SIZE
				padded_batch = tf.concat([image_patches, tf.zeros([self.BATCH_SIZE - tf.shape(image_patches)[0], self.SIZE, self.SIZE, 3], # image_patches instead of next_batch
															dtype=self.DTYPE)], 0)
				padded_batch.set_shape([self.BATCH_SIZE, self.SIZE, self.SIZE, 3])

			with arg_scope(inception_arg_scope()):
				_, end_points = inception_v4.inception_v4(self.padded_batch_pl, num_classes = self.NUM_CLASSES)
				prelogits = end_points['PreLogitsFlatten']
				slogits = end_points['Predictions']
				num_tensors_final_layer = prelogits.get_shape().as_list()[1]
				vars_to_restore = []

				for var_to_restore in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
					if ((var_to_restore.name[12:21] != "AuxLogits") and 
						((var_to_restore.name[:25] != "InceptionV4/Logits/Logits") or not final_layer)):
						vars_to_restore.append(var_to_restore)

				saver = tf.train.Saver(vars_to_restore)
				
				with tf.Session() as sess:
					init = (tf.global_variables_initializer(), tf.local_variables_initializer())
					sess.run(init)

					ckpt = tf.train.get_checkpoint_state(model_dir)
					if ckpt and ckpt.model_checkpoint_path:
						print("Restoring saved checkpoint model.")
						saver.restore(sess, ckpt.model_checkpoint_path)
					else:
						raise Exception('Unable to find checkpoint file.')