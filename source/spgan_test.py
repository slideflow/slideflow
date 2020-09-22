import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from slideflow import gan

policy = tf.keras.mixed_precision.experimental.Policy('float32')
mixed_precision.set_policy(policy)
gan.gan_test(project='/home/shawarma/Thyroid-Paper-Final/projects/TCGA',
			  model='/home/shawarma/Thyroid-Paper-Final/projects/TCGA/models/brs-BRS_VGG16_FULL_NEWT/trained_model_epoch1.h5',
			  checkpoint_dir='/home/shawarma/test_log',
			  batch_size=4,
			  load_checkpoint=0,
			  starting_step=0,
			  enable_features=False,
			  generator_steps=4,
			  discriminator_steps=1,
			  image_size=256,
			  use_mixed_precision=False)
