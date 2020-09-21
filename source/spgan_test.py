import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from slideflow import gan

gan.gan_test(project='/scratch/t.cri.jdolezal/projects/TCGA_THCA_MANUSCRIPT',
	     model='/scratch/t.cri.jdolezal/models/vgg_model_partially_trained.h5',
	     checkpoint_dir='/scratch/t.cri.jdolezal/checkpoints/reworked/',
	     batch_size=64,
	     epochs=200,
	     generator_steps=10,
	     discriminator_steps=1,
	     summary_step=40,
	     load_checkpoint=None,#'/scratch/t.cri.jdolezal/checkpoints/ckpt-63',
	     adversarial_loss_weight=1.0,
	     diversity_loss_weight=50.0,
	     reconstruction_loss_weight=0,
	     enable_features=False,
	     use_mixed_precision=False)
