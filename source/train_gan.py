import argparse
import os
import shutil

from os.path import join, exists

def get_paths(args):
	scratch = join('/scratch', args.user)
	labshare = '/gpfs/data/pearson-lab'
	project_folder = join(scratch, 'projects', 'TCGA_THCA_MANUSCRIPT')
	paths = {
		'scratch': scratch,
		'labshare': labshare,
		'annotations_labshare': join(labshare, 'PROJECT_BACKUPS/THCA/Thyroid-Manuscript-TCGA/annotations.csv'),
		'tfrecords_labshare': join(labshare, 'TFRECORDS_BACKUPS/TCGA_THCA_MANUSCRIPT'),
		'trained_model_labshare': join(labshare, 'TRAINED_MODELS/TCGA_THCA - Thyroid manuscript/vgg_model_partially_trained.h5'),
		'project_folder': project_folder,
		'annotations_scratch': join(project_folder, 'annotations.csv'),
		'tfrecords_scratch': join(scratch, 'tfrecords'),
		'models_scratch': join(scratch, 'models'),
		'trained_model_scratch': join(scratch, 'models', 'vgg_model_partially_trained.h5'),
		'checkpoint_dir': join(scratch, 'checkpoints', args.name)
	}
	if not exists(paths['checkpoint_dir']): os.makedirs(p['checkpoint_dir'])
	return paths

def gan_train(args):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	from slideflow import SlideflowProject
	from slideflow import gan as sfgan
	p = get_paths(args)
	sfgan.gan_test(project=p['project_folder'],
	     model=p['trained_model_scratch'],
	     checkpoint_dir=p['checkpoint_dir'],
	     batch_size=args.batch_size,
	     epochs=args.epochs,
	     generator_steps=args.gen_steps,
	     discriminator_steps=args.dis_steps,
		 z_dim=args.z_dim,
		 image_size=args.size,
	     summary_step=args.summary_steps,
	     load_checkpoint=args.ckpt,
	     adversarial_loss_weight=args.adv_loss_wt,
	     diversity_loss_weight=args.div_loss_wt,
	     reconstruction_loss_weight=args.rec_loss_wt,
	     enable_features=False,
	     use_mixed_precision=args.mixed_precision)

def setup(args):
	from slideflow import SlideflowProject
	p = get_paths(args)
	print("Settting up scratch and project directories...")
	if not exists(p['project_folder']): os.makedirs(p['project_folder'])
	if not exists(p['tfrecords_scratch']): os.makedirs(p['tfrecords_scratch'])
	if not exists(p['models_scratch']): os.makedirs(p['models_scratch'])

	print("Copying annotation file...")
	shutil.copy(p['annotations_labshare'], p['annotations_scratch'])
	print("Copying TFRecords...")
	shutil.copy(p['tfrecords_labshare'], p['tfrecords_scratch'])
	print("Copying trained model...")
	shutil.copy(p['trained_model_labshare'], p['models_scratch'])

	print("Setting up project...")
	SFP = SlideflowProject(p['project_folder'], ignore_gpu=True, interactive=False)
	SFP.add_dataset('TCGA_THCA_MANUSCRIPT', p['scratch'], p['scratch'], p['scratch'], p['tfrecords_scratch'])

if __name__=='__main__':
	parser = argparse.ArgumentParser(description = "Helper to guide GAN setup on CRI cluster.")
	parser.add_argument('--setup', action='store_true', help='Whether to perform initial setup.')
	parser.add_argument('-u', '--user', required=True, help='Username.')
	parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size.')
	parser.add_argument('-e', '--epochs', type=int, default=200, help='Total epochs.')
	parser.add_argument('-z', '--z_dim', type=int, default=128, help='Noise z dim. [128]')
	parser.add_argument('-gs', '--gen_steps', type=int, default=2, help='Number of generator steps.')
	parser.add_argument('-ds', '--dis_steps', type=int, default=1, help='Number of discriminator steps.')
	parser.add_argument('-ss', '--summary_steps', type=int, default=40, help='Number steps before summary.')
	parser.add_argument('-adv', '--adv_loss_wt', type=float, default=10.0, help='Adversarial loss weight.')
	parser.add_argument('-div', '--div_loss_wt', type=float, default=1.0, help='Diversity loss weight.')
	parser.add_argument('-rec', '--rec_loss_wt', type=float, default=1.0, help='Reconstruction loss weight.')
	parser.add_argument('--size', type=int, default=256, help='Image size. [256]')
	parser.add_argument('--ckpt', type=str, default='', help='Path to checkpoint to load.')
	parser.add_argument('--name', type=str, default='train', help='Name of checkpoint subfolder')
	parser.add_argument('-g', '--gpu', type=str, default='-1', help='Which GPU to use for training. Defaults to -1 (all).')
	parser.add_argument('--mixed_precision', action='store_true', help='Whether to used mixed precision.')

	args = parser.parse_args()
	if args.setup:
		setup(args)
	gan_train(args)
	