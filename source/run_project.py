import slideflow as sf
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Helper to guide through the SlideFlow pipeline")
	parser.add_argument('-p', '--project', help='Path to project directory.')
	args = parser.parse_args()

	# here's a comment

	SFP = sf.SlideFlowProject(args.project)
	#SFP.prepare_tiles()
	#SFP.extract_tiles(slide_filters = {'dataset': ['train']})
	#SFP.separate_training_and_eval()
	#SFP.generate_tfrecord()
	#SFP.create_blank_batch_config()
	SFP.batch_train()
	#SFP.train_model('model2')
