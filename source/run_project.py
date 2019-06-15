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
	#SFP.batch_train()
	#SFP.evaluate(model="/home/shawarma/data/slideflow_projects/thyroid_5_cat/models/NFCOMP_c/trained_model.h5", dataset='test')
	#SFP.train_model('model2')
	SFP.generate_heatmaps('PT_ModelB', slide_filters={'dataset': ['eval']})
