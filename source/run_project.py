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
	#SFP.generate_manifest()
	#SFP.batch_train(resume_training="/home/shawarma/data/slideflow_projects/thyroid_5_cat/models/retrain_balanced2/trained_model.h5")
	#SFP.evaluate(model="/home/shawarma/data/slideflow_projects/thyroid_5_cat/models/retrain_balanced3/trained_model.h5")
	#SFP.train_model('model2')
	SFP.generate_heatmaps('retrain_balanced3', slide_filters={'dataset': ['eval']})
	