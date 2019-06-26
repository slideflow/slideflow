import slideflow as sf
import argparse

def main(SFP):
	#SFP.extract_tiles(filter_header='dataset', filter_values=['eval'])
	#SFP.separate_training_and_eval()
	#SFP.generate_tfrecord()
	#SFP.create_hyperparameter_sweep(toplayer_epochs=0, finetune_epochs=2, model=['InceptionV3', 'Xception', 'VGG16'], pooling=['avg'], loss='sparse_categorical_crossentropy', 
	#								learning_rate=[0.001, 0.01], batch_size=64, hidden_layers=[0,1], optimizer='Adam', early_stop=False, early_stop_patience=0, balanced_training='BALANCE_BY_CATEGORY',
	#								balanced_validation='NO_BALANCE', augment=True, filename=None)
	#SFP.train(category_header="description",
	#		  filter_header="dataset",
	#		  filter_values=['train'])#resume_training="/home/shawarma/data/slideflow_projects/thyroid_5_cat/models/retrain_balanced2/trained_model.h5")
	#SFP.evaluate('HPSweep1', filter_header='dataset', filter_values=['eval'])
	SFP.generate_heatmaps('HPSweep0', category_header='description', filter_header='dataset', filter_values=['eval'])

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Helper to guide through the SlideFlow pipeline")
	parser.add_argument('-p', '--project', help='Path to project directory.')
	args = parser.parse_args()
	SFP = sf.SlideFlowProject(args.project)
	main(SFP)

	
	
