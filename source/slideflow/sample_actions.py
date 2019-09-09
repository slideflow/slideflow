def main(SFP):
	#SFP.create_blank_annotations_file()
	#SFP.associate_slide_names()
	
	#SFP.extract_tiles(filters = {'to_extract': 'yes'}, subfolder='all')
		   
	#SFP.create_hyperparameter_sweep(finetune_epochs=[5], toplayer_epochs=0, model=['Xception'], pooling=['avg'], loss='sparse_categorical_crossentropy', 
	#								learning_rate=[0.00001, 0.001], batch_size=64, hidden_layers=[1], optimizer='Adam', early_stop=True, early_stop_patience=15, balanced_training=['BALANCE_BY_CATEGORY'],
	#								balanced_validation='NO_BALANCE', augment=True, filename=None)
	#SFP.train(subfolder='all',
	#	  category_header="category",
	#	  filters = {
	#		  'dataset': 'train',
	#		  'category': ['negative', 'positive']
	#	  },
	#	  batch_file='batch_train.tsv')

	#SFP.evaluate(model='HPSweep0-kfold3', subfolder='all', category_header="category", filters = {'dataset': 'eval'})
	#SFP.generate_heatmaps('HPSweep0')
	#SFP.generate_mosaic('HPSweep0')
	pass