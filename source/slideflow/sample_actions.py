def main(SFP):
	#SFP.create_blank_annotations_file()
	#SFP.associate_slide_names()
	
	#SFP.extract_tiles()
		   
	#SFP.create_hyperparameter_sweep(finetune_epochs=[5], toplayer_epochs=0, model=['Xception'], pooling=['avg'], loss='sparse_categorical_crossentropy', 
	#								learning_rate=[0.00001, 0.001], batch_size=64, hidden_layers=[1], optimizer='Adam', early_stop=True, early_stop_patience=15, balanced_training=['BALANCE_BY_CATEGORY'],
	#								balanced_validation='NO_BALANCE', augment=True, filename=None)
	#SFP.train(
	#	  outcome_header="category",
	#	  filters = {
	#		  'dataset': 'train',
	#		  'category': ['negative', 'positive']
	#	  },
	#	  batch_file='batch_train.tsv')

	#SFP.evaluate(model='HPSweep0-kfold3', outcome_header="category", filters = {'dataset': 'eval'})
	#SFP.generate_heatmaps('HPSweep0')
	#SFP.resize_tfrecords(598)
	#SFP.generate_mosaic('/home/shawarma/data/slideflow_projects/TCGA_HNSC_HPV_598px_604um/external_models/TCGA_HNSC_HPV_598px_604um/trained_model_epoch5.h5', resolution='high')

	#AV = SFP.generate_activations_analytics(outcome_header="HPV", filters={"HPV": ["HPV+", "HPV-"]})
	#AV.generate_box_plots()
	#umap = AV.plot_2D_umap()
	#top_nodes = AV.get_top_nodes_by_slide()
	#for node in top_nodes[:10]:
	#	AV.plot_3D_umap(node)
	#filtered_tiles = AV.filter_tiles_by_umap(umap, x_lower = 0.4,
	#											   x_upper = 0.45,
	#											   y_lower = 0.2,
	#											   y_upper = 0.25)
	#AV.save_example_tiles_gradient(tile_filter=filtered_tiles)
	pass