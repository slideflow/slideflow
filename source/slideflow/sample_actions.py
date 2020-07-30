def main(SFP):
	# First run only: automatically associate slide names in the annotations file
	# ---------------------------------------------------------------------------
	#SFP.associate_slide_names()
	
	# Perform tile extraction
	# -----------------------
	#SFP.extract_tiles()
		   
	# Train with a hyperparameter sweep
	# ---------------------------------
	#SFP.create_hyperparameter_sweep(tile_px=[299, 331], tile_um=[302], finetune_epochs=[5], toplayer_epochs=0, model=['Xception'], pooling=['avg'], loss='sparse_categorical_crossentropy', 
	#								learning_rate=[0.00001, 0.001], batch_size=64, hidden_layers=[1], optimizer='Adam', early_stop=True, early_stop_patience=15, balanced_training=['BALANCE_BY_CATEGORY'],
	#								balanced_validation='NO_BALANCE', hidden_layer_width=500, trainable_layers=0, L2_weight=0, early_stop_method='loss', augment=True, filename=None)
	#SFP.train(
	#	  outcome_header="category",
	#	  filters = {
	#		  'dataset': 'train',
	#		  'category': ['negative', 'positive']
	#	  },
	#	  batch_file='batch_train.tsv')

	# Evaluate model performance with separate data
	# ---------------------------------------------
	#SFP.evaluate(model='/path/to/trained_model.h5', outcome_header="category", filters = {'dataset': ['eval']})

	# Create heatmaps of predictions with a certain model
	# ---------------------------------------------------
	#SFP.generate_heatmaps(model='/path/to/trained_model.h5',
	# 					   filters = {'dataset': ['eval']})

	# Generate a mosaic map of tiles using a certain model
	# ----------------------------------------------------
	#SFP.generate_mosaic(model='/path/to/trained_model.h5', 
	# 					 filters = {'dataset': ['eval']}, 
	# 					 resolution='high')

	# Visualize and analyze penultimate layer activations
	# ---------------------------------------------------
	#from slideflow.statistics import TFRecordMap
	#from os.path import join
	#AV = SFP.generate_activations_analytics(model='/path/to/trained_model.h5', outcome_header="HPV", filters={"HPV": ["HPV+", "HPV-"]})
	#AV.generate_box_plots()
	#umap = TFRecordMap.from_activations(AV)
	#umap.save_2d_plot(join(SFP.PROJECT['root'], 'stats', '2d_umap.png'))
	#top_nodes = AV.get_top_nodes_by_slide()
	#for node in top_nodes[:10]:
	#	umap.save_3d_node_plot(node, join(SFP.PROJECT['root'], 'stats', f'3d_node{node}.png'))
	pass