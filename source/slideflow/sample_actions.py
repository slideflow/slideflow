def main(SFP):
	# First run only: automatically associate slide names in the annotations file
	# ---------------------------------------------------------------------------
	#SFP.associate_slide_names()
	
	# Perform tile extraction
	# -----------------------
	#SFP.extract_tiles()
		   
	# Train with a hyperparameter sweep
	# ---------------------------------
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

	# Evaluate model performance with separate data
	# ---------------------------------------------
	#SFP.evaluate(model='HPSweep0-kfold3', outcome_header="category", filters = {'dataset': ['eval']})

	# Create heatmaps of predictions with a certain model
	# ---------------------------------------------------
	#SFP.generate_heatmaps('HPSweep0', filters = {'dataset': ['eval']})

	# Generate a mosaic map of tiles using a certain model
	# ----------------------------------------------------
	#SFP.generate_mosaic('/home/shawarma/data/slideflow_projects/TCGA_HNSC_HPV_598px_604um/external_models/TCGA_HNSC_HPV_598px_604um/trained_model_epoch5.h5', 
	# 						filters = {'dataset': ['eval']}, resolution='high')

	# Visualize and analyze penultimate layer activations
	# ---------------------------------------------------
	#AV = SFP.generate_activations_analytics(outcome_header="HPV", filters={"HPV": ["HPV+", "HPV-"]})
	#AV.generate_box_plots()
	#AV.plot_2D_umap()
	#top_nodes = AV.get_top_nodes_by_slide()
	#for node in top_nodes[:10]:
	#	AV.plot_3D_umap(node)
	pass