def main(SFP):
    # First run only: automatically associate slide names in the annotations file
    # ---------------------------------------------------------------------------
    #SFP.associate_slide_names()

    # Perform tile extraction
    # -----------------------
    #SFP.extract_tiles(tile_px=299, tile_um=302)

    # Train with a hyperparameter sweep
    # ---------------------------------
    #SFP.create_hyperparameter_sweep(tile_px=[299, 331],
    #                                tile_um=[302],
    #                                epochs=[5],
    #                                toplayer_epochs=0,
    #                                model=['Xception'],
    #                                pooling=['avg'],
    #                                loss='sparse_categorical_crossentropy',
    #                                learning_rate=[0.00001, 0.001],
    #                                batch_size=64,
    #                                hidden_layers=[1],
    #                                optimizer='Adam',
    #                                early_stop=True,
    #                                early_stop_patience=15,
    #                                balanced_training=['category'],
    #                                balanced_validation='none',
    #                                hidden_layer_width=500,
    #                                trainable_layers=0,
    #                                L2_weight=0,
    #                                early_stop_method='loss',
    #                                augment=True,
    #                                filename=None)
    #SFP.train(
    #      outcome_label_headers='category',
    #      hyperparameters='sweep',
    #      filters = {
    #          'dataset': 'train',
    #          'category': ['negative', 'positive']
    #      },)

    # Evaluate model performance with separate data
    # ---------------------------------------------
    #SFP.evaluate(model='/path/to/trained_model',
    #             outcome_label_headers='category',
    #             filters = {'dataset': ['eval']})

    # Create heatmaps of predictions with a certain model
    # ---------------------------------------------------
    #SFP.generate_heatmaps(model='/path/to/trained_model',
    #                      filters = {'dataset': ['eval']})

    # Visualize and analyze layer activations
    # ---------------------------------------------------
    #AV = SFP.generate_activations(model='/path/to/trained_model',
    #                              outcome_label_header="HPV",
    #                              filters={"HPV": ["HPV+", "HPV-"]})

    # Generate a mosaic map of tiles using a certain model
    # ----------------------------------------------------
    #mosaic = SFP.generate_mosaic(AV, resolution='high')
    #mosaic.save('/path.png')
    #mosaic.slide_map.save('/path.png')
    pass