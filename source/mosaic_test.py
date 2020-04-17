import slideflow as sf
import numpy as np

from slideflow.activations import ActivationsVisualizer

project_folder = '/home/shawarma/data/slideflow_projects/TCGA_HNSC_OP_299px_302um/'
filters = None
umap_category = 'type'

SFP = sf.SlideflowProject(project_folder)
dataset = sf.io.datasets.Dataset(config_file=SFP.PROJECT['dataset_config'], sources=SFP.PROJECT['datasets'])
dataset.load_annotations(SFP.PROJECT['annotations'])
tfrecords = dataset.get_filtered_tfrecords(filters)
dataset_slides = [tfrecord.split('/')[-1][:-10] for tfrecord in tfrecords]
manifest = dataset.get_manifest()

outcome_headers = ["umap-all-x", "umap-all-y"]

outcomes, _ = dataset.get_outcomes_from_annotations(outcome_headers, filters=filters, 
												 				  	 filter_blank=outcome_headers,
																	 use_float=True)
slides = [slide for slide in list(outcomes.keys()) if slide in dataset_slides]

print("Setup complete.")

umap_x = np.array([outcomes[slide]['outcome'][0] for slide in slides])
umap_y = np.array([outcomes[slide]['outcome'][1] for slide in slides])
umap_meta = [{'slide': slide, 'index': 0} for slide in slides]

umap = {
	'nodes': None,
	'slides': slides,
	'umap_x': umap_x,
	'umap_y': umap_y,
	'umap_meta': umap_meta
}

AV = ActivationsVisualizer(model=None,
						   tfrecords=tfrecords, 
						   root_dir=SFP.PROJECT['root'],
						   image_size=SFP.PROJECT['tile_px'],
						   annotations=SFP.PROJECT['annotations'],
						   outcome_header=umap_category,
						   focus_nodes=None,
						   use_fp16=SFP.PROJECT['use_fp16'],
						   batch_size=32,
						   export_csv=False)

AV.generate_mosaic(umap=umap)
AV.plot_2D_umap(umap=umap)
