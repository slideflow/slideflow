import slideflow as sf
import numpy as np

from slideflow.mosaic import Mosaic
from slideflow.statistics import TFRecordUMAP
from slideflow.io.datasets import Dataset
from os.path import join

project_folder = '/home/shawarma/data/slideflow_projects/TCGA_HNSC_OP_299px_302um/'
filters = None
umap_category = 'type'
outcome_headers = ["umap-all-x", "umap-all-y"]

SFP = sf.SlideflowProject(project_folder)
dataset = Dataset(config_file=SFP.PROJECT['dataset_config'], sources=SFP.PROJECT['datasets'])
dataset.load_annotations(SFP.PROJECT['annotations'])
dataset.apply_filters(filters=filters, filter_blank=outcome_headers)
tfrecords = dataset.get_tfrecords()
slides = dataset.get_slides()
outcomes, _ = dataset.get_outcomes_from_annotations(outcome_headers, use_float=True)

umap_x = np.array([outcomes[slide]['outcome'][0] for slide in slides])
umap_y = np.array([outcomes[slide]['outcome'][1] for slide in slides])
umap_meta = [{'slide': slide, 'index': 0} for slide in slides]

umap = TFRecordUMAP(tfrecords=dataset.get_tfrecords(), slides=dataset.get_slides())
umap.load_precalculated(umap_x, umap_y, umap_meta)

mosaic_map = Mosaic(umap, leniency=1.5,
						  expanded=False,
						  tile_zoom=15,
						  num_tiles_x=100,
						  resolution='high')

mosaic_map.save(join(SFP.PROJECT['root'], 'stats'))