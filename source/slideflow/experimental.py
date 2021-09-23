import os
from io import BytesIO
from PIL import Image
from os.path import exists, join
from random import shuffle
from functools import partial
from multiprocessing.dummy import Pool as DPool
import slideflow as sf
import slideflow.io
from slideflow.util import log, ProgressBar
from tqdm import tqdm

'''# ===== RNA-SEQ ===================================
# This goes in the trainer() function of sf.Project
patient_to_slide = {}
for s in slide_labels_dict:
    slide_labels_dict[s]['outcome_label'] = []
    patient = sf.util._shortname(s)
    if patient not in patient_to_slide:
        patient_to_slide[patient] = [s]
    else:
        patient_to_slide[patient] += [s]

rna_seq_csv = '/mnt/data/TCGA_HNSC/hnsc_tcga_pan_can_atlas_2018/data_RNA_Seq_v2_mRNA_median_all_sample_Zscores.txt'
print ('Importing csv data...')
num_genes = 0
with open(rna_seq_csv, 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    header = next(reader)
    pt_with_rna_seq = [h[:12] for h in header[2:]]
    slide_labels_dict = {s:v for s,v in slide_labels_dict.items() if sf.util._shortname(s) in pt_with_rna_seq}
    for row in reader:
        exp_data = row[2:]
        if 'NA' in exp_data:
            continue
        num_genes += 1
        for p, exp in enumerate(exp_data):
            if pt_with_rna_seq[p] in patient_to_slide:
                for s in patient_to_slide[pt_with_rna_seq[p]]:
                    slide_labels_dict[s]['outcome_label'] += [float(exp)]
print(f'Loaded {num_genes} genes for {len(slide_labels_dict)} patients.')
outcome_label_headers = None

if True:
    outcome_labels=None

# ========================================='''

def extract_dual_tiles(project,
                       tile_um,
                       tile_px,
                       stride_div=1,
                       filters=None,
                       buffer=True,
                       normalizer=None,
                       normalizer_source=None):

    '''Experimental function to extract dual tiles at two different px/um sizes, saving both in the same TFRecord.'''
    from slideflow.slide import WSI
    import tensorflow as tf

    log.info('Extracting dual-image tiles...')
    extracting_dataset = project.get_dataset(filters=filters, tile_px=tile_px, tile_um=tile_um)

    def extract_tiles_from_slide(slide_path, roi_list, dataset_config, pb):
        root_path = join(dataset_config['tfrecords'], dataset_config['label'])
        if not exists(root_path):
            os.makedirs(root_path)

        whole_slide = WSI(slide_path,
                          tile_px,
                          tile_um,
                          stride_div,
                          roi_list=roi_list,
                          buffer=buffer,
                          pb_counter=pb.get_counter(),
                          counter_lock=pb.get_lock(),
                          skip_missing_roi=True)

        small_tile_generator = whole_slide.build_generator(dual_extract=True,
                                                           normalizer=normalizer,
                                                           normalizer_source=normalizer_source)

        tfrecord_name = sf.util.path_to_name(slide_path)
        tfrecord_path = join(root_path, f'{tfrecord_name}.tfrecords')
        records = []

        for image_dict in tqdm(small_tile_generator(), total=whole_slide.estimated_num_tiles):
            label = bytes(tfrecord_name, 'utf-8')
            image_string_dict = {}
            for image_label in image_dict:
                np_image = image_dict[image_label]
                image = Image.fromarray(np_image).convert('RGB')
                with BytesIO() as output:
                    image.save(output, format='JPEG')
                    image_string = output.getvalue()
                    image_string_dict.update({
                        image_label: image_string
                    })
            records += [[label, image_string_dict]]

        shuffle(records)

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for label, image_string_dict in records:
                tf_example = sf.io.tfrecords.multi_image_example(label, image_string_dict)
                writer.write(tf_example.SerializeToString())

    for dataset_name in project.datasets:
        log.info(f'Working on dataset {sf.util.bold(dataset_name)}')
        slide_list = extracting_dataset.get_slide_paths(source=dataset_name)
        roi_list = extracting_dataset.get_rois()
        dataset_config = extracting_dataset.datasets[dataset_name]
        log.info(f'Extracting tiles from {len(slide_list)} slides ({tile_um} um, {tile_px} px)')
        #TODO: ending_val needs to be calculated from total number of tiles
        pb = ProgressBar(ending_val=0, bar_length=5, counter_text='tiles')
        pb.auto_refresh()

        if project.default_threads > 1:
            pool = DPool(project.default_threads)
            pool.map(partial(extract_tiles_from_slide,
                             roi_list=roi_list,
                             dataset_config=dataset_config,
                             pb=pb),
                     slide_list)
            pool.close()
        else:
            for slide_path in slide_list:
                extract_tiles_from_slide(slide_path, roi_list, dataset_config, pb)

    extracting_dataset.update_manifest()

def visualize_tiles(model, node, tfrecord_dict=None, directory=None, mask_width=None,
                    normalizer=None, normalizer_source=None):
        '''Visualizes node activations across a set of image tiles through progressive convolutional masking.

        Args:
            model:              Path to Tensorflow model
            node:               Int, node to analyze
            tfrecord_dict:      Dictionary mapping tfrecord paths to tile indices.
                                    Visualization will be performed on these tiles.
            directory:          Directory in which to save images.
            mask_width:         Width of mask to convolutionally apply. Defaults to 1/6 of tile_px
            normalizer:         Normalization strategy to use on image tiles.
            normalizer_source:  Path to normalizer source image.
        '''
        from slideflow.activations import TileVisualizer

        hp_data = sf.util.get_model_hyperparameters(model)
        tile_px = hp_data['hp']['tile_px']
        TV = TileVisualizer(model=model,
                            node=node,
                            tile_px=tile_px,
                            mask_width=mask_width,
                            normalizer=normalizer,
                            normalizer_source=normalizer_source)

        if tfrecord_dict:
            for tfrecord in tfrecord_dict:
                for tile_index in tfrecord_dict[tfrecord]:
                    TV.visualize_tile(tfrecord=tfrecord, index=tile_index, export_folder=directory)

        else:
            tiles = [o for o in os.listdir(directory) if not os.path.isdir(join(directory, o))]
            tiles.sort(key=lambda x: int(x.split('-')[0]))
            tiles.reverse()
            for tile in tiles[:20]:
                tile_loc = join(directory, tile)
                TV.visualize_tile(image_jpg=tile_loc, export_folder=directory)