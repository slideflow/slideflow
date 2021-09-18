import os
from io import BytesIO
from PIL import Image
from os.path import exists, join
from random import shuffle
from functools import partial
from multiprocessing.dummy import Pool as DPool
from slideflow.util import log, ProgressBar
from tqdm import tqdm
import slideflow.io as sfio
import slideflow.util as sfutil

def extract_dual_tiles(project,
                       tile_um,
                       tile_px,
                       stride_div=1,
                       filters=None,
                       buffer=True,
                       normalizer=None,
                       normalizer_source=None):

    '''Experimental function to extract dual tiles at two different px/um sizes, saving both in the same TFRecord.'''
    import slideflow.slide as sfslide
    import tensorflow as tf

    log.info('Extracting dual-image tiles...')
    extracting_dataset = project.get_dataset(filters=filters, tile_px=tile_px, tile_um=tile_um)

    def extract_tiles_from_slide(slide_path, roi_list, dataset_config, pb):
        root_path = join(dataset_config['tfrecords'], dataset_config['label'])
        if not exists(root_path):
            os.makedirs(root_path)

        whole_slide = sfslide.SlideReader(slide_path,
                                          tile_px,
                                          tile_um,
                                          stride_div,
                                          roi_list=roi_list,
                                          buffer=buffer,
                                          pb_counter=pb.get_counter(),
                                          counter_lock=pb.get_lock(),
                                          skip_missing_roi=True,
                                          print_fn=pb.print)

        small_tile_generator = whole_slide.build_generator(dual_extract=True,
                                                           normalizer=normalizer,
                                                           normalizer_source=normalizer_source)

        tfrecord_name = sfutil.path_to_name(slide_path)
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
                tf_example = sfio.tfrecords.multi_image_example(label, image_string_dict)
                writer.write(tf_example.SerializeToString())

    for dataset_name in project.datasets:
        log.info(f'Working on dataset {sfutil.bold(dataset_name)}')
        slide_list = extracting_dataset.get_slide_paths(dataset=dataset_name)
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
