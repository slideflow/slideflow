# Generates a mosaic map from exported GAN images
import os

import slideflow as sf
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from os.path import join
from slideflow.io import tfrecords
from slideflow.statistics import gen_umap, SlideMap
from slideflow.mosaic import Mosaic

out_dir = '/home/t.cri.jdolezal/stylegan2-slideflow/out2/'

# First, put all tiles into a tfrecord
# and collect associated latent space vectors
latent_vectors = {}
images = tfrecords._get_images_by_dir(out_dir)
tfrecord_dest = join(out_dir, 'generated.tfrecords')
with tf.io.TFRecordWriter(tfrecord_dest) as writer:
    for image in tqdm(images):
        seed = image[4:8]
        seed_encoded = bytes(seed, 'utf-8')
        img_string = open(join(out_dir, image), 'rb').read()
        tf_example = tfrecords.tfrecord_example(seed_encoded, img_string)
        writer.write(tf_example.SerializeToString())

        latent_file = join(out_dir, f'projected_w_{seed}.npz')
        latent_npz = np.load(latent_file)
        latent = latent_npz['w'].flatten()
        latent_vectors[seed] = latent

# Generate UMAP from latent space vectors
print('Calculating UMAP...')
seeds, vectors = zip(*latent_vectors.items())
seeds = list(seeds)
vectors = np.array(list(vectors))
umap = gen_umap(vectors)

# Create mosaic
print('Setting up mosaic...')
meta = [{'slide': 'generated', 'index': i} for i in range(len(seeds))]
tfrecord_map = SlideMap.from_precalculated(slides=['generated'],
                                              x=umap[:,0],
                                              y=umap[:,1],
                                              meta=meta)
mosaic_map = Mosaic(tfrecord_map, tfrecords=[tfrecord_dest], num_tiles_x=40)
mosaic_map.save(join(out_dir, 'mosaic', 'mosaic.png'))