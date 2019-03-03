# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017, Updated 3/2/19
# ==========================================================================

'''Convolutionally applies a saved Tensorflow model to a larger image, displaying
the result as a heatmap overlay.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import warnings

from datetime import datetime
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

import inception_v4
from inception_utils import inception_arg_scope
import progress_bar

from multiprocessing import Pool
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mcol

# NOTE!! THIS MODULE IS CURRENTLY NOT WORKING, THERE IS A BUG WITH WINDOWING
# Not currently working; overlay is offset
# Also not efficient; an alternative solution using multiprocessing has been half implemented here

class Convoluter:
    # Model variables
    SIZE = 64
    NUM_CLASSES = 1
    BATCH_SIZE = 32

    # Display variables
    stride_divisor = 4
    STRIDES = [1, int(SIZE/stride_divisor), int(SIZE/stride_divisor), 1]
    WINDOW_SIZE = 6000
    VERBOSE = True

    WHOLE_IMAGE = "images/234781-2.jpg" # Filename of whole image (JPG) to evaluate with saved model
    DATA_DIR = '/Users/james/thyroid' # Path to histcon data directory
    CONV_DIR = '/Users/james/thyroid/conv' # Directory where to write logs and summaries for the convoluter
    MODEL_DIR = '/Users/james/thyroid/models/active' # Directory where to write event logs and checkpoints.
    USE_FP16 = True

    def __init__(self):
        self.DTYPE = tf.float16 if self.USE_FP16 else tf.float32
        self.DTYPE_INT = tf.int16 if self.USE_FP16 else tf.int32

    def batch(self, iterable, n=1):
        '''Organizes image tiles into workable batches for transfer to the GPU.'''
        l =len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx+n, l)]

    def concat_output(self, arr):
        '''Converts a 2D collection of 2D patches into a single large 2D array'''
        new_output = []
        for row in arr:
            row_height = len(row[0])
            for y in range(row_height):
                new_x = []
                for window in row:
                    new_x += window[y].tolist()
                new_output.append(new_x)
        return np.array(new_output)

    def scan_image(self):
        '''Opens a whole-slide image, sections it into tiles, loads a saved Tensorflow model,
        and applies the model to each of the image tiles. The output - a 3D array of logits
        that correspond to model predictions at each location on the whole-slide iamge, is then
        restructured into an array that can be displayed as a heatmap overlay.'''

        warnings.simplefilter('ignore', Image.DecompressionBombWarning)
        with tf.Graph().as_default() as g:
            filename = os.path.join(self.DATA_DIR, self.WHOLE_IMAGE)

            ri = tf.placeholder(self.DTYPE_INT, shape=[None, None, 3])

            unshaped_patches = tf.extract_image_patches(images=[ri], ksizes=[1, self.SIZE, self.SIZE, 1],
                                                        strides = self.STRIDES, rates = [1, 1, 1, 1],
                                                        padding = "VALID")

            patches = tf.cast(tf.reshape(unshaped_patches, [-1, self.SIZE, self.SIZE, 3]), self.DTYPE)

            batch_pl = tf.placeholder(self.DTYPE, shape=[self.BATCH_SIZE, self.SIZE, self.SIZE, 3])
            standardized_batch = tf.map_fn(lambda patch: tf.cast(tf.image.per_image_standardization(patch), self.DTYPE), batch_pl, dtype=self.DTYPE)

            with arg_scope(inception_arg_scope()):
                _, end_points = inception_v4.inception_v4(standardized_batch, num_classes=self.NUM_CLASSES)
            slogits = end_points['Predictions']
            saver = tf.train.Saver()

            #logits = histcon.inference(standardized_batch)
            #slogits = tf.nn.softmax(logits)

            # Restore model
            #variable_averages = tf.train.ExponentialMovingAverage(histcon.MOVING_AVERAGE_DECAY)
            #variables_to_restore = variable_averages.variables_to_restore()
            #saver = tf.train.Saver(variables_to_restore)
            
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.CONV_DIR, g)

            with tf.Session() as sess:
                print("\n" + "="*20 + "\n")
                # Load image through PIL
                im = Image.open(filename)
                imported_image = np.array(im, dtype=self.DTYPE)
                print("Image size: ", imported_image.shape)

                init = (tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init)

                # Restore checkpoint
                ckpt = tf.train.get_checkpoint_state(self.MODEL_DIR)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found.')
                    return

                coord = tf.train.Coordinator()
                try:
                    threads = []
                    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                    len_y = imported_image.shape[0]
                    len_x = imported_image.shape[1]
                    margin = int(self.SIZE/2)

                    # First, determine how to divide x and y
                    num_full_windows_y = int((len_y - margin*2) / (self.WINDOW_SIZE - margin*2))
                    leftover_y = (len_y - margin*2) % (self.WINDOW_SIZE - margin*2)

                    num_full_windows_x = int((len_x - margin*2) / (self.WINDOW_SIZE - margin*2))
                    leftover_x = (len_x - margin*2) % (self.WINDOW_SIZE - margin*2)

                    if self.VERBOSE: print("Leftovers:       x: %s   y: %s" % (leftover_x, leftover_y))
                    if self.VERBOSE: print("Whole patches:   x: %s   y: %s" % (num_full_windows_x, num_full_windows_y))

                    total_patches = (num_full_windows_x + (1 if leftover_x else 0)) * (num_full_windows_y + (1 if leftover_y else 0))
                    patch_index = 0

                    # {start: [x, y], end: [x,y]}
                    window_list = []

                    # Create list of coordinates for full y-length windows
                    for j in range(num_full_windows_y):
                        y_start = j * (self.WINDOW_SIZE - margin*2)
                        y_end = y_start + self.WINDOW_SIZE
                        window_x_list = []

                        for i in range(num_full_windows_x):
                            x_start = i * (self.WINDOW_SIZE - margin*2)
                            x_end = x_start + self.WINDOW_SIZE
                            window_x_list.append({  'start' : [x_start, y_start],
                                                    'end'   : [x_end, y_end]      })
                        if leftover_x:
                            x_start = num_full_windows_x * (self.WINDOW_SIZE - margin*2)
                            x_end = x_start + leftover_x + margin*2
                            window_x_list.append({  'start' : [x_start, y_start],
                                                    'end'   : [x_end, y_end]      })
                        window_list.append(window_x_list)

                    if leftover_y:
                        y_start = num_full_windows_y * (self.WINDOW_SIZE - margin*2)
                        y_end = y_start + leftover_y + margin*2
                        window_x_list = []

                        for i in range(num_full_windows_x):
                            x_start = i * (self.WINDOW_SIZE - margin*2)
                            x_end = x_start + self.WINDOW_SIZE
                            window_x_list.append({  'start' : [x_start, y_start],
                                                    'end'   : [x_end, y_end]      })
                        if leftover_x:
                            x_start = num_full_windows_x * (self.WINDOW_SIZE - margin*2)
                            x_end = x_start + leftover_x + margin*2
                            window_x_list.append({  'start' : [x_start, y_start],
                                                    'end'   : [x_end, y_end]      })
                        window_list.append(window_x_list)

                    output = np.zeros([len(window_list), len(window_list[0])]).tolist()

                    for y_index, window_x_list in enumerate(window_list):
                        for x_index, window in enumerate(window_x_list):
                            x_start = window['start'][0]
                            x_end = window['end'][0]
                            y_start = window['start'][1]
                            y_end = window['end'][1]
                            index = (x_index, y_index)

                            image_window = np.array([x[x_start:x_end] for x in imported_image[y_start:y_end]])

                            all_slogs = np.array([])
                            patch_array = np.array(sess.run(patches, feed_dict = {ri:image_window}))

                            if self.VERBOSE:
                                print("\nPatch array %s (size: %s)" % (index, patch_array.shape[0]))
                                print("X: %s - %s,  Y: %s - %s" %(x_start, x_end, y_start, y_end))
                            else:
                                progress_bar.bar(patch_index, total_patches)
                                patch_index += 1

                            num_batches = int(patch_array.shape[0]/self.BATCH_SIZE)

                            for i, x in enumerate(self.batch(patch_array, self.BATCH_SIZE)):
                                if x.shape[0] == self.BATCH_SIZE:
                                    # Full batch
                                    if self.VERBOSE: progress_bar.bar(i, num_batches)
                                    sl = sess.run(slogits, feed_dict = {batch_pl: x})

                                    if not all_slogs.any(): 
                                        all_slogs = sl
                                    else: 
                                        all_slogs = np.concatenate((all_slogs, sl), axis=0)
                                else:
                                    if self.VERBOSE: progress_bar.bar(1, 1)
                                    num_pad = self.BATCH_SIZE - x.shape[0]
                                    z = np.zeros([num_pad, self.SIZE, self.SIZE, 3])
                                    padded_x = np.concatenate((x, z), axis=0)
                                    padded_sl = sess.run(slogits, feed_dict = {batch_pl: padded_x})
                                    trimmed_sl = padded_sl[:x.shape[0]]

                                    if not all_slogs.any():
                                        all_slogs = trimmed_sl
                                    else:
                                        all_slogs = np.concatenate((all_slogs, trimmed_sl), axis=0)

                            patches_height = 1 + int((int(y_end - y_start) - int(self.SIZE)) / self.STRIDES[1])
                            patches_width  = 1 + int((int(x_end - x_start) - int(self.SIZE)) / self.STRIDES[2])

                            reshaped_slogs = np.reshape(all_slogs, [patches_height, patches_width, self.NUM_CLASSES])

                            output[y_index][x_index] = reshaped_slogs

                            if self.VERBOSE:
                                progress_bar.end()
                                print(" ", reshaped_slogs.shape)

                    output = self.concat_output(output)
                    if not self.VERBOSE:
                        progress_bar.end()
                    print("\nFormatted output shape:", output.shape)

                except Exception as e:
                        coord.request_stop(e)
                        print("\n")

                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)
                print("\nFinished.")
                self.display(filename, output, self.SIZE, self.STRIDES)

    def display(self, image_file, logits, size, stride):
        '''Displays logits calculated using scan_image as a heatmap overlay.'''
        print("Received logits, size=%s, stride=%sx%s (%s x %s)" % (size, stride[1], stride[2], len(logits), len(logits[0])))
        print("Calculating overlay matrix...")

        im = plt.imread(image_file)
        implot = plt.imshow(im,zorder=0)
        gca = plt.gca()
        gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

        im_extent = implot.get_extent()
        extent = [im_extent[0] + size/2, im_extent[1] - size/2, im_extent[2] - size/2, im_extent[3] + size/2]

        # Define color map
        jetMap = np.linspace(0.45, 0.95, 255)
        cmMap = cm.nipy_spectral(jetMap)
        newMap = mcol.ListedColormap(cmMap)

        sl = logits[:, :, 0]
        
        # Consider alternate interpolations: none, bicubic, quadric, lanczos
        heatmap = plt.imshow(sl, extent=extent, cmap=newMap, alpha = 0.3, interpolation='bicubic', zorder=10)

        def update_opacity(val):
            heatmap.set_alpha(val)

        # Show sliders to adjust heatmap overlay
        ax_opac = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgoldenrodyellow')
        opac = Slider(ax_opac, 'Opacity', 0, 1, valinit = 1)
        opac.on_changed(update_opacity)

        plt.axis('scaled')
        plt.show()

def main(argv=None):
    c = Convoluter()
    c.scan_image()

if __name__ == '__main__':
    tf.app.run()