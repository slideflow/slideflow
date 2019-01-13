# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017, Updated 12/16/18
# ==========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import warnings

from datetime import datetime
from PIL import Image

import numpy as np
import tensorflow as tf

import histcon
import inception
import conv_display
import progress_bar

# TODO: fix compatibility with float16 and 32

parser = argparse.ArgumentParser()

parser.add_argument('--whole_image', type=str, default="half_images/234807-1.jpg",
    help='Filename of whole image (JPG) to evaluate with saved model.')

parser.add_argument('--data_dir', type=str, default='/home/falafel/histcon',
    help='Path to the HISTCON data directory.')

parser.add_argument('--batch_size', type=int, default=32,
    help='Number of images to process in a batch.')

parser.add_argument('--conv_dir', type=str, default='/home/falafel/histcon/conv',
    help='Directory where to write logs and summaries for the convoluter.')

parser.add_argument('--model_dir', type=str, default='/home/falafel/histcon/model',
    help='Directory where to write event logs and checkpoints.')

SIZE = histcon.IMAGE_SIZE
THREADS = 2
stride_divisor = 4
STRIDES = [1, int(SIZE/stride_divisor), int(SIZE/stride_divisor), 1]
WINDOW_SIZE = 2000
VERBOSE = True
DTYPE = tf.float16 if histcon.FLAGS.use_fp16 else tf.float32
DTYPE_INT = tf.int16 if histcon.FLAGS.use_fp16 else tf.int32

def batch(iterable, n=1):
    l =len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx+n, l)]

def concat_output(arr):
    new_output = []
    for row in arr:
        row_height = len(row[0])
        for y in range(row_height):
            new_x = []
            for window in row:
                new_x += window[y].tolist()
            new_output.append(new_x)
    return np.array(new_output)

def scan_image():
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)
    with tf.Graph().as_default() as g:
        filename = os.path.join(FLAGS.data_dir, FLAGS.whole_image)

        ri = tf.placeholder(DTYPE_INT, shape=[None, None, 3])

        unshaped_patches = tf.extract_image_patches(images=[ri], ksizes=[1, SIZE, SIZE, 1],
                                                    strides = STRIDES, rates = [1, 1, 1, 1],
                                                    padding = "VALID")

        patches = tf.cast(tf.reshape(unshaped_patches, [-1, SIZE, SIZE, 3]), DTYPE)

        batch_pl = tf.placeholder(DTYPE, shape=[FLAGS.batch_size, SIZE, SIZE, 3])
        standardized_batch = tf.map_fn(lambda patch: tf.cast(tf.image.per_image_standardization(patch), DTYPE), batch_pl, dtype=DTYPE)
        _, end_points = inception.inception_v4(standardized_batch, num_classes=2)
        slogits = end_points['Predictions']
        saver = tf.train.Saver()

        #logits = histcon.inference(standardized_batch)
        #slogits = tf.nn.softmax(logits)

        # Restore model
        #variable_averages = tf.train.ExponentialMovingAverage(histcon.MOVING_AVERAGE_DECAY)
        #variables_to_restore = variable_averages.variables_to_restore()
        #saver = tf.train.Saver(variables_to_restore)
        

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.conv_dir, g)

        with tf.Session() as sess:
            print("\n" + "="*20 + "\n")
            # Load image through PIL
            im = Image.open(filename)
            imported_image = np.array(im, dtype=np.int32)
            print("Image size: ", imported_image.shape)

            init = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            # Restore checkpoint
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
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
                margin = int(SIZE/2)

                # First, determine how to divide x and y
                num_full_windows_y = int((len_y - margin*2) / (WINDOW_SIZE - margin*2))
                leftover_y = (len_y - margin*2) % (WINDOW_SIZE - margin*2)

                num_full_windows_x = int((len_x - margin*2) / (WINDOW_SIZE - margin*2))
                leftover_x = (len_x - margin*2) % (WINDOW_SIZE - margin*2)

                if VERBOSE: print("Leftovers:       x: %s   y: %s" % (leftover_x, leftover_y))
                if VERBOSE: print("Whole patches:   x: %s   y: %s" % (num_full_windows_x, num_full_windows_y))

                total_patches = (num_full_windows_x + (1 if leftover_x else 0)) * (num_full_windows_y + (1 if leftover_y else 0))
                patch_index = 0

                # {start: [x, y], end: [x,y]}
                window_list = []

                # Create list of coordinates for full y-length windows
                for j in range(num_full_windows_y):
                    y_start = j * (WINDOW_SIZE - margin*2)
                    y_end = y_start + WINDOW_SIZE
                    window_x_list = []

                    for i in range(num_full_windows_x):
                        x_start = i * (WINDOW_SIZE - margin*2)
                        x_end = x_start + WINDOW_SIZE
                        window_x_list.append({  'start' : [x_start, y_start],
                                                'end'   : [x_end, y_end]      })
                    if leftover_x:
                        x_start = num_full_windows_x * (WINDOW_SIZE - margin*2)
                        x_end = x_start + leftover_x + margin*2
                        window_x_list.append({  'start' : [x_start, y_start],
                                                'end'   : [x_end, y_end]      })
                    window_list.append(window_x_list)

                if leftover_y:
                    y_start = num_full_windows_y * (WINDOW_SIZE - margin*2)
                    y_end = y_start + leftover_y + margin*2
                    window_x_list = []

                    for i in range(num_full_windows_x):
                        x_start = i * (WINDOW_SIZE - margin*2)
                        x_end = x_start + WINDOW_SIZE
                        window_x_list.append({  'start' : [x_start, y_start],
                                                'end'   : [x_end, y_end]      })
                    if leftover_x:
                        x_start = num_full_windows_x * (WINDOW_SIZE - margin*2)
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

                        if VERBOSE:
                            print("\nPatch array %s (size: %s)" % (index, patch_array.shape[0]))
                            print("X: %s - %s,  Y: %s - %s" %(x_start, x_end, y_start, y_end))
                        else:
                            progress_bar.bar(patch_index, total_patches)
                            patch_index += 1

                        num_batches = int(patch_array.shape[0]/FLAGS.batch_size)

                        for i, x in enumerate(batch(patch_array, FLAGS.batch_size)):
                            if x.shape[0] == FLAGS.batch_size:
                                # Full batch
                                if VERBOSE: progress_bar.bar(i, num_batches)
                                sl = sess.run(slogits, feed_dict = {batch_pl: x})

                                if not all_slogs.any(): 
                                    all_slogs = sl
                                else: 
                                    all_slogs = np.concatenate((all_slogs, sl), axis=0)
                            else:
                                if VERBOSE: progress_bar.bar(1, 1)
                                num_pad = FLAGS.batch_size - x.shape[0]
                                z = np.zeros([num_pad, SIZE, SIZE, 3])
                                padded_x = np.concatenate((x, z), axis=0)
                                padded_sl = sess.run(slogits, feed_dict = {batch_pl: padded_x})
                                trimmed_sl = padded_sl[:x.shape[0]]

                                if not all_slogs.any():
                                    all_slogs = trimmed_sl
                                else:
                                    all_slogs = np.concatenate((all_slogs, trimmed_sl), axis=0)

                        patches_height = 1 + int((int(y_end - y_start) - int(SIZE)) / STRIDES[1])
                        patches_width  = 1 + int((int(x_end - x_start) - int(SIZE)) / STRIDES[2])

                        reshaped_slogs = np.reshape(all_slogs, [patches_height, patches_width, 2])

                        output[y_index][x_index] = reshaped_slogs

                        if VERBOSE:
                            progress_bar.end()
                            print(" ", reshaped_slogs.shape)

                output = concat_output(output)
                if not VERBOSE:
                    progress_bar.end()
                print("\nFormatted output shape:", output.shape)

            except Exception as e:
                    coord.request_stop(e)
                    print("\n")

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            print("\nFinished.")
            conv_display.display(filename, output, SIZE, STRIDES)

def main(argv=None):
    scan_image()

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()