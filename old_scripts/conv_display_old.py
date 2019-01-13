# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

''' Displays results of Tensorflow-based convoluter on a designated image.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display(image_file, logits, size, stride):
	im = mpimg.imread(image_file)
	plt.axes()
	gca = plt.gca()
	#gca.invert_yaxis()
	gca.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

	plt.imshow(im)

	for row in range(len(logits)):
		for col in range(len(logits[0])):
			area = [[col*stride[1], row*stride[2]], [col*stride[1]+size, row*stride[2]], [col*stride[1]+size, row*stride[2]+size], [col*stride[1], row*stride[2]+size]]
			zero = logits[row][col][0]
			one = logits[row][col][1]
			logit_poly = plt.Polygon(area, facecolor=(one, zero, 0), alpha=0.02)
			gca.add_patch(logit_poly)

	plt.axis('scaled')
	plt.show()

