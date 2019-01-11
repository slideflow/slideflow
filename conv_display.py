# Copyright (C) James Dolezal - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by James Dolezal <jamesmdolezal@gmail.com>, October 2017
# ==========================================================================

''' Displays results of Tensorflow-based convoluter on a designated image.'''

# Not currently working; overlay is offset
# Also not efficient; an alternative solution using multiprocessing has been half implemented here

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mcol
import numpy as np
import progress_bar as pb
import sys

from multiprocessing import Pool
from matplotlib.widgets import Slider

def display(image_file, logits, size, stride):
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

	sl = logits[:, :, 1]
	
	# Consider alternate interpolations: none, bicubic, quadric, lanczos
	heatmap = plt.imshow(logits[:, :, 1], extent=extent, cmap=newMap, alpha = 0.3, interpolation='bicubic', zorder=10)

	def update_opacity(val):
		heatmap.set_alpha(val)

	# Show sliders to adjust heatmap overlay
	ax_opac = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='lightgoldenrodyellow')
	opac = Slider(ax_opac, 'Opacity', 0, 1, valinit = 1)
	opac.on_changed(update_opacity)

	plt.axis('scaled')
	plt.show()