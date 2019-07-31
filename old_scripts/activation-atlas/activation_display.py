import math
import tensorflow as tf
import numpy as np
import pickle

import matplotlib.pyplot as plt
from IPython.display import display, HTML

import umap
from sklearn.manifold import TSNE

from lucid.misc.io import save, show, load, writing
import lucid.modelzoo.vision_models as models

import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

with open('html_dump', 'rb') as handle:
	show_canvas, canvas = pickle.load(handle)

show(canvas)