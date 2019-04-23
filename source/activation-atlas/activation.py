import math
import tensorflow as tf
import numpy as np
import pickle

import matplotlib.pyplot as plt

import umap
from sklearn.manifold import TSNE

from lucid.misc.io import save, show, load
import lucid.modelzoo.vision_models as models

import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

model = models.InceptionV1()
model.load_graphdef()

# TODO:
# https://stackoverflow.com/questions/45864363/tensorflow-how-to-convert-meta-data-and-index-model-files-into-one-graph-pb

# model.layers[7] is mixed4c
layer = "mixed4c"
print(model.layers[7])
raw_activations = model.layers[7].activations
activations = raw_activations[:100000]
print(activations.shape)

def whiten(full_activations):
    correl = np.matmul(full_activations.T, full_activations) / len(full_activations)
    correl = correl.astype("float32")
    S = np.linalg.inv(correl)
    S = S.astype("float32")
    return S

S = whiten(raw_activations)

def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
    """Removes outliers and scales layout to between [0,1]."""

    # compute percentiles
    mins = np.percentile(layout, min_percentile, axis=(0))
    maxs = np.percentile(layout, max_percentile, axis=(0))

    # add margins
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)

    # `clip` broadcasts, `[None]`s added only for readability
    clipped = np.clip(layout, mins, maxs)

    # embed within [0,1] along both axes
    clipped -= clipped.min(axis=0)
    clipped /= clipped.max(axis=0)

    return clipped

layout = umap.UMAP(n_components=2, verbose=True, n_neighbors=20, min_dist=0.01, metric="cosine").fit_transform(activations)

## You can optionally use TSNE as well
# layout = TSNE(n_components=2, verbose=True, metric="cosine", learning_rate=10, perplexity=50).fit_transform(d)

layout = normalize_layout(layout)

plt.figure(figsize=(10, 10))
plt.scatter(x=layout[:,0],y=layout[:,1], s=2)
plt.show()

# FEATURE VISUALIZATION
# 
# Whitened, euclidean neuron objective
# 
@objectives.wrap_objective
def direction_neuron_S(layer_name, vec, batch=None, x=None, y=None, S=None):
    def inner(T):
        layer = T(layer_name)
        shape = tf.shape(layer)
        x_ = shape[1] // 2 if x is None else x
        y_ = shape[2] // 2 if y is None else y
        if batch is None:
            raise RuntimeError("requires batch")

        acts = layer[batch, x_, y_]
        vec_ = vec
        if S is not None: vec_ = tf.matmul(vec_[None], S)[0]
        # mag = tf.sqrt(tf.reduce_sum(acts**2))
        dot = tf.reduce_mean(acts * vec_)
        # cossim = dot/(1e-4 + mag)
        return dot
    return inner

# 
# Whitened, cosine similarity objective
# 
@objectives.wrap_objective
def direction_neuron_cossim_S(layer_name, vec, batch=None, x=None, y=None, cossim_pow=1, S=None):
    def inner(T):
        layer = T(layer_name)
        shape = tf.shape(layer)
        x_ = shape[1] // 2 if x is None else x
        y_ = shape[2] // 2 if y is None else y
        if batch is None:
          raise RuntimeError("requires batch")

        acts = layer[batch, x_, y_]
        vec_ = vec
        if S is not None: vec_ = tf.matmul(vec_[None], S)[0]
        mag = tf.sqrt(tf.reduce_sum(acts**2))
        dot = tf.reduce_mean(acts * vec_)
        cossim = dot/(1e-4 + mag)
        cossim = tf.maximum(0.1, cossim)
        return dot * cossim ** cossim_pow
    return inner

#
# Renders a batch of activations as icons
#
def render_icons(directions, model, layer, size=80, n_steps=128, verbose=False, S=None, num_attempts=2, cossim=True, alpha=True):
    image_attempts = []
    loss_attempts = []

    # Render multiple attempts, and pull the one with the lowest loss score.
    for attempt in range(num_attempts):

      # Render an image for each activation vector
      param_f = lambda: param.image(size, batch=directions.shape[0], fft=True, decorrelate=True, alpha=alpha)
      if(S is not None):
          if(cossim is True):
              obj_list = ([
                direction_neuron_cossim_S(layer, v, batch=n, S=S, cossim_pow=4) for n,v in enumerate(directions)
              ]) 
          else: 
              obj_list = ([
                direction_neuron_S(layer, v, batch=n, S=S) for n,v in enumerate(directions)
              ])    
      else: 
          obj_list = ([
            objectives.direction_neuron(layer, v, batch=n) for n,v in enumerate(directions)
          ])

      obj = objectives.Objective.sum(obj_list)

      transforms = []
      if alpha:
          transforms.append(transform.collapse_alpha_random())
      transforms.append(transform.pad(2, mode='constant', constant_value=1))
      transforms.append(transform.jitter(4))
      transforms.append(transform.jitter(4))
      transforms.append(transform.jitter(8))
      transforms.append(transform.jitter(8))
      transforms.append(transform.jitter(8))
      transforms.append(transform.random_scale([0.995**n for n in range(-5,80)] + [0.998**n for n in 2*list(range(20,40))]))
      transforms.append(transform.random_rotate(list(range(-20,20))+list(range(-10,10))+list(range(-5,5))+5*[0]))
      transforms.append(transform.jitter(2))

      # This is the tensorflow optimization process.
      # We can't use the lucid helpers here because we need to know the loss.

      print("attempt: ", attempt)
      with tf.Graph().as_default(), tf.Session() as sess:
          learning_rate = 0.05
          losses = []
          trainer = tf.train.AdamOptimizer(learning_rate)
          T = render.make_vis_T(model, obj, param_f, trainer, transforms)
          loss_t, vis_op, t_image = T("loss"), T("vis_op"), T("input")
          losses_ = [obj_part(T) for obj_part in obj_list]
          tf.global_variables_initializer().run()
          for i in range(n_steps):
              loss, _ = sess.run([losses_, vis_op])
              losses.append(loss)
              if (i % 100 == 0):
                  print(i)

          img = t_image.eval()
          img_rgb = img[:,:,:,:3]
          if alpha:
              print("alpha true")
              k = 0.8
              bg_color = 0.0
              img_a = img[:,:,:,3:]
              img_merged = img_rgb*((1-k)+k*img_a) + bg_color * k*(1-img_a)
              image_attempts.append(img_merged)
          else:
              print("alpha false")
              image_attempts.append(img_rgb)

          loss_attempts.append(losses[-1])

    # Use the icon with the lowest loss
    loss_attempts = np.asarray(loss_attempts)   
    loss_final = []
    image_final = []
    print("Merging best scores from attempts...")
    for i, d in enumerate(directions):
        # note, this should be max, it is not a traditional loss
        mi = np.argmax(loss_attempts[:,i])
        image_final.append(image_attempts[mi][i])

    return (image_final, loss_final)
  
# GRID
# # 
# Takes a list of x,y layout and bins them into grid cells
# 
def grid(xpts=None, ypts=None, grid_size=(8,8), x_extent=(0., 1.), y_extent=(0., 1.)):
    xpx_length = grid_size[0]
    ypx_length = grid_size[1]

    xpt_extent = x_extent
    ypt_extent = y_extent

    xpt_length = xpt_extent[1] - xpt_extent[0]
    ypt_length = ypt_extent[1] - ypt_extent[0]

    xpxs = ((xpts - xpt_extent[0]) / xpt_length) * xpx_length
    ypxs = ((ypts - ypt_extent[0]) / ypt_length) * ypx_length

    ix_s = range(grid_size[0])
    iy_s = range(grid_size[1])
    xs = []
    for xi in ix_s:
        ys = []
        for yi in iy_s:
            xpx_extent = (xi, (xi + 1))
            ypx_extent = (yi, (yi + 1))

            in_bounds_x = np.logical_and(xpx_extent[0] <= xpxs, xpxs <= xpx_extent[1])
            in_bounds_y = np.logical_and(ypx_extent[0] <= ypxs, ypxs <= ypx_extent[1])
            in_bounds = np.logical_and(in_bounds_x, in_bounds_y)

            in_bounds_indices = np.where(in_bounds)[0]
            ys.append(in_bounds_indices)
        xs.append(ys)
    return np.asarray(xs)
  
def render_layout(model, layer, S, xs, ys, activ, n_steps=512, n_attempts=2, min_density=10, grid_size=(10, 10), icon_size=80, x_extent=(0., 1.0), y_extent=(0., 1.0)):
    grid_layout = grid(xpts=xs, ypts=ys, grid_size=grid_size, x_extent=x_extent, y_extent=y_extent)
    icons = []

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            indices = grid_layout[x, y]
            if len(indices) > min_density:
                average_activation = np.average(activ[indices], axis=0)
                icons.append((average_activation, x, y))

    icons = np.asarray(icons)
    icon_batch, losses = render_icons(icons[:,0], model, alpha=False, layer=layer, S=S, n_steps=n_steps, size=icon_size, num_attempts=n_attempts)

    canvas = np.ones((icon_size * grid_size[0], icon_size * grid_size[1], 3))
    for i, icon in enumerate(icon_batch):
        y = int(icons[i, 1])
        x = int(icons[i, 2])
        canvas[(grid_size[0] - x - 1) * icon_size:(grid_size[0] - x) * icon_size, (y) * icon_size:(y + 1) * icon_size] = icon

    return canvas

# 
# Given a layout, renders an icon for the average of all the activations in each grid cell.
# 

xs = layout[:, 0]
ys = layout[:, 1]
canvas = render_layout(model, layer, S, xs, ys, raw_activations, n_steps=512, grid_size=(20, 20), n_attempts=1)
show(canvas)

with open('html_dump', 'wb') as handle:
	pickle.dump([show(canvas), canvas], handle)