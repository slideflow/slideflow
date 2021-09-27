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


class TileVisualizer:
    '''Class to supervize visualization of node activations across an image tile.
    Visualization is accomplished by performing sequential convolutional masking
        and determining impact of masking on node activation. In this way,
        the masking reveals spatial importance with respect to activation of the given node.
    '''

    def __init__(self,
                 model,
                 node,
                 tile_px,
                 mask_width=None,
                 normalizer=None,
                 normalizer_source=None):

        '''Object initializer.

        Args:
            model:              Path to Tensorflow model
            node:               Int, activation node to analyze
            tile_px:            Int, width/height of image tiles
            mask_width:         Width of mask to convolutionally apply. Defaults to 1/6 of tile_px
            normalizer:         String, normalizer to apply to tiles in real-time.
            normalizer_source:  Path to normalizer source image.
        '''
        self.NODE = node
        self.IMAGE_SHAPE = (tile_px, tile_px, 3)
        self.MASK_WIDTH = mask_width if mask_width else int(self.IMAGE_SHAPE[0]/6)
        self.normalizer = None if not normalizer else sf.util.StainNormalizer(method=normalizer, source=normalizer_source)

        log.info('Initializing tile visualizer')
        log.info(f'Node: {sf.util.bold(str(node))} | Shape: ({self.IMAGE_SHAPE}) | Window size: {self.MASK_WIDTH}')
        log.info(f'Loading Tensorflow model at {sf.util.green(model)}...')

        self.interface = ActivationsInterface(model)

    def _calculate_activation_map(self, stride_div=4):
        '''Creates map of importance through convolutional masking and
        examining changes in node activations.'''
        sx = self.IMAGE_SHAPE[0]
        sy = self.IMAGE_SHAPE[1]
        w  = self.MASK_WIDTH
        stride = int(self.MASK_WIDTH / stride_div)
        min_x  = int(w/2)
        max_x  = int(sx - w/2)
        min_y  = int(w/2)
        max_y  = int(sy - w/2)

        act_array = []
        for yi in range(min_y, max_y, stride):
            for xi in range(min_x, max_x, stride):
                mask = self._create_bool_mask(xi, yi, w, sx, sy)
                masked = self.tf_processed_image.numpy() * mask
                act, _ = self.interface(np.array([masked]))
                act_array += [act[0][self.NODE]]
                print(f'Calculating activations at x:{xi}, y:{yi}; act={act[0][self.NODE]}', end='\033[K\r')
        max_center_x = max(range(min_x, max_x, stride))
        max_center_y = max(range(min_y, max_y, stride))
        reshaped_array = np.reshape(np.array(act_array), [len(range(min_x, max_x, stride)),
                                                          len(range(min_y, max_y, stride))])
        print()
        return reshaped_array, max_center_x, max_center_y

    def _create_bool_mask(self, x, y, w, sx, sy):
        l = max(0,  int(x-(w/2.)))
        r = min(sx, int(x+(w/2.)))
        t = max(0,  int(y-(w/2.)))
        b = min(sy, int(y+(w/2.)))
        m = np.array([[[True]*3]*sx]*sy)
        for yi in range(m.shape[1]):
            for xi in range(m.shape[0]):
                if (t < yi < b) and (l < xi < r):
                    m[yi][xi] = [False, False, False]
        return m

    def _predict_masked(self, x, y, index):
        mask = self._create_bool_mask(x, y, self.MASK_WIDTH, self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1])
        masked = self.tf_processed_image.numpy() * mask
        act, _ = self.loaded_model.predict(np.array([masked]))
        return act[0][index]

    def visualize_tile(self,
                       tfrecord=None,
                       index=None,
                       image_jpg=None,
                       export_folder=None,
                       zoomed=True,
                       interactive=False):

        '''Visualizes tiles, either interactively or saving to directory.

        Args:
            tfrecord:           If provided, will visualize tile from the designated tfrecord.
                                    Must supply either a tfrecord and index or image_jpg
            index:              Index of tile to visualize within tfrecord, if provided
            image_jpeg:         JPG image to perform analysis on
            export_folder:      Folder in which to save heatmap visualization
            zoomed:             Bool. If true, will crop image to space containing heatmap
                                    (otherwise a small border will be seen)
            interactive:        If true, will display as interactive map using matplotlib
        '''
        if not (image_jpg or tfrecord):
            raise ActivationsError('Must supply either tfrecord or image_jpg')

        if image_jpg:
            log.info(f'Processing tile at {sf.util.green(image_jpg)}...')
            tilename = sf.util.path_to_name(image_jpg)
            self.tile_image = Image.open(image_jpg)
            image_file = open(image_jpg, 'rb')
            tf_decoded_image = tf.image.decode_png(image_file.read(), channels=3)
        else:
            slide, tf_decoded_image = sf.io.tfrecords.get_tfrecord_by_index(tfrecord, index, decode=True)
            tilename = f"{slide.numpy().decode('utf-8')}-{index}"
            self.tile_image = Image.fromarray(tf_decoded_image.numpy())

        # Normalize PIL image & TF image
        if self.normalizer:
            self.tile_image = self.normalizer.pil_to_pil(self.tile_image)
            tf_decoded_image = tf.py_function(self.normalizer.tf_to_rgb, [self.tile_image], tf.int32)

        # Next, process image with Tensorflow
        self.tf_processed_image = tf.image.per_image_standardization(tf_decoded_image)
        self.tf_processed_image = tf.image.convert_image_dtype(self.tf_processed_image, tf.float16)
        self.tf_processed_image.set_shape(self.IMAGE_SHAPE)

        # Now create the figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.implot = plt.imshow(self.tile_image)

        if interactive:
            self.rect = patches.Rectangle((0, 0), self.MASK_WIDTH, self.MASK_WIDTH, facecolor='white', zorder=20)
            self.ax.add_patch(self.rect)

        activation_map, max_center_x, max_center_y = self._calculate_activation_map()

        # Prepare figure
        filename = join(export_folder, f'{tilename}-heatmap.png')

        def hover(event):
            if event.xdata:
                self.rect.set_xy((event.xdata-self.MASK_WIDTH/2, event.ydata-self.MASK_WIDTH/2))
                print(self._predict_masked(event.xdata, event.ydata, index=self.NODE), end='\r')
                self.fig.canvas.draw_idle()

        def click(event):
            if event.button == 1:
                self.MASK_WIDTH = min(min(self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1]), self.MASK_WIDTH + 25)
                self.rect.set_width(self.MASK_WIDTH)
                self.rect.set_height(self.MASK_WIDTH)
            else:
                self.MASK_WIDTH = max(0, self.MASK_WIDTH - 25)
                self.rect.set_width(self.MASK_WIDTH)
                self.rect.set_height(self.MASK_WIDTH)
            self.fig.canvas.draw_idle()

        if interactive:
            self.fig.canvas.mpl_connect('motion_notify_event', hover)
            self.fig.canvas.mpl_connect('button_press_event', click)

        if activation_map is not None:
            # Calculate boundaries of heatmap
            hw = int(self.MASK_WIDTH/2)
            if zoomed:
                extent = (hw, max_center_x, max_center_y, hw)
            else:
                extent = (0, max_center_x+hw, max_center_y+hw, 0)

            # Heatmap
            divnorm = mcol.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1.0)
            self.ax.imshow(activation_map,
                           extent=extent,
                           cmap='coolwarm',
                           norm=divnorm,
                           alpha=0.6 if not interactive else 0.0,
                           interpolation='bicubic',
                           zorder=10)
        if filename:
            plt.savefig(filename, bbox_inches='tight')
            log.info(f'Heatmap saved to {filename}')
        if interactive:
            plt.show()

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

def neighbors(AV, n_AV, neighbor_slides, n_neighbors=5, algorithm='ball_tree'):
    """Finds neighboring tiles for a given ActivationsVisualizer and list of slides.
    WARNING: not confirmed to be working after a refactor. In need of further testing.

    Args:
        n_AV (:class:`slideflow.activations.ActivationsVisualizer`): Neighboring ActivationsVisualizer.
            Search neighboring activations in ActivationsVisualizer for neighbors.
        neighbor_slides (list(str)): Either a single slide name or a list of slide names.
            Corresponds to slides in the provided neighboring AV. Look for neighbors to all tiles in these slides.
        n_neighbors (int, optional): Number of neighbors to find for each tile. Defaults to 5.
        algorithm (str, optional): NearestNeighbors algorithm, either 'kd_tree', 'ball_tree', or 'brute'.
            Defaults to 'ball_tree'.

    Returns:
        dict: Dict mapping slide names to tile indices for tiles found to be neighbors
            to the provided n_AV and neighbor_slides.
    """

    if not isinstance(neighbor_slides, list): neighbor_slides = [neighbor_slides]
    if not all(slide in n_AV.activations for slide in neighbor_slides):
        raise ActivationsError(f'Not all neighbor slides exist in neighboring activations.')

    tiles = [(slide, ti) for slide in AV.activations for ti in range(len(AV.activations[slide]))]
    X = np.stack([AV.activations[slide][ti] for slide, ti in tiles]) #stack -> array
    neighbor_X = np.concatenate([n_AV.activations[slide] for slide in neighbor_slides])

    log.info('Searching for nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, n_jobs=-1).fit(X)
    _, neighbor_idx = nbrs.kneighbors(neighbor_X)

    neighbors = defaultdict(list)
    for nn_idx in neighbor_idx.flatten():
        slide, tile_idx = tiles[nn_idx]
        neighbors[slide].append(tile_idx)

    return neighbors

class RNASeqModel(sf.model.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.RNA_SEQ_TABLE = {self.slides[i]:outcome_labels[i] for i in range(len(self.slides))}
    def _build_model(self, *args, **kwargs):
        # In the hidden layers section:
        '''merged_model = tf.keras.layers.Dense(512, name=f'hidden_0', activation=None, kernel_regularizer=regularizer)(merged_model)
        merged_model = tf.keras.layers.LeakyReLU()(merged_model)
        merged_model = tf.keras.layers.Dense(128, name=f'hidden_1', activation=None, kernel_regularizer=regularizer)(merged_model)
        merged_model = tf.keras.layers.LeakyReLU()(merged_model)
        merged_model = tf.keras.layers.Dense(32, name=f'autoencoder', activation='tanh', kernel_regularizer=regularizer)(merged_model)
        merged_model = tf.keras.layers.Dense(64, name=f'reencode_0', activation=None, kernel_regularizer=regularizer)(merged_model)
        merged_model = tf.keras.layers.LeakyReLU()(merged_model)
        merged_model = tf.keras.layers.Dense(128, name=f'reencode_1', activation=None, kernel_regularizer=regularizer)(merged_model)
        merged_model = tf.keras.layers.LeakyReLU()(merged_model)
        merged_model = tf.keras.layers.Dense(256, name=f'reencode_2', activation=None, kernel_regularizer=regularizer)(merged_model)
        merged_model = tf.keras.layers.LeakyReLU()(merged_model)
        merged_model = tf.keras.layers.Dense(512, name=f'reencode_3', activation=None, kernel_regularizer=regularizer)(merged_model)
        merged_model = tf.keras.layers.LeakyReLU()(merged_model)
        merged_model = tf.keras.layers.Dense(1024, name=f'reencode_4', activation=None, kernel_regularizer=regularizer)(merged_model)
        merged_model = tf.keras.layers.LeakyReLU()(merged_model)
        merged_model = tf.keras.layers.Dense(2048, name=f'reencode_5', activation=None, kernel_regularizer=regularizer)(merged_model)
        merged_model = tf.keras.layers.LeakyReLU()(merged_model)'''
        #final_dense_layer = tf.keras.layers.Dropout(0.2)(final_dense_layer) # include for rna seq
        pass
    def _parse_tfrecord_labels(self, *args, **kwargs):
        # === RNA SEQ ==========
        #def rna_seq_lookup(s): return self.RNA_SEQ_TABLE[s.numpy().decode('utf-8')]

        #label = tf.py_function(func=rna_seq_lookup,
        #                        inp=[slide],
        #                        Tout=tf.float32)
        # ====================
        pass