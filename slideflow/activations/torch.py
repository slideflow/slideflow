import torch
import numpy as np
import slideflow as sf
from slideflow.model.torch import ModelParams
from slideflow.util import log

class ActivationsError(Exception):
    pass

class ActivationsInterface:
    """Interface for obtaining logits and intermediate layer activations from Slideflow models.

    Use by calling on either a batch of images (returning outputs for a single batch), or by calling on a
    :class:`slideflow.WSI` object, which will generate an array of spatially-mapped activations matching
    the slide.

    Examples
        *Calling on batch of images:*

        .. code-block:: python

            interface = ActivationsInterface('/model/path', layers='postconv')
            for image_batch in train_data:
                # Return shape: (batch_size, num_features)
                batch_activations = interface(image_batch)

        *Calling on a slide:*

        .. code-block:: python

            slide = sf.slide.WSI(...)
            interface = ActivationsInterface('/model/path', layers='postconv')
            # Return shape: (slide.grid.shape[0], slide.grid.shape[1], num_features):
            activations_grid = interface(slide)

    Note:
        When this interface is called on a batch of images, no image processing or stain normalization will be
        performed, as it is assumed that normalization will occur during data loader image processing.
        When the interface is called on a `slideflow.WSI`, the normalization strategy will be read from the model
        configuration file, and normalization will be performed on image tiles extracted from the WSI. If this interface
        was created from an existing model and there is no model configuration file to read, a
        slideflow.slide.StainNormalizer object may be passed during initialization via the argument `wsi_normalizer`.

    """

    def __init__(self, path, layers='postconv', include_logits=False, mixed_precision=True, device=None):
        """Creates an activations interface from a saved slideflow model which outputs feature activations
        at the designated layers.

        Intermediate layers are returned in the order of layers. Logits are returned last.

        Args:
            path (str): Path to saved Slideflow model.
            layers (list(str), optional): Layers from which to generate activations.  The post-convolution activation layer
                is accessed via 'postconv'. Defaults to 'postconv'.
            include_logits (bool, optional): Include logits in output. Will be returned last. Defaults to False.
            mixed_precision (bool, optional): Use mixed precision. Defaults to True.
            device (:class:`torch.device`, optional): Device for model. Defaults to torch.device('cuda')
        """

        if layers and isinstance(layers, str): layers = [layers]
        self.path = path
        self.num_logits = 0
        self.num_features = 0
        self.mixed_precision = mixed_precision
        self.activation = {}
        self.layers = layers
        self.include_logits = include_logits
        self.device = device if device is not None else torch.device('cuda')

        if path is not None:
            try:
                config = sf.util.get_model_config(path)
            except:
                raise ActivationsError(f"Unable to find configuration for model {path}")

            self.hp = ModelParams()
            self.hp.load_dict(config['hp'])
            self.wsi_normalizer = self.hp.get_normalizer()
            self.tile_px = self.hp.tile_px
            self._model = self.hp.build_model(num_classes=len(config['outcome_labels'])) #labels=
            self._model.load_state_dict(torch.load(path))
            self._model.to(self.device)
            self.model_type = self._model.__class__.__name__
            self._build()
            self._model.eval()

    @classmethod
    def from_model(cls, model, tile_px, layers='postconv', include_logits=False, mixed_precision=True,
                   wsi_normalizer=None, device=None):
        """Creates an activations interface from a loaded slideflow model which outputs feature activations
        at the designated layers.

        Intermediate layers are returned in the order of layers. Logits are returned last.

        Args:
            model (:class:`tensorflow.keras.models.Model`): Loaded model.
            tile_px (int): Width/height of input image size.
            layers (list(str), optional): Layers from which to generate activations.  The post-convolution activation layer
                is accessed via 'postconv'. Defaults to 'postconv'.
            include_logits (bool, optional): Include logits in output. Will be returned last. Defaults to False.
            wsi_normalizer (:class:`slideflow.slide.StainNormalizer`): Stain normalizer to use on whole-slide images.
                Is not used on individual tile datasets via __call__. Defaults to None.
            device (:class:`torch.device`, optional): Device for model. Defaults to torch.device('cuda')
        """

        obj = cls(None, layers, include_logits, mixed_precision, device)
        if isinstance(model, torch.nn.Module):
            obj._model = model.to(obj.device)
            obj._model.eval()
        else:
            raise TypeError("Provided model is not a valid PyTorch model.")
        obj.hp = None
        obj.model_type = obj._model.__class__.__name__
        obj.tile_px = tile_px
        obj.wsi_normalizer = wsi_normalizer
        obj._build()
        return obj

    def __call__(self, inp, **kwargs):
        """Process a given input and return activations and/or logits. Expects either a batch of images or
        a :class:`slideflow.slide.WSI` object."""

        if isinstance(inp, sf.slide.WSI):
            return self._predict_slide(inp, **kwargs)
        else:
            return self._predict(inp)

    def _predict_slide(self, slide, batch_size=128, dtype=np.float16, **kwargs):
        """Generate activations from slide => activation grid array."""
        total_out = self.num_features + self.num_logits
        activations_grid = np.zeros((slide.grid.shape[1], slide.grid.shape[0], total_out), dtype=dtype)
        generator = slide.build_generator(shuffle=False, include_loc='grid', show_progress=True, **kwargs)

        if not generator:
            log.error(f"No tiles extracted from slide {sf.util.green(slide.name)}")
            return

        class SlideIterator(torch.utils.data.IterableDataset):
            def __init__(self, *args, **kwargs):
                super(SlideIterator).__init__(*args, **kwargs)
            def __iter__(self):
                for image_dict in generator():
                    np_image = torch.from_numpy(image_dict['image'])
                    if self.wsi_normalizer:
                        np_image = self.wsi_normalizer.rgb_to_rgb(np_image)
                    np_image = np_image.permute(2, 0, 1) # WHC => CWH
                    loc = np.array(image_dict['loc'])
                    np_image = np_image / 127.5 - 1
                    yield np_image, loc

        tile_dataset = torch.utils.data.DataLoader(SlideIterator(), batch_size=batch_size)

        act_arr = []
        loc_arr = []
        for i, (batch_images, batch_loc) in enumerate(tile_dataset):
            model_out = self._predict(batch_images)
            if not isinstance(model_out, list): model_out = [model_out]
            act_arr += [np.concatenate([m.cpu().detach().numpy() for m in model_out])]
            loc_arr += [batch_loc]

        act_arr = np.concatenate(act_arr)
        loc_arr = np.concatenate(loc_arr)

        for i, act in enumerate(act_arr):
            xi = loc_arr[i][0]
            yi = loc_arr[i][1]
            activations_grid[yi][xi] = act

        return activations_grid

    def _predict(self, inp):
        """Return activations for a single batch of images."""
        with torch.cuda.amp.autocast() if self.mixed_precision else sf.model.no_scope():
            with torch.no_grad():
                logits = self._model(inp.to(self.device))

        layer_activations = []
        if self.layers:
            for l in self.layers:
                act = self.activation[l]
                if l == 'postconv':
                    act = self._postconv_processing(act)
                layer_activations.append(act)

        if self.include_logits:
            layer_activations += [logits]
        self.activation = {}
        return layer_activations

    def _get_postconv(self):
        """Returns post-convolutional layer."""

        if self.model_type == 'ViT':
            return self._model.to_latent
        if self.model_type in ('ResNet', 'Inception3', 'GoogLeNet'):
            return self._model.avgpool
        if self.model_type in ('AlexNet', 'SqueezeNet', 'VGG', 'MobileNetV2', 'MobileNetV3', 'MNASNet'):
            return next(self._model.classifier.children())
        if self.model_type == 'DenseNet':
            return self._model.features.norm5
        if self.model_type == 'ShuffleNetV2':
            return list(self._model.conv5.children())[1]
        if self.model_type == 'Xception':
            return self._model.bn4

    def _postconv_processing(self, output):
        """Applies processing (pooling, resizing) to post-convolutional outputs,
        to convert output to the shape (batch_size, num_features)"""

        def pool(x):
            return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

        def squeeze(x):
            return x.view(x.size(0), -1)

        if self.model_type in ('ViT', 'AlexNet', 'VGG', 'MobileNetV2', 'MobileNetV3', 'MNASNet'):
            return output
        if self.model_type in ('ResNet', 'Inception3', 'GoogLeNet'):
            return squeeze(output)
        if self.model_type in ('SqueezeNet', 'DenseNet', 'ShuffleNetV2', 'Xception'):
            return squeeze(pool(output))

    def _build(self):
        """Builds the interface model that outputs feature activations at the designated layers and/or logits.
            Intermediate layers are returned in the order of layers. Logits are returned last."""

        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook

        if isinstance(self.layers, list):
            for l in self.layers:
                if l == 'postconv':
                    self._get_postconv().register_forward_hook(get_activation('postconv'))
                else:
                    getattr(self._model, l).register_forward_hook(get_activation(l))
        elif self.layers is not None:
            raise TypeError(f"Unrecognized type {type(self.layers)} for self.layers")

        # Calculate output and layer sizes
        rand_data = torch.rand(1, 3, self.tile_px, self.tile_px)
        output = self._model(rand_data.to(self.device))
        self.num_logits = output.shape[1] if self.include_logits else 0
        self.num_features = sum([f.shape[1] for f in self.activation.values()])

        if self.include_logits:
            log.debug(f'Number of logits: {self.num_logits}')
        log.debug(f'Number of activation features: {self.num_features}')