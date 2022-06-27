from typing import (TYPE_CHECKING, Any, Callable, Generator, List, Optional,
                    Tuple, Union)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import slideflow as sf
import tensorflow as tf
import torch
from PIL import Image
from slideflow.gan import tf_utils
from slideflow.gan.search import seed_search
from slideflow.gan.stylegan2 import embedding, utils
from tqdm import tqdm


class StyleGAN2Interpolator:

    def __init__(
        self,
        gan_pkl: str,
        start: int,
        end: int,
        device: torch.device,
        **gan_kwargs
    ) -> None:
        """Coordinates class and embedding interpolation for a trained
        class-conditional StyleGAN2.

        Args:
            gan_pkl (str): Path to saved network pkl.
            start (int): Starting class index.
            end (int): Ending class index.
            device (torch.device): Torch device.
        """
        self.E_G, self.G = embedding.load_embedding_gan(gan_pkl, device)
        self.device = device
        self.gan_kwargs = gan_kwargs
        self.decode_kwargs = dict(standardize=False, resize_px=299)
        self.embed0, self.embed1 = embedding.get_class_embeddings(
            self.G,
            start=start,
            end=end,
            device=device
        )
        self.features = None
        self.normalizer = None

    def z(self, seed: int) -> torch.tensor:
        """Returns a noise tensor for a given seed.

        Args:
            seed (int): Seed.

        Returns:
            torch.tensor: Noise tensor for the corresponding seed.
        """
        return utils.noise_tensor(seed, self.E_G.z_dim).to(self.device)

    def set_feature_model(
        self,
        path: str,
        layers: Union[str, List[str]] = 'postconv'
    ) -> None:
        """Configures a classifier model to be used for generating features
        and predictions during interpolation.

        Args:
            path (str): Path to trained model.
            layers (Union[str, List[str]], optional): Layers from which to
                calculate activations for interpolated images.
                Defaults to 'postconv'.
        """
        self.features = sf.model.Features(path, layers=layers, include_logits=True)
        self.normalizer = self.features.wsi_normalizer

    def seed_search(
        self,
        seeds: List[int],
        batch_size: int = 32,
        verbose: bool = False
    ) -> pd.core.frame.DataFrame:
        """Generates images for starting and ending classes for many seeds,
        calculating layer activations from a set classifier.

        Args:
            seeds (List[int]): Seeds.
            batch_size (int, optional): Batch size for GAN during generation.
                Defaults to 32.
            verbose (bool, optional): Verbose output. Defaults to False.

        Raises:
            Exception: If classifier model has not been been set with
                .set_feature_model()

        Returns:
            pd.core.frame.DataFrame: Dataframe of results.
        """

        if self.features is None:
            raise Exception("Feature model not set; use .set_feature_model()")
        return seed_search(
            seeds,
            self.embed0,
            self.embed1,
            self.E_G,
            self.features,
            self.device,
            batch_size,
            normalizer=self.normalizer,
            verbose=verbose,
            **self.gan_kwargs
        )

    def plot_comparison(self, seeds: Union[int, List[int]]) -> None:
        """Plots side-by-side comparison of images from the starting
        and ending interpolation classes.

        Args:
            seeds (int or list(int)): Seeds to display.
        """
        if not isinstance(seeds, list):
            seeds = [seeds]

        scale = 5
        fig, ax = plt.subplots(len(seeds), 2, figsize=(2 * scale, len(seeds) * scale))
        for s, seed in enumerate(seeds):
            # First image (starting embedding, BRAF-like)
            img0 = self.E_G(self.z(seed), self.embed0, **self.gan_kwargs)
            img0 = tf_utils.process_gan_batch(img0)
            img0 = tf_utils.decode_batch(img0, **self.decode_kwargs)
            img0 = Image.fromarray(img0['tile_image'].numpy()[0])

            # Second image (ending embedding, RAS-like)
            img1 = self.E_G(self.z(seed), self.embed1, **self.gan_kwargs)
            img1 = tf_utils.process_gan_batch(img1)
            img1 = tf_utils.decode_batch(img1, **self.decode_kwargs)
            img1 = Image.fromarray(img1['tile_image'].numpy()[0])

            if len(seeds) == 1:
                _ax0 = ax[0]
                _ax1 = ax[1]
            else:
                _ax0 = ax[s, 0]
                _ax1 = ax[s, 1]
            if s == 0:
                _ax0.set_title('BRAF-like')
                _ax1.set_title('RAS-like')
            _ax0.imshow(img0)
            _ax1.imshow(img1)
            _ax0.axis('off')
            _ax1.axis('off')

        fig.subplots_adjust(wspace=0.05, hspace=0)

    def generate_tf_from_embedding(
        self,
        seed: int,
        embedding: torch.tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create a processed Tensorflow image from the GAN output from a given
        seed and embedding.

        Args:
            seed (int): Seed for noise vector.
            embedding (torch.tensor): Class embedding.

        Returns:
            tf.Tensor: Unprocessed image (tf.Tensor), uint8.

            tf.Tensor: Processed image (tf.Tensor), standardized and normalized.
        """
        z = self.z(seed)
        gan_out = self.E_G(z, embedding, **self.gan_kwargs)
        raw, processed = tf_utils.process_gan_raw(
            gan_out,
            normalizer=self.normalizer,
            **self.decode_kwargs
        )
        return raw, processed

    def generate_tf_start(self, seed: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create a processed Tensorflow image from the GAN output of a given
        seed and the starting class embedding.

        Args:
            seed (int): Seed for noise vector.

        Returns:
            tf.Tensor: Unprocessed image (tf.Tensor), uint8.

            tf.Tensor: Processed image (tf.Tensor), standardized and normalized.
        """
        return self.generate_tf_from_embedding(seed, self.embed0)

    def generate_tf_end(self, seed: int) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create a processed Tensorflow image from the GAN output of a given
        seed and the ending class embedding.

        Args:
            seed (int): Seed for noise vector.

        Returns:
            tf.Tensor: Unprocessed image (tf.Tensor), uint8.

            tf.Tensor: Processed image (tf.Tensor), standardized and normalized.
        """
        return self.generate_tf_from_embedding(seed, self.embed1)

    def class_interpolate(self, seed: int, steps: int = 100) -> Generator:
        """Sets up a generator that returns images during class embedding
        interpolation.

        Args:
            seed (int): Seed for random noise vector.
            steps (int, optional): Number of steps for interpolation.
                Defaults to 100.

        Returns:
            Generator: Generator which yields images (torch.tensor, uint8)
                during interpolation.

        Yields:
            Generator: images (torch.tensor, dtype=uint8)
        """
        return embedding.class_interpolate(
            self.E_G,
            self.z(seed),
            self.embed0,
            self.embed1,
            device=self.device,
            steps=steps,
            **self.gan_kwargs
        )

    def linear_interpolate(self, seed: int, steps: int = 100) -> Generator:
        """Sets up a generator that returns images during linear label
        interpolation.

        Args:
            seed (int): Seed for random noise vector.
            steps (int, optional): Number of steps for interpolation.
                Defaults to 100.

        Returns:
            Generator: Generator which yields images (torch.tensor, uint8)
                during interpolation.

        Yields:
            Generator: images (torch.tensor, dtype=uint8)
        """
        return embedding.linear_interpolate(
            self.G,
            self.z(seed),
            device=self.device,
            steps=steps,
            **self.gan_kwargs
        )

    def interpolate(
        self,
        seed: int,
        steps: int = 100,
        watch: Callable = None
    ) -> Tuple[List, ...]:
        """Interpolates between starting and ending classes for a seed,
        recording raw images, processed images, predictions, and optionally
        calling a function on each image during interpolation and recording
        results.

        Args:
            seed (int): Seed for random noise vector.
            steps (int, optional): Number of steps during interpolation.
                Defaults to 100.
            watch (Callable, optional): Function to call on every processed
                image during interpolation. If provided, will provide list of
                results as last returned item. Defaults to None.

        Returns:
            Tuple[List, ...]: List of raw images, processed images, predictions,
                and optionally a list of results from the watch function called
                on each processed image.
        """
        imgs = []
        proc_imgs = []
        preds = []
        watch_out = []

        for img in tqdm(self.class_interpolate(seed, steps), total=steps):
            img = torch.from_numpy(np.expand_dims(img, axis=0)).permute(0, 3, 1, 2)
            img = (img / 127.5) - 1
            img = tf_utils.process_gan_batch(img)
            img = tf_utils.decode_batch(img, **self.decode_kwargs)
            if self.normalizer:
                img = self.normalizer.batch_to_batch(img['tile_image'])[0]
            else:
                img = img['tile_image']
            processed_img = tf.image.per_image_standardization(img)
            img = img.numpy()[0]
            pred = self.features(processed_img)[-1].numpy()
            preds += [pred[0][0]]
            if watch is not None:
                watch_out += [watch(processed_img)]
            imgs += [img]
            proc_imgs += [processed_img[0]]

        sns.lineplot(x=range(len(preds)), y=preds, label="Prediction")
        plt.axhline(y=0, color='black', linestyle='--')
        plt.title("Prediction during interpolation")
        plt.xlabel("Interpolation Step (BRAF -> RAS)")
        plt.ylabel("Prediction")

        if watch is not None:
            return imgs, proc_imgs, preds, watch_out
        else:
            return imgs, proc_imgs, preds
