import torch
import numpy as np
import slideflow as sf
from typing import Union, Optional, Callable

from .utils import topleft_pad, make_tiles, average_tiles

# -----------------------------------------------------------------------------

try:
    import pytorch_lightning as pl
except ImportError:
    raise ImportError("pytorch_lightning is required for training segmentation models. "
                        "This library can be installed with `pip install pytorch-lightning`.")
try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError("segmentation_models_pytorch is required for training segmentation models. "
                      "This library can be installed with `pip install segmentation-models-pytorch`.")

# -----------------------------------------------------------------------------

class SegmentModel(pl.LightningModule):

    losses = {
        'dice': smp.losses.DiceLoss,
        'jaccard': smp.losses.JaccardLoss,
        'focal': smp.losses.FocalLoss,
        'tversky': smp.losses.TverskyLoss,
        'lovasz': smp.losses.LovaszLoss,
        'bce': smp.losses.SoftBCEWithLogitsLoss,
        'ce': smp.losses.SoftCrossEntropyLoss,
        'mcc': smp.losses.MCCLoss,
    }

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        in_channels: int,
        out_classes: int,
        *,
        mpp: Optional[float] = None,
        lr: float = 1e-4,
        loss: Union[str, Callable] = 'dice',
        mode: str = 'binary',
        **kwargs
    ):
        super().__init__()

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs
        )
        self.mpp = mpp
        self.lr = lr
        self.out_classes = out_classes

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.mode = mode
        self.loss_fn = self.get_loss_fn(loss, mode)
        self.outputs = {
            'train': [],
            'valid': []
        }

    @staticmethod
    def get_loss_fn(loss: Union[str, Callable], mode: str) -> Callable:
        if not isinstance(loss, str):
            return loss
        if loss in SegmentModel.losses:
            loss_fn = SegmentModel.losses[loss]
        else:
            raise ValueError("Invalid loss: {}".format(loss))

        if loss in ('bce', 'ce', 'mcc'):
            if mode and mode != 'binary':
                raise ValueError("Invalid loss mode for loss {!r}: Expected 'binary', got: {!r}".format(loss, mode))
            return loss_fn()
        else:
            return loss_fn(mode=SegmentModel.get_loss_mode(mode))

    @staticmethod
    def get_loss_mode(mode):
        if mode == 'binary':
            return smp.losses.BINARY_MODE
        elif mode == 'multiclass':
            return smp.losses.MULTICLASS_MODE
        elif mode == 'multilabel':
            return smp.losses.MULTILABEL_MODE
        else:
            raise ValueError("Invalid loss mode: {}".format(mode))

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        if self.mode == 'binary':
            # Shape of the mask should be [batch_size, num_classes, height, width]
            # for binary segmentation num_classes = 1
            assert mask.ndim == 4
        elif self.mode == 'multiclass':
            # Shape of the mask should be [batch_size, height, width]
            # for multiclass segmentation, values are the classes.
            assert mask.ndim == 3
        elif self.mode == 'multilabel':
            # Shape of the mask should be [batch_size, num_classes, height, width]
            assert mask.ndim == 4

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        if self.mode == 'multiclass':
            prob_mask = torch.softmax(logits_mask, dim=1)
            pred_mask = torch.argmax(prob_mask, dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(),
                mask.long(),
                mode=self.mode,
                num_classes=self.out_classes
            )
        else:
            # Lets compute metrics for some threshold
            # first convert mask values to probabilities, then
            # apply thresholding
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_mask.long(),
                mask.long(),
                mode=self.mode
            )

        output = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        self.outputs[stage].append(output)
        return output

    def shared_epoch_end(self, stage):
        outputs = self.outputs[stage]

        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou.to(self.device).float(),
            f"{stage}_dataset_iou": dataset_iou.to(self.device).float(),
        }

        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.outputs[stage].clear()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def run_tiled_inference(self, img: np.ndarray):
        """Run inference on an image, with tiling."""

        # Pad to at least the target size.
        if img.shape[-1] == 4:
            img = img[..., :3]
        orig_dims = img.shape
        img = topleft_pad(img, 1024).transpose(2, 0, 1)

        # Tile the thumbnail.
        tiles, ysub, xsub, ly, lx = make_tiles(img, 1024)
        batched_tiles = tiles.reshape(-1, 3, 1024, 1024)

        # Generate UNet predictions.
        with torch.no_grad():
            tile_preds = []
            for tile in batched_tiles:
                pred = self.forward(torch.from_numpy(tile).unsqueeze(0).to(self.device))
                tile_preds.append(pred)
            tile_preds = torch.cat(tile_preds)

        # Merge predictions across the tiles.
        tiled_preds = average_tiles(tile_preds.cpu().numpy(), ysub, xsub, ly, lx)

        # Crop predictions to the original size.
        tiled_preds = tiled_preds[:orig_dims[0], :orig_dims[1]]

        # Softmax, if multiclass.
        if self.mode == 'binary':
            tiled_preds = tiled_preds[0]
        elif self.mode == 'multiclass':
            tiled_preds = torch.from_numpy(tiled_preds).softmax(dim=0).numpy()

        return tiled_preds

    def run_slide_inference(self, slide: Union[str, "sf.WSI"], mpp=None):
        """Run model inference on a slide thumbnail."""

        # Validation
        if self.mpp is None:
            raise ValueError("Must specify mpp when running inference on a slide. "
                             "This can be done by setting the model .mpp parameter, "
                             "or by passing an mpp value to this function.")
        elif mpp is not None and mpp != self.mpp:
            sf.log.warning("Overriding model mpp with mpp parameter.")
        else:
            mpp = self.mpp
        if not isinstance(slide, (str, sf.WSI)):
            raise TypeError("slide must be a string or sf.WSI object.")

        # Load the slide.
        if isinstance(slide, str):
            slide = sf.WSI(slide, 299, 512)

        # Get the slide thumbnail.
        thumb = np.array(slide.thumb(mpp=mpp))

        # Return predictions.
        return self.run_tiled_inference(thumb)


# -----------------------------------------------------------------------------
