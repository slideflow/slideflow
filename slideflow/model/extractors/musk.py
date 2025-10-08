import timm
import torch
from torchvision import transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.transforms import InterpolationMode

from ._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------

class MUSKFeatures(TorchFeatureExtractor):
    """MUSK pretrained feature extractor.

    MUSK (Multi-Scale Unmasked Pathology Transformer) is a large-scale vision
    transformer pretrained on 1.5M pathology image patches from over 5,000
    whole-slide images across 20 cancer types. The model uses a ViT-Large
    architecture with patch size 16x16 and is trained to handle multi-scale
    inputs at 384x384 pixel resolution.

    Feature dimensions: 1024 (ms_aug=False) or 2048 (ms_aug=True)

    The model is distributed on Hugging Face at hf-hub:xiangjx/musk and requires
    installation of the MUSK package.

    Installation:
        pip install fairscale git+https://github.com/lilab-stanford/MUSK

    Reference:
        Xiang, J., et al. (2024). "MUSK: Multi-Scale Unmasked Pathology Transformer"

    Hugging Face: https://huggingface.co/xiangjx/musk
    GitHub: https://github.com/lilab-stanford/MUSK
    """

    tag = 'musk'
    license = """Please check the original license at https://huggingface.co/xiangjx/musk"""
    citation = """
@article{xiang2024musk,
  title={MUSK: Multi-Scale Unmasked Pathology Transformer},
  author={Xiang, Jiawen and others},
  journal={arXiv preprint},
  year={2024}
}
"""

    def __init__(self, device=None, ms_aug=True, with_head=False, out_norm=True, **kwargs):
        super().__init__(**kwargs)

        # Disable force_uint8 for MUSK to handle TFRecord float32 inputs
        self.force_uint8 = False

        from slideflow.model import torch_utils

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
                print("⚠️  CUDA not available, using CPU")

        self.device = torch_utils.get_device(device)
        self.ms_aug = ms_aug  # Multi-scale augmentation
        self.with_head = with_head  # Use retrieval head (for zero-shot tasks)
        self.out_norm = out_norm  # Output normalization

        # Import MUSK dependencies
        try:
            from musk import utils, modeling
        except ImportError:
            raise ImportError(
                "MUSK package not found. Please install with: "
                "pip install fairscale git+https://github.com/lilab-stanford/MUSK"
            )

        # Load model
        try:
            self.model = timm.create_model("musk_large_patch16_384")
            utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", self.model, 'model|module', '')
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MUSK model. Make sure you have access to the model "
                f"and have registered your Hugging Face token. Error: {e}"
            )

        # Use float16 for GPU, float32 for CPU
        if self.device.type == 'cuda':
            self.model.to(self.device, dtype=torch.float16)
        else:
            self.model.to(self.device, dtype=torch.float32)
            print("ℹ️  Using float32 precision for CPU")
        self.model.eval()

        # ---------------------------------------------------------------------
        # Feature dimensions depend on ms_aug: 1024 (False) or 2048 (True)
        self.num_features = 2048 if ms_aug else 1024

        # MUSK uses 384x384 input size with IMAGENET_INCEPTION normalization
        self.transform = self.build_transform(
            img_size=384,
            interpolation='bicubic',
            center_crop=True,
            norm_mean=IMAGENET_INCEPTION_MEAN,
            norm_std=IMAGENET_INCEPTION_STD
        )
        self.preprocess_kwargs = dict(standardize=False)

    def _process_output(self, output):
        """Process MUSK model output."""
        # MUSK forward pass returns (vision_cls, text_cls). We only need vision_cls.
        if isinstance(output, tuple):
            output = output[0]
        return output.to(torch.float32)

    def __call__(self, obj, **kwargs):
        """Generate features for a batch of images or a WSI."""
        # Handle data type conversion for TFRecord compatibility
        if hasattr(obj, 'dtype') and hasattr(obj, 'to'):
            # MUSK can handle float32 inputs directly
            # Move to appropriate device with correct dtype for model
            if self.device.type == 'cuda':
                obj = obj.to(self.device, dtype=torch.float16)
            else:
                obj = obj.to(self.device, dtype=torch.float32)

        # Call parent method
        return super().__call__(obj, **kwargs)

    def forward(self, x):
        """Forward pass through MUSK model."""
        return self.model(
            image=x,
            with_head=self.with_head,
            out_norm=self.out_norm,
            ms_aug=self.ms_aug
        )[0]  # Forward pass yields (vision_cls, text_cls). We only need vision_cls.

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.
        """
        return self._dump_config(
            class_name='slideflow.model.extractors.musk.MUSKFeatures',
            ms_aug=self.ms_aug,
            with_head=self.with_head,
            out_norm=self.out_norm
        )